"""Reinforcement-learning signal generator using stable-baselines3 PPO.

Provides a custom Gymnasium environment (``OptionsTradeEnv``) that frames
options-signal generation as a sequential decision problem.  The PPO agent
observes feature vectors and outputs discrete signal actions; rewards are
risk-adjusted PnL.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from hedgefund.signals.base import SignalGenerator
from hedgefund.types import (
    MarketRegime,
    SentimentResult,
    SignalAction,
    SignalDirection,
    TradeSignal,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Action space mapping (discrete)
# ---------------------------------------------------------------------------
# 0 -> NO_TRADE, 1 -> BUY_CALL, 2 -> BUY_PUT, 3 -> SELL_CALL, 4 -> SELL_PUT
_ACTION_MAP: dict[int, tuple[SignalAction, SignalDirection]] = {
    0: (SignalAction.NO_TRADE, SignalDirection.NEUTRAL),
    1: (SignalAction.BUY_CALL, SignalDirection.LONG),
    2: (SignalAction.BUY_PUT, SignalDirection.SHORT),
    3: (SignalAction.SELL_CALL, SignalDirection.SHORT),
    4: (SignalAction.SELL_PUT, SignalDirection.LONG),
}

_NUM_ACTIONS = len(_ACTION_MAP)

# Default observation features the environment exposes to the agent.
_DEFAULT_OBS_FEATURES: list[str] = [
    "close",
    "volume",
    "rsi",
    "macd",
    "macd_hist",
    "atr",
    "iv_rank",
    "delta",
    "gamma",
    "theta",
    "vega",
    "put_call_ratio",
    "sentiment_score",
]


# ---------------------------------------------------------------------------
# Custom Gymnasium environment
# ---------------------------------------------------------------------------

def _make_env_class() -> type:
    """Lazily construct the Gym environment class to avoid hard import."""
    import gymnasium as gym  # type: ignore[import-untyped]
    from gymnasium import spaces  # type: ignore[import-untyped]

    class OptionsTradeEnv(gym.Env):
        """Gymnasium environment for options signal generation.

        **State**: normalised feature vector from the DataFrame.
        **Action**: discrete choice from ``_ACTION_MAP``.
        **Reward**: risk-adjusted PnL computed as ``direction * return / volatility``.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            df: pd.DataFrame,
            feature_columns: list[str],
            *,
            transaction_cost: float = 0.001,
            sharpe_window: int = 20,
        ) -> None:
            super().__init__()
            self._feature_cols = [c for c in feature_columns if c in df.columns]
            if not self._feature_cols:
                raise ValueError("No matching feature columns")

            self._df = df.reset_index(drop=True)
            self._n = len(self._df)
            self._tx_cost = transaction_cost
            self._sharpe_window = sharpe_window

            self.action_space = spaces.Discrete(_NUM_ACTIONS)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._feature_cols),),
                dtype=np.float32,
            )

            self._step_idx = 0
            self._returns: list[float] = []

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            self._step_idx = 1  # need at least one prior bar
            self._returns = []
            obs = self._get_obs()
            return obs, {}

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            signal_action, direction = _ACTION_MAP.get(
                action, (SignalAction.NO_TRADE, SignalDirection.NEUTRAL)
            )

            # Compute single-step return
            prev_close = float(self._df.iloc[self._step_idx - 1]["close"])
            curr_close = float(self._df.iloc[self._step_idx]["close"])
            raw_return = (curr_close - prev_close) / prev_close if prev_close else 0.0

            # Apply direction
            if direction == SignalDirection.LONG:
                pnl = raw_return - self._tx_cost
            elif direction == SignalDirection.SHORT:
                pnl = -raw_return - self._tx_cost
            else:
                pnl = 0.0

            self._returns.append(pnl)

            # Risk-adjusted reward (rolling Sharpe-like)
            reward = self._risk_adjusted_reward(pnl)

            self._step_idx += 1
            terminated = self._step_idx >= self._n
            truncated = False

            obs = self._get_obs() if not terminated else np.zeros(
                len(self._feature_cols), dtype=np.float32
            )

            info = {
                "pnl": pnl,
                "action": signal_action.value,
                "direction": direction.value,
            }
            return obs, reward, terminated, truncated, info

        def _get_obs(self) -> np.ndarray:
            row = self._df.iloc[self._step_idx][self._feature_cols]
            return np.array(row.fillna(0).values, dtype=np.float32)

        def _risk_adjusted_reward(self, pnl: float) -> float:
            if len(self._returns) < 2:
                return pnl
            window = self._returns[-self._sharpe_window:]
            mean_r = np.mean(window)
            std_r = np.std(window)
            if std_r < 1e-8:
                return pnl
            return float(mean_r / std_r)

    return OptionsTradeEnv


# ---------------------------------------------------------------------------
# RL Signal Generator
# ---------------------------------------------------------------------------

class RLSignalGenerator(SignalGenerator):
    """PPO-based signal generator.

    Parameters:
        model_path: Path to a saved ``stable_baselines3`` PPO model.
        feature_columns: Features the environment observes.
        min_confidence: Minimum action probability to emit a signal.
        atr_stop_mult: ATR multiplier for stop-loss placement.
        min_rr: Minimum risk/reward ratio.
        strategy_name: Name for generated signals.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        feature_columns: list[str] | None = None,
        min_confidence: float = 0.50,
        atr_stop_mult: float = 1.5,
        min_rr: float = 2.0,
        strategy_name: str = "rl_ppo",
    ) -> None:
        self._model: Any | None = None
        self._model_path = Path(model_path) if model_path else None
        self._features = feature_columns or _DEFAULT_OBS_FEATURES
        self._min_conf = min_confidence
        self._atr_mult = atr_stop_mult
        self._min_rr = min_rr
        self._name = strategy_name

    # -- model lifecycle ----------------------------------------------------

    def load_model(self, path: str | Path | None = None) -> None:
        from stable_baselines3 import PPO  # type: ignore[import-untyped]

        p = Path(path) if path else self._model_path
        if p is None:
            raise FileNotFoundError("No model path provided")
        self._model = PPO.load(str(p))
        log.info("rl_model_loaded", path=str(p))

    def train(
        self,
        df: pd.DataFrame,
        *,
        total_timesteps: int = 50_000,
        learning_rate: float = 3e-4,
        save_path: str | Path | None = None,
    ) -> None:
        """Train a PPO agent on historical data.

        Args:
            df: Historical feature DataFrame.
            total_timesteps: Training budget.
            learning_rate: PPO learning rate.
            save_path: Optional path to save the trained model.
        """
        from stable_baselines3 import PPO  # type: ignore[import-untyped]

        EnvCls = _make_env_class()
        env = EnvCls(df, self._features)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=min(2048, max(len(df) - 2, 64)),
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
        self._model = model

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(p))
            log.info("rl_model_saved", path=str(p))

        log.info("rl_model_trained", timesteps=total_timesteps)

    # -- signal generation --------------------------------------------------

    async def generate(
        self,
        features_df: pd.DataFrame,
        regime: MarketRegime,
        sentiment: SentimentResult,
    ) -> list[TradeSignal]:
        if self._model is None:
            if self._model_path and self._model_path.exists():
                self.load_model()
            else:
                log.warning("rl_no_model_available")
                return []

        if features_df.empty:
            return []

        available = [c for c in self._features if c in features_df.columns]
        if not available:
            log.warning("rl_no_matching_features")
            return []

        obs = np.array(
            features_df.iloc[-1][available].fillna(0).values, dtype=np.float32
        )

        action, action_probs = await asyncio.to_thread(self._predict, obs)
        signal_action, direction = _ACTION_MAP.get(
            action, (SignalAction.NO_TRADE, SignalDirection.NEUTRAL)
        )

        if direction == SignalDirection.NEUTRAL:
            return []

        confidence = float(action_probs[action]) if action_probs is not None else 0.5
        if confidence < self._min_conf:
            return []

        close = float(features_df.iloc[-1]["close"])
        atr = float(features_df.iloc[-1].get("atr", 0))
        if atr <= 0:
            return []

        signal = self._build_signal(
            signal_action, direction, close, atr, confidence, regime
        )
        if signal is None:
            return []

        signal.underlying = str(features_df.iloc[-1].get("symbol", ""))
        signal.metadata["action_probabilities"] = {
            _ACTION_MAP[i][0].value: round(float(p), 4)
            for i, p in enumerate(action_probs)
        } if action_probs is not None else {}

        return [signal]

    # -- internals ----------------------------------------------------------

    def _predict(self, obs: np.ndarray) -> tuple[int, np.ndarray | None]:
        action, _state = self._model.predict(obs, deterministic=True)  # type: ignore[union-attr]
        action_int = int(action)
        # Extract action probabilities from the policy
        try:
            import torch  # type: ignore[import-untyped]

            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
            dist = self._model.policy.get_distribution(obs_tensor)  # type: ignore[union-attr]
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            return action_int, probs
        except Exception:
            return action_int, None

    def _build_signal(
        self,
        action: SignalAction,
        direction: SignalDirection,
        close: float,
        atr: float,
        confidence: float,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        stop_dist = atr * self._atr_mult
        target_dist = stop_dist * self._min_rr

        if direction == SignalDirection.LONG:
            stop = close - stop_dist
            target = close + target_dist
        else:
            stop = close + stop_dist
            target = close - target_dist

        rr = target_dist / stop_dist if stop_dist > 0 else 0.0
        if rr < self._min_rr:
            return None

        return TradeSignal(
            signal_id=TradeSignal.generate_id(),
            timestamp=datetime.now(timezone.utc),
            underlying="",
            action=action,
            direction=direction,
            confidence=round(confidence, 4),
            strategy_name=self._name,
            entry_price=close,
            stop_loss=round(stop, 4),
            target_price=round(target, 4),
            risk_reward_ratio=round(rr, 2),
            reasoning=f"RL-PPO {action.value} (p={confidence:.2%})",
            metadata={"regime": regime.value},
        )
