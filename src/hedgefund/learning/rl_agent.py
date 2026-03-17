"""Reinforcement learning agent for options trading using stable-baselines3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

from hedgefund.types import (
    SignalAction,
)

log = structlog.get_logger(__name__)


# ── Environment Configuration ─────────────────────────────────────────────────


@dataclass(slots=True)
class TradingEnvConfig:
    """Configuration for the options trading gym environment."""

    initial_capital: float = 100_000.0
    max_positions: int = 10
    transaction_cost: float = 1.50
    max_steps: int = 252  # one trading year
    risk_free_rate: float = 0.05
    drawdown_penalty: float = 2.0
    sharpe_window: int = 20
    # Feature dimensions.
    n_price_features: int = 10
    n_greek_features: int = 5
    n_portfolio_features: int = 5
    n_sentiment_features: int = 3

    @property
    def observation_size(self) -> int:
        return (
            self.n_price_features
            + self.n_greek_features
            + self.n_portfolio_features
            + self.n_sentiment_features
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# ── Gymnasium Environment ─────────────────────────────────────────────────────

# Discrete action mapping matching SignalAction enum.
ACTION_MAP: dict[int, SignalAction] = {
    0: SignalAction.BUY_CALL,
    1: SignalAction.BUY_PUT,
    2: SignalAction.SELL_CALL,
    3: SignalAction.SELL_PUT,
    4: SignalAction.NO_TRADE,
}


class OptionsTradingEnv(gym.Env):
    """Custom gymnasium environment for options trading.

    **State space** (continuous):
        price features | greeks | portfolio state | sentiment

    **Action space** (discrete):
        0 = buy_call, 1 = buy_put, 2 = sell_call, 3 = sell_put, 4 = hold

    **Reward**:
        Risk-adjusted PnL (Sharpe-like ratio over a rolling window) with a
        penalty proportional to drawdown.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: np.ndarray,
        config: TradingEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or TradingEnvConfig()
        self._data = np.asarray(data, dtype=np.float32)
        if self._data.ndim != 2:

            raise RuntimeError("data must be (T, obs_size)")

        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.observation_size,),
            dtype=np.float32,
        )

        # Mutable episode state.
        self._step_idx: int = 0
        self._capital: float = self.cfg.initial_capital
        self._positions: int = 0
        self._portfolio_values: list[float] = []
        self._returns: list[float] = []
        self._high_water_mark: float = self.cfg.initial_capital

    # ── Gym API ───────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._capital = self.cfg.initial_capital
        self._positions = 0
        self._portfolio_values = [self.cfg.initial_capital]
        self._returns = []
        self._high_water_mark = self.cfg.initial_capital
        return self._get_observation(), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        signal = ACTION_MAP[action]

        # Simulate P&L from action (simplified: use next-bar return proxy).
        pnl = self._simulate_action(signal)
        self._capital += pnl

        # Track portfolio value and returns.
        self._portfolio_values.append(self._capital)
        ret = pnl / max(self._portfolio_values[-2], 1.0)
        self._returns.append(ret)

        # Update high-water mark.
        if self._capital > self._high_water_mark:
            self._high_water_mark = self._capital

        reward = self._compute_reward(ret)
        self._step_idx += 1

        terminated = self._step_idx >= min(self.cfg.max_steps, len(self._data) - 1)
        truncated = self._capital <= 0

        info: dict[str, Any] = {
            "capital": self._capital,
            "pnl": pnl,
            "positions": self._positions,
            "drawdown": self._current_drawdown(),
        }
        return self._get_observation(), reward, terminated, truncated, info

    # ── Internals ─────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        idx = min(self._step_idx, len(self._data) - 1)
        obs = self._data[idx].copy()
        # Clip extreme values for numerical stability.
        return np.clip(obs, -10.0, 10.0)

    def _simulate_action(self, signal: SignalAction) -> float:
        """Simplified P&L simulation based on next-bar price movement.

        In production this would interface with the broker or paper-trading
        engine.  Here we derive a proxy return from the data matrix.
        """
        if self._step_idx + 1 >= len(self._data):
            return 0.0

        # Use first feature column as a price-return proxy.
        next_return = float(self._data[self._step_idx + 1, 0])

        cost = self.cfg.transaction_cost
        position_size = self._capital * 0.02  # risk 2% per trade

        if signal == SignalAction.BUY_CALL:
            pnl = position_size * next_return - cost
            self._positions = min(self._positions + 1, self.cfg.max_positions)
        elif signal == SignalAction.BUY_PUT:
            pnl = position_size * (-next_return) - cost
            self._positions = min(self._positions + 1, self.cfg.max_positions)
        elif signal == SignalAction.SELL_CALL:
            pnl = position_size * (-next_return) - cost
            self._positions = max(self._positions - 1, 0)
        elif signal == SignalAction.SELL_PUT:
            pnl = position_size * next_return - cost
            self._positions = max(self._positions - 1, 0)
        else:
            pnl = 0.0

        return pnl

    def _compute_reward(self, current_return: float) -> float:
        """Sharpe-like reward with drawdown penalty."""
        window = self._returns[-self.cfg.sharpe_window :]
        if len(window) < 2:
            sharpe = current_return
        else:
            arr = np.array(window)
            excess = arr - self.cfg.risk_free_rate / 252.0
            std = arr.std()
            sharpe = float(excess.mean() / max(std, 1e-8))

        dd = self._current_drawdown()
        penalty = self.cfg.drawdown_penalty * dd
        return sharpe - penalty

    def _current_drawdown(self) -> float:
        if self._high_water_mark <= 0:
            return 0.0
        return (self._high_water_mark - self._capital) / self._high_water_mark


# ── RL Agent Wrapper ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class RLAgentConfig:
    """Configuration for the RL training loop."""

    algorithm: str = "PPO"  # "PPO" or "SAC"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # PPO rollout length
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_eval_episodes: int = 10
    eval_freq: int = 5000
    verbose: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class TradingRLAgent:
    """Reinforcement learning agent for options trading.

    Wraps stable-baselines3 PPO or SAC with the custom
    :class:`OptionsTradingEnv`.
    """

    def __init__(
        self,
        env_config: TradingEnvConfig | None = None,
        agent_config: RLAgentConfig | None = None,
    ) -> None:
        self.env_config = env_config or TradingEnvConfig()
        self.agent_config = agent_config or RLAgentConfig()
        self._model: Any = None  # sb3 model
        self._log = log.bind(algorithm=self.agent_config.algorithm)

    # ── Training ──────────────────────────────────────────────────────

    def train(self, train_data: np.ndarray, eval_data: np.ndarray | None = None) -> dict[str, float]:
        """Train the RL agent on historical feature data.

        Args:
            train_data: array of shape (T, obs_size) for the training environment.
            eval_data: optional held-out data for periodic evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor

        train_env = Monitor(OptionsTradingEnv(train_data, self.env_config))

        algo_cls = PPO if self.agent_config.algorithm.upper() == "PPO" else SAC
        common_kwargs: dict[str, Any] = {
            "policy": "MlpPolicy",
            "env": train_env,
            "learning_rate": self.agent_config.learning_rate,
            "batch_size": self.agent_config.batch_size,
            "gamma": self.agent_config.gamma,
            "verbose": self.agent_config.verbose,
        }
        if algo_cls is PPO:
            common_kwargs.update(
                n_steps=self.agent_config.n_steps,
                gae_lambda=self.agent_config.gae_lambda,
                clip_range=self.agent_config.clip_range,
                ent_coef=self.agent_config.ent_coef,
            )

        self._model = algo_cls(**common_kwargs)

        callbacks = []
        if eval_data is not None:
            eval_env = Monitor(OptionsTradingEnv(eval_data, self.env_config))
            callbacks.append(
                EvalCallback(
                    eval_env,
                    n_eval_episodes=self.agent_config.n_eval_episodes,
                    eval_freq=self.agent_config.eval_freq,
                    best_model_save_path=None,
                    verbose=0,
                )
            )

        self._log.info(
            "rl_training_started",
            timesteps=self.agent_config.total_timesteps,
            algorithm=self.agent_config.algorithm,
        )
        self._model.learn(
            total_timesteps=self.agent_config.total_timesteps,
            callback=callbacks or None,
        )
        self._log.info("rl_training_complete")

        return self.evaluate(train_data)

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate(
        self, data: np.ndarray, n_episodes: int = 5
    ) -> dict[str, float]:
        """Run the agent on *data* for *n_episodes* and aggregate metrics."""
        if self._model is None:
            raise RuntimeError("Agent has not been trained. Call train() first.")

        env = OptionsTradingEnv(data, self.env_config)
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        final_capitals: list[float] = []
        max_drawdowns: list[float] = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            peak = env.cfg.initial_capital
            max_dd = 0.0
            done = False

            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                steps += 1
                capital = info["capital"]
                if capital > peak:
                    peak = capital
                dd = (peak - capital) / max(peak, 1e-8)
                max_dd = max(max_dd, dd)
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            final_capitals.append(info["capital"])
            max_drawdowns.append(max_dd)

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_final_capital": float(np.mean(final_capitals)),
            "mean_return": float(
                np.mean(
                    [(c - self.env_config.initial_capital) / self.env_config.initial_capital for c in final_capitals]
                )
            ),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }
        self._log.info("rl_evaluation", **metrics)
        return metrics

    # ── Inference ─────────────────────────────────────────────────────

    def act(self, observation: np.ndarray, deterministic: bool = True) -> SignalAction:
        """Select an action for a single observation.

        Returns the corresponding :class:`SignalAction`.
        """
        if self._model is None:
            raise RuntimeError("Agent has not been trained. Call train() first.")
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return ACTION_MAP[int(action)]

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> Path:
        """Save sb3 model and config to *path*."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "rl_model"))
        (path / "env_config.json").write_text(json.dumps(self.env_config.to_dict()))
        (path / "agent_config.json").write_text(json.dumps(self.agent_config.to_dict()))
        self._log.info("rl_model_saved", path=str(path))
        return path

    def load(self, path: Path) -> None:
        """Load a previously saved sb3 model."""
        from stable_baselines3 import PPO, SAC

        path = Path(path)
        self.env_config = TradingEnvConfig(
            **json.loads((path / "env_config.json").read_text())
        )
        self.agent_config = RLAgentConfig(
            **json.loads((path / "agent_config.json").read_text())
        )
        algo_cls = PPO if self.agent_config.algorithm.upper() == "PPO" else SAC
        self._model = algo_cls.load(str(path / "rl_model"))
        self._log.info("rl_model_loaded", path=str(path))
