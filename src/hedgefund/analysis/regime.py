"""Market regime detection using HMM and rule-based classification.

Classifies the current market into one of the :class:`MarketRegime` enum
values using a combination of Hidden Markov Model inference (when hmmlearn
is available) and a deterministic rule engine based on trend, volatility, and
momentum signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from hedgefund.types import MarketRegime

log = structlog.get_logger(__name__)

# Try importing hmmlearn; fall back to pure rule-based if unavailable.
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]

    _HAS_HMM = True
except ImportError:  # pragma: no cover
    _HAS_HMM = False


@dataclass
class RegimeDetector:
    """Detect market regimes from OHLCV + indicator data.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states for the HMM (default matches ``MarketRegime``
        enum cardinality).
    vol_lookback : int
        Lookback (bars) for volatility z-score calculation.
    trend_lookback : int
        Lookback (bars) for trend slope calculation.
    momentum_lookback : int
        Lookback (bars) for momentum (rate-of-change) calculation.
    use_hmm : bool
        Whether to attempt HMM-based detection (requires hmmlearn).
    """

    n_regimes: int = 6
    vol_lookback: int = 21
    trend_lookback: int = 50
    momentum_lookback: int = 14
    use_hmm: bool = True

    # Fitted HMM model (populated by ``fit_hmm``)
    _hmm_model: Optional[object] = field(default=None, repr=False)
    # Transition probability matrix (n_regimes x n_regimes)
    _transition_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Public API ───────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* with ``regime`` and ``regime_probability`` columns.

        Uses HMM when a fitted model is available; otherwise falls back to
        the deterministic rule engine.
        """
        df = df.copy()

        if self.use_hmm and _HAS_HMM and self._hmm_model is not None:
            df = self._hmm_predict(df)
        else:
            df = self._rule_based(df)

        return df

    def fit_hmm(self, df: pd.DataFrame) -> None:
        """Fit the HMM on historical data.

        Features used: log returns, realised volatility, momentum.
        """
        if not _HAS_HMM:
            log.warning("hmm_not_available", msg="hmmlearn not installed; skipping HMM fit")
            return

        features = self._extract_features(df)
        if features is None:
            return

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(features)
        self._hmm_model = model
        self._transition_matrix = model.transmat_
        log.info(
            "hmm_fitted",
            n_regimes=self.n_regimes,
            log_likelihood=float(model.score(features)),
        )

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        """Return the transition probability matrix (or ``None`` if not fitted)."""
        return self._transition_matrix

    # ── HMM prediction ───────────────────────────────────────────────────

    def _hmm_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._extract_features(df)
        if features is None or self._hmm_model is None:
            return self._rule_based(df)

        model = self._hmm_model  # type: ignore[union-attr]
        hidden_states = model.predict(features)  # type: ignore[union-attr]
        proba = model.predict_proba(features)  # type: ignore[union-attr]

        regime_map = self._map_states_to_regimes(model, features)

        regimes: list[MarketRegime] = []
        probabilities: list[float] = []
        for i, state in enumerate(hidden_states):
            regime = regime_map.get(int(state), MarketRegime.MEAN_REVERTING)
            regimes.append(regime)
            probabilities.append(float(proba[i, state]))

        # Pad leading NaN rows (from feature computation)
        pad = len(df) - len(regimes)
        df["regime"] = [None] * pad + [r.value for r in regimes]
        df["regime_probability"] = [np.nan] * pad + probabilities
        return df

    def _map_states_to_regimes(self, model, features: np.ndarray) -> dict[int, MarketRegime]:
        """Heuristically map HMM hidden states to ``MarketRegime`` labels
        based on the mean return and volatility of each state.
        """
        means = model.means_  # (n_states, n_features)
        mapping: dict[int, MarketRegime] = {}
        regime_list = list(MarketRegime)

        for state_idx in range(means.shape[0]):
            ret_mean = means[state_idx, 0]   # log return
            vol_mean = means[state_idx, 1]   # volatility

            bullish = ret_mean > 0
            high_vol = vol_mean > np.median(means[:, 1])

            if bullish and not high_vol:
                regime = MarketRegime.LOW_VOL_BULLISH
            elif bullish and high_vol:
                regime = MarketRegime.HIGH_VOL_BULLISH
            elif not bullish and not high_vol:
                regime = MarketRegime.LOW_VOL_BEARISH
            else:
                regime = MarketRegime.HIGH_VOL_BEARISH

            mapping[state_idx] = regime

        return mapping

    # ── Rule-based fallback ──────────────────────────────────────────────

    def _rule_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each bar using deterministic rules on trend, vol, and
        momentum."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1))

        # Trend: slope of close over lookback (via linear regression)
        trend = close.rolling(window=self.trend_lookback, min_periods=self.trend_lookback).apply(
            self._linear_slope, raw=True
        )

        # Volatility z-score
        vol = log_ret.rolling(window=self.vol_lookback).std() * np.sqrt(252)
        vol_mean = vol.rolling(window=self.vol_lookback * 4, min_periods=self.vol_lookback).mean()
        vol_std = vol.rolling(window=self.vol_lookback * 4, min_periods=self.vol_lookback).std().replace(0, np.nan)
        vol_z = (vol - vol_mean) / vol_std

        # Momentum (ROC)
        momentum = close.pct_change(self.momentum_lookback)

        regimes: list[str | None] = []
        probs: list[float] = []

        for i in range(len(df)):
            t = trend.iloc[i] if not np.isnan(trend.iloc[i]) else 0.0
            vz = vol_z.iloc[i] if not np.isnan(vol_z.iloc[i]) else 0.0
            mom = momentum.iloc[i] if not np.isnan(momentum.iloc[i]) else 0.0

            regime, conf = self._classify(t, vz, mom)
            regimes.append(regime.value if regime else None)
            probs.append(conf)

        df["regime"] = regimes
        df["regime_probability"] = probs
        return df

    @staticmethod
    def _classify(
        trend: float, vol_z: float, momentum: float
    ) -> tuple[MarketRegime, float]:
        """Map continuous signals to a discrete regime with confidence."""
        bullish = trend > 0 and momentum > 0
        bearish = trend < 0 and momentum < 0
        high_vol = vol_z > 0.5
        low_vol = vol_z < -0.5
        trending = abs(trend) > 0.5

        # Confidence heuristic: stronger signals → higher confidence
        conf = min(1.0, 0.5 + 0.2 * abs(trend) + 0.15 * abs(vol_z) + 0.15 * abs(momentum))

        if trending and abs(momentum) > 0.05:
            return MarketRegime.TRENDING, conf

        if bullish and low_vol:
            return MarketRegime.LOW_VOL_BULLISH, conf
        if bullish and high_vol:
            return MarketRegime.HIGH_VOL_BULLISH, conf
        if bearish and low_vol:
            return MarketRegime.LOW_VOL_BEARISH, conf
        if bearish and high_vol:
            return MarketRegime.HIGH_VOL_BEARISH, conf

        return MarketRegime.MEAN_REVERTING, max(0.3, conf - 0.2)

    @staticmethod
    def _linear_slope(values: np.ndarray) -> float:
        """Normalised slope of OLS fit to *values*."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = values.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        slope = ((x - x_mean) * (values - y_mean)).sum() / denom
        # Normalise by mean price so it's comparable across assets
        return slope / max(abs(y_mean), 1e-10)
