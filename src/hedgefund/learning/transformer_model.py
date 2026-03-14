"""Temporal Fusion Transformer for multi-horizon time-series forecasting."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hedgefund.learning.base import TradingModel

log = structlog.get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TFTConfig:
    """Hyperparameters for the Temporal Fusion Transformer."""

    input_size: int = 1
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    forecast_horizon: int = 5
    sequence_length: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    min_delta: float = 1e-5
    num_static_vars: int = 0
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k in self.__slots__:
            v = getattr(self, k)
            d[k] = list(v) if isinstance(v, tuple) else v
        return d


# ── Positional Encoding ──────────────────────────────────────────────────────


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Variable Selection Network ────────────────────────────────────────────────


class _VariableSelectionNetwork(nn.Module):
    """Learns per-variable importance weights via a GRN + softmax gate."""

    def __init__(self, input_size: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_size = input_size
        # Per-variable transform.
        self.variable_transforms = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(input_size)]
        )
        # Gating network: flattened features -> softmax weights.
        self.gate = nn.Sequential(
            nn.Linear(input_size * d_model, input_size),
            nn.Dropout(dropout),
            nn.Softmax(dim=-1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        transformed = []
        for i in range(F):
            transformed.append(self.variable_transforms[i](x[:, :, i : i + 1]))  # (B,T,d)
        stacked = torch.stack(transformed, dim=2)  # (B, T, F, d_model)
        flat = stacked.reshape(B, T, -1)  # (B, T, F*d_model)
        weights = self.gate(flat).unsqueeze(-1)  # (B, T, F, 1)
        selected = (stacked * weights).sum(dim=2)  # (B, T, d_model)
        return self.dropout(selected)


# ── Full Transformer Network ─────────────────────────────────────────────────


class _TemporalFusionNetwork(nn.Module):
    """Encoder-only Transformer with variable selection and quantile output."""

    def __init__(self, cfg: TFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.variable_selection = _VariableSelectionNetwork(
            cfg.input_size, cfg.d_model, cfg.dropout
        )
        self.positional_encoding = _PositionalEncoding(
            cfg.d_model, max_len=cfg.sequence_length + cfg.forecast_horizon, dropout=cfg.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_encoder_layers
        )
        self.output_head = nn.Linear(
            cfg.d_model, cfg.forecast_horizon * len(cfg.quantiles)
        )
        self._n_quantiles = len(cfg.quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        selected = self.variable_selection(x)  # (B, T, d_model)
        encoded = self.positional_encoding(selected)
        transformed = self.transformer_encoder(encoded)  # (B, T, d_model)
        last = transformed[:, -1, :]  # (B, d_model)
        raw = self.output_head(last)  # (B, horizon * n_quantiles)
        return raw.view(-1, self.cfg.forecast_horizon, self._n_quantiles)


# ── Quantile Loss ─────────────────────────────────────────────────────────────


def _quantile_loss(
    preds: torch.Tensor, targets: torch.Tensor, quantiles: tuple[float, ...]
) -> torch.Tensor:
    """Pinball loss summed over all quantile levels."""
    # preds: (B, H, Q), targets: (B, H)
    losses = []
    for i, q in enumerate(quantiles):
        errors = targets - preds[:, :, i]
        losses.append(torch.max(q * errors, (q - 1.0) * errors).mean())
    return torch.stack(losses).sum()


# ── Preprocessor ──────────────────────────────────────────────────────────────


class _TFTPreprocessor:
    """Z-score normalisation and sliding-window sequence creation."""

    def __init__(self, sequence_length: int, forecast_horizon: int) -> None:
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        self._means = data.mean(axis=0)
        self._stds = data.std(axis=0)
        self._stds[self._stds < 1e-8] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self._means is not None
        return (data - self._means) / self._stds

    def inverse_transform_target(self, data: np.ndarray, col: int = 0) -> np.ndarray:
        assert self._means is not None
        return data * self._stds[col] + self._means[col]

    def create_sequences(
        self, data: np.ndarray, target_col: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where y has shape (N, forecast_horizon)."""
        X, y = [], []
        total = self.sequence_length + self.forecast_horizon
        for i in range(len(data) - total + 1):
            X.append(data[i : i + self.sequence_length])
            y.append(
                data[
                    i + self.sequence_length : i + total, target_col
                ]
            )
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "means": self._means.tolist() if self._means is not None else None,
            "stds": self._stds.tolist() if self._stds is not None else None,
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._means = np.array(state["means"]) if state["means"] else None
        self._stds = np.array(state["stds"]) if state["stds"] else None
        self.sequence_length = state["sequence_length"]
        self.forecast_horizon = state["forecast_horizon"]


# ── Public Model ──────────────────────────────────────────────────────────────


class TemporalFusionTransformer(TradingModel):
    """Transformer-based multi-horizon forecaster with quantile outputs.

    Produces forecasts at configurable quantile levels (default: 10th, 50th,
    90th percentile) for each step in the forecast horizon.
    """

    def __init__(self, config: TFTConfig | None = None, version: str = "0.1.0") -> None:
        super().__init__(name="temporal_fusion_transformer", version=version)
        self.config = config or TFTConfig()
        self.metadata.hyperparameters = self.config.to_dict()
        self.metadata.compute_config_hash()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _TemporalFusionNetwork | None = None
        self._preprocessor = _TFTPreprocessor(
            self.config.sequence_length, self.config.forecast_horizon
        )

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> dict[str, float]:
        raw = np.asarray(X_train, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        self.config.input_size = raw.shape[1]

        self._preprocessor.fit(raw)
        normed = self._preprocessor.transform(raw)
        X_seq, y_seq = self._preprocessor.create_sequences(normed)

        if X_val is None:
            split = int(len(X_seq) * 0.9)
            X_tr, y_tr = X_seq[:split], y_seq[:split]
            X_va, y_va = X_seq[split:], y_seq[split:]
        else:
            val_raw = np.asarray(X_val, dtype=np.float32)
            if val_raw.ndim == 1:
                val_raw = val_raw.reshape(-1, 1)
            val_normed = self._preprocessor.transform(val_raw)
            X_va, y_va = self._preprocessor.create_sequences(val_normed)
            X_tr, y_tr = X_seq, y_seq

        train_loader = self._make_loader(X_tr, y_tr, shuffle=True)
        val_loader = self._make_loader(X_va, y_va, shuffle=False)

        self._net = _TemporalFusionNetwork(self.config).to(self._device)
        optimiser = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_state: dict[str, Any] = {}
        epochs_no_improve = 0

        self._log.info(
            "training_started",
            train_seqs=len(X_tr),
            val_seqs=len(X_va),
            device=str(self._device),
        )

        train_loss = float("inf")
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, optimiser)
            val_loss = self._evaluate(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epochs_no_improve == 0:
                self._log.debug(
                    "epoch", epoch=epoch,
                    train_loss=round(train_loss, 6),
                    val_loss=round(val_loss, 6),
                )

            if epochs_no_improve >= self.config.patience:
                self._log.info("early_stopping", epoch=epoch, best_val=best_val_loss)
                break

        if best_state:
            self._net.load_state_dict(best_state)

        self._is_trained = True
        metrics = {"train_loss": train_loss, "val_loss": best_val_loss}
        self.metadata.metrics = metrics
        self.metadata.training_rows = len(raw)
        self._log.info("training_complete", **metrics)
        return metrics

    # ── Prediction ────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return median (50th percentile) multi-horizon forecast, de-normalised."""
        quantile_preds = self.predict_quantiles(X)
        median_idx = list(self.config.quantiles).index(0.5)
        return quantile_preds[:, :, median_idx]

    def predict_quantiles(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return quantile predictions of shape (N, horizon, n_quantiles), de-normalised."""
        self._validate_trained()
        assert self._net is not None

        raw = np.asarray(X, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        normed = self._preprocessor.transform(raw)
        X_seq, _ = self._preprocessor.create_sequences(normed)
        tensor = torch.tensor(X_seq, device=self._device)

        self._net.eval()
        with torch.no_grad():
            preds_normed = self._net(tensor).cpu().numpy()  # (N, H, Q)

        # De-normalise each quantile channel.
        result = np.empty_like(preds_normed)
        for q_idx in range(preds_normed.shape[2]):
            result[:, :, q_idx] = self._preprocessor.inverse_transform_target(
                preds_normed[:, :, q_idx]
            )
        return result

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> Path:
        self._validate_trained()
        assert self._net is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), path / "weights.pt")
        (path / "preprocessor.json").write_text(json.dumps(self._preprocessor.state, default=str))
        (path / "config.json").write_text(json.dumps(self.config.to_dict(), default=str))
        self._save_metadata(path)
        self._log.info("model_saved", path=str(path))
        return path

    def load(self, path: Path) -> None:
        path = Path(path)
        cfg_data = json.loads((path / "config.json").read_text())
        # Restore quantiles as a tuple.
        if "quantiles" in cfg_data:
            cfg_data["quantiles"] = tuple(cfg_data["quantiles"])
        self.config = TFTConfig(**cfg_data)

        prep_data = json.loads((path / "preprocessor.json").read_text())
        self._preprocessor.load_state(prep_data)

        self._net = _TemporalFusionNetwork(self.config).to(self._device)
        state_dict = torch.load(path / "weights.pt", map_location=self._device, weights_only=True)
        self._net.load_state_dict(state_dict)
        self._net.eval()

        self._load_metadata(path)
        self._is_trained = True
        self._log.info("model_loaded", path=str(path))

    # ── Internal ──────────────────────────────────────────────────────

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, *, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle)

    def _run_epoch(self, loader: DataLoader, optimiser: torch.optim.Optimizer) -> float:
        assert self._net is not None
        self._net.train()
        total = 0.0
        n = 0
        for X_b, y_b in loader:
            X_b = X_b.to(self._device)
            y_b = y_b.to(self._device)
            optimiser.zero_grad()
            preds = self._net(X_b)
            loss = _quantile_loss(preds, y_b, self.config.quantiles)
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            optimiser.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        assert self._net is not None
        self._net.eval()
        total = 0.0
        n = 0
        for X_b, y_b in loader:
            X_b = X_b.to(self._device)
            y_b = y_b.to(self._device)
            preds = self._net(X_b)
            total += _quantile_loss(preds, y_b, self.config.quantiles).item()
            n += 1
        return total / max(n, 1)
