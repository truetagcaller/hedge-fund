"""LSTM-based price and volatility forecasting model."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hedgefund.learning.base import TradingModel

log = structlog.get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LSTMConfig:
    """All tuneable hyperparameters for the LSTM predictor."""

    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 60
    output_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    min_delta: float = 1e-5
    mc_dropout_samples: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# ── PyTorch Module ────────────────────────────────────────────────────────────


class _LSTMNetwork(nn.Module):
    """Multi-layer LSTM followed by a fully connected head."""

    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size // 2, cfg.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        last_hidden = lstm_out[:, -1, :]  # (B, H)
        return self.fc(self.dropout(last_hidden))  # (B, O)


# ── Preprocessing ─────────────────────────────────────────────────────────────


class _SequencePreprocessor:
    """Normalise features and carve rolling windows for the LSTM."""

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        """Compute per-feature mean and std from training data."""
        self._means = data.mean(axis=0)
        self._stds = data.std(axis=0)
        # Prevent division by zero for constant features.
        self._stds[self._stds < 1e-8] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self._means is not None, "Call fit() before transform()."
        return (data - self._means) / self._stds

    def inverse_transform(self, data: np.ndarray, col_idx: int = 0) -> np.ndarray:
        """Undo normalisation for a single output column."""
        assert self._means is not None
        return data * self._stds[col_idx] + self._means[col_idx]

    def create_sequences(
        self, data: np.ndarray, target_col: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where X has shape (N, T, F) and y has shape (N,)."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i, target_col])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "means": self._means.tolist() if self._means is not None else None,
            "stds": self._stds.tolist() if self._stds is not None else None,
            "sequence_length": self.sequence_length,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._means = np.array(state["means"]) if state["means"] else None
        self._stds = np.array(state["stds"]) if state["stds"] else None
        self.sequence_length = state["sequence_length"]


# ── Public model class ────────────────────────────────────────────────────────


class LSTMPricePredictor(TradingModel):
    """Multi-layer LSTM for price / volatility forecasting.

    Supports MC-Dropout for uncertainty estimation at inference time.
    """

    def __init__(self, config: LSTMConfig | None = None, version: str = "0.1.0") -> None:
        super().__init__(name="lstm_price_predictor", version=version)
        self.config = config or LSTMConfig()
        self.metadata.hyperparameters = self.config.to_dict()
        self.metadata.compute_config_hash()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: _LSTMNetwork | None = None
        self._preprocessor = _SequencePreprocessor(self.config.sequence_length)

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the LSTM on raw feature arrays.

        *X_train* should contain all feature columns (the first column is
        treated as the prediction target unless *y_train* is explicit).
        If *X_val* / *y_val* are ``None`` the last 10 % of the training
        data is held out automatically.
        """
        raw = np.asarray(X_train, dtype=np.float32)
        self.config.input_size = raw.shape[1] if raw.ndim == 2 else 1
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        # Fit normaliser on training split only.
        self._preprocessor.fit(raw)
        normed = self._preprocessor.transform(raw)

        X_seq, y_seq = self._preprocessor.create_sequences(normed)

        # Auto-split if no explicit validation set.
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

        self._net = _LSTMNetwork(self.config).to(self._device)
        optimiser = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state: dict[str, Any] = {}
        epochs_no_improve = 0

        self._log.info(
            "training_started",
            train_sequences=len(X_tr),
            val_sequences=len(X_va),
            device=str(self._device),
        )

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimiser)
            val_loss = self._evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epochs_no_improve == 0:
                self._log.debug(
                    "epoch_complete",
                    epoch=epoch,
                    train_loss=round(train_loss, 6),
                    val_loss=round(val_loss, 6),
                )

            if epochs_no_improve >= self.config.patience:
                self._log.info("early_stopping", epoch=epoch, best_val_loss=best_val_loss)
                break

        # Restore best weights.
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
        """Point predictions (de-normalised)."""
        self._validate_trained()
        assert self._net is not None
        preds_normed = self._forward_pass(X)
        return self._preprocessor.inverse_transform(preds_normed)

    def predict_with_uncertainty(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC-Dropout uncertainty: return (mean_prediction, std_prediction).

        Runs *mc_dropout_samples* forward passes with dropout enabled
        to approximate the predictive distribution.
        """
        self._validate_trained()
        assert self._net is not None

        raw = np.asarray(X, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        normed = self._preprocessor.transform(raw)
        X_seq, _ = self._preprocessor.create_sequences(normed)
        tensor = torch.tensor(X_seq, device=self._device)

        # Enable dropout at inference time for MC sampling.
        self._net.train()
        samples = []
        with torch.no_grad():
            for _ in range(self.config.mc_dropout_samples):
                out = self._net(tensor).cpu().numpy().squeeze()
                samples.append(out)
        self._net.eval()

        stacked = np.stack(samples, axis=0)  # (S, N)
        mean_normed = stacked.mean(axis=0)
        std_normed = stacked.std(axis=0)

        mean = self._preprocessor.inverse_transform(mean_normed)
        # Scale uncertainty back to original magnitude.
        std = std_normed * (self._preprocessor._stds[0] if self._preprocessor._stds is not None else 1.0)
        return mean, std

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> Path:
        self._validate_trained()
        assert self._net is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), path / "weights.pt")

        import json
        (path / "preprocessor.json").write_text(
            json.dumps(self._preprocessor.state, default=str)
        )
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), default=str)
        )
        self._save_metadata(path)
        self._log.info("model_saved", path=str(path))
        return path

    def load(self, path: Path) -> None:
        path = Path(path)
        import json

        cfg_data = json.loads((path / "config.json").read_text())
        self.config = LSTMConfig(**cfg_data)

        prep_data = json.loads((path / "preprocessor.json").read_text())
        self._preprocessor.load_state(prep_data)

        self._net = _LSTMNetwork(self.config).to(self._device)
        state_dict = torch.load(path / "weights.pt", map_location=self._device, weights_only=True)
        self._net.load_state_dict(state_dict)
        self._net.eval()

        self._load_metadata(path)
        self._is_trained = True
        self._log.info("model_loaded", path=str(path))

    # ── Internal helpers ──────────────────────────────────────────────

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, *, shuffle: bool
    ) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
    ) -> float:
        assert self._net is not None
        self._net.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            optimiser.zero_grad()
            preds = self._net(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self._net is not None
        self._net.eval()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            preds = self._net(X_batch).squeeze()
            total_loss += criterion(preds, y_batch).item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _forward_pass(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Normalise, sequence, and run a single forward pass."""
        assert self._net is not None
        raw = np.asarray(X, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        normed = self._preprocessor.transform(raw)
        X_seq, _ = self._preprocessor.create_sequences(normed)
        tensor = torch.tensor(X_seq, device=self._device)
        self._net.eval()
        with torch.no_grad():
            return self._net(tensor).cpu().numpy().squeeze()
