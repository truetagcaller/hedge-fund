"""Historical data loading and replay for backtesting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd
import structlog

from hedgefund.types import OHLCV, OptionContract, OptionQuote, OptionType, Greeks

log = structlog.get_logger(__name__)


class HistoricalDataHandler:
    """Load and replay historical market data for the backtesting engine.

    Supports CSV and Parquet files as well as in-memory DataFrames.
    Handles corporate actions (splits and dividends) if adjustment columns
    are present.
    """

    # Expected column mapping (source name -> canonical name).
    _COLUMN_MAP: dict[str, str] = {
        "date": "timestamp",
        "Date": "timestamp",
        "datetime": "timestamp",
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
    }

    def __init__(self) -> None:
        self._bars: list[OHLCV] = []
        self._options_chains: dict[str, list[OptionQuote]] = {}
        self._log = log.bind(component="data_handler")

    # ── Loading ───────────────────────────────────────────────────────

    def load_file(
        self,
        path: str | Path,
        *,
        symbol: str = "UNKNOWN",
        adjust: bool = True,
    ) -> list[OHLCV]:
        """Load OHLCV data from a CSV or Parquet file.

        Args:
            path: Filesystem path to the data file.
            symbol: Ticker symbol (used for logging only).
            adjust: Whether to apply split/dividend adjustments.

        Returns:
            Chronologically ordered list of :class:`OHLCV` bars.
        """
        path = Path(path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in (".csv", ".tsv"):
            df = pd.read_csv(path, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self._log.info("file_loaded", path=str(path), rows=len(df), symbol=symbol)
        return self.load_dataframe(df, symbol=symbol, adjust=adjust)

    def load_dataframe(
        self,
        df: pd.DataFrame,
        *,
        symbol: str = "UNKNOWN",
        adjust: bool = True,
    ) -> list[OHLCV]:
        """Convert a DataFrame into a list of OHLCV bars.

        The DataFrame must contain at least: open, high, low, close, volume.
        A ``timestamp`` or ``date`` column (or the index) is used for timing.
        """
        df = df.copy()
        df.rename(columns=self._COLUMN_MAP, inplace=True)

        # Use index as timestamp if no explicit column.
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index
            else:
                df["timestamp"] = pd.to_datetime(df.index)

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Canonical lower-case columns.
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                lower_candidates = [c for c in df.columns if c.lower() == col]
                if lower_candidates:
                    df.rename(columns={lower_candidates[0]: col}, inplace=True)

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if adjust:
            df = self._apply_adjustments(df)

        df.sort_values("timestamp", inplace=True)
        bars = [
            OHLCV(
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
            )
            for row in df.itertuples(index=False)
        ]
        self._bars = bars
        self._log.info("bars_prepared", count=len(bars), symbol=symbol)
        return bars

    # ── Options chain reconstruction ──────────────────────────────────

    def load_options_chain(
        self,
        df: pd.DataFrame,
        *,
        underlying: str = "UNKNOWN",
    ) -> dict[str, list[OptionQuote]]:
        """Build a date-keyed options chain from historical options data.

        Expected columns: expiration, strike, option_type, bid, ask, last,
        volume, open_interest, iv, delta, gamma, theta, vega, timestamp.

        Returns:
            Dict mapping date strings (YYYY-MM-DD) to lists of OptionQuote.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        chains: dict[str, list[OptionQuote]] = {}

        for _, row in df.iterrows():
            date_key = row["timestamp"].strftime("%Y-%m-%d")
            contract = OptionContract(
                symbol=f"{underlying}_{row['strike']}_{row['option_type']}",
                underlying=underlying,
                option_type=OptionType(row["option_type"]),
                strike=float(row["strike"]),
                expiration=pd.to_datetime(row["expiration"]).date(),
            )
            greeks = Greeks(
                delta=float(row.get("delta", 0)),
                gamma=float(row.get("gamma", 0)),
                theta=float(row.get("theta", 0)),
                vega=float(row.get("vega", 0)),
                iv=float(row.get("iv", 0)),
            )
            quote = OptionQuote(
                contract=contract,
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                last=float(row.get("last", (row["bid"] + row["ask"]) / 2)),
                volume=int(row.get("volume", 0)),
                open_interest=int(row.get("open_interest", 0)),
                greeks=greeks,
                timestamp=row["timestamp"],
            )
            chains.setdefault(date_key, []).append(quote)

        self._options_chains = chains
        self._log.info(
            "options_chain_loaded",
            underlying=underlying,
            dates=len(chains),
            total_quotes=sum(len(v) for v in chains.values()),
        )
        return chains

    # ── Replay iterator ───────────────────────────────────────────────

    def replay(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Iterator[OHLCV]:
        """Yield bars one at a time within the optional date range."""
        for bar in self._bars:
            if start and bar.timestamp < start:
                continue
            if end and bar.timestamp > end:
                break
            yield bar

    def get_options_for_date(self, date_str: str) -> list[OptionQuote]:
        """Return reconstructed options chain for a given date."""
        return self._options_chains.get(date_str, [])

    # ── Corporate action adjustments ──────────────────────────────────

    @staticmethod
    def _apply_adjustments(df: pd.DataFrame) -> pd.DataFrame:
        """Adjust OHLCV for splits and dividends when adjustment columns exist.

        Recognises ``adj_close``, ``split_ratio``, and ``dividend`` columns.
        """
        if "adj_close" in df.columns and "close" in df.columns:
            ratio = df["adj_close"] / df["close"]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * ratio
            df["volume"] = (df["volume"] / ratio).astype(int)
            df.drop(columns=["adj_close"], inplace=True, errors="ignore")

        if "split_ratio" in df.columns:
            cumulative = df["split_ratio"].cumprod()
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] / cumulative
            df["volume"] = (df["volume"] * cumulative).astype(int)
            df.drop(columns=["split_ratio"], inplace=True, errors="ignore")

        if "dividend" in df.columns:
            cum_div = df["dividend"].cumsum()
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] - cum_div
            df.drop(columns=["dividend"], inplace=True, errors="ignore")

        return df

    # ── Convenience ───────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return loaded bars as a DataFrame."""
        return pd.DataFrame(
            [
                {
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in self._bars
            ]
        )

    @property
    def bars(self) -> list[OHLCV]:
        return list(self._bars)
