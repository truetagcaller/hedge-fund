#!/usr/bin/env python3
"""Download historical market data for backtesting."""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def download_yahoo_data(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: str = "data/historical",
) -> None:
    """Download OHLCV data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        print(f"Downloading {symbol}...")
        ticker = yf.Ticker(symbol)

        # Daily data
        df = ticker.history(start=start, end=end, interval="1d")
        if not df.empty:
            filepath = output_path / f"{symbol}_daily.parquet"
            df.to_parquet(filepath)
            print(f"  Saved {len(df)} daily bars to {filepath}")

        # Hourly data (limited to ~730 days)
        hourly_start = max(
            datetime.strptime(start, "%Y-%m-%d"),
            datetime.now() - timedelta(days=729),
        ).strftime("%Y-%m-%d")
        df_h = ticker.history(start=hourly_start, end=end, interval="1h")
        if not df_h.empty:
            filepath = output_path / f"{symbol}_hourly.parquet"
            df_h.to_parquet(filepath)
            print(f"  Saved {len(df_h)} hourly bars to {filepath}")

        # Options chain snapshot
        try:
            expirations = ticker.options
            if expirations:
                chains = []
                for exp in expirations[:5]:  # First 5 expirations
                    chain = ticker.option_chain(exp)
                    calls = chain.calls.copy()
                    calls["option_type"] = "CALL"
                    calls["expiration"] = exp
                    puts = chain.puts.copy()
                    puts["option_type"] = "PUT"
                    puts["expiration"] = exp
                    chains.extend([calls, puts])

                if chains:
                    options_df = pd.concat(chains, ignore_index=True)
                    filepath = output_path / f"{symbol}_options.parquet"
                    options_df.to_parquet(filepath)
                    print(f"  Saved {len(options_df)} option contracts to {filepath}")
        except Exception as e:
            print(f"  Warning: Could not fetch options for {symbol}: {e}")

    print("Download complete.")


def main():
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMZN", "MSFT"],
        help="Symbols to download",
    )
    parser.add_argument(
        "--start",
        default=(datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        default="data/historical",
        help="Output directory",
    )

    args = parser.parse_args()
    download_yahoo_data(args.symbols, args.start, args.end, args.output)


if __name__ == "__main__":
    main()
