#!/usr/bin/env python3
"""Offline model training entrypoint."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hedgefund.logger import setup_logging, get_logger
from hedgefund.config.loader import load_config
from hedgefund.learning.trainer import ModelTrainer
from hedgefund.learning.model_store import ModelStore


logger = get_logger(__name__)


async def train_all_models(config_env: str = "development") -> None:
    """Train all ML models using historical data."""
    setup_logging(level="INFO")

    config = load_config(env=config_env)
    model_store = ModelStore(base_path=Path("models"))
    trainer = ModelTrainer(config=config.learning, model_store=model_store)

    logger.info("starting_model_training", env=config_env)

    # Load historical data
    data_dir = Path("data/historical")
    if not data_dir.exists():
        logger.error("no_historical_data", msg="Run download_historical.py first")
        return

    import pandas as pd

    # Load available data
    parquet_files = list(data_dir.glob("*_daily.parquet"))
    if not parquet_files:
        logger.error("no_data_files")
        return

    datasets = {}
    for f in parquet_files:
        symbol = f.stem.replace("_daily", "")
        datasets[symbol] = pd.read_parquet(f)
        logger.info("loaded_data", symbol=symbol, rows=len(datasets[symbol]))

    # Train models
    await trainer.train_all(datasets)

    logger.info("training_complete")


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--env", default="development", help="Config environment")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer", "rl", "classifier", "all"],
        default="all",
        help="Model to train",
    )
    args = parser.parse_args()

    asyncio.run(train_all_models(config_env=args.env))


if __name__ == "__main__":
    main()
