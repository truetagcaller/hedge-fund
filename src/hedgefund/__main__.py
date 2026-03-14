"""CLI entrypoint for the HedgeFund AI trading system.

Usage::

    hedgefund run          # start live trading
    hedgefund backtest     # run backtesting
    hedgefund train        # train ML models
    hedgefund dashboard    # start dashboard only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from hedgefund.config.settings import build_settings
from hedgefund.logger import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedgefund",
        description="AI-Powered Options Trading Agent",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config directory (default: <repo>/config)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (development, production, etc.)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log level",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────
    run_parser = sub.add_parser("run", help="Start live trading")
    run_parser.add_argument(
        "--tick-interval",
        type=float,
        default=60.0,
        help="Seconds between trading ticks (default: 60)",
    )

    # ── backtest ─────────────────────────────────────────────────
    bt_parser = sub.add_parser("backtest", help="Run backtesting")
    bt_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name to backtest",
    )
    bt_parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to include (default: from config)",
    )
    bt_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    bt_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    # ── train ────────────────────────────────────────────────────
    train_parser = sub.add_parser("train", help="Train ML models")
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["lstm", "transformer", "rl", "random_forest", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    # ── dashboard ────────────────────────────────────────────────
    dash_parser = sub.add_parser("dashboard", help="Start dashboard only")
    dash_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override dashboard bind host",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override dashboard bind port",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate command handler."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build settings from config + env
    settings = build_settings(config_dir=args.config, env=args.env)

    # Configure logging
    log_level = args.log_level or settings.app.log_level
    json_logs = settings.app.json_logs
    setup_logging(level=log_level, json_output=json_logs)

    if args.command == "run":
        _cmd_run(settings, args)
    elif args.command == "backtest":
        _cmd_backtest(settings, args)
    elif args.command == "train":
        _cmd_train(settings, args)
    elif args.command == "dashboard":
        _cmd_dashboard(settings, args)
    else:
        parser.print_help()
        sys.exit(1)


# ── Command handlers ─────────────────────────────────────────────────────────


def _cmd_run(settings, args) -> None:  # type: ignore[no-untyped-def]
    """Start the live trading loop."""
    from hedgefund.app import TradingApplication

    app = TradingApplication(
        settings=settings,
        tick_interval=args.tick_interval,
    )
    asyncio.run(app.start())


def _cmd_backtest(settings, args) -> None:  # type: ignore[no-untyped-def]
    """Run a backtest."""
    from hedgefund.logger import get_logger

    log = get_logger("cli.backtest")

    async def _run() -> None:
        try:
            from hedgefund.backtest import BacktestEngine  # type: ignore[attr-defined]

            symbols = args.symbols or settings.data.symbols
            engine = BacktestEngine(settings.backtest)
            results = await engine.run(
                strategy=args.strategy,
                symbols=symbols,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            log.info("backtest_complete", metrics=str(results))
        except ImportError:
            log.error("backtest_engine_not_implemented")
            sys.exit(1)

    asyncio.run(_run())


def _cmd_train(settings, args) -> None:  # type: ignore[no-untyped-def]
    """Train ML models."""
    from hedgefund.logger import get_logger

    log = get_logger("cli.train")

    async def _run() -> None:
        try:
            from hedgefund.learning import ModelTrainer  # type: ignore[attr-defined]

            overrides = {}
            if args.epochs is not None:
                overrides["epochs"] = args.epochs

            trainer = ModelTrainer(settings.learning)
            await trainer.train(model=args.model, **overrides)
            log.info("training_complete", model=args.model)
        except ImportError:
            log.error("model_trainer_not_implemented")
            sys.exit(1)

    asyncio.run(_run())


def _cmd_dashboard(settings, args) -> None:  # type: ignore[no-untyped-def]
    """Start the dashboard web server."""
    from hedgefund.app import run_dashboard

    overrides = {}
    if args.host is not None:
        overrides["dashboard__host"] = args.host
    if args.port is not None:
        overrides["dashboard__port"] = args.port

    if overrides:
        # Rebuild settings with dashboard overrides
        settings = build_settings(
            config_dir=None,
            env=None,
            overrides={"dashboard": {
                k.split("__")[1]: v for k, v in overrides.items()
            }},
        )

    asyncio.run(run_dashboard(settings))


if __name__ == "__main__":
    main()
