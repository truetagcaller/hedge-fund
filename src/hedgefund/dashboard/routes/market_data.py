"""FastAPI routes for live market data and historical charts via Zerodha Kite Connect.

Provides REST endpoints for live quotes, LTP, historical OHLCV candles,
instrument listing, and instrument search.  All data is sourced from the
Zerodha Kite Connect API with no synthetic/mock fallback.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/market-data", tags=["market-data"])

_SOURCE = "Zerodha Kite API"


def _get_feed(request: Request) -> Any:
    """Retrieve the ZerodhaMarketFeed from app state, or raise 503."""
    feed = getattr(request.app.state, "zerodha_feed", None)
    if feed is None:
        raise HTTPException(
            status_code=503,
            detail="Zerodha market feed is not available. Check Zerodha credentials.",
        )
    return feed


def _get_historical(request: Request) -> Any:
    """Retrieve or create a ZerodhaHistorical client from app state."""
    hist = getattr(request.app.state, "zerodha_historical", None)
    if hist is not None:
        return hist

    # Build from credential store on first access
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        api_key = store.retrieve("zerodha", "api_key")
        access_token = store.retrieve("zerodha", "access_token")
        if not api_key or not access_token:
            raise ValueError("Missing Zerodha credentials")

        from hedgefund.data.zerodha_historical import ZerodhaHistorical
        hist = ZerodhaHistorical(api_key=api_key, access_token=access_token)
        request.app.state.zerodha_historical = hist
        return hist
    except Exception as exc:
        log.warning("zerodha_historical.init_failed", error=str(exc), source=_SOURCE)
        raise HTTPException(
            status_code=503,
            detail=(
                "Zerodha historical data is not available. "
                "Check Zerodha credentials in the credential store."
            ),
        ) from exc


# ── Live quote endpoints ──────────────────────────────────────────────────────


@router.get("/quote/{symbol:path}")
async def get_quote(symbol: str, request: Request) -> dict[str, Any]:
    """Get full live quote for a symbol.

    Example: ``/api/market-data/quote/NSE:RELIANCE``

    Returns OHLC, depth, volume, OI, and last traded price.
    """
    feed = _get_feed(request)
    try:
        data = await feed.get_quote([symbol])
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No quote data returned for {symbol}",
            )
        log.info(
            "market_data.quote_served",
            message="Market data received from Zerodha Kite API",
            symbol=symbol,
            source=_SOURCE,
        )
        return {"status": "ok", "data": data, "source": _SOURCE}
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/ltp")
async def get_ltp(
    request: Request,
    symbols: str = Query(..., description="Comma-separated symbols, e.g. NSE:NIFTY+50,NSE:RELIANCE"),
) -> dict[str, Any]:
    """Get last traded price for comma-separated symbols.

    Example: ``/api/market-data/ltp?symbols=NSE:NIFTY+50,NSE:RELIANCE``
    """
    feed = _get_feed(request)
    instrument_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not instrument_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    try:
        data = await feed.get_ltp(instrument_list)
        log.info(
            "market_data.ltp_served",
            message="Market data received from Zerodha Kite API",
            symbols=len(instrument_list),
            source=_SOURCE,
        )
        return {"status": "ok", "data": data, "source": _SOURCE}
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── Historical candle endpoints ───────────────────────────────────────────────


@router.get("/historical/{instrument_token}/{interval}")
async def get_historical(
    instrument_token: str,
    interval: str,
    request: Request,
    from_date: str = Query(..., description="Start date YYYY-MM-DD or YYYY-MM-DD+HH:MM:SS"),
    to_date: str = Query(..., description="End date YYYY-MM-DD or YYYY-MM-DD+HH:MM:SS"),
) -> dict[str, Any]:
    """Get historical OHLCV candles for an instrument.

    Example:
        ``/api/market-data/historical/256265/15minute?from_date=2026-03-01&to_date=2026-03-14``

    Intervals: ``minute``, ``3minute``, ``5minute``, ``15minute``,
    ``30minute``, ``60minute``, ``day``.
    """
    hist = _get_historical(request)
    try:
        candles = await hist.get_candles(
            instrument_token=instrument_token,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )
        log.info(
            "market_data.historical_served",
            message="Market data received from Zerodha Kite API",
            instrument_token=instrument_token,
            interval=interval,
            candles=len(candles),
            source=_SOURCE,
        )
        return {"status": "ok", "data": candles, "source": _SOURCE}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ── Instrument endpoints ──────────────────────────────────────────────────────


@router.get("/instruments")
async def get_instruments(
    request: Request,
    exchange: str = Query("NSE", description="Exchange: NSE, NFO, BSE, BFO, MCX, CDS"),
) -> dict[str, Any]:
    """Get instrument list for an exchange.

    Example: ``/api/market-data/instruments?exchange=NSE``
    """
    hist = _get_historical(request)
    try:
        instruments = await hist.get_instruments(exchange=exchange)
        log.info(
            "market_data.instruments_served",
            message="Market data received from Zerodha Kite API",
            exchange=exchange,
            count=len(instruments),
            source=_SOURCE,
        )
        return {"status": "ok", "data": instruments, "source": _SOURCE}
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/search")
async def search_instruments(
    request: Request,
    q: str = Query(..., description="Search query (symbol or company name)"),
    exchange: str = Query("NSE", description="Exchange to search within"),
) -> dict[str, Any]:
    """Search instruments by name or trading symbol.

    Example: ``/api/market-data/search?q=reliance&exchange=NSE``
    """
    hist = _get_historical(request)
    try:
        results = await hist.search_instrument(query=q, exchange=exchange)
        log.info(
            "market_data.search_served",
            message="Market data received from Zerodha Kite API",
            query=q,
            exchange=exchange,
            results=len(results),
            source=_SOURCE,
        )
        return {"status": "ok", "data": results, "source": _SOURCE}
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
