"""FastAPI routes for live market data and historical charts.

Provides REST endpoints for live quotes, LTP, historical OHLCV candles,
instrument listing, and instrument search.  Data is sourced from the active
broker's market data API (Zerodha Kite Connect, Groww Trading API, etc.)
with no synthetic/mock fallback.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from hedgefund.auth.middleware import get_current_user_optional

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/market-data", tags=["market-data"])

_SOURCE = "Zerodha Kite API"
_GROWW_SOURCE = "Groww Trading API"


async def _get_active_broker_type(request: Request) -> str:
    """Return the active broker type, checking in-memory router first."""
    from hedgefund.execution.capabilities import BROKER_CAPABILITIES

    def _type_from_id(broker_id: str) -> str:
        for known in BROKER_CAPABILITIES:
            if broker_id == known or broker_id.startswith(known + "_"):
                return known
        return ""

    # 1. In-memory BrokerRouter (always current after switch)
    br = getattr(request.app.state, "broker_router", None)
    if br:
        try:
            # Check all users (dashboard may not have user context here)
            for uid, bid in getattr(br, "_user_active_broker", {}).items():
                bt = _type_from_id(bid)
                if bt:
                    return bt
        except Exception:
            pass

    # 2. Fall back to MongoDB
    db = getattr(request.app.state, "db", None)
    if db is not None:
        try:
            pref = await db.user_preferences.find_one({})
            if pref:
                broker_id = pref.get("active_broker", "")
                bt = _type_from_id(broker_id)
                if bt:
                    return bt
        except Exception:
            pass
    return ""


def _get_feed(request: Request) -> Any:
    """Retrieve the ZerodhaMarketFeed from app state, or raise 503."""
    feed = getattr(request.app.state, "zerodha_feed", None)
    if feed is None:
        raise HTTPException(
            status_code=503,
            detail="Market feed is not available. Check broker credentials.",
        )
    return feed


async def _get_groww_access_token() -> str | None:
    """Exchange Groww API key + secret for an access token."""
    try:
        import asyncio
        from hedgefund.security.credential_store import CredentialStore
        from growwapi import GrowwAPI
        store = CredentialStore()
        api_key = store.retrieve("groww", "api_key")
        api_secret = store.retrieve("groww", "api_secret")
        if not api_key:
            return None
        return await asyncio.to_thread(
            GrowwAPI.get_access_token, api_key=api_key, secret=api_secret,
        )
    except Exception as exc:
        log.warning("groww_token_exchange_failed", error=str(exc))
        return None


def _to_groww_symbol(symbol: str) -> str:
    """Convert 'NSE:RELIANCE' or 'RELIANCE' to Groww format 'NSE_RELIANCE'."""
    if ":" in symbol:
        return symbol.replace(":", "_")
    return f"NSE_{symbol}"


async def _groww_ltp(symbols: list[str]) -> dict[str, Any] | None:
    """Fetch LTP from Groww Trading API for the given symbols."""
    try:
        access_token = await _get_groww_access_token()
        if not access_token:
            return None

        import httpx
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-VERSION": "1.0",
            "Accept": "application/json",
        }
        # Convert symbols to Groww format: NSE:RELIANCE → NSE_RELIANCE
        exchange_symbols = ",".join(_to_groww_symbol(s) for s in symbols)
        params = {"segment": "CASH", "exchange_symbols": exchange_symbols}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.groww.in/v1/live-data/ltp",
                headers=headers,
                params=params,
            )
            if resp.status_code != 200:
                log.warning("groww_ltp_failed", status=resp.status_code)
                return None
            return resp.json()
    except Exception as exc:
        log.warning("groww_ltp_error", error=str(exc))
        return None


async def _groww_quote(symbol: str) -> dict[str, Any] | None:
    """Fetch a full quote from Groww Trading API."""
    try:
        access_token = await _get_groww_access_token()
        if not access_token:
            return None

        import httpx
        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-API-VERSION": "1.0",
            "Accept": "application/json",
        }
        # Parse exchange:symbol format
        parts = symbol.split(":")
        exchange = parts[0] if len(parts) > 1 else "NSE"
        trading_symbol = parts[1] if len(parts) > 1 else parts[0]

        params = {
            "exchange": exchange,
            "segment": "CASH",
            "trading_symbol": trading_symbol,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.groww.in/v1/live-data/quote",
                headers=headers,
                params=params,
            )
            if resp.status_code != 200:
                log.warning("groww_quote_failed", status=resp.status_code)
                return None
            return resp.json()
    except Exception as exc:
        log.warning("groww_quote_error", error=str(exc))
        return None


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
    Routes to the active broker's market data API.
    """
    active = await _get_active_broker_type(request)

    # Try Groww if it's the active broker
    if active == "groww":
        data = await _groww_quote(symbol)
        if data is not None:
            return {"status": "ok", "data": data, "source": _GROWW_SOURCE}

    # Try Zerodha feed
    feed = getattr(request.app.state, "zerodha_feed", None)
    if feed is not None:
        try:
            data = await feed.get_quote([symbol])
            if data:
                log.info(
                    "market_data.quote_served",
                    symbol=symbol,
                    source=_SOURCE,
                )
                return {"status": "ok", "data": data, "source": _SOURCE}
        except (PermissionError, RuntimeError):
            pass

    # Groww fallback if not already tried
    if active != "groww":
        data = await _groww_quote(symbol)
        if data is not None:
            return {"status": "ok", "data": data, "source": _GROWW_SOURCE}

    raise HTTPException(
        status_code=503,
        detail="No market data feed available. Check broker credentials.",
    )


@router.get("/ltp")
async def get_ltp(
    request: Request,
    symbols: str = Query(..., description="Comma-separated symbols, e.g. NSE:NIFTY+50,NSE:RELIANCE"),
) -> dict[str, Any]:
    """Get last traded price for comma-separated symbols.

    Example: ``/api/market-data/ltp?symbols=NSE:NIFTY+50,NSE:RELIANCE``
    Routes to the active broker's market data API.
    """
    instrument_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not instrument_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    active = await _get_active_broker_type(request)

    # Try Groww if active
    if active == "groww":
        data = await _groww_ltp(instrument_list)
        if data is not None:
            return {"status": "ok", "data": data, "source": _GROWW_SOURCE}

    # Try Zerodha feed
    feed = getattr(request.app.state, "zerodha_feed", None)
    if feed is not None:
        try:
            data = await feed.get_ltp(instrument_list)
            log.info(
                "market_data.ltp_served",
                symbols=len(instrument_list),
                source=_SOURCE,
            )
            return {"status": "ok", "data": data, "source": _SOURCE}
        except (PermissionError, RuntimeError):
            pass

    # Groww fallback
    if active != "groww":
        data = await _groww_ltp(instrument_list)
        if data is not None:
            return {"status": "ok", "data": data, "source": _GROWW_SOURCE}

    raise HTTPException(
        status_code=503,
        detail="No market data feed available. Check broker credentials.",
    )


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
