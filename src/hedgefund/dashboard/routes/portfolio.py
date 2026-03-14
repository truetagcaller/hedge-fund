"""Portfolio, positions, trades, and P&L endpoints with multi-user data isolation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, Query, Request

from hedgefund.auth.middleware import get_current_user
from hedgefund.types import Position, PortfolioSnapshot, TradeRecord

log = structlog.get_logger(__name__)

router = APIRouter(tags=["portfolio"])


def _serialize_position(pos: Position) -> Dict[str, Any]:
    """Convert a Position dataclass to a JSON-safe dict."""
    data = asdict(pos)
    data["market_value"] = pos.market_value
    data["notional_value"] = pos.notional_value
    return data


def _serialize_snapshot(snap: PortfolioSnapshot) -> Dict[str, Any]:
    """Convert a PortfolioSnapshot to a JSON-safe dict."""
    return {
        "timestamp": snap.timestamp.isoformat(),
        "cash": snap.cash,
        "net_liquidation": snap.net_liquidation,
        "total_market_value": snap.total_market_value,
        "position_count": snap.position_count,
        "total_delta": snap.total_delta,
        "total_gamma": snap.total_gamma,
        "total_theta": snap.total_theta,
        "total_vega": snap.total_vega,
        "daily_pnl": snap.daily_pnl,
        "total_pnl": snap.total_pnl,
        "drawdown_pct": snap.drawdown_pct,
        "high_water_mark": snap.high_water_mark,
    }


def _get_db(request: Request):
    """Extract MongoDB instance from app state, or None."""
    return getattr(request.app.state, "db", None)


def _empty_snapshot() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cash": 0.0,
        "net_liquidation": 0.0,
        "total_market_value": 0,
        "position_count": 0,
        "total_delta": 0.0,
        "total_gamma": 0.0,
        "total_theta": 0.0,
        "total_vega": 0.0,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "drawdown_pct": 0.0,
        "high_water_mark": 0.0,
    }


async def _get_active_broker_type(request: Request, user_id: str) -> str:
    """Return the active broker type for this user, or empty string."""
    from hedgefund.execution.capabilities import BROKER_CAPABILITIES

    def _type_from_id(broker_id: str) -> str:
        for known in BROKER_CAPABILITIES:
            if broker_id == known or broker_id.startswith(known + "_"):
                return known
        return ""

    # 1. Check in-memory BrokerRouter (always up to date after switch)
    br = getattr(request.app.state, "broker_router", None)
    if br:
        try:
            bid = await br.get_active_broker(user_id)
            if bid:
                bt = _type_from_id(bid)
                if bt:
                    return bt
        except Exception:
            pass

    # 2. Fall back to MongoDB
    db = _get_db(request)
    if db is not None:
        try:
            pref = await db.user_preferences.find_one({"user_id": user_id})
            if pref:
                broker_id = pref.get("active_broker", "")
                bt = _type_from_id(broker_id)
                if bt:
                    return bt
        except Exception:
            pass
    return ""


async def _fetch_groww_portfolio() -> Dict[str, Any] | None:
    """Fetch portfolio directly from Groww Trading API."""
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        api_key = store.retrieve("groww", "api_key")
        api_secret = store.retrieve("groww", "api_secret")
        if not api_key:
            return None

        import httpx
        token = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
        headers = {
            "Authorization": token,
            "X-API-VERSION": "1.0",
            "Accept": "application/json",
        }
        base = "https://api.groww.in/v1"

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Fetch user profile
            profile_resp = await client.get(f"{base}/user/profile", headers=headers)
            if profile_resp.status_code != 200:
                log.warning(
                    "portfolio.groww_profile_failed",
                    status=profile_resp.status_code,
                )
                return None

            profile_data = profile_resp.json()
            success = profile_data.get("success", profile_data)
            user_info = (
                success.get("data", success)
                if isinstance(success, dict) else {}
            )

            # Fetch holdings
            positions: list[Dict[str, Any]] = []
            holdings_value = 0.0
            holdings_pnl = 0.0

            hold_resp = await client.get(f"{base}/holdings/user", headers=headers)
            if hold_resp.status_code == 200:
                hold_data = hold_resp.json()
                holdings_list = _extract_groww_list(hold_data, "holdings")
                for h in holdings_list:
                    symbol = h.get("trading_symbol", h.get("tradingSymbol", ""))
                    qty = int(h.get("quantity", 0))
                    if qty == 0:
                        continue
                    avg_price = float(h.get("average_price", h.get("avgPrice", 0)))
                    last_price = float(h.get("ltp", h.get("lastPrice", avg_price)))
                    pnl = (last_price - avg_price) * qty
                    holdings_value += last_price * qty
                    holdings_pnl += pnl
                    positions.append({
                        "symbol": symbol,
                        "exchange": h.get("exchange", "NSE"),
                        "quantity": qty,
                        "avg_price": avg_price,
                        "last_price": last_price,
                        "pnl": round(pnl, 2),
                        "product": "CNC",
                        "source": "groww",
                        "is_holding": True,
                    })

            # Fetch intraday positions
            pos_resp = await client.get(f"{base}/positions/user", headers=headers)
            pos_value = 0.0
            pos_pnl = 0.0
            if pos_resp.status_code == 200:
                pos_data = pos_resp.json()
                pos_list = _extract_groww_list(pos_data, "positions")
                for p in pos_list:
                    symbol = p.get("trading_symbol", p.get("tradingSymbol", ""))
                    qty = int(p.get("quantity", 0))
                    if qty == 0:
                        continue
                    net_price = float(p.get("net_price", p.get("netPrice", 0)))
                    realised = float(p.get("realised_pnl", p.get("realisedPnl", 0)))
                    pos_value += abs(qty) * net_price
                    pos_pnl += realised
                    positions.append({
                        "symbol": symbol,
                        "exchange": p.get("exchange", "NSE"),
                        "quantity": qty,
                        "avg_price": net_price,
                        "last_price": net_price,
                        "pnl": round(realised, 2),
                        "product": p.get("product", "MIS"),
                        "source": "groww",
                    })

            total_value = holdings_value + pos_value
            total_pnl = holdings_pnl + pos_pnl

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cash": 0.0,
                "net_liquidation": total_value,
                "total_market_value": total_value,
                "position_count": len(positions),
                "positions": positions,
                "total_delta": 0.0,
                "total_gamma": 0.0,
                "total_theta": 0.0,
                "total_vega": 0.0,
                "daily_pnl": total_pnl,
                "total_pnl": total_pnl,
                "drawdown_pct": 0.0,
                "high_water_mark": total_value,
                "broker": "groww",
                "source": "Groww Trading API",
                "ucc": user_info.get("ucc", ""),
                "segments": user_info.get("activeSegments", []),
            }
    except Exception as exc:
        log.warning("portfolio.groww_fetch_failed", error=str(exc))
        return None


def _extract_groww_list(result: Any, key: str) -> list[Dict[str, Any]]:
    """Extract a list from Groww API response shapes."""
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    success = result.get("success", result)
    if isinstance(success, dict):
        data = success.get("data", success)
        if isinstance(data, dict):
            items = data.get(key, [])
            if isinstance(items, list):
                return items
        if isinstance(data, list):
            return data
    items = result.get(key, [])
    return items if isinstance(items, list) else []


async def _fetch_zerodha_portfolio() -> Dict[str, Any] | None:
    """Fetch portfolio directly from Zerodha Kite API."""
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        api_key = store.retrieve("zerodha", "api_key")
        access_token = store.retrieve("zerodha", "access_token")
        if not api_key or not access_token:
            return None

        import httpx
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {api_key}:{access_token}",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Fetch margins
            margins_resp = await client.get(
                "https://api.kite.trade/user/margins", headers=headers,
            )
            if margins_resp.status_code != 200:
                log.warning(
                    "portfolio.zerodha_margins_failed",
                    status=margins_resp.status_code,
                )
                return None

            margins = margins_resp.json().get("data", {})
            equity = margins.get("equity", {})
            commodity = margins.get("commodity", {})

            eq_net = equity.get("net", 0)
            eq_cash = equity.get("available", {}).get("cash", 0)
            eq_opening = equity.get("available", {}).get("opening_balance", 0)
            eq_live = equity.get("available", {}).get("live_balance", 0)
            eq_collateral = equity.get("available", {}).get("collateral", 0)

            utilised = equity.get("utilised", {})
            eq_debits = utilised.get("debits", 0)

            # Fetch positions
            pos_resp = await client.get(
                "https://api.kite.trade/portfolio/positions",
                headers=headers,
            )
            positions = []
            net_positions = []
            if pos_resp.status_code == 200:
                pos_data = pos_resp.json().get("data", {})
                net_positions = pos_data.get("net", [])
                day_positions = pos_data.get("day", [])

            # Fetch holdings
            hold_resp = await client.get(
                "https://api.kite.trade/portfolio/holdings",
                headers=headers,
            )
            holdings = []
            if hold_resp.status_code == 200:
                holdings = hold_resp.json().get("data", [])

            # Calculate totals
            total_pnl = sum(p.get("pnl", 0) for p in net_positions)
            total_market_value = sum(
                abs(p.get("quantity", 0)) * p.get("last_price", 0)
                for p in net_positions
            )
            holdings_value = sum(
                h.get("quantity", 0) * h.get("last_price", 0)
                for h in holdings
            )

            # Build positions list for the UI
            for p in net_positions:
                positions.append({
                    "symbol": p.get("tradingsymbol", ""),
                    "exchange": p.get("exchange", ""),
                    "quantity": p.get("quantity", 0),
                    "avg_price": p.get("average_price", 0),
                    "last_price": p.get("last_price", 0),
                    "pnl": p.get("pnl", 0),
                    "product": p.get("product", ""),
                    "source": "zerodha",
                })

            for h in holdings:
                positions.append({
                    "symbol": h.get("tradingsymbol", ""),
                    "exchange": h.get("exchange", ""),
                    "quantity": h.get("quantity", 0),
                    "avg_price": h.get("average_price", 0),
                    "last_price": h.get("last_price", 0),
                    "pnl": h.get("pnl", 0),
                    "product": "CNC",
                    "source": "zerodha",
                    "is_holding": True,
                })

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cash": eq_cash or eq_opening or eq_live,
                "net_liquidation": eq_net,
                "total_market_value": total_market_value + holdings_value,
                "position_count": len(net_positions) + len(holdings),
                "positions": positions,
                "total_delta": 0.0,
                "total_gamma": 0.0,
                "total_theta": 0.0,
                "total_vega": 0.0,
                "daily_pnl": total_pnl,
                "total_pnl": total_pnl,
                "drawdown_pct": 0.0,
                "high_water_mark": eq_net,
                "broker": "zerodha",
                "source": "Zerodha Kite API",
                "margin_available": equity.get("available", {}),
                "margin_utilised": utilised,
                "collateral": eq_collateral,
            }
    except Exception as exc:
        log.warning("portfolio.zerodha_fetch_failed", error=str(exc))
        return None


@router.get("/portfolio")
async def get_portfolio(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the current portfolio snapshot for the authenticated user."""
    user_id = user["user_id"]
    db = _get_db(request)

    # 1. Fetch from the user's active broker
    active_broker = await _get_active_broker_type(request, user_id)

    _broker_fetchers: Dict[str, Any] = {
        "zerodha": _fetch_zerodha_portfolio,
        "groww": _fetch_groww_portfolio,
    }

    # Try active broker first
    if active_broker in _broker_fetchers:
        data = await _broker_fetchers[active_broker]()
        if data is not None:
            return data

    # Fall back to other brokers
    for broker_type, fetcher in _broker_fetchers.items():
        if broker_type == active_broker:
            continue
        data = await fetcher()
        if data is not None:
            return data

    # 2. Try MongoDB positions
    if db is not None:
        positions_cursor = db.positions.find({"user_id": user_id})
        positions = await positions_cursor.to_list(length=1000)
        if positions:
            total_market_value = sum(
                p.get("current_price", 0) * p.get("quantity", 0)
                for p in positions
            )
            total_unrealized = sum(
                p.get("unrealized_pnl", 0.0) for p in positions
            )
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cash": 0.0,
                "net_liquidation": total_market_value,
                "total_market_value": total_market_value,
                "position_count": len(positions),
                "total_delta": 0.0,
                "total_gamma": 0.0,
                "total_theta": 0.0,
                "total_vega": 0.0,
                "daily_pnl": 0.0,
                "total_pnl": total_unrealized,
                "drawdown_pct": 0.0,
                "high_water_mark": 0.0,
            }

    # 3. Fall back to broker_manager
    broker_mgr = getattr(request.app.state, "broker_manager", None)
    if broker_mgr is not None:
        try:
            snapshot = await broker_mgr.get_aggregate_portfolio()
            return _serialize_snapshot(snapshot)
        except Exception:
            log.warning("portfolio.broker_manager_failed", exc_info=True)

    portfolio_mgr = getattr(request.app.state, "portfolio_manager", None)
    if portfolio_mgr is not None:
        snapshot: PortfolioSnapshot = await portfolio_mgr.get_snapshot()
        return _serialize_snapshot(snapshot)

    return _empty_snapshot()


@router.get("/positions")
async def get_positions(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return all open positions for the authenticated user."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Try loading from MongoDB first
    if db is not None:
        cursor = db.positions.find({"user_id": user_id})
        docs = await cursor.to_list(length=1000)
        if docs:
            positions = []
            for doc in docs:
                doc.pop("_id", None)
                doc.pop("user_id", None)
                positions.append(doc)
            return {"positions": positions, "count": len(positions)}

    # Fall back to broker_manager aggregate positions
    broker_mgr = getattr(request.app.state, "broker_manager", None)
    if broker_mgr is not None:
        try:
            agg_positions = await broker_mgr.get_aggregate_positions()
            positions = [_serialize_position(p) for p in agg_positions]
            return {"positions": positions, "count": len(positions)}
        except Exception:
            log.warning("positions.broker_manager_fallback_failed", exc_info=True)

    portfolio_mgr = getattr(request.app.state, "portfolio_manager", None)
    if portfolio_mgr is not None:
        snapshot: PortfolioSnapshot = await portfolio_mgr.get_snapshot()
        positions = [_serialize_position(p) for p in snapshot.positions]
        return {"positions": positions, "count": len(positions)}

    return {"positions": [], "count": 0}


@router.get("/trades")
async def get_trades(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None, description="Filter by underlying symbol"),
    mode: Optional[str] = Query(None, description="Filter by mode: live, paper, backtest"),
) -> Dict[str, Any]:
    """Return trade history for the authenticated user with pagination."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Try MongoDB first
    if db is not None:
        query: Dict[str, Any] = {"user_id": user_id}
        if symbol:
            query["symbol"] = symbol.upper()
        if mode:
            query["mode"] = mode

        total = await db.trades.count_documents(query)
        cursor = db.trades.find(query).sort("entry_time", -1).skip(offset).limit(limit)
        trades = []
        async for doc in cursor:
            doc.pop("_id", None)
            doc.pop("user_id", None)
            # Serialize datetime fields
            for key in ("entry_time", "exit_time"):
                if isinstance(doc.get(key), datetime):
                    doc[key] = doc[key].isoformat()
            trades.append(doc)

        return {
            "trades": trades,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # Fall back to in-memory trade store
    trade_store = getattr(request.app.state, "trade_store", None)
    if trade_store is None:
        return {"trades": [], "total": 0, "limit": limit, "offset": offset}

    trades_list: List[TradeRecord] = await trade_store.get_trades(
        limit=limit,
        offset=offset,
        symbol=symbol,
    )
    total_count: int = await trade_store.count_trades(symbol=symbol)

    return {
        "trades": [asdict(t) for t in trades_list],
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@router.get("/pnl")
async def get_pnl(
    request: Request,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return P&L breakdown for the authenticated user."""
    user_id = user["user_id"]
    db = _get_db(request)

    # Try computing from user's MongoDB data
    if db is not None:
        # Per-position P&L from positions collection
        positions_cursor = db.positions.find({"user_id": user_id})
        per_position = []
        total_unrealized = 0.0
        total_realized = 0.0

        async for pos in positions_cursor:
            unrealized = pos.get("unrealized_pnl", 0.0)
            total_unrealized += unrealized
            per_position.append({
                "symbol": pos.get("symbol", ""),
                "underlying": pos.get("underlying", ""),
                "unrealized_pnl": unrealized,
                "realized_pnl": pos.get("realized_pnl", 0.0),
                "market_value": pos.get("current_price", 0.0) * pos.get("quantity", 0),
            })

        # Strategy-level P&L from trades
        strategy_pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": "$strategy_name",
                    "total_pnl": {"$sum": "$pnl"},
                }
            },
        ]
        strategy_pnl: Dict[str, float] = {}
        async for doc in db.trades.aggregate(strategy_pipeline):
            if doc["_id"]:
                strategy_pnl[doc["_id"]] = round(doc["total_pnl"], 2)

        if per_position or strategy_pnl:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "daily_pnl": 0.0,
                "total_pnl": round(total_unrealized + total_realized, 2),
                "drawdown_pct": 0.0,
                "high_water_mark": 0.0,
                "per_position": per_position,
                "per_strategy": strategy_pnl,
            }

    # Fall back to portfolio manager
    portfolio_mgr = getattr(request.app.state, "portfolio_manager", None)
    if portfolio_mgr is None:
        return _empty_pnl()

    snapshot: PortfolioSnapshot = await portfolio_mgr.get_snapshot()

    per_position = [
        {
            "symbol": p.contract.symbol,
            "underlying": p.contract.underlying,
            "unrealized_pnl": p.unrealized_pnl,
            "realized_pnl": p.realized_pnl,
            "market_value": p.market_value,
        }
        for p in snapshot.positions
    ]

    trade_store = getattr(request.app.state, "trade_store", None)
    strategy_pnl_data: Dict[str, float] = {}
    if trade_store is not None:
        strategy_pnl_data = await trade_store.get_strategy_pnl()

    return {
        "timestamp": snapshot.timestamp.isoformat(),
        "daily_pnl": snapshot.daily_pnl,
        "total_pnl": snapshot.total_pnl,
        "drawdown_pct": snapshot.drawdown_pct,
        "high_water_mark": snapshot.high_water_mark,
        "per_position": per_position,
        "per_strategy": strategy_pnl_data,
    }


def _empty_snapshot() -> Dict[str, Any]:
    """Fallback response when portfolio manager is not attached."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cash": 0.0,
        "net_liquidation": 0.0,
        "total_market_value": 0.0,
        "position_count": 0,
        "total_delta": 0.0,
        "total_gamma": 0.0,
        "total_theta": 0.0,
        "total_vega": 0.0,
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "drawdown_pct": 0.0,
        "high_water_mark": 0.0,
    }


def _empty_pnl() -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "drawdown_pct": 0.0,
        "high_water_mark": 0.0,
        "per_position": [],
        "per_strategy": {},
    }
