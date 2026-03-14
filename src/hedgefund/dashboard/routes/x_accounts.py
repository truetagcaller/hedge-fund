"""X (Twitter) account management and sentiment endpoints.

Provides connecting/disconnecting X accounts, managing tracked tickers and
influencers, viewing X feed posts, and querying aggregated sentiment data.
All data is scoped to the authenticated user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from hedgefund.auth.database import MongoDB
from hedgefund.auth.middleware import get_current_user
from hedgefund.auth.models import encrypt_credentials

log = structlog.get_logger(__name__)

router = APIRouter(tags=["x-accounts"])


# ── Pydantic models ──────────────────────────────────────────────────────────


class ConnectXRequest(BaseModel):
    auth_method: str = Field(
        ..., description="Authentication method: 'api_key', 'oauth', or 'token'"
    )
    credentials: Dict[str, str] = Field(
        ...,
        description=(
            "Credentials dict. Keys depend on auth_method: "
            "api_key/api_secret, bearer_token, access_token/access_token_secret"
        ),
    )


class ConnectXResponse(BaseModel):
    x_user_id: str
    username: str
    display_name: str
    connected_at: str


class DisconnectXResponse(BaseModel):
    status: str = "disconnected"


class XAccountInfo(BaseModel):
    x_user_id: str
    username: str
    connected_at: str
    is_active: bool


class XAccountListResponse(BaseModel):
    accounts: List[XAccountInfo]


class TrackedTickersRequest(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to track, e.g. ['SPY', 'NIFTY', '$BTC']")


class TrackedInfluencersRequest(BaseModel):
    influencers: List[str] = Field(
        ..., description="Influencer handles to track, e.g. ['@elonmusk']"
    )


class XPostItem(BaseModel):
    tweet_id: str
    text: str
    author: str
    sentiment_score: float
    sentiment_label: str
    ticker_mentions: List[str]
    timestamp: str


class XFeedResponse(BaseModel):
    posts: List[XPostItem]


class SymbolSentiment(BaseModel):
    score: float
    label: str
    magnitude: float
    tweet_count: int


class XSentimentResponse(BaseModel):
    sentiments: Dict[str, SymbolSentiment]


class TrendingTicker(BaseModel):
    symbol: str
    mentions: int
    sentiment_score: float


class XTrendingResponse(BaseModel):
    trending: List[TrendingTicker]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_db(request: Request) -> MongoDB:
    """Extract MongoDB instance from app state."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    return db


async def _validate_x_credentials(credentials: Dict[str, str]) -> Dict[str, Any]:
    """Validate X credentials by calling the Twitter API v2 /2/users/me endpoint.

    Returns user info dict on success, raises HTTPException on failure.
    """
    import httpx

    # Determine which token to use for the API call
    bearer_token = credentials.get("bearer_token")
    access_token = credentials.get("access_token")

    headers: Dict[str, str] = {}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    elif access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either bearer_token or access_token is required",
        )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.twitter.com/2/users/me",
                headers=headers,
                params={"user.fields": "id,name,username"},
            )
    except httpx.RequestError as exc:
        log.error("x_accounts.validation_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to reach Twitter API",
        )

    if resp.status_code != 200:
        log.warning("x_accounts.invalid_credentials", status=resp.status_code)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X credentials",
        )

    data = resp.json().get("data", {})
    return {
        "x_user_id": data.get("id", ""),
        "username": data.get("username", ""),
        "display_name": data.get("name", ""),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/x/connect", status_code=status.HTTP_201_CREATED)
async def connect_x_account(
    request: Request,
    body: ConnectXRequest,
    user: dict = Depends(get_current_user),
) -> ConnectXResponse:
    """Connect an X (Twitter) account for the current user."""
    db = _get_db(request)
    user_id = user["user_id"]

    # Validate credentials against Twitter API
    x_info = await _validate_x_credentials(body.credentials)

    # Encrypt the credentials before storing
    encrypted = encrypt_credentials(body.credentials)
    now = datetime.now(timezone.utc)

    doc = {
        "user_id": user_id,
        "x_user_id": x_info["x_user_id"],
        "username": x_info["username"],
        "display_name": x_info["display_name"],
        "auth_method": body.auth_method,
        "credentials_encrypted": encrypted,
        "connected_at": now,
        "is_active": True,
        "tracked_tickers": [],
        "tracked_influencers": [],
    }

    # Upsert: if this X account is already connected for this user, update it
    await db.x_accounts.update_one(
        {"user_id": user_id, "x_user_id": x_info["x_user_id"]},
        {"$set": doc},
        upsert=True,
    )

    log.info(
        "x_accounts.connected",
        user_id=user_id,
        x_username=x_info["username"],
    )

    return ConnectXResponse(
        x_user_id=x_info["x_user_id"],
        username=x_info["username"],
        display_name=x_info["display_name"],
        connected_at=now.isoformat(),
    )


@router.delete("/x/{x_account_id}")
async def disconnect_x_account(
    request: Request,
    x_account_id: str,
    user: dict = Depends(get_current_user),
) -> DisconnectXResponse:
    """Disconnect an X account for the current user."""
    db = _get_db(request)
    user_id = user["user_id"]

    result = await db.x_accounts.delete_one(
        {"user_id": user_id, "x_user_id": x_account_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"X account {x_account_id!r} not found",
        )

    log.info("x_accounts.disconnected", user_id=user_id, x_account_id=x_account_id)
    return DisconnectXResponse()


@router.get("/x/accounts")
async def list_x_accounts(
    request: Request,
    user: dict = Depends(get_current_user),
) -> XAccountListResponse:
    """List connected X accounts for the current user (tokens masked)."""
    db = _get_db(request)
    user_id = user["user_id"]

    accounts: List[XAccountInfo] = []
    async for doc in db.x_accounts.find(
        {"user_id": user_id},
        {"credentials_encrypted": 0},  # never expose tokens
    ):
        accounts.append(
            XAccountInfo(
                x_user_id=doc["x_user_id"],
                username=doc.get("username", ""),
                connected_at=doc["connected_at"].isoformat()
                if isinstance(doc.get("connected_at"), datetime)
                else str(doc.get("connected_at", "")),
                is_active=doc.get("is_active", True),
            )
        )

    return XAccountListResponse(accounts=accounts)


@router.put("/x/tracked-tickers")
async def update_tracked_tickers(
    request: Request,
    body: TrackedTickersRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update tracked tickers for this user's sentiment analysis."""
    db = _get_db(request)
    user_id = user["user_id"]

    # Update all X accounts for this user with the new tickers list
    result = await db.x_accounts.update_many(
        {"user_id": user_id},
        {"$set": {"tracked_tickers": body.tickers}},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No X accounts found. Connect an X account first.",
        )

    log.info("x_accounts.tickers_updated", user_id=user_id, tickers=body.tickers)
    return {"status": "updated", "tickers": body.tickers}


@router.put("/x/tracked-influencers")
async def update_tracked_influencers(
    request: Request,
    body: TrackedInfluencersRequest,
    user: dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Update tracked influencers for this user's X feed."""
    db = _get_db(request)
    user_id = user["user_id"]

    result = await db.x_accounts.update_many(
        {"user_id": user_id},
        {"$set": {"tracked_influencers": body.influencers}},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No X accounts found. Connect an X account first.",
        )

    log.info("x_accounts.influencers_updated", user_id=user_id, count=len(body.influencers))
    return {"status": "updated", "influencers": body.influencers}


@router.get("/x/feed")
async def get_x_feed(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
) -> XFeedResponse:
    """Return recent tweets collected for this user."""
    db = _get_db(request)
    user_id = user["user_id"]

    query: Dict[str, Any] = {"user_id": user_id}
    if ticker:
        query["ticker_mentions"] = ticker.upper()

    posts: List[XPostItem] = []
    cursor = db.x_posts.find(query).sort("timestamp", -1).limit(limit)
    async for doc in cursor:
        posts.append(
            XPostItem(
                tweet_id=doc.get("tweet_id", ""),
                text=doc.get("text", ""),
                author=doc.get("author", ""),
                sentiment_score=doc.get("sentiment_score", 0.0),
                sentiment_label=doc.get("sentiment_label", "neutral"),
                ticker_mentions=doc.get("ticker_mentions", []),
                timestamp=doc["timestamp"].isoformat()
                if isinstance(doc.get("timestamp"), datetime)
                else str(doc.get("timestamp", "")),
            )
        )

    return XFeedResponse(posts=posts)


@router.get("/x/sentiment")
async def get_x_sentiment(
    request: Request,
    user: dict = Depends(get_current_user),
    symbols: str = Query(..., description="Comma-separated symbols, e.g. SPY,QQQ,NIFTY"),
) -> XSentimentResponse:
    """Return aggregated sentiment per symbol from this user's X data."""
    db = _get_db(request)
    user_id = user["user_id"]
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    sentiments: Dict[str, SymbolSentiment] = {}
    for sym in symbol_list:
        pipeline = [
            {"$match": {"user_id": user_id, "ticker_mentions": sym}},
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$sentiment_score"},
                    "count": {"$sum": 1},
                    "magnitude": {"$avg": {"$abs": "$sentiment_score"}},
                }
            },
        ]
        results = await db.x_posts.aggregate(pipeline).to_list(1)
        if results:
            agg = results[0]
            score = agg.get("avg_score", 0.0)
            label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
            sentiments[sym] = SymbolSentiment(
                score=round(score, 4),
                label=label,
                magnitude=round(agg.get("magnitude", 0.0), 4),
                tweet_count=agg.get("count", 0),
            )
        else:
            sentiments[sym] = SymbolSentiment(
                score=0.0,
                label="neutral",
                magnitude=0.0,
                tweet_count=0,
            )

    return XSentimentResponse(sentiments=sentiments)


@router.get("/x/trending")
async def get_x_trending(
    request: Request,
    user: dict = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
) -> XTrendingResponse:
    """Return trending tickers from this user's X feed."""
    db = _get_db(request)
    user_id = user["user_id"]

    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$unwind": "$ticker_mentions"},
        {
            "$group": {
                "_id": "$ticker_mentions",
                "mentions": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
            }
        },
        {"$sort": {"mentions": -1}},
        {"$limit": limit},
    ]

    trending: List[TrendingTicker] = []
    async for doc in db.x_posts.aggregate(pipeline):
        trending.append(
            TrendingTicker(
                symbol=doc["_id"],
                mentions=doc["mentions"],
                sentiment_score=round(doc.get("avg_sentiment", 0.0), 4),
            )
        )

    return XTrendingResponse(trending=trending)
