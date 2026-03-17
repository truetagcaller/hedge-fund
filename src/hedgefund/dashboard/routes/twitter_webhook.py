"""X (Twitter) webhook receiver endpoint.

Handles:
- **GET /twitter** — CRC (Challenge-Response Check) for webhook registration.
  Twitter sends a ``crc_token`` query param; we must respond with an
  HMAC-SHA256 signature using the app's consumer secret.
- **POST /twitter** — Incoming Account Activity events (tweets, follows,
  DMs, etc.).  We extract tweet_create_events, run sentiment analysis,
  and publish SENTIMENT events to the EventBus.

Webhook URL: https://hedgefund.viewfir.com/twitter
"""

from __future__ import annotations

import hashlib
import hmac
import base64
from datetime import datetime, timezone
from typing import Any, Dict

import structlog
from fastapi import APIRouter, Query, Request, status

log = structlog.get_logger(__name__)

router = APIRouter(tags=["twitter-webhook"])


# ---------------------------------------------------------------------------
# GET /twitter — CRC challenge-response
# ---------------------------------------------------------------------------

@router.get("/twitter")
async def twitter_crc_challenge(
    request: Request,
    crc_token: str = Query(..., description="CRC token sent by Twitter"),
) -> Dict[str, str]:
    """Respond to Twitter's CRC validation request.

    Twitter periodically sends a GET with ``crc_token``.  We must return
    ``{"response_token": "sha256=<HMAC-SHA256(crc_token, consumer_secret)>"}``
    to prove we own the webhook.
    """
    consumer_secret = _get_consumer_secret(request)
    if not consumer_secret:
        log.error("twitter_webhook.crc_no_secret")
        return {"response_token": "sha256="}

    signature = hmac.new(
        consumer_secret.encode("utf-8"),
        crc_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    response_token = "sha256=" + base64.b64encode(signature).decode("utf-8")

    log.info("twitter_webhook.crc_responded")
    return {"response_token": response_token}


# ---------------------------------------------------------------------------
# POST /twitter — Incoming Account Activity events
# ---------------------------------------------------------------------------

@router.post("/twitter", status_code=status.HTTP_200_OK)
async def twitter_webhook_event(request: Request) -> Dict[str, str]:
    """Receive and process Twitter Account Activity webhook events.

    Twitter POSTs JSON payloads containing events like:
    - ``tweet_create_events`` — new tweets from subscribed users
    - ``favorite_events`` — likes
    - ``follow_events`` — follows

    We focus on ``tweet_create_events`` for sentiment analysis.
    """
    try:
        payload = await request.json()
    except Exception:
        log.warning("twitter_webhook.invalid_payload")
        return {"status": "error", "message": "invalid payload"}

    # Identify the user this event belongs to
    for_user_id = payload.get("for_user_id", "unknown")

    log.info(
        "twitter_webhook.event_received",
        for_user_id=for_user_id,
        keys=list(payload.keys()),
        source="X (Twitter) Webhook",
    )

    # ── Process tweet_create_events ───────────────────────────────────
    tweet_events = payload.get("tweet_create_events", [])
    if tweet_events:
        await _process_tweet_events(request, tweet_events, for_user_id)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_consumer_secret(request: Request) -> str:
    """Retrieve the Twitter consumer secret from app state or settings."""
    # Check if stored in app state (set during X account connection)
    settings = getattr(request.app.state, "settings", None)
    if settings:
        # Look in auth config or environment
        secret = getattr(settings, "_twitter_consumer_secret", "")
        if secret:
            return secret

    # Check credential store
    try:
        from hedgefund.security.credential_store import CredentialStore
        store = CredentialStore()
        secret = store.retrieve("twitter", "consumer_secret")
        if secret:
            return secret
    except Exception:  # noqa: S110
            log.debug("unexpected_error", exc_info=True)

    # Fall back to environment variable
    import os
    return os.environ.get("TWITTER_CONSUMER_SECRET", "")


async def _process_tweet_events(
    request: Request,
    tweets: list[Dict[str, Any]],
    for_user_id: str,
) -> None:
    """Process incoming tweet events and publish sentiment to EventBus."""
    import re

    ticker_pattern = re.compile(r"\$([A-Z]{1,6})\b")
    event_bus = _get_event_bus(request)
    social_manager = _get_social_manager(request)
    db = _get_db(request)

    for tweet in tweets:
        text = tweet.get("text", "")
        tweet_id = tweet.get("id_str", tweet.get("id", ""))
        user_info = tweet.get("user", {})
        author = user_info.get("screen_name", "")
        author_name = user_info.get("name", "")
        followers = user_info.get("followers_count", 0)
        created_at_str = tweet.get("created_at", "")

        # Extract tickers mentioned
        tickers = [m.group(1) for m in ticker_pattern.finditer(text)]

        # Sentiment scoring
        sentiment_score = 0.0
        magnitude = 0.5
        try:
            from hedgefund.data.social_feed import _lexicon_sentiment
            sentiment_score, magnitude = _lexicon_sentiment(text)
        except ImportError:
            # Basic fallback: count positive/negative words
            positive = {"bullish", "buy", "long", "moon", "pump", "calls", "up", "green"}
            negative = {"bearish", "sell", "short", "crash", "dump", "puts", "down", "red"}
            words = set(text.lower().split())
            pos_count = len(words & positive)
            neg_count = len(words & negative)
            total = pos_count + neg_count
            if total > 0:
                sentiment_score = (pos_count - neg_count) / total
                magnitude = min(1.0, total / 5.0)

        log.info(
            "twitter_webhook.tweet_processed",
            tweet_id=tweet_id,
            author=f"@{author}",
            tickers=tickers,
            sentiment=round(sentiment_score, 3),
            source="X (Twitter) Webhook",
        )

        # Publish to EventBus for each mentioned ticker
        if event_bus:
            from hedgefund.streaming.event_bus import Event, EventType
            for ticker in tickers:
                event = Event(
                    event_type=EventType.SENTIMENT,
                    timestamp=datetime.now(timezone.utc),
                    symbol=ticker,
                    data={
                        "score": sentiment_score,
                        "sentiment_score": sentiment_score,
                        "magnitude": magnitude,
                        "source": "twitter",
                        "tweet_id": str(tweet_id),
                        "author": author,
                        "author_name": author_name,
                        "followers": followers,
                        "text": text[:500],
                        "tickers": tickers,
                        "mention_count": len(tickers),
                        "is_live": True,
                    },
                    source="X (Twitter) Webhook",
                )
                await event_bus.publish(event)

        # Feed into SocialStreamManager if available
        if social_manager:
            try:
                tweet_data = {
                    "id": str(tweet_id),
                    "text": text,
                    "author_id": user_info.get("id_str", ""),
                    "_author_name": author_name,
                    "_author_followers": followers,
                    "created_at": created_at_str,
                    "public_metrics": {
                        "like_count": tweet.get("favorite_count", 0),
                        "retweet_count": tweet.get("retweet_count", 0),
                    },
                }
                social_manager._process_tweet_webhook(tweet_data)
            except Exception:
                log.debug("twitter_webhook.social_manager_feed_failed", exc_info=True)

        # Store in MongoDB (with write guard)
        if db and tickers:
            try:
                from hedgefund.data.write_guard import WriteGuard
                doc = {
                    "tweet_id": str(tweet_id),
                    "text": text[:500],
                    "author": author,
                    "author_name": author_name,
                    "followers": followers,
                    "tickers": tickers,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": (
                        "bullish" if sentiment_score > 0.1
                        else "bearish" if sentiment_score < -0.1
                        else "neutral"
                    ),
                    "ticker_mentions": tickers,
                    "source": "twitter",
                    "timestamp": datetime.now(timezone.utc),
                    "for_user_id": for_user_id,
                    "is_live": True,
                }
                WriteGuard.validate_sentiment(doc)
                await db.x_posts.insert_one(doc)
            except Exception:
                log.debug("twitter_webhook.db_store_failed", exc_info=True)


def _get_event_bus(request: Request) -> Any:
    """Get the EventBus from app state if available."""
    # TradingApplication sets this; dashboard-only mode may not have it
    dsm = getattr(request.app.state, "data_source_manager", None)
    if dsm:
        return getattr(dsm, "_event_bus", None)
    return None


def _get_social_manager(request: Request) -> Any:
    """Get the SocialStreamManager from app state."""
    return getattr(request.app.state, "social_stream_manager", None)


def _get_db(request: Request) -> Any:
    """Get MongoDB from app state."""
    return getattr(request.app.state, "db", None)
