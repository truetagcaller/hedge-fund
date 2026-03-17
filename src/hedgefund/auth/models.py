"""MongoDB document models and helper functions for the multi-user auth system.

All documents are plain dicts (no ODM dependency).  Helper functions handle
password hashing (bcrypt) and credential encryption (Fernet via the existing
CredentialStore infrastructure).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import bcrypt

from hedgefund.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fernet encryption helpers (reuse key derivation from credential_store)
# ---------------------------------------------------------------------------

try:
    from cryptography.fernet import Fernet
    _HAS_FERNET = True
except ImportError:
    _HAS_FERNET = False

_HEDGEFUND_DIR_PATH = os.path.join(os.path.expanduser("~"), ".hedgefund")
_MASTER_KEY_FILE_PATH = os.path.join(_HEDGEFUND_DIR_PATH, ".master_key")


def _get_fernet() -> "Fernet":
    """Return a Fernet instance keyed from the shared master key."""
    if not _HAS_FERNET:
        raise RuntimeError(
            "cryptography library is required for credential encryption. "
            "Install it with: pip install cryptography"
        )

    env_key = os.environ.get("HEDGEFUND_MASTER_KEY")
    if env_key:
        raw_key = env_key.encode()
    elif os.path.exists(_MASTER_KEY_FILE_PATH):
        with open(_MASTER_KEY_FILE_PATH, "rb") as f:
            raw_key = f.read().strip()
    else:
        os.makedirs(_HEDGEFUND_DIR_PATH, exist_ok=True)
        raw_key = Fernet.generate_key()
        with open(_MASTER_KEY_FILE_PATH, "wb") as f:
            f.write(raw_key)
        os.chmod(_MASTER_KEY_FILE_PATH, 0o600)

    derived = hashlib.sha256(raw_key).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt_credentials(data: dict) -> str:
    """Encrypt a dict of credentials into a Fernet-encrypted string."""
    fernet = _get_fernet()
    payload = json.dumps(data).encode()
    return fernet.encrypt(payload).decode()


def decrypt_credentials(encrypted: str) -> dict:
    """Decrypt a Fernet-encrypted string back into a dict."""
    fernet = _get_fernet()
    decrypted = fernet.decrypt(encrypted.encode())
    return json.loads(decrypted.decode())


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------


def _hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(plain_password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Document factory helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_user_document(email: str, username: str, password: str) -> dict:
    """Create a new user document dict with a bcrypt-hashed password."""
    return {
        "email": email.lower().strip(),
        "username": username.strip(),
        "password_hash": _hash_password(password),
        "role": "user",
        "created_at": _utcnow(),
        "last_login": None,
        "is_active": True,
        "preferences": {},
    }


# ---------------------------------------------------------------------------
# Document type annotations (for reference / IDE support)
# ---------------------------------------------------------------------------

UserDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "email": str,           # unique, indexed
    "username": str,         # unique, indexed
    "password_hash": str,    # bcrypt
    "role": "user" | "admin",
    "created_at": datetime,
    "last_login": datetime | None,
    "is_active": bool,
    "preferences": dict,     # dashboard settings, default timeframe, etc.
}
"""

BrokerAccountDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,                # indexed
    "broker_type": str,            # zerodha, binance, groww, indmoney, paper
    "broker_id": str,
    "credentials_encrypted": str,  # Fernet-encrypted JSON blob
    "status": str,                 # connected, disconnected, error
    "connected_at": datetime | None,
    "last_refresh": datetime | None,
    "account_info": dict,
    "mode": "live" | "paper",
}
"""

TradeDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,          # indexed
    "trade_id": str,         # unique per user
    "signal_id": str,
    "underlying": str,
    "symbol": str,
    "side": str,
    "entry_price": float,
    "exit_price": float | None,
    "quantity": int,
    "pnl": float,
    "pnl_pct": float,
    "entry_time": datetime,
    "exit_time": datetime | None,
    "strategy_name": str,
    "mode": "live" | "paper" | "backtest",
}
"""

SignalDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,          # indexed
    "signal_id": str,
    "timestamp": datetime,
    "underlying": str,
    "action": str,
    "direction": str,
    "confidence": float,
    "entry_price": float,
    "stop_loss": float,
    "target_price": float,
    "risk_reward_ratio": float,
    "reasoning": str,
    "outcome": str | None,
    "mode": "live" | "paper" | "backtest",
}
"""

PositionDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,          # indexed
    "broker_id": str,
    "symbol": str,
    "underlying": str,
    "option_type": str | None,
    "strike": float | None,
    "quantity": int,
    "avg_entry": float,
    "current_price": float,
    "unrealized_pnl": float,
    "mode": "live" | "paper",
    "updated_at": datetime,
}
"""

XAccountDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,                    # indexed
    "x_user_id": str,
    "username": str,
    "display_name": str,
    "access_token_encrypted": str,
    "refresh_token_encrypted": str,
    "connected_at": datetime,
    "is_active": bool,
}
"""

XPostDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,           # indexed
    "tweet_id": str,          # unique
    "text": str,
    "author": str,
    "author_id": str,
    "timestamp": datetime,
    "sentiment_score": float,
    "sentiment_label": str,   # bullish, bearish, neutral
    "ticker_mentions": list[str],
    "impact_score": float,
}
"""

SentimentDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,          # indexed
    "symbol": str,
    "source": str,           # news, twitter, options_flow
    "score": float,
    "magnitude": float,
    "timestamp": datetime,
    "mode": "live" | "paper",
}
"""

BacktestDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,          # indexed
    "backtest_id": str,
    "strategy": str,
    "symbols": list[str],
    "start_date": str,
    "end_date": str,
    "initial_capital": float,
    "status": str,
    "metrics": dict | None,
    "submitted_at": datetime,
    "completed_at": datetime | None,
}
"""

TrainingDataDocument = Dict[str, Any]
"""
{
    "_id": ObjectId,
    "user_id": str,          # indexed
    "model_name": str,
    "version": str,
    "training_date": datetime,
    "metrics": dict,
    "config_hash": str,
}
"""
