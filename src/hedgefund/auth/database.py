"""MongoDB connection and collection management for the multi-user hedge fund.

Provides an async MongoDB wrapper built on motor (async MongoDB driver).
All collections are accessed as properties; indexes are created on startup
via ``create_indexes()``.
"""

from __future__ import annotations

import os
from typing import Optional

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING, IndexModel

from hedgefund.logger import get_logger

log = get_logger(__name__)

_DEFAULT_URI = os.environ.get("HEDGEFUND_MONGO_URI", "mongodb://172.21.0.2:27017")
_DEFAULT_DB = "hedgefund"

# Singleton instance
_instance: Optional["MongoDB"] = None


class MongoDB:
    """Async MongoDB connection manager with per-collection access."""

    def __init__(
        self,
        uri: str = _DEFAULT_URI,
        db_name: str = _DEFAULT_DB,
    ) -> None:
        self._uri = uri
        self._db_name = db_name
        self._client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self._db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the Motor client and select the database."""
        if self._client is not None:
            return
        self._client = motor.motor_asyncio.AsyncIOMotorClient(self._uri)
        self._db = self._client[self._db_name]
        # Quick connectivity check
        await self._client.admin.command("ping")
        log.info("mongodb.connected", uri=self._uri, db=self._db_name)

    async def disconnect(self) -> None:
        """Close the Motor client."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            log.info("mongodb.disconnected")

    # -- index management ----------------------------------------------------

    async def create_indexes(self) -> None:
        """Create all required indexes for every collection."""
        assert self._db is not None, "Call connect() first"

        # users
        await self._db.users.create_indexes([
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("username", ASCENDING)], unique=True),
        ])

        # broker_accounts
        await self._db.broker_accounts.create_indexes([
            IndexModel([("user_id", ASCENDING), ("broker_id", ASCENDING)]),
        ])

        # trades
        await self._db.trades.create_indexes([
            IndexModel([("user_id", ASCENDING), ("entry_time", DESCENDING)]),
            IndexModel([("user_id", ASCENDING), ("mode", ASCENDING)]),
        ])

        # signals
        await self._db.signals.create_indexes([
            IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)]),
        ])

        # positions
        await self._db.positions.create_indexes([
            IndexModel([("user_id", ASCENDING), ("broker_id", ASCENDING)]),
        ])

        # x_accounts
        await self._db.x_accounts.create_indexes([
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("x_user_id", ASCENDING)], unique=True),
        ])

        # x_posts
        await self._db.x_posts.create_indexes([
            IndexModel([("user_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("tweet_id", ASCENDING)], unique=True),
        ])

        # sentiment_data
        await self._db.sentiment_data.create_indexes([
            IndexModel([
                ("user_id", ASCENDING),
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING),
            ]),
        ])

        # backtests
        await self._db.backtests.create_indexes([
            IndexModel([("user_id", ASCENDING)]),
        ])

        # training_data
        await self._db.training_data.create_indexes([
            IndexModel([("user_id", ASCENDING), ("model_name", ASCENDING)]),
        ])

        # data_sources
        await self._db.data_sources.create_indexes([
            IndexModel([("user_id", ASCENDING)]),
            IndexModel([("user_id", ASCENDING), ("source_id", ASCENDING)], unique=True),
        ])

        log.info("mongodb.indexes_created")

    # -- collection properties -----------------------------------------------

    @property
    def users(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.users

    @property
    def broker_accounts(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.broker_accounts

    @property
    def trades(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.trades

    @property
    def signals(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.signals

    @property
    def positions(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.positions

    @property
    def x_accounts(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.x_accounts

    @property
    def x_posts(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.x_posts

    @property
    def sentiment_data(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.sentiment_data

    @property
    def backtests(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.backtests

    @property
    def training_data(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.training_data

    @property
    def data_sources(self) -> motor.motor_asyncio.AsyncIOMotorCollection:
        assert self._db is not None, "Call connect() first"
        return self._db.data_sources


def get_database() -> MongoDB:
    """Return a cached singleton MongoDB instance.

    The URI and database name are read from environment variables
    ``HEDGEFUND_MONGO_URI`` and ``HEDGEFUND_MONGO_DB``, falling back to
    localhost defaults.
    """
    global _instance
    if _instance is None:
        uri = os.environ.get("HEDGEFUND_MONGO_URI", _DEFAULT_URI)
        db_name = os.environ.get("HEDGEFUND_MONGO_DB", _DEFAULT_DB)
        _instance = MongoDB(uri=uri, db_name=db_name)
    return _instance
