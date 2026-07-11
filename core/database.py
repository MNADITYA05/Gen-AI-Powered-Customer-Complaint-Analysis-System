"""
MongoDB database layer.

Replaces the previous SQLAlchemy setup.  All application code that needs the
database should call get_db() to receive a pymongo.database.Database instance
and operate on its collections directly.

Collections used:
    users            — application users
    complaints       — customer complaint records
    complaint_notes  — agent notes attached to complaints
    model_runs       — training run history
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pymongo import DESCENDING, MongoClient
from pymongo.database import Database

from core.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> MongoClient:
    """Return a cached MongoClient (connection-pooled singleton)."""
    settings = get_settings()
    return MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=10_000)


def get_db() -> Database:
    """Return the application MongoDB database."""
    settings = get_settings()
    return _get_client()[settings.mongodb_db_name]


def ensure_indexes() -> None:
    """
    Create all required indexes (idempotent — safe to call on every startup).
    Equivalent to SQLAlchemy's create_tables().
    """
    db = get_db()

    db.users.create_index("username", unique=True, background=True)
    db.users.create_index("email",    unique=True, background=True)

    db.complaints.create_index([("created_at", DESCENDING)], background=True)
    db.complaints.create_index("category", background=True)
    db.complaints.create_index("severity", background=True)
    db.complaints.create_index("status",   background=True)

    db.complaint_notes.create_index("complaint_id", background=True)
    db.model_runs.create_index([("created_at", DESCENDING)], background=True)

    logger.info("MongoDB indexes ensured.")
