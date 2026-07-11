"""
Shared pytest fixtures.
Uses an in-memory SQLite database so tests never touch the dev database.

Key design decisions:
- StaticPool forces SQLite :memory: to reuse a single connection across all
  sessions, so tables created in one session are visible in another.
- env vars are set BEFORE any project imports so lru_cache settings pick them up.
"""
import os

# Must be set before any project module is imported (lru_cache on settings)
os.environ["DATABASE_URL"]          = "sqlite:///:memory:"
os.environ["MODEL_DIR"]             = "/tmp/test_models"
os.environ["MLFLOW_TRACKING_URI"]   = "sqlite:////tmp/test_mlruns.db"

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from core import db_models  # noqa: F401 — registers ORM models with Base
from core.database import Base, get_db

# ── Single shared in-memory DB (StaticPool = same connection always) ──────────
_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,           # ← ensures :memory: isn't re-created per session
)
_TestSession = sessionmaker(bind=_test_engine, autocommit=False, autoflush=False)

# Create tables once at import time — StaticPool keeps them alive for the session
Base.metadata.create_all(bind=_test_engine)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def sample_complaints_df() -> pd.DataFrame:
    """Minimal labelled dataset sufficient for classifier training."""
    rows = []
    for i in range(30):
        rows.append({
            "complaint_text": f"The ATM swallowed my card and I am very frustrated complaint {i}",
            "category":  "ATM_FAILURE",
            "emotion":   "frustrated",
            "severity":  "high",
        })
        rows.append({
            "complaint_text": f"There are unauthorized transactions on my account I am scared {i}",
            "category":  "FRAUD_DETECTION",
            "emotion":   "scared",
            "severity":  "critical",
        })
        rows.append({
            "complaint_text": f"Your mobile banking app keeps crashing and login fails {i}",
            "category":  "UX_ISSUES",
            "emotion":   "annoyed",
            "severity":  "medium",
        })
    return pd.DataFrame(rows)
