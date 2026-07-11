"""
Shared pytest fixtures.
Uses mongomock for an in-memory MongoDB so tests never touch a real database.

Key design decisions:
- mongomock provides a drop-in MongoClient replacement with no network calls
- env vars are set BEFORE any project imports so lru_cache settings pick them up
"""
import os

# Must be set before any project module is imported (lru_cache on settings)
os.environ["MONGODB_URL"]           = "mongodb://localhost:27017"  # overridden by mock
os.environ["MONGODB_DB_NAME"]       = "test_complaint_analysis"
os.environ["MODEL_DIR"]             = "/tmp/test_models"
os.environ["MLFLOW_TRACKING_URI"]   = "sqlite:////tmp/test_mlruns.db"

import pandas as pd
import pytest

try:
    import mongomock
    _MOCK_CLIENT = mongomock.MongoClient()
except ImportError:
    _MOCK_CLIENT = None  # type: ignore[assignment]

from fastapi.testclient import TestClient

from api.main import app
from core.database import get_db


def _mock_get_db():
    if _MOCK_CLIENT is None:
        pytest.skip("mongomock not installed — run: pip install mongomock")
    return _MOCK_CLIENT["test_complaint_analysis"]


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = _mock_get_db
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
