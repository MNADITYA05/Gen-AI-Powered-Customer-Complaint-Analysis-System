"""
FastAPI dependency injection.
Singletons for ML components + auth helpers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.analysis.classifier import ComplaintClassifier
from core.analysis.multi_task_model import MultiTaskClassifier
from core.auth import decode_token
from core.database import get_db
from core.db_models import User
from core.settings import get_settings

settings = get_settings()


# ── ML singleton ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_classifier():
    """
    Returns MultiTaskClassifier (RoBERTa-base, avg_acc=92.10%) if trained weights
    exist in models/multitask_model/, otherwise falls back to the classical
    TF-IDF + RF classifier.  Both expose identical predict() / get_info() interfaces.
    """
    mt = MultiTaskClassifier(model_dir=settings.model_dir)
    if mt.load():
        return mt
    clf = ComplaintClassifier()
    clf.load()
    return clf


# ── Auth helpers ──────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Decode the Bearer token and return the matching User row."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require the caller to have the 'admin' role."""
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


def require_agent(current_user: User = Depends(get_current_user)) -> User:
    """Require the caller to be logged in (any role)."""
    return current_user
