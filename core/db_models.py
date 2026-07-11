"""
MongoDB document factories and lightweight wrapper classes.

Replaces the previous SQLAlchemy ORM models.

Usage:
    # Create a new document dict to insert:
    doc = new_user(username="alice", email="a@b.com", hashed_password="...")
    db.users.insert_one(doc)

    # Wrap a retrieved dict for attribute-style access:
    user = UserDoc(db.users.find_one({"username": "alice"}))
    print(user.role)          # "agent"
    print(user.to_dict())     # safe dict (no hashed_password)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ── Document factories ────────────────────────────────────────────────────────

def new_user(
    username: str,
    email: str,
    hashed_password: str,
    role: str = "agent",
) -> dict:
    return {
        "_id":             _uuid(),
        "username":        username,
        "email":           email,
        "hashed_password": hashed_password,
        "role":            role,
        "is_active":       True,
        "created_at":      _now(),
    }


def new_complaint(
    complaint_text: str,
    category: str,
    severity: str,
    emotion: str,
    source: str = "api",
    status: str = "open",
    **kwargs: Any,
) -> dict:
    return {
        "_id":             _uuid(),
        "complaint_text":  complaint_text,
        "category":        category,
        "severity":        severity,
        "emotion":         emotion,
        "source":          source,
        "status":          status,
        "created_at":      _now(),
        "updated_at":      _now(),
        **kwargs,
    }


def new_complaint_note(complaint_id: str, user_id: str, content: str) -> dict:
    return {
        "_id":          _uuid(),
        "complaint_id": complaint_id,
        "user_id":      user_id,
        "content":      content,
        "created_at":   _now(),
    }


def new_model_run(
    category_accuracy: float,
    emotion_accuracy: float,
    severity_accuracy: float,
    training_samples: int,
    is_active: bool = True,
    mlflow_run_id: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> dict:
    return {
        "_id":               _uuid(),
        "category_accuracy": category_accuracy,
        "emotion_accuracy":  emotion_accuracy,
        "severity_accuracy": severity_accuracy,
        "training_samples":  training_samples,
        "mlflow_run_id":     mlflow_run_id,
        "parameters":        parameters or {},
        "is_active":         is_active,
        "created_at":        _now(),
    }


# ── Wrapper classes ───────────────────────────────────────────────────────────

class _DocWrapper:
    """
    Wraps a raw MongoDB document dict, providing attribute-style field access
    and a safe to_dict() serialiser.
    """

    def __init__(self, doc: dict) -> None:
        # Store the raw document in __dict__ directly to avoid __getattr__ recursion.
        object.__setattr__(self, "_doc", doc)

    # Attribute access — delegates to the underlying document
    def __getattr__(self, name: str) -> Any:
        doc = object.__getattribute__(self, "_doc")
        try:
            return doc[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        doc = object.__getattribute__(self, "_doc")
        doc[name] = value

    # Convenience: id → _id alias
    @property
    def id(self) -> str:
        return object.__getattribute__(self, "_doc")["_id"]

    def to_dict(self) -> dict:
        doc = object.__getattribute__(self, "_doc")
        d = dict(doc)
        d["id"] = d.pop("_id")
        d.pop("hashed_password", None)
        return d


class UserDoc(_DocWrapper):
    """Wraps a users collection document."""


class ComplaintDoc(_DocWrapper):
    """Wraps a complaints collection document."""

    def to_dict(self) -> dict:
        doc = object.__getattribute__(self, "_doc")
        d = dict(doc)
        d["id"] = d.pop("_id")
        return d


class ComplaintNoteDoc(_DocWrapper):
    """Wraps a complaint_notes collection document."""

    def to_dict(self) -> dict:
        doc = object.__getattribute__(self, "_doc")
        d = dict(doc)
        d["id"] = d.pop("_id")
        return d


class ModelRunDoc(_DocWrapper):
    """Wraps a model_runs collection document."""

    def to_dict(self) -> dict:
        doc = object.__getattribute__(self, "_doc")
        d = dict(doc)
        d["id"] = d.pop("_id")
        return d
