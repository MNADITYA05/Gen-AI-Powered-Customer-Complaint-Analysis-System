"""
SQLAlchemy ORM models.

User         — application users (admin / agent roles)
Complaint    — every real or synthetic complaint
ComplaintNote — agent notes attached to a complaint
ModelRun     — records each training run with metrics
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)

from core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ── User ──────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id              = Column(String, primary_key=True, default=_uuid)
    username        = Column(String(64), unique=True, nullable=False, index=True)
    email           = Column(String(128), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role            = Column(String(16), nullable=False, default="agent")  # admin | agent
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d.pop("hashed_password", None)  # never expose the hash
        return d


# ── Complaint ─────────────────────────────────────────────────────────────────

class Complaint(Base):
    __tablename__ = "complaints"

    id             = Column(String, primary_key=True, default=_uuid)
    complaint_text = Column(Text, nullable=False)
    category       = Column(String(64), nullable=False, index=True)
    specific_issue = Column(String(128))
    severity       = Column(String(32), nullable=False, index=True)
    emotion        = Column(String(64), nullable=False)

    # Case management
    status            = Column(String(32), default="open", index=True)   # open | in_progress | resolved | closed
    source            = Column(String(32), default="synthetic")           # web_form | csv_upload | synthetic | api
    assignee_id       = Column(String, ForeignKey("users.id"), nullable=True)
    resolution_notes  = Column(Text, nullable=True)

    # Demographics
    age_group          = Column(String(16))
    tech_savviness     = Column(String(16))
    banking_experience = Column(String(32))

    # Metadata
    channel               = Column(String(32))
    account_type          = Column(String(32))
    resolution_time_hours = Column(Integer)
    location              = Column(String(128))
    customer_id           = Column(String(64))
    customer_name         = Column(String(128))

    # Stats
    word_count        = Column(Integer)
    character_count   = Column(Integer)
    generation_method = Column(String(64), default="template")

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ── ComplaintNote ─────────────────────────────────────────────────────────────

class ComplaintNote(Base):
    __tablename__ = "complaint_notes"

    id           = Column(String, primary_key=True, default=_uuid)
    complaint_id = Column(String, ForeignKey("complaints.id"), nullable=False, index=True)
    user_id      = Column(String, ForeignKey("users.id"), nullable=False)
    content      = Column(Text, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ── ModelRun ──────────────────────────────────────────────────────────────────

class ModelRun(Base):
    __tablename__ = "model_runs"

    id                = Column(String, primary_key=True, default=_uuid)
    category_accuracy = Column(Float)
    emotion_accuracy  = Column(Float)
    severity_accuracy = Column(Float)
    training_samples  = Column(Integer)
    mlflow_run_id     = Column(String(64))
    parameters        = Column(JSON)
    is_active         = Column(Boolean, default=False)
    created_at        = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
