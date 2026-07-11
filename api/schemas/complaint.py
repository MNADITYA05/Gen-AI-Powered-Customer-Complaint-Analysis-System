"""Pydantic schemas for complaint-related API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Category = Literal["ATM_FAILURE", "FRAUD_DETECTION", "UX_ISSUES"]
Severity = Literal["low", "medium", "high", "critical"]


# ── Request schemas ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000, description="Raw complaint text")


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class StatusUpdateRequest(BaseModel):
    status:           Optional[str] = None   # open | in_progress | resolved | closed
    assignee_id:      Optional[str] = None
    resolution_notes: Optional[str] = None
    note:             Optional[str] = None   # freeform note appended to complaint history


# ── Response schemas ──────────────────────────────────────────────────────────

class ConfidenceScores(BaseModel):
    category: float
    emotion:  float
    severity: float


class AnalyzeResponse(BaseModel):
    category:   str
    emotion:    str
    severity:   str
    confidence: ConfidenceScores


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    count:   int


class ComplaintListItem(BaseModel):
    id:                str
    complaint_text:    str
    category:          str
    severity:          str
    emotion:           str
    status:            Optional[str] = "open"
    source:            Optional[str] = None
    generation_method: str
    created_at:        Optional[datetime] = None


class ComplaintListResponse(BaseModel):
    items:  list[ComplaintListItem]
    total:  int
    page:   int
    limit:  int
