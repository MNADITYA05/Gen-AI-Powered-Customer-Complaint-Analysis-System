"""Pydantic schemas for model management endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TrainResponse(BaseModel):
    category_accuracy:  float
    emotion_accuracy:   float
    severity_accuracy:  float
    training_samples:   int
    mlflow_run_id:      Optional[str] = None
    message:            str = "Training complete"


class ModelInfoResponse(BaseModel):
    is_trained:        bool
    category_classes:  Optional[list[str]] = None
    emotion_classes:   Optional[list[str]] = None
    severity_classes:  Optional[list[str]] = None
    vectorizer_vocab_size: Optional[int] = None


class ModelRunRecord(BaseModel):
    id:                str
    category_accuracy: Optional[float] = None
    emotion_accuracy:  Optional[float] = None
    severity_accuracy: Optional[float] = None
    training_samples:  Optional[int]   = None
    mlflow_run_id:     Optional[str]   = None
    is_active:         bool
    created_at:        Optional[datetime] = None


class ModelRunListResponse(BaseModel):
    runs:  list[ModelRunRecord]
    total: int
