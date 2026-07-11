"""
Model training and management endpoints.

POST /api/v1/models/train   — train on DB data
GET  /api/v1/models/info    — current model metadata
GET  /api/v1/models/runs    — history of training runs
"""
from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_classifier, require_admin, require_agent
from core.db_models import User
from api.schemas.model import (
    ModelInfoResponse,
    ModelRunListResponse,
    ModelRunRecord,
    TrainResponse,
)
from core.analysis.classifier import ComplaintClassifier
from core.database import get_db
from core.db_models import Complaint, ModelRun

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post("/train", response_model=TrainResponse)
def train_model(
    db: Session = Depends(get_db),
    classifier: ComplaintClassifier = Depends(get_classifier),
    _admin: User = Depends(require_admin),
):
    """
    Train the classifier on all complaints stored in the database.
    At least 20 complaints are required to produce a meaningful model.
    """
    rows = db.query(Complaint).all()
    if len(rows) < 20:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 20 complaints to train, found {len(rows)}. "
                   "Generate more via POST /api/v1/complaints/generate.",
        )

    df = pd.DataFrame([r.to_dict() for r in rows])

    try:
        metrics = classifier.train(df)
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    # Mark all previous runs inactive, record this one
    db.query(ModelRun).update({"is_active": False})
    run = ModelRun(
        category_accuracy = metrics["category_accuracy"],
        emotion_accuracy  = metrics["emotion_accuracy"],
        severity_accuracy = metrics["severity_accuracy"],
        training_samples  = metrics["training_samples"],
        mlflow_run_id     = metrics.get("mlflow_run_id"),
        parameters        = classifier.get_info(),
        is_active         = True,
    )
    db.add(run)
    db.commit()

    return TrainResponse(
        category_accuracy = metrics["category_accuracy"],
        emotion_accuracy  = metrics["emotion_accuracy"],
        severity_accuracy = metrics["severity_accuracy"],
        training_samples  = metrics["training_samples"],
        mlflow_run_id     = metrics.get("mlflow_run_id"),
        message           = f"Training complete on {metrics['training_samples']} samples.",
    )


@router.get("/info", response_model=ModelInfoResponse)
def model_info(
    classifier: ComplaintClassifier = Depends(get_classifier),
    _user: User = Depends(require_agent),
):
    """Return metadata about the currently loaded model."""
    info = classifier.get_info()
    return ModelInfoResponse(**info)


@router.get("/runs", response_model=ModelRunListResponse)
def list_runs(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    """Return all historical training runs, newest first."""
    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    return ModelRunListResponse(
        runs=[ModelRunRecord(**r.to_dict()) for r in runs],
        total=len(runs),
    )
