"""
ML analysis endpoints.

POST /api/v1/analysis/upload        — upload real complaints via CSV
POST /api/v1/analysis/analyze       — classify a single complaint
POST /api/v1/analysis/batch-analyze — classify many at once
POST /api/v1/analysis/rag/rebuild   — admin: rebuild FAISS similarity index
"""
from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_classifier, require_admin, require_agent
from api.schemas.complaint import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    ConfidenceScores,
)
from core.analysis.rag_engine import get_rag_engine
from core.database import get_db
from core.db_models import UserDoc, new_complaint
from core.validation.complaint_validator import ComplaintValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

_validator = ComplaintValidator()


# ── CSV Upload ────────────────────────────────────────────────────────────────

@router.post("/upload", status_code=201)
def upload_complaints(
    file: UploadFile = File(...),
    db=Depends(get_db),
    classifier=Depends(get_classifier),
    current_user: UserDoc = Depends(require_agent),
):
    """
    Upload a CSV of real complaints.
    Required column: complaint_text.
    Optional: category, severity, emotion, customer_name, customer_id, channel, location.
    Missing classification fields are auto-predicted by the RoBERTa model.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are accepted")

    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if "complaint_text" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must contain a 'complaint_text' column")

    df = df.dropna(subset=["complaint_text"])
    if df.empty:
        raise HTTPException(status_code=422, detail="No valid rows found in CSV")

    stored = 0
    skipped = 0
    docs = []
    for _, row in df.iterrows():
        text = str(row["complaint_text"]).strip()

        validation = _validator.validate(text)
        if not validation:
            skipped += 1
            continue

        category = str(row.get("category", "")) if "category" in df.columns else ""
        severity = str(row.get("severity", "")) if "severity" in df.columns else ""
        emotion  = str(row.get("emotion", ""))  if "emotion"  in df.columns else ""

        if getattr(classifier, "is_trained", False) or getattr(classifier, "is_loaded", False):
            if not category or not severity or not emotion:
                try:
                    pred = classifier.predict(text)
                    category = category or pred["category"]
                    severity = severity or pred["severity"]
                    emotion  = emotion  or pred["emotion"]
                except Exception:
                    pass

        docs.append(new_complaint(
            complaint_text    = text,
            category          = category or "UNKNOWN",
            severity          = severity or "medium",
            emotion           = emotion  or "neutral",
            source            = "csv_upload",
            status            = "open",
            customer_name     = str(row.get("customer_name", "")) or None,
            customer_id       = str(row.get("customer_id", ""))   or None,
            channel           = str(row.get("channel", ""))       or None,
            location          = str(row.get("location", ""))      or None,
            word_count        = len(text.split()),
            character_count   = len(text),
            generation_method = "csv_upload",
        ))
        stored += 1

    if docs:
        db.complaints.insert_many(docs)

    return {"stored": stored, "skipped": skipped, "total_rows": len(df)}


# ── Single analysis ───────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_complaint(
    req: AnalyzeRequest,
    db=Depends(get_db),
    classifier=Depends(get_classifier),
    current_user: UserDoc = Depends(require_agent),
):
    """Classify a single complaint text using the RoBERTa multi-task model."""
    validation = _validator.validate(req.text)
    if not validation:
        raise HTTPException(status_code=422, detail=validation.reason)

    if not getattr(classifier, "is_trained", None) and not getattr(classifier, "is_loaded", None):
        raise HTTPException(status_code=409, detail="No trained model available.")

    try:
        result = classifier.predict(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    db.complaints.insert_one(new_complaint(
        complaint_text    = req.text,
        category          = result["category"],
        severity          = result["severity"],
        emotion           = result["emotion"],
        source            = "api",
        status            = "open",
        word_count        = len(req.text.split()),
        character_count   = len(req.text),
        generation_method = "api_submission",
    ))

    return AnalyzeResponse(
        category   = result["category"],
        emotion    = result["emotion"],
        severity   = result["severity"],
        confidence = ConfidenceScores(**result["confidence"]),
    )


# ── Batch analysis ────────────────────────────────────────────────────────────

@router.post("/batch-analyze", response_model=BatchAnalyzeResponse)
def batch_analyze(
    req: BatchAnalyzeRequest,
    classifier=Depends(get_classifier),
    current_user: UserDoc = Depends(require_agent),
):
    """Classify up to 500 complaints in a single call."""
    if not getattr(classifier, "is_trained", None) and not getattr(classifier, "is_loaded", None):
        raise HTTPException(status_code=409, detail="No trained model available.")
    results = []
    for text in req.texts:
        r = classifier.predict(text)
        results.append(AnalyzeResponse(
            category=r["category"], emotion=r["emotion"], severity=r["severity"],
            confidence=ConfidenceScores(**r["confidence"]),
        ))
    return BatchAnalyzeResponse(results=results, count=len(results))


# ── RAG rebuild ───────────────────────────────────────────────────────────────

@router.post("/rag/rebuild")
def rebuild_rag_index(
    db=Depends(get_db),
    _admin: UserDoc = Depends(require_admin),
):
    """Admin: rebuild the RAG similarity index from all DB complaints."""
    engine = get_rag_engine()
    try:
        count = engine.rebuild_index(db)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"indexed": count, "engine_info": engine.get_info()}
