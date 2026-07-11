"""
Complaint endpoints.

POST /api/v1/complaints/upload        — agent: upload real complaints via CSV
POST /api/v1/complaints/analyze       — agent: classify a single complaint
POST /api/v1/complaints/batch-analyze — agent: classify many at once
GET  /api/v1/complaints               — agent: list stored complaints (paginated)
PATCH /api/v1/complaints/{id}/status  — agent: update case status / assign / add notes
GET  /api/v1/complaints/{id}/similar  — agent: find semantically similar complaints (RAG)
POST /api/v1/complaints/rag/rebuild   — admin: rebuild FAISS similarity index
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.dependencies import get_classifier, require_admin, require_agent
from core.rag_engine import get_rag_engine
from api.schemas.complaint import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    ComplaintListResponse,
    ConfidenceScores,
    StatusUpdateRequest,
)
from core.database import get_db
from core.db_models import ComplaintDoc, ComplaintNoteDoc, UserDoc, new_complaint, new_complaint_note

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/complaints", tags=["complaints"])


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
    errors = 0
    docs = []
    for _, row in df.iterrows():
        text = str(row["complaint_text"]).strip()
        if len(text) < 10:
            errors += 1
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

        doc = new_complaint(
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
        )
        docs.append(doc)
        stored += 1

    if docs:
        db.complaints.insert_many(docs)

    return {"stored": stored, "skipped": errors, "total_rows": len(df)}


# ── Analyze ───────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_complaint(
    req: AnalyzeRequest,
    db=Depends(get_db),
    classifier=Depends(get_classifier),
    current_user: UserDoc = Depends(require_agent),
):
    """Classify a single complaint text using the RoBERTa multi-task model."""
    if not getattr(classifier, "is_trained", None) and not getattr(classifier, "is_loaded", None):
        raise HTTPException(status_code=409, detail="No trained model available.")
    try:
        result = classifier.predict(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    doc = new_complaint(
        complaint_text    = req.text,
        category          = result["category"],
        severity          = result["severity"],
        emotion           = result["emotion"],
        source            = "api",
        status            = "open",
        word_count        = len(req.text.split()),
        character_count   = len(req.text),
        generation_method = "api_submission",
    )
    db.complaints.insert_one(doc)

    return AnalyzeResponse(
        category   = result["category"],
        emotion    = result["emotion"],
        severity   = result["severity"],
        confidence = ConfidenceScores(**result["confidence"]),
    )


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


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=ComplaintListResponse)
def list_complaints(
    page:     int = Query(default=1, ge=1),
    limit:    int = Query(default=50, ge=1, le=500),
    category: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    status:   Optional[str] = Query(default=None),
    db=Depends(get_db),
    current_user: UserDoc = Depends(require_agent),
):
    """Return stored complaints with optional filters."""
    filt: dict = {}
    if category:
        filt["category"] = category
    if severity:
        filt["severity"] = severity
    if status:
        filt["status"] = status

    total = db.complaints.count_documents(filt)
    rows  = list(
        db.complaints.find(filt)
        .sort("created_at", -1)
        .skip((page - 1) * limit)
        .limit(limit)
    )

    return ComplaintListResponse(
        items=[
            {
                "id":                r["_id"],
                "complaint_text":    r.get("complaint_text", ""),
                "category":          r.get("category", ""),
                "severity":          r.get("severity", ""),
                "emotion":           r.get("emotion", ""),
                "status":            r.get("status", "open"),
                "source":            r.get("source", "unknown"),
                "generation_method": r.get("generation_method", "unknown"),
                "created_at":        r.get("created_at"),
            }
            for r in rows
        ],
        total=total,
        page=page,
        limit=limit,
    )


# ── Status update ─────────────────────────────────────────────────────────────

@router.patch("/{complaint_id}/status")
def update_status(
    complaint_id: str,
    req: StatusUpdateRequest,
    db=Depends(get_db),
    current_user: UserDoc = Depends(require_agent),
):
    """Update a complaint's status, assignee, or add a resolution note."""
    doc = db.complaints.find_one({"_id": complaint_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Complaint not found")

    updates: dict = {}
    if req.status:
        updates["status"] = req.status
    if req.assignee_id:
        updates["assignee_id"] = req.assignee_id
    if req.resolution_notes:
        updates["resolution_notes"] = req.resolution_notes

    if updates:
        db.complaints.update_one({"_id": complaint_id}, {"$set": updates})

    if req.note:
        note_doc = new_complaint_note(
            complaint_id = complaint_id,
            user_id      = current_user.id,
            content      = req.note,
        )
        db.complaint_notes.insert_one(note_doc)

    new_status = updates.get("status", doc.get("status", "open"))
    return {"id": complaint_id, "status": new_status}


# ── Similar Cases (RAG) ───────────────────────────────────────────────────────

@router.get("/{complaint_id}/similar")
def similar_complaints(
    complaint_id: str,
    k: int = Query(default=5, ge=1, le=10),
    db=Depends(get_db),
    current_user: UserDoc = Depends(require_agent),
):
    """
    Return the k most semantically similar complaints using the local RAG engine.
    Results are ranked by cosine similarity of sentence-transformer embeddings.
    """
    doc = db.complaints.find_one({"_id": complaint_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Complaint not found")

    engine = get_rag_engine()
    if not engine.is_ready:
        try:
            engine.load_or_build(db)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    results = engine.find_similar(
        doc["complaint_text"], k=k, exclude_id=complaint_id
    )
    return {
        "complaint_id": complaint_id,
        "similar":      results,
        "engine_info":  engine.get_info(),
    }


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


# ── Notes ─────────────────────────────────────────────────────────────────────

@router.get("/{complaint_id}/notes")
def get_notes(
    complaint_id: str,
    db=Depends(get_db),
    current_user: UserDoc = Depends(require_agent),
):
    """Return all notes for a complaint."""
    notes = list(
        db.complaint_notes.find({"complaint_id": complaint_id})
        .sort("created_at", 1)
    )
    return {"notes": [ComplaintNoteDoc(n).to_dict() for n in notes]}
