"""
Complaint CRUD endpoints.

GET   /api/v1/complaints              — list stored complaints (paginated + filtered)
PATCH /api/v1/complaints/{id}/status  — update case status / assignee / add note
GET   /api/v1/complaints/{id}/similar — find semantically similar complaints (RAG)
GET   /api/v1/complaints/{id}/notes   — list notes on a complaint
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import require_agent
from api.schemas.complaint import ComplaintListResponse, StatusUpdateRequest
from core.analysis.rag_engine import get_rag_engine
from core.database import get_db
from core.db_models import ComplaintNoteDoc, UserDoc, new_complaint_note

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/complaints", tags=["complaints"])


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
        db.complaint_notes.insert_one(new_complaint_note(
            complaint_id = complaint_id,
            user_id      = current_user.id,
            content      = req.note,
        ))

    return {"id": complaint_id, "status": updates.get("status", doc.get("status", "open"))}


# ── Similar Cases (RAG) ───────────────────────────────────────────────────────

@router.get("/{complaint_id}/similar")
def similar_complaints(
    complaint_id: str,
    k: int = Query(default=5, ge=1, le=10),
    db=Depends(get_db),
    current_user: UserDoc = Depends(require_agent),
):
    """Return the k most semantically similar complaints via sentence-transformers + FAISS."""
    doc = db.complaints.find_one({"_id": complaint_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Complaint not found")

    engine = get_rag_engine()
    if not engine.is_ready:
        try:
            engine.load_or_build(db)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    results = engine.find_similar(doc["complaint_text"], k=k, exclude_id=complaint_id)
    return {
        "complaint_id": complaint_id,
        "similar":      results,
        "engine_info":  engine.get_info(),
    }


# ── Notes ─────────────────────────────────────────────────────────────────────

@router.get("/{complaint_id}/notes")
def get_notes(
    complaint_id: str,
    db=Depends(get_db),
    current_user: UserDoc = Depends(require_agent),
):
    """Return all notes for a complaint, oldest first."""
    notes = list(
        db.complaint_notes.find({"complaint_id": complaint_id}).sort("created_at", 1)
    )
    return {"notes": [ComplaintNoteDoc(n).to_dict() for n in notes]}
