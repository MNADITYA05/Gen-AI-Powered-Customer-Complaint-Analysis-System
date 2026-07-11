"""Health and readiness endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.database import get_db
from api.dependencies import get_classifier

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def readiness(
    db: Session = Depends(get_db),
    classifier=Depends(get_classifier),
) -> dict:
    """Returns 200 only when DB is reachable and models are loaded."""
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        return {"status": "not_ready", "reason": f"DB error: {exc}"}

    return {
        "status": "ready",
        "models_loaded": classifier.is_trained,
    }
