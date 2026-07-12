"""
core/startup.py
---------------
Shared startup routines called by api/main.py (and any future entry points).

Keeps entry-point files thin: they import and call these functions rather than
defining them inline.
"""
from __future__ import annotations

import logging
from pathlib import Path

from core.database import ensure_indexes, get_db
from core.settings import get_settings

logger = logging.getLogger(__name__)

_GITHUB_RELEASE_BASE = (
    "https://github.com/MNADITYA05/Gen-AI-Powered-Customer-Complaint-Analysis-System"
    "/releases/download/v1.0"
)
_MODEL_FILES = ["pytorch_model.pt", "config.json", "label_maps.json"]


def seed_admin() -> None:
    """Create a default admin account if no users exist yet."""
    from core.auth import get_password_hash
    from core.db_models import new_user

    settings = get_settings()
    db = get_db()
    if db.users.count_documents({}) == 0:
        admin = new_user(
            username        = settings.default_admin_username,
            email           = "admin@localhost",
            hashed_password = get_password_hash(settings.default_admin_password),
            role            = "admin",
        )
        db.users.insert_one(admin)
        logger.info("Default admin created — username: '%s'", settings.default_admin_username)


def download_model_weights() -> None:
    """Download model weights from GitHub Releases if not already present."""
    import requests

    settings = get_settings()
    model_dir    = Path(settings.model_dir) / "multitask_model"
    weights_file = model_dir / "pytorch_model.pt"

    if weights_file.exists():
        logger.info("Model weights already present at %s — skipping download.", model_dir)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in _MODEL_FILES:
        url  = f"{_GITHUB_RELEASE_BASE}/{filename}"
        dest = model_dir / filename
        logger.info("Downloading %s ...", filename)
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("Downloaded %s", filename)
        except Exception as exc:
            import traceback
            logger.error("Failed to download %s: %s\n%s", filename, exc, traceback.format_exc())
            return
    logger.info("All model weights downloaded to %s", model_dir)


def init_rag() -> None:
    """Build or load the FAISS similarity index."""
    try:
        from core.analysis.rag_engine import get_rag_engine
        engine = get_rag_engine()
        db = get_db()
        engine.load_or_build(db)
    except Exception as exc:
        import traceback
        logger.warning("RAG index init skipped: %s\n%s", exc, traceback.format_exc())


def run_startup() -> None:
    """Run the full startup sequence. Call once at application boot."""
    ensure_indexes()
    seed_admin()
    download_model_weights()
    init_rag()
