"""
app.py — HuggingFace Gradio Space entry point.

Mounts our FastAPI routers onto Gradio's internal FastAPI instance so the Space
serves both a minimal status UI (at /) and the full REST API on port 7860.
"""
import logging
from pathlib import Path

import gradio as gr
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Initialise app dependencies ────────────────────────────────────────────────

from core.settings import get_settings
from core.database import create_tables, SessionLocal

settings = get_settings()


def _seed_admin() -> None:
    from core.auth import get_password_hash
    from core.db_models import User
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username=settings.default_admin_username,
                email="admin@localhost",
                hashed_password=get_password_hash(settings.default_admin_password),
                role="admin",
            )
            db.add(admin)
            db.commit()
            logger.info("Default admin created — username: '%s'", settings.default_admin_username)
    finally:
        db.close()


_GITHUB_RELEASE_BASE = (
    "https://github.com/MNADITYA05/Gen-AI-Powered-Customer-Complaint-Analysis-System"
    "/releases/download/v1.0"
)
_MODEL_FILES = ["pytorch_model.pt", "config.json", "label_maps.json"]


def _download_model_weights() -> None:
    import requests
    model_dir = Path(settings.model_dir) / "multitask_model"
    if (model_dir / "pytorch_model.pt").exists():
        logger.info("Model weights already present — skipping download.")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in _MODEL_FILES:
        url = f"{_GITHUB_RELEASE_BASE}/{filename}"
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


def _init_rag() -> None:
    try:
        from core.rag_engine import get_rag_engine
        engine = get_rag_engine()
        db = SessionLocal()
        try:
            engine.load_or_build(db)
        finally:
            db.close()
    except Exception as exc:
        import traceback
        logger.warning("RAG init skipped: %s\n%s", exc, traceback.format_exc())


# Run startup sequence
create_tables()
_seed_admin()
_download_model_weights()
_init_rag()

# ── Gradio UI (minimal status page) ───────────────────────────────────────────

with gr.Blocks(title="Complaint Analysis API") as demo:
    gr.Markdown("## Customer Complaint Analysis System")
    gr.Markdown(
        "The REST API is running on this Space. "
        "Access interactive docs at [/docs](/docs)."
    )

# ── Attach our FastAPI routers to Gradio's internal app ───────────────────────

from api.routers import complaints, health, models
from api.routers import auth as auth_router

app = demo.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth_router.router)
app.include_router(complaints.router)
app.include_router(models.router)

# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
