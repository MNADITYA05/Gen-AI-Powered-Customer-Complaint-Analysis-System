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


def _download_model_weights() -> None:
    model_dir = Path(settings.model_dir) / "multitask_model"
    weights_file = model_dir / "pytorch_model.pt"
    if weights_file.exists():
        logger.info("Model weights already present — skipping download.")
        return
    if not settings.huggingface_token:
        logger.warning("HUGGINGFACE_TOKEN not set — cannot download model weights.")
        return
    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading model weights from ADI2005/complaint-roberta-multitask ...")
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="ADI2005/complaint-roberta-multitask",
            token=settings.huggingface_token,
            local_dir=str(model_dir),
        )
        logger.info("Model weights downloaded to %s", model_dir)
    except Exception as exc:
        import traceback
        logger.error("Model download failed: %s\n%s", exc, traceback.format_exc())


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
