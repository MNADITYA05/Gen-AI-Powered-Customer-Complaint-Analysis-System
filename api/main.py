"""
FastAPI application entry point.
Run with:  uvicorn api.main:app --reload
Or:        make api
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import complaints, health, models
from api.routers import auth as auth_router
from core.database import create_tables, SessionLocal
from core.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


def _seed_admin() -> None:
    """Create a default admin account if no users exist yet."""
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
            logger.info(
                "Default admin created — username: '%s'  password: '%s'  "
                "(change this immediately in production!)",
                settings.default_admin_username,
                settings.default_admin_password,
            )
    finally:
        db.close()


_GITHUB_RELEASE_BASE = (
    "https://github.com/MNADITYA05/Gen-AI-Powered-Customer-Complaint-Analysis-System"
    "/releases/download/v1.0"
)
_MODEL_FILES = ["pytorch_model.pt", "config.json", "label_maps.json"]


def _download_model_weights() -> None:
    """Download model weights from GitHub Releases if not already present."""
    import requests
    from pathlib import Path

    model_dir = Path(settings.model_dir) / "multitask_model"
    weights_file = model_dir / "pytorch_model.pt"

    if weights_file.exists():
        logger.info("Model weights already present at %s — skipping download.", model_dir)
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
    """Build or load the RAG similarity index in the background."""
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
        logger.warning("RAG index init skipped: %s\n%s", exc, traceback.format_exc())


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()            # idempotent — creates any missing tables
    _seed_admin()              # no-op if users already exist
    _download_model_weights()  # download from HF Hub if not present locally
    _init_rag()                # load or build the FAISS similarity index
    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "REST API for banking customer complaint management, "
        "ML-based classification, and analytics."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

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


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "docs":    "/docs",
    }
