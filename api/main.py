"""
FastAPI application entry point.
Run with:  uvicorn api.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analysis, complaints, health, models
from api.routers import auth as auth_router
from core.settings import get_settings
from core.startup import run_startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_startup()
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
app.include_router(analysis.router)
app.include_router(models.router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "docs":    "/docs",
    }
