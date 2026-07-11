"""
SQLAlchemy database engine and session factory.
Uses SQLite by default (DATABASE_URL in .env).
Swap to postgresql+psycopg2://... for production with zero code changes.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from core.settings import get_settings

settings = get_settings()

_connect_args = (
    {"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a DB session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables (idempotent). Call once on startup."""
    from core import db_models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
