from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Customer Complaint Analysis API"
    api_version: str = "1.0.0"

    # Auth / JWT
    secret_key: str = "change-me-in-production-use-a-long-random-string"
    access_token_expire_minutes: int = 480  # 8 hours
    default_admin_username: str = "admin"
    default_admin_password: str = "admin123"

    # Frontend
    api_base_url: str = "http://localhost:8000"

    # HuggingFace
    huggingface_token: str = ""

    # ML
    model_dir: str = "models"

    # Database — MongoDB Atlas (or local mongod)
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "complaint_analysis"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///./mlruns.db"
    mlflow_experiment_name: str = "complaint_classification"

    # Data
    data_dir: str = "data"
    seeds_dir: str = "data/seeds"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
