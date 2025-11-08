"""Application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[5]

class Settings(BaseSettings):
    app_name: str = "sentiment-backend"
    environment: str = "local"
    imdb_weights_path: str = str(PROJECT_ROOT / "artifacts" / "imdb_dense" / "weights.01.keras")
    imdb_max_length: int = 256
    imdb_word_index_path: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="SENTIMENT_BACKEND_",
        env_file=".env",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
