"""Application settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "sentiment-backend"
    environment: str = "local"

    model_config = SettingsConfigDict(
        env_prefix="SENTIMENT_BACKEND_",
        env_file=".env",
    )


def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
