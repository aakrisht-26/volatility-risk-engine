"""Application settings, loaded from the environment and .env via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every database connection flows through DATABASE_URL."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str


def get_settings() -> Settings:
    """Read settings fresh from the environment; call sites hold no globals."""
    return Settings()
