"""Application settings loaded from the environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    app_env: str = "development"

    # Database: SQLite for local dev, Postgres via DATABASE_URL in production.
    database_url: str = "sqlite:///./dropshipping_dev.db"

    # AI provider: mock | anthropic | openai | gemini
    ai_provider: str = "mock"

    # Two-way approval model
    default_approval_expiry_minutes: int = 60

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()
