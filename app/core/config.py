"""Application settings: Pydantic v2 ``BaseSettings`` loaded from ``.env``.

Design constraints for this project
-----------------------------------
* **Zero-setup local dev.** Every field has a working default, so a fresh clone
  boots on an 8 GB RAM Windows machine without Docker and without a ``.env``
  file. ``copy .env.example .env`` is optional.
* **SQLite by default, PostgreSQL by env var.** Swapping backends is a single
  ``DATABASE_URL`` change - no code edits (see ``app/db/session.py``).
* **Fail fast on typos.** Unknown environments, malformed database URLs and
  non-positive approval windows are rejected at import time with a readable
  error instead of blowing up mid-request.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AI_PROVIDER_CHOICES",
    "API_V1_PREFIX",
    "APP_NAME",
    "APP_VERSION",
    "ARCHITECTURE_PHASE",
    "ENVIRONMENT_CHOICES",
    "Settings",
    "get_settings",
    "settings",
]

# ── Application identity ──────────────────────────────────────────────────
APP_NAME: str = "Dropshipping AI System"
APP_VERSION: str = "0.1.0"
API_V1_PREFIX: str = "/api/v1"

#: Which slice of the target architecture this codebase currently implements.
#: Reported by ``GET /api/v1/health`` so a running instance can always be
#: identified without reading the source tree.
ARCHITECTURE_PHASE: str = "1A-foundation"

ENVIRONMENT_CHOICES: tuple[str, ...] = (
    "development",
    "testing",
    "staging",
    "production",
)
AI_PROVIDER_CHOICES: tuple[str, ...] = ("mock", "anthropic", "openai", "gemini")

#: Dialect prefixes accepted in ``DATABASE_URL`` (``postgresql+psycopg`` and
#: friends are matched on their part before the ``+``).
_SUPPORTED_DB_DIALECTS: tuple[str, ...] = ("sqlite", "postgresql", "postgres")


class Settings(BaseSettings):
    """Runtime configuration. Values come from process env vars first, then
    ``.env``, then the defaults declared below."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ── Environment ───────────────────────────────────────────────────────
    app_env: str = Field(
        default="development",
        description="development | testing | staging | production",
    )

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./dropshipping_dev.db",
        description="SQLAlchemy 2.0 URL. SQLite for local dev, PostgreSQL in prod.",
    )

    #: Echo SQL to stdout. Handy while debugging, off by default: it is noisy
    #: and measurably slower on a small machine.
    sql_echo: bool = Field(default=False)

    # ── AI provider ───────────────────────────────────────────────────────
    ai_provider: Literal["mock", "anthropic", "openai", "gemini"] = Field(
        default="mock",
        description="mock = deterministic, offline, zero spend.",
    )

    # ── Two-way approval model ────────────────────────────────────────────
    default_approval_expiry_minutes: int = Field(
        default=60,
        gt=0,
        le=60 * 24 * 7,
        description="How long a LEVEL_2_APPROVAL request stays open before it expires.",
    )

    # ── Normalisation / validation ────────────────────────────────────────
    @field_validator("app_env", mode="before")
    @classmethod
    def _normalise_app_env(cls, value: Any) -> str:
        env = str(value).strip().lower()
        if env not in ENVIRONMENT_CHOICES:
            raise ValueError(
                f"APP_ENV must be one of {', '.join(ENVIRONMENT_CHOICES)} (got {value!r})"
            )
        return env

    @field_validator("ai_provider", mode="before")
    @classmethod
    def _normalise_ai_provider(cls, value: Any) -> str:
        return str(value).strip().lower()

    @field_validator("database_url", mode="before")
    @classmethod
    def _validate_database_url(cls, value: Any) -> str:
        url = str(value).strip()
        if not url:
            raise ValueError("DATABASE_URL must not be empty")
        dialect = url.split("://", 1)[0].split("+", 1)[0].lower()
        if dialect not in _SUPPORTED_DB_DIALECTS:
            raise ValueError(
                f"Unsupported DATABASE_URL dialect {dialect!r}. "
                f"Use sqlite:///... for local dev or postgresql+psycopg://... for production."
            )
        return url

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith(("postgresql", "postgres"))

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def approval_expiry(self) -> timedelta:
        """Default approval window as a ``timedelta``."""
        return timedelta(minutes=self.default_approval_expiry_minutes)

    @property
    def approval_expiry_timedelta(self) -> timedelta:
        """Alias of :attr:`approval_expiry`.

        Milestone 1B/1C services (``app.services.approval``) use this name; kept
        as the canonical spelling there so the governance layer needs no edits.
        """
        return self.approval_expiry


#: Process-wide singleton. Imported directly by the db/session layers.
settings = Settings()


def get_settings() -> Settings:
    """FastAPI dependency wrapper around :data:`settings`.

    Using a dependency (instead of importing the singleton inline) lets tests
    override configuration with ``app.dependency_overrides``.
    """
    return settings
