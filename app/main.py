"""FastAPI application — Milestone 1A: Phase 1 Foundation.

Dev-mode lifespan auto-creates tables so a fresh clone runs with zero setup.
Alembic takes over migrations from Milestone 1B onward.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.approvals import router as governance_router
from app.api.v1.orchestrator import router as orchestrator_router

from app.core.config import settings
from app.db.base import Base

# Import models so they register on Base.metadata before table creation.
from app.models import core as _core_models  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.app_env == "development":
        from app.db.session import engine

        Base.metadata.create_all(bind=engine)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dropshipping AI System",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(governance_router, prefix="/api/v1")
    app.include_router(orchestrator_router, prefix="/api/v1")

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": settings.app_env,
            "architecture_phase": "1A-foundation",
        }

    return app


app = create_app()
