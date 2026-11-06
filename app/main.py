"""FastAPI application entry point - Milestone 1A (Phase 1 Foundation).

Modular monolith: one process, one dependency graph, strict internal module
boundaries (``core`` -> ``db`` -> ``models`` -> ``main``). That keeps the memory
footprint small enough for an 8 GB RAM Windows machine and postpones the
operational cost of services until there is real load to justify them.

Run locally (no Docker)::

    python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/docs or GET /api/v1/health.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict

from app.api.v1.approvals import router as governance_router
from app.api.v1.orchestrator import router as orchestrator_router
from app.api.v1.research import router as research_router
from app.core.config import (
    API_V1_PREFIX,
    APP_NAME,
    APP_VERSION,
    ARCHITECTURE_PHASE,
    settings,
)
from app.db.session import dispose_engine, init_db

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Payload of ``GET /api/v1/health``."""

    model_config = ConfigDict(frozen=True)

    status: str
    environment: str
    architecture_phase: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown.

    Tables are auto-created in development so a fresh clone runs with zero
    setup; staging and production are expected to run ``alembic upgrade head``
    (Milestone 1B) instead of letting the app mutate its own schema.
    """
    if settings.is_development:
        init_db()
        logger.info("Development startup: schema ensured for %s", settings.database_url)
    else:
        logger.info(
            "Skipping create_all in %s - run 'alembic upgrade head'", settings.app_env
        )
    try:
        yield
    finally:
        dispose_engine()
        logger.info("Shutdown complete: database connections released")


def create_app() -> FastAPI:
    """Application factory.

    A factory (rather than only a module-level ``app``) lets the test suite
    build isolated instances and lets later milestones compose extra routers.
    """
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "Autonomous dropshipping agent system with a two-way human "
            "approval model. Architecture phase: " + ARCHITECTURE_PHASE + "."
        ),
        lifespan=lifespan,
    )

    api_v1 = APIRouter(prefix=API_V1_PREFIX)

    @api_v1.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness probe",
        tags=["system"],
    )
    def health() -> HealthResponse:
        """Report liveness, the active environment and the built phase."""
        return HealthResponse(
            status="ok",
            environment=settings.app_env,
            architecture_phase=ARCHITECTURE_PHASE,
        )

    app.include_router(api_v1)

    # Milestone 1B/1C surface: governance (permissions, approvals, audit) and
    # the orchestrator/ops API. They declare paths relative to /api/v1.
    app.include_router(governance_router, prefix=API_V1_PREFIX)
    app.include_router(orchestrator_router, prefix=API_V1_PREFIX)
    # Milestone 2B surface: product research opportunity reports (Agent 1).
    app.include_router(research_router, prefix=API_V1_PREFIX)
    return app


#: ASGI instance used by ``uvicorn app.main:app``.
app = create_app()
