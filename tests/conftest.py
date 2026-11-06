"""Shared pytest fixtures for the Milestone 1A foundation suite.

Hermetic by construction
------------------------
``APP_ENV`` / ``DATABASE_URL`` / ``AI_PROVIDER`` are forced into the process
environment *before* any ``app.*`` module is imported, because
:data:`app.core.config.settings` is built once at import time. That means
neither a developer's local ``.env`` nor a production ``DATABASE_URL`` can
change what the tests do - the suite always runs against a throwaway in-memory
SQLite database.

Every test gets a **fresh engine** (and therefore a fresh in-memory database),
so no state can leak between tests in either direction.

The HTTP client is Starlette's ``TestClient``, which subclasses
``httpx.Client`` - i.e. FastAPI's httpx-based test client. It is used as a
context manager so the application lifespan (startup schema init, shutdown
connection release) runs exactly like it does under uvicorn.
"""

from __future__ import annotations

# ── Must run before any app import (settings are read at import time) ──────
import os

os.environ["APP_ENV"] = "development"
os.environ["DATABASE_URL"] = "sqlite://"  # in-memory; build_engine adds StaticPool
os.environ["AI_PROVIDER"] = "mock"
os.environ["SQL_ECHO"] = "false"

from collections.abc import Generator  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.core.enums import AgentType, PermissionLevel, TaskStatus  # noqa: E402
from app import main as app_main  # noqa: E402
from app.db import session as session_module  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import build_engine, get_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models.core import Task  # noqa: E402

#: In-memory SQLite: zero disk I/O, zero cleanup, dies with the process.
TEST_DATABASE_URL = "sqlite://"


@pytest.fixture()
def engine() -> Generator[Engine, None, None]:
    """A brand-new in-memory database, created from the real model metadata."""
    test_engine = build_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=test_engine)
    try:
        yield test_engine
    finally:
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


@pytest.fixture()
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )


@pytest.fixture()
def db(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """An ORM session bound to this test's database."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def app(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """An app whose engine is this test's engine.

    Two patches make the app safe to drive from a test:

    * ``session_module.engine`` -> this test's engine, so the lifespan's
      ``init_db()`` builds the throwaway database instead of touching
      ``dropshipping_dev.db`` on disk.
    * ``app_main.dispose_engine`` -> no-op. In production the shutdown hook
      releases pooled connections, which is correct and is asserted separately
      by ``test_lifespan_releases_connections_on_shutdown``. Here the engine is
      a single-connection ``StaticPool`` that the test's own ORM session is
      still using, so disposing it at client teardown would close the database
      underneath that session and its teardown rollback would fail. The
      ``engine`` fixture owns disposal instead - after the session is closed.
    """
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(app_main, "dispose_engine", lambda: None)
    return create_app()


@pytest.fixture()
def client(app: FastAPI, db: Session) -> Generator[TestClient, None, None]:
    """httpx-backed ``TestClient`` with ``get_db`` pointed at the test session."""

    def override_get_db() -> Generator[Session, None, None]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_task(db: Session) -> Task:
    """A committed LEVEL_2_APPROVAL task used by most relationship tests."""
    task = Task(
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Find winning products",
        objective="Research 10 trending pet products under $20 with 4.5+ rating",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
        input_payload={"niche": "pet accessories", "max_price": 20},
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task
