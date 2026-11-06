"""Engine, session factory and the FastAPI ``get_db()`` dependency.

SQLite is the local-dev default (no Docker, no setup, tiny memory footprint);
any PostgreSQL URL dropped into ``DATABASE_URL`` works unchanged. Dialect
specifics are isolated in :func:`build_engine` so the rest of the app never
branches on the backend:

* SQLite gets ``check_same_thread=False`` (FastAPI may serve a request from a
  worker thread), ``PRAGMA foreign_keys=ON`` (SQLite does **not** enforce
  foreign keys by default, which would silently disable our ``ON DELETE``
  rules) and ``PRAGMA busy_timeout`` so concurrent readers wait instead of
  raising "database is locked" on Windows.
* In-memory SQLite (``sqlite://``, used by the test suite) gets a
  ``StaticPool`` so every connection sees the same database.
* PostgreSQL gets a small, bounded pool (5 + 5 overflow) plus ``pool_pre_ping``
  and ``pool_recycle`` - deliberately conservative for an 8 GB RAM machine.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base

__all__ = [
    "SessionLocal",
    "build_engine",
    "dispose_engine",
    "engine",
    "get_db",
    "init_db",
    "session_factory",
]

logger = logging.getLogger(__name__)

#: Bound the Postgres pool: each connection costs server memory, and this box
#: also runs the OS, a browser and an editor.
_POSTGRES_POOL_SIZE = 5
_POSTGRES_MAX_OVERFLOW = 5
_POSTGRES_POOL_RECYCLE_SECONDS = 1800
_SQLITE_BUSY_TIMEOUT_MS = 5000


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _is_in_memory_sqlite(url: str) -> bool:
    """True for ``sqlite://``, ``sqlite:///`` and ``sqlite:///:memory:``."""
    if not _is_sqlite(url):
        return False
    path = url.split("://", 1)[-1]
    return path.strip("/") in {"", ":memory:"}


def _enable_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
    """Per-connection SQLite pragmas.

    ``foreign_keys`` is connection-scoped in SQLite, so it must be switched on
    for every new connection - otherwise ``ON DELETE CASCADE``/``SET NULL``
    rules defined on the models are ignored in local dev and in tests.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()


def build_engine(database_url: str | None = None, *, echo: bool | None = None) -> Engine:
    """Create an engine tuned for the dialect of ``database_url``."""
    url = (database_url if database_url is not None else settings.database_url).strip()
    kwargs: dict[str, Any] = {
        "echo": settings.sql_echo if echo is None else echo,
    }

    if _is_sqlite(url):
        kwargs["connect_args"] = {"check_same_thread": False}
        if _is_in_memory_sqlite(url):
            # One shared connection, otherwise every new connection would see a
            # brand-new empty in-memory database.
            kwargs["poolclass"] = StaticPool
    else:
        kwargs.update(
            pool_size=_POSTGRES_POOL_SIZE,
            max_overflow=_POSTGRES_MAX_OVERFLOW,
            pool_recycle=_POSTGRES_POOL_RECYCLE_SECONDS,
            pool_pre_ping=True,
        )

    new_engine = create_engine(url, **kwargs)
    if _is_sqlite(url):
        event.listen(new_engine, "connect", _enable_sqlite_pragmas)
    return new_engine


#: The application engine. Tests swap this attribute (via ``monkeypatch``) or
#: build their own with :func:`build_engine`.
engine: Engine = build_engine()

#: Session factory used by the FastAPI dependency. ``expire_on_commit=False``
#: keeps objects readable after ``commit()`` - essential for request handlers
#: that return ORM-backed Pydantic models.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def session_factory(bind: Engine | None = None) -> sessionmaker[Session]:
    """Build a session factory for an arbitrary engine (used by the tests)."""
    return sessionmaker(
        bind=bind if bind is not None else engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a session, always close it afterwards.

    Usage::

        @router.get("/tasks")
        def list_tasks(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine: Engine | None = None) -> None:
    """Create any missing tables.

    Development convenience only - from Milestone 1B onwards Alembic owns
    schema changes and production starts with ``alembic upgrade head``.
    """
    import app.models  # noqa: F401  - importing registers the tables on metadata

    active_engine = engine if target_engine is None else target_engine
    Base.metadata.create_all(bind=active_engine)
    logger.info("Database schema ensured on %s", active_engine.url)


def dispose_engine() -> None:
    """Release pooled connections (called on application shutdown)."""
    engine.dispose()
