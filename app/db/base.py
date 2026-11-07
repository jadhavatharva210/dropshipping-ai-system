"""SQLAlchemy 2.0 declarative base, UTC timestamp mixin and portable types.

Everything in here exists to make one database schema behave identically on
SQLite (local dev on Windows, no Docker) and PostgreSQL (production):

* :class:`UTCDateTime` - SQLite has no timezone support and silently strips
  ``tzinfo`` on read. This decorator normalises to UTC on the way in and
  re-attaches ``timezone.utc`` on the way out, so application code can always
  compare timestamps safely. On PostgreSQL it maps to ``TIMESTAMP WITH TIME
  ZONE``.
* :data:`NAMING_CONVENTION` - deterministic constraint/index names, which is
  what makes Alembic autogenerate (Milestone 1B) stable across machines.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, MetaData, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "Base",
    "NAMING_CONVENTION",
    "TimestampMixin",
    "UTCDateTime",
    "new_uuid",
    "utcnow",
]


def utcnow() -> datetime:
    """Timezone-aware current time in UTC.

    Never use ``datetime.utcnow()``: it returns a *naive* datetime, which
    compares badly against aware values and is deprecated in Python 3.12+.
    """
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    """Primary-key factory: UUID4 rendered as a 36-char string.

    Strings (rather than a native UUID column) keep the schema identical on
    SQLite and PostgreSQL and make ids safe to paste into logs and URLs.
    """
    return str(uuid.uuid4())


#: Explicit names for every constraint class, so migrations are reproducible.
#: ``ck`` must keep the ``constraint_name`` token: SQLAlchemy names enum CHECK
#: constraints after the enum class (``ck_tasks_taskstatus``), and a convention
#: without that token is silently skipped for them. Pass ``constraint_name=``
#: to :func:`app.models.core.enum_column` if one table ever uses the same enum
#: on two columns - ``tests`` assert constraint names stay unique per table.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Root base class for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC ``DateTime`` that behaves the same on every backend.

    * bind: naive values are assumed to already be UTC and are tagged as such;
      aware values are converted to UTC.
    * result: naive values (what SQLite always returns) are tagged UTC; aware
      values (PostgreSQL ``timestamptz``) are converted to UTC.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class TimestampMixin:
    """Adds ``created_at`` / ``updated_at`` (UTC, auto-maintained) to a model.

    ``updated_at`` is refreshed by SQLAlchemy on every UPDATE, so models that
    need it never have to remember to set it by hand.
    """

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime,
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )
