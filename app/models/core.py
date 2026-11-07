"""Core domain models: tasks, approvals, audit log and shared memory.

These four tables are the entire persistence surface of Milestone 1A and the
contract every later milestone builds on:

``tasks``
    A unit of work owned by exactly one agent, optionally nested under a parent
    task (the orchestrator decomposes goals into a tree).
``approval_requests``
    The human gate for ``LEVEL_2_APPROVAL`` actions. A task parks in
    ``WAITING_APPROVAL`` while one of these is ``PENDING``.
``audit_logs``
    Append-only trail of *what happened, who did it and under which permission
    level*. Rows deliberately outlive their task (``ON DELETE SET NULL``):
    destroying evidence when a task is deleted would defeat the purpose.
``shared_memory``
    Namespaced key/value store the agents use to hand context to each other
    without talking directly.

Portability notes
-----------------
* JSON payloads use SQLAlchemy's generic :class:`~sqlalchemy.JSON` type - TEXT
  with serialisation on SQLite, native ``JSON`` on PostgreSQL, no dialect
  branching anywhere. Object-shaped payloads are wrapped in
  :class:`~sqlalchemy.ext.mutable.MutableDict` (see :func:`json_column`) so an
  in-place ``payload["k"] = v`` is not silently lost; ``SharedMemory.value``
  stays untracked so it can hold any JSON shape.
* Enums are VARCHAR-backed with a real ``CHECK`` constraint (see
  :func:`enum_column`), so an invalid value is rejected by the database itself
  and not only by Python, and so the DDL is identical on both backends.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Enum as SAEnum,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, MappedColumn, mapped_column, relationship

from app.core.config import settings
from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus
from app.db.base import Base, TimestampMixin, UTCDateTime, new_uuid, utcnow

__all__ = [
    "ApprovalRequest",
    "AuditLog",
    "SharedMemory",
    "Task",
    "enum_column",
    "json_column",
]

#: Generous width for every enum VARCHAR: the longest current member is
#: ``ADVERTISING_ANALYTICS`` (21 chars).
_ENUM_MAX_LENGTH = 32

#: JSON that tells the unit of work when a payload is edited *in place*, so
#: ``task.output_payload["x"] = 1`` is persisted instead of silently lost.
MutableJSON = MutableDict.as_mutable(JSON)


def enum_column(
    enum_cls: type[enum.Enum],
    *,
    constraint_name: str | None = None,
    **kwargs: Any,
) -> MappedColumn[Any]:
    """``mapped_column`` for a domain enum, validated at two layers.

    ``native_enum=False`` stores the value in a VARCHAR instead of creating a
    database enum type: the DDL is then identical on SQLite and PostgreSQL, and
    adding a member later is a plain Alembic ``ALTER`` rather than a risky enum
    migration.

    * ``validate_strings=True`` - the ORM rejects an unknown value immediately
      (``StatementError`` wrapping ``LookupError``) instead of shipping a typo
      down to the database.
    * ``create_constraint=True`` - a real ``CHECK`` constraint, so raw SQL,
      hand-edited rows and bad migrations are rejected by the database too.

    :param constraint_name: override for the generated CHECK constraint name.
        By default SQLAlchemy derives it from the enum class, so a table using
        the *same* enum on two columns must disambiguate them here, e.g.
        ``enum_column(AgentType, constraint_name="from_agent")``.
    """
    enum_kwargs: dict[str, Any] = {
        "native_enum": False,
        "length": _ENUM_MAX_LENGTH,
        "create_constraint": True,
        "validate_strings": True,
    }
    if constraint_name:
        # Passing name=None explicitly would make SQLAlchemy emit an *anonymous*
        # constraint, which the naming convention then rejects - so only set it
        # when a real override was given.
        enum_kwargs["name"] = constraint_name
    return mapped_column(SAEnum(enum_cls, **enum_kwargs), **kwargs)


def json_column(**kwargs: Any) -> MappedColumn[Any]:
    """Change-tracked JSON **object** column (see :data:`MutableJSON`).

    Use for payloads annotated ``dict[str, Any] | None``. Two rules follow from
    :class:`~sqlalchemy.ext.mutable.MutableDict`, both covered by tests:

    1. Only edits of the *outer* object are tracked. ``payload["k"] = v`` is
       persisted; ``payload["k"].append(v)`` is not - reassign the nested
       container instead.
    2. The owning ORM instance must still be reachable. A dict grabbed from an
       instance that is then garbage collected has no parent to notify, so the
       change is dropped. Keep a reference to the object, not just its payload.

    Assigning a non-dict (list/str/…) raises ``ValueError``: these columns are
    contractually JSON objects. For free-form values use a plain ``JSON``
    column, as ``SharedMemory.value`` does.
    """
    return mapped_column(MutableJSON, **kwargs)


def _default_expiry() -> datetime:
    """``now + DEFAULT_APPROVAL_EXPIRY_MINUTES`` - keeps approvals from rotting."""
    return utcnow() + settings.approval_expiry


class Task(Base, TimestampMixin):
    """A unit of work assigned to one agent, optionally part of a task tree."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    parent_task_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    assigned_agent: Mapped[AgentType] = enum_column(
        AgentType, nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = enum_column(
        TaskStatus,
        default=TaskStatus.PENDING,
        nullable=False,
        index=True,
    )
    permission_level: Mapped[PermissionLevel] = enum_column(
        PermissionLevel, nullable=False
    )
    #: JSON object describing what the agent was asked to do.
    input_payload: Mapped[dict[str, Any] | None] = json_column(nullable=True)
    #: JSON object describing what the agent produced.
    output_payload: Mapped[dict[str, Any] | None] = json_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # ── Relationships ─────────────────────────────────────────────────────
    #: Sub-tasks. ``passive_deletes`` lets the database apply the FK cascade
    #: instead of the ORM loading the whole subtree into memory first.
    children: Mapped[list["Task"]] = relationship(
        "Task",
        back_populates="parent",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Task.created_at",
    )
    parent: Mapped["Task | None"] = relationship(
        "Task",
        back_populates="children",
        remote_side="Task.id",
    )
    approvals: Mapped[list["ApprovalRequest"]] = relationship(
        "ApprovalRequest",
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ApprovalRequest.created_at",
    )
    #: Audit rows keep living after their task is deleted (``SET NULL``), so
    #: this side must not cascade - hence no ``delete-orphan`` here.
    audit_entries: Mapped[list["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="task",
        passive_deletes=True,
        order_by="AuditLog.created_at",
    )

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Task id={self.id!r} title={self.title!r} "
            f"agent={self.assigned_agent} status={self.status}>"
        )


class ApprovalRequest(Base):
    """A ``LEVEL_2_APPROVAL`` action waiting for (or settled by) a human."""

    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    requesting_agent: Mapped[AgentType] = enum_column(
        AgentType, nullable=False, index=True
    )
    action_description: Mapped[str] = mapped_column(Text, nullable=False)
    #: Why the agent wants to do this - the context a human needs to decide.
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Exact parameters the agent intends to use if approved.
    requested_parameters: Mapped[dict[str, Any] | None] = json_column(nullable=True)
    status: Mapped[ApprovalStatus] = enum_column(
        ApprovalStatus,
        default=ApprovalStatus.PENDING,
        nullable=False,
        index=True,
    )
    approver_identity: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, default=_default_expiry, nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    #: What actually happened when the approved action was executed.
    execution_result: Mapped[dict[str, Any] | None] = json_column(nullable=True)

    task: Mapped["Task"] = relationship("Task", back_populates="approvals")

    @property
    def is_expired(self) -> bool:
        """True when a still-pending request has passed its deadline."""
        if self.status is ApprovalStatus.EXPIRED:
            return True
        if self.status is not ApprovalStatus.PENDING or self.expires_at is None:
            return False
        return self.expires_at <= utcnow()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<ApprovalRequest id={self.id!r} task_id={self.task_id!r} "
            f"status={self.status} agent={self.requesting_agent}>"
        )


class AuditLog(Base):
    """Append-only record of an agent action and the permission it ran under."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    #: Nullable on purpose: system-level events (startup, sweeps) have no task,
    #: and rows survive task deletion via ``ON DELETE SET NULL``.
    task_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    agent_type: Mapped[AgentType] = enum_column(
        AgentType, nullable=False, index=True
    )
    action_name: Mapped[str] = mapped_column(String(255), nullable=False)
    permission_level: Mapped[PermissionLevel] = enum_column(
        PermissionLevel, nullable=False
    )
    #: Free-form outcome (``success``, ``failed``, ``blocked``, ``approved``…).
    #: Deliberately a string, not an enum: the vocabulary grows every milestone
    #: and an audit trail must never fail to write.
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[dict[str, Any] | None] = json_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, nullable=False, index=True
    )

    task: Mapped["Task | None"] = relationship("Task", back_populates="audit_entries")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<AuditLog id={self.id!r} action={self.action_name!r} "
            f"agent={self.agent_type} status={self.status!r}>"
        )


class SharedMemory(Base, TimestampMixin):
    """Namespaced key/value store used for inter-agent context sharing.

    ``(namespace, key)`` is unique, so agents can upsert shared state without
    racing each other into duplicates.
    """

    __tablename__ = "shared_memory"
    __table_args__ = (
        UniqueConstraint("namespace", "key", name="uq_shared_memory_namespace_key"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=new_uuid
    )
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    #: Any JSON-serialisable value: object, list or scalar.
    #:
    #: Deliberately a plain ``JSON`` column rather than :func:`json_column`:
    #: ``MutableDict`` refuses to coerce lists and scalars, and shared memory
    #: must be able to hold any shape an agent writes. The rule here is simply
    #: **reassign ``value`` to persist a change** (assigning a whole new object
    #: is always tracked).
    value: Mapped[Any] = mapped_column(JSON, nullable=True)
    updated_by_agent: Mapped[AgentType] = enum_column(AgentType, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<SharedMemory namespace={self.namespace!r} key={self.key!r} "
            f"updated_by={self.updated_by_agent}>"
        )
