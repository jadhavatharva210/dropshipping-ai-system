"""Core domain models: tasks, approvals, audit log, shared memory.

JSON columns use SQLAlchemy's generic JSON type (TEXT on SQLite, JSONB-ready
on Postgres) so the models run unchanged on both backends.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Enum as SAEnum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus
from app.db.base import Base, TimestampMixin, utcnow


def _uuid() -> str:
    return str(uuid.uuid4())


class Task(Base, TimestampMixin):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    parent_task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )
    assigned_agent: Mapped[AgentType] = mapped_column(
        SAEnum(AgentType, native_enum=False, length=32), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        SAEnum(TaskStatus, native_enum=False, length=32),
        default=TaskStatus.PENDING,
        nullable=False,
    )
    permission_level: Mapped[PermissionLevel] = mapped_column(
        SAEnum(PermissionLevel, native_enum=False, length=32), nullable=False
    )
    input_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Self-referential hierarchy.
    children: Mapped[list["Task"]] = relationship(
        "Task", back_populates="parent", cascade="all, delete-orphan"
    )
    parent: Mapped["Task | None"] = relationship("Task", back_populates="children", remote_side="Task.id")
    # Approvals raised by this task.
    approvals: Mapped[list["ApprovalRequest"]] = relationship(
        "ApprovalRequest", back_populates="task", cascade="all, delete-orphan"
    )
    # Audit entries referencing this task.
    audit_entries: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="task", cascade="all, delete-orphan"
    )


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    requesting_agent: Mapped[AgentType] = mapped_column(
        SAEnum(AgentType, native_enum=False, length=32), nullable=False
    )
    action_description: Mapped[str] = mapped_column(Text, nullable=False)
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    requested_parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[ApprovalStatus] = mapped_column(
        SAEnum(ApprovalStatus, native_enum=False, length=32),
        default=ApprovalStatus.PENDING,
        nullable=False,
    )
    approver_identity: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(nullable=True)
    execution_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    task: Mapped["Task"] = relationship("Task", back_populates="approvals")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=True
    )
    agent_type: Mapped[AgentType] = mapped_column(
        SAEnum(AgentType, native_enum=False, length=32), nullable=False
    )
    action_name: Mapped[str] = mapped_column(String(255), nullable=False)
    permission_level: Mapped[PermissionLevel] = mapped_column(
        SAEnum(PermissionLevel, native_enum=False, length=32), nullable=False
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=utcnow)

    task: Mapped["Task | None"] = relationship("Task", back_populates="audit_entries")


class SharedMemory(Base, TimestampMixin):
    __tablename__ = "shared_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_by_agent: Mapped[AgentType] = mapped_column(
        SAEnum(AgentType, native_enum=False, length=32), nullable=False
    )
