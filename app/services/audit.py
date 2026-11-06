"""Audit logging service: every governance decision leaves a trail."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import AgentType, PermissionLevel
from app.models.core import AuditLog


class AuditService:
    """Writes and queries AuditLog entries.

    log_event only flushes: the caller owns the transaction boundary and
    commits, so audit rows always land atomically with the state change
    they describe.
    """

    @staticmethod
    def log_event(
        db: Session,
        *,
        agent_type: AgentType,
        action_name: str,
        permission_level: PermissionLevel,
        status: str,
        task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            task_id=task_id,
            agent_type=agent_type,
            action_name=action_name,
            permission_level=permission_level,
            status=status,
            details=details,
        )
        db.add(entry)
        db.flush()
        return entry

    @staticmethod
    def list_logs(
        db: Session,
        *,
        task_id: str | None = None,
        agent_type: AgentType | None = None,
        permission_level: PermissionLevel | None = None,
    ) -> list[AuditLog]:
        stmt = select(AuditLog)
        if task_id is not None:
            stmt = stmt.where(AuditLog.task_id == task_id)
        if agent_type is not None:
            stmt = stmt.where(AuditLog.agent_type == agent_type)
        if permission_level is not None:
            stmt = stmt.where(AuditLog.permission_level == permission_level)
        # Deterministic order: chronological, id breaks same-timestamp ties.
        stmt = stmt.order_by(AuditLog.created_at, AuditLog.id)
        return list(db.scalars(stmt).all())
