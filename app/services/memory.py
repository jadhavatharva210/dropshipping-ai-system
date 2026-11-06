"""Shared business memory service: cross-agent context with clean upserts."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import AgentType
from app.models.core import SharedMemory


class SharedMemoryService:
    """Namespaced key/value memory shared between agents."""

    @staticmethod
    def set_memory(
        db: Session,
        *,
        namespace: str,
        key: str,
        value: Any,
        agent_type: AgentType,
    ) -> SharedMemory:
        record = SharedMemoryService.get_memory(db, namespace=namespace, key=key)
        if record is None:
            record = SharedMemory(
                namespace=namespace,
                key=key,
                value=value,
                updated_by_agent=agent_type,
            )
            db.add(record)
        else:
            record.value = value
            record.updated_by_agent = agent_type
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def get_memory(db: Session, *, namespace: str, key: str) -> SharedMemory | None:
        stmt = select(SharedMemory).where(
            SharedMemory.namespace == namespace, SharedMemory.key == key
        )
        return db.scalars(stmt).first()

    @staticmethod
    def list_namespace(db: Session, *, namespace: str) -> list[SharedMemory]:
        stmt = (
            select(SharedMemory)
            .where(SharedMemory.namespace == namespace)
            .order_by(SharedMemory.created_at, SharedMemory.id)
        )
        return list(db.scalars(stmt).all())
