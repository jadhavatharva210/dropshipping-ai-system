"""Milestone 1A foundation suite.

Covers: health endpoint, CRUD on all four core models, FK relationships,
and enum constraint enforcement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.enums import (
    AgentType,
    ApprovalStatus,
    PermissionLevel,
    TaskStatus,
)
from app.models.core import ApprovalRequest, AuditLog, SharedMemory, Task


# ── Health ────────────────────────────────────────────────────────────────


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "development"
    assert body["architecture_phase"] == "1A-foundation"


# ── Task ──────────────────────────────────────────────────────────────────


def test_create_task_with_enums(db: Session) -> None:
    task = Task(
        assigned_agent=AgentType.MAIN_ORCHESTRATOR,
        title="Coordinate product launch",
        objective="Delegate research, ads and delivery subtasks",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        input_payload={"launch_date": "2026-12-01"},
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    assert task.id is not None
    assert task.assigned_agent == AgentType.MAIN_ORCHESTRATOR
    assert task.status == TaskStatus.PENDING
    assert task.retry_count == 0
    assert task.created_at is not None
    assert task.updated_at is not None


def test_task_parent_child_hierarchy(db: Session, sample_task: Task) -> None:
    child = Task(
        parent_task_id=sample_task.id,
        assigned_agent=AgentType.ADVERTISING_ANALYTICS,
        title="Analyse ad creatives",
        objective="Score top 5 creatives for the niche",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
    )
    db.add(child)
    db.commit()
    db.refresh(sample_task)

    assert len(sample_task.children) == 1
    assert sample_task.children[0].id == child.id
    assert child.parent.id == sample_task.id


def test_task_status_transitions(db: Session, sample_task: Task) -> None:
    sample_task.status = TaskStatus.IN_PROGRESS
    db.commit()
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.IN_PROGRESS

    sample_task.status = TaskStatus.WAITING_APPROVAL
    db.commit()
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.WAITING_APPROVAL


def test_task_output_and_error_fields(db: Session, sample_task: Task) -> None:
    sample_task.output_payload = {"products_found": 10, "top_pick": "LED collar"}
    sample_task.status = TaskStatus.COMPLETED
    db.commit()
    db.refresh(sample_task)
    assert sample_task.output_payload["top_pick"] == "LED collar"

    sample_task.status = TaskStatus.FAILED
    sample_task.error_message = "Supplier API timeout"
    sample_task.retry_count = 2
    db.commit()
    db.refresh(sample_task)
    assert sample_task.error_message == "Supplier API timeout"
    assert sample_task.retry_count == 2


# ── ApprovalRequest ───────────────────────────────────────────────────────


def test_create_approval_request_linked_to_task(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.PRODUCT_RESEARCH,
        action_description="Place test order with supplier",
        context_summary="Need to verify product quality before scaling",
        requested_parameters={"supplier": "AliExpress", "budget_usd": 50},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=60),
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)

    assert approval.id is not None
    assert approval.status == ApprovalStatus.PENDING
    assert approval.task.id == sample_task.id
    assert len(sample_task.approvals) == 1


def test_approval_decision_flow(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.DELIVERY_OPERATIONS,
        action_description="Book courier pickup",
        requested_parameters={"courier": "Delhivery"},
    )
    db.add(approval)
    db.commit()

    approval.status = ApprovalStatus.APPROVED
    approval.approver_identity = "atharva"
    approval.decided_at = datetime.now(timezone.utc)
    approval.execution_result = {"awb": "1234567890"}
    db.commit()
    db.refresh(approval)

    assert approval.status == ApprovalStatus.APPROVED
    assert approval.approver_identity == "atharva"
    assert approval.execution_result == {"awb": "1234567890"}


def test_approval_rejection_flow(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.ADVERTISING_ANALYTICS,
        action_description="Increase daily ad budget to $500",
        requested_parameters={"daily_budget_usd": 500},
    )
    db.add(approval)
    db.commit()

    approval.status = ApprovalStatus.REJECTED
    approval.rejection_reason = "Budget cap exceeded for this week"
    approval.decided_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(approval)

    assert approval.status == ApprovalStatus.REJECTED
    assert approval.rejection_reason is not None


# ── AuditLog ──────────────────────────────────────────────────────────────


def test_create_audit_log_records(db: Session, sample_task: Task) -> None:
    entry = AuditLog(
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name="task.created",
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
        status="success",
        details={"title": sample_task.title},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    assert entry.id is not None
    assert entry.task.id == sample_task.id
    assert len(sample_task.audit_entries) == 1


def test_audit_log_without_task(db: Session) -> None:
    entry = AuditLog(
        task_id=None,
        agent_type=AgentType.MAIN_ORCHESTRATOR,
        action_name="system.startup",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        status="success",
        details={"version": "0.1.0"},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    assert entry.task is None


# ── SharedMemory ──────────────────────────────────────────────────────────


def test_create_and_update_shared_memory(db: Session) -> None:
    record = SharedMemory(
        namespace="product-research",
        key="winning_products",
        value={"products": ["LED collar", "pet camera"]},
        updated_by_agent=AgentType.PRODUCT_RESEARCH,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    assert record.id is not None
    assert record.created_at is not None

    record.value = {"products": ["LED collar", "pet camera", "slow feeder"]}
    record.updated_by_agent = AgentType.MAIN_ORCHESTRATOR
    db.commit()
    db.refresh(record)
    assert len(record.value["products"]) == 3


# ── Relationship integrity ────────────────────────────────────────────────


def test_task_approval_audit_chain(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.PRODUCT_RESEARCH,
        action_description="Approve supplier shortlist",
    )
    audit = AuditLog(
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name="approval.requested",
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
        status="success",
    )
    db.add_all([approval, audit])
    db.commit()
    db.refresh(sample_task)

    assert len(sample_task.approvals) == 1
    assert len(sample_task.audit_entries) == 1
    assert sample_task.approvals[0].requesting_agent == AgentType.PRODUCT_RESEARCH
    assert sample_task.audit_entries[0].action_name == "approval.requested"


# ── Enum constraints ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "enum_cls,value",
    [
        (AgentType, "NOT_AN_AGENT"),
        (TaskStatus, "DOING_STUFF"),
        (PermissionLevel, "LEVEL_99"),
        (ApprovalStatus, "MAYBE"),
    ],
)
def test_invalid_enum_values_rejected(enum_cls, value: str) -> None:
    with pytest.raises(ValueError):
        enum_cls(value)
