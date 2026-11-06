"""Milestone 1B governance suite.

Covers: 3-tier classification (incl. deny-by-default), Level 1 auto-execution,
Level 2 approval state machine (approve / reject / expiry / double-decision),
the stale-approval sweep, the audit service, and every /api/v1 governance
endpoint end-to-end over HTTPX.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus
from app.core.permissions import PermissionManager, PermissionPolicy, TaskNotFoundError
from app.models.core import ApprovalRequest, AuditLog, Task
from app.schemas.governance import ActionEvaluationRequest
from app.services.approval import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalService,
)
from app.services.audit import AuditService

LEVEL_1_ACTIONS = [
    "research_products",
    "analyze_competitors",
    "compare_suppliers",
    "calculate_margins",
    "draft_product_description",
    "draft_ad_copy",
    "prepare_creative_brief",
    "analyze_campaign_metrics",
    "check_order_status",
    "draft_customer_reply",
    "generate_report",
]

LEVEL_2_ACTIONS = [
    "publish_advertisement",
    "allocate_ad_budget",
    "place_supplier_order",
    "commit_customer_order",
    "issue_refund",
    "cancel_order",
    "update_live_price",
    "send_customer_communication",
    "update_account_settings",
]

LEVEL_3_ACTIONS = [
    "delete_audit_logs",
    "bypass_permission_check",
    "export_raw_credentials",
    "execute_arbitrary_shell",
]


def _eval_request(task_id: str, action_name: str) -> ActionEvaluationRequest:
    return ActionEvaluationRequest(
        task_id=task_id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name=action_name,
        action_description=f"Perform {action_name}",
        context_summary="unit test context",
        parameters={"dry_run": True},
    )


def _make_level2_approval(db: Session, task: Task) -> str:
    """Run a Level 2 evaluation and return the new approval request id."""
    result = PermissionManager.evaluate_action(
        db, _eval_request(task.id, "place_supplier_order")
    )
    assert result.approval_request_id is not None
    return result.approval_request_id


def _force_expired(db: Session, approval_id: str) -> None:
    approval = db.get(ApprovalRequest, approval_id)
    approval.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()


# ── Classification (deny-by-default) ─────────────────────────────────────


@pytest.mark.parametrize("action", LEVEL_1_ACTIONS)
def test_level_1_actions_classified(action: str) -> None:
    assert PermissionPolicy.level_for(action) == PermissionLevel.LEVEL_1_AUTO


@pytest.mark.parametrize("action", LEVEL_2_ACTIONS)
def test_level_2_actions_classified(action: str) -> None:
    assert PermissionPolicy.level_for(action) == PermissionLevel.LEVEL_2_APPROVAL


@pytest.mark.parametrize("action", LEVEL_3_ACTIONS)
def test_level_3_actions_classified(action: str) -> None:
    assert PermissionPolicy.level_for(action) == PermissionLevel.LEVEL_3_BLOCKED


@pytest.mark.parametrize("action", ["rm_rf_everything", "send_all_money", "", "RESEARCH_PRODUCTS"])
def test_unknown_actions_denied_by_default(action: str) -> None:
    assert PermissionPolicy.level_for(action) == PermissionLevel.LEVEL_3_BLOCKED


# ── Level 1: immediate grant + audit trail ────────────────────────────────


def test_level1_evaluation_allows_and_transitions_task(
    db: Session, sample_task: Task
) -> None:
    result = PermissionManager.evaluate_action(
        db, _eval_request(sample_task.id, "research_products")
    )
    assert result.allowed_to_proceed is True
    assert result.permission_level == PermissionLevel.LEVEL_1_AUTO
    assert result.task_status == TaskStatus.IN_PROGRESS
    assert result.approval_request_id is None

    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.IN_PROGRESS
    assert sample_task.permission_level == PermissionLevel.LEVEL_1_AUTO

    audit = db.get(AuditLog, result.audit_log_id)
    assert audit is not None
    assert audit.status == "ALLOWED_AUTO"
    assert audit.task_id == sample_task.id
    assert audit.action_name == "research_products"


def test_level1_keeps_non_pending_task_status(db: Session, sample_task: Task) -> None:
    sample_task.status = TaskStatus.COMPLETED
    db.commit()
    result = PermissionManager.evaluate_action(
        db, _eval_request(sample_task.id, "generate_report")
    )
    assert result.allowed_to_proceed is True
    assert result.task_status == TaskStatus.COMPLETED


# ── Level 2: pause + approval request ────────────────────────────────────


def test_level2_evaluation_pauses_task_and_creates_approval(
    db: Session, sample_task: Task
) -> None:
    before = datetime.now(timezone.utc)
    result = PermissionManager.evaluate_action(
        db, _eval_request(sample_task.id, "issue_refund")
    )
    assert result.allowed_to_proceed is False
    assert result.permission_level == PermissionLevel.LEVEL_2_APPROVAL
    assert result.task_status == TaskStatus.WAITING_APPROVAL

    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.WAITING_APPROVAL

    approval = db.get(ApprovalRequest, result.approval_request_id)
    assert approval is not None
    assert approval.status == ApprovalStatus.PENDING
    assert approval.task_id == sample_task.id
    assert approval.requested_parameters == {"dry_run": True}
    expiry = approval.expires_at
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    assert before + timedelta(minutes=59) < expiry < before + timedelta(minutes=61)

    audit = db.get(AuditLog, result.audit_log_id)
    assert audit.status == "WAITING_HUMAN_APPROVAL"


# ── Level 3: block ────────────────────────────────────────────────────────


def test_level3_evaluation_blocks_task(db: Session, sample_task: Task) -> None:
    result = PermissionManager.evaluate_action(
        db, _eval_request(sample_task.id, "execute_arbitrary_shell")
    )
    assert result.allowed_to_proceed is False
    assert result.permission_level == PermissionLevel.LEVEL_3_BLOCKED
    assert result.task_status == TaskStatus.BLOCKED
    assert result.approval_request_id is None
    audit = db.get(AuditLog, result.audit_log_id)
    assert audit.status == "BLOCKED_PROHIBITED_ACTION"


def test_unregistered_action_evaluation_blocks_task(db: Session, sample_task: Task) -> None:
    result = PermissionManager.evaluate_action(
        db, _eval_request(sample_task.id, "exfiltrate_database")
    )
    assert result.allowed_to_proceed is False
    assert result.permission_level == PermissionLevel.LEVEL_3_BLOCKED
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.BLOCKED


def test_evaluate_missing_task_raises(db: Session) -> None:
    with pytest.raises(TaskNotFoundError):
        PermissionManager.evaluate_action(db, _eval_request("no-such-task", "research_products"))


# ── Approval flow ─────────────────────────────────────────────────────────


def test_approve_request_approves_and_resumes_task(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    approval = ApprovalService.approve_request(db, approval_id, "atharva")

    assert approval.status == ApprovalStatus.APPROVED
    assert approval.approver_identity == "atharva"
    assert approval.decided_at is not None
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.IN_PROGRESS

    logs = AuditService.list_logs(db, task_id=sample_task.id)
    assert any(entry.status == "APPROVED_BY_HUMAN" for entry in logs)


def test_approve_with_execution_result_completes_task(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    approval = ApprovalService.approve_request(
        db, approval_id, "atharva", execution_result={"order_id": "SUP-42"}
    )
    assert approval.execution_result == {"order_id": "SUP-42"}
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.COMPLETED


# ── Rejection flow ────────────────────────────────────────────────────────


def test_reject_request_blocks_task_and_records_reason(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    approval = ApprovalService.reject_request(db, approval_id, "atharva", "Too expensive")

    assert approval.status == ApprovalStatus.REJECTED
    assert approval.rejection_reason == "Too expensive"
    assert approval.approver_identity == "atharva"
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.BLOCKED

    logs = AuditService.list_logs(db, task_id=sample_task.id)
    assert any(entry.status == "REJECTED_BY_HUMAN" for entry in logs)


@pytest.mark.parametrize("reason", ["", "   "])
def test_reject_requires_reason(db: Session, sample_task: Task, reason: str) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    with pytest.raises(ValueError):
        ApprovalService.reject_request(db, approval_id, "atharva", reason)
    db.refresh(db.get(ApprovalRequest, approval_id))
    assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.PENDING


# ── Expiry ────────────────────────────────────────────────────────────────


def test_expired_approval_cannot_be_approved(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    _force_expired(db, approval_id)

    with pytest.raises(ApprovalExpiredError):
        ApprovalService.approve_request(db, approval_id, "atharva")

    approval = db.get(ApprovalRequest, approval_id)
    assert approval.status == ApprovalStatus.EXPIRED
    assert approval.execution_result is None
    logs = AuditService.list_logs(db, task_id=sample_task.id)
    assert any(entry.status == "APPROVAL_EXPIRED" for entry in logs)


def test_expired_approval_cannot_be_rejected(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    _force_expired(db, approval_id)

    with pytest.raises(ApprovalExpiredError):
        ApprovalService.reject_request(db, approval_id, "atharva", "too slow")

    assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.EXPIRED


# ── Double-decision prevention ────────────────────────────────────────────


def test_double_approval_prevented(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    ApprovalService.approve_request(db, approval_id, "atharva")
    with pytest.raises(ApprovalAlreadyDecidedError):
        ApprovalService.approve_request(db, approval_id, "atharva")


def test_reject_after_approve_prevented(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    ApprovalService.approve_request(db, approval_id, "atharva")
    with pytest.raises(ApprovalAlreadyDecidedError):
        ApprovalService.reject_request(db, approval_id, "atharva", "changed my mind")


def test_approve_after_reject_prevented(db: Session, sample_task: Task) -> None:
    approval_id = _make_level2_approval(db, sample_task)
    ApprovalService.reject_request(db, approval_id, "atharva", "no budget")
    with pytest.raises(ApprovalAlreadyDecidedError):
        ApprovalService.approve_request(db, approval_id, "atharva")


# ── Stale sweep ───────────────────────────────────────────────────────────


def test_expire_stale_requests_sweeps_and_blocks(db: Session, sample_task: Task) -> None:
    stale_ids = [_make_level2_approval(db, sample_task) for _ in range(2)]
    for approval_id in stale_ids:
        _force_expired(db, approval_id)
    fresh_id = _make_level2_approval(db, sample_task)

    expired_count = ApprovalService.expire_stale_requests(db)
    assert expired_count == 2

    for approval_id in stale_ids:
        assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.EXPIRED
    assert db.get(ApprovalRequest, fresh_id).status == ApprovalStatus.PENDING

    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.BLOCKED
    assert sample_task.error_message == "Approval request expired"

    logs = AuditService.list_logs(db, task_id=sample_task.id)
    assert sum(entry.status == "APPROVAL_EXPIRED" for entry in logs) == 2


def test_expire_stale_requests_noop_when_nothing_stale(db: Session, sample_task: Task) -> None:
    _make_level2_approval(db, sample_task)
    assert ApprovalService.expire_stale_requests(db) == 0


# ── Audit service ─────────────────────────────────────────────────────────


def test_audit_service_log_and_filtered_list(db: Session, sample_task: Task) -> None:
    AuditService.log_event(
        db,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name="research_products",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        status="ALLOWED_AUTO",
        task_id=sample_task.id,
    )
    AuditService.log_event(
        db,
        agent_type=AgentType.MAIN_ORCHESTRATOR,
        action_name="system.review",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        status="ALLOWED_AUTO",
    )
    db.commit()

    all_logs = AuditService.list_logs(db)
    assert len(all_logs) == 2
    assert [entry.created_at for entry in all_logs] == sorted(
        entry.created_at for entry in all_logs
    )

    by_task = AuditService.list_logs(db, task_id=sample_task.id)
    assert len(by_task) == 1
    by_agent = AuditService.list_logs(db, agent_type=AgentType.MAIN_ORCHESTRATOR)
    assert len(by_agent) == 1 and by_agent[0].task_id is None


# ── HTTP API: permissions/evaluate ────────────────────────────────────────


def _evaluate_payload(task_id: str, action_name: str) -> dict:
    return {
        "task_id": task_id,
        "agent_type": AgentType.PRODUCT_RESEARCH.value,
        "action_name": action_name,
        "action_description": f"Perform {action_name}",
        "context_summary": "api test",
        "parameters": {"dry_run": True},
    }


def test_api_evaluate_level1(client: TestClient, sample_task: Task) -> None:
    response = client.post(
        "/api/v1/permissions/evaluate",
        json=_evaluate_payload(sample_task.id, "research_products"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["allowed_to_proceed"] is True
    assert body["permission_level"] == "LEVEL_1_AUTO"
    assert body["task_status"] == "IN_PROGRESS"
    assert body["approval_request_id"] is None
    assert body["audit_log_id"]


def test_api_evaluate_level2_creates_approval(client: TestClient, sample_task: Task) -> None:
    response = client.post(
        "/api/v1/permissions/evaluate",
        json=_evaluate_payload(sample_task.id, "publish_advertisement"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["allowed_to_proceed"] is False
    assert body["permission_level"] == "LEVEL_2_APPROVAL"
    assert body["task_status"] == "WAITING_APPROVAL"
    assert body["approval_request_id"]


def test_api_evaluate_level3_and_unknown_blocked(client: TestClient, sample_task: Task) -> None:
    for action in ("delete_audit_logs", "totally_made_up_action"):
        response = client.post(
            "/api/v1/permissions/evaluate", json=_evaluate_payload(sample_task.id, action)
        )
        assert response.status_code == 200
        body = response.json()
        assert body["allowed_to_proceed"] is False
        assert body["permission_level"] == "LEVEL_3_BLOCKED"
        assert body["task_status"] == "BLOCKED"


def test_api_evaluate_missing_task_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/permissions/evaluate",
        json=_evaluate_payload("missing-task", "research_products"),
    )
    assert response.status_code == 404


# ── HTTP API: approvals ───────────────────────────────────────────────────


def _api_raise_approval(client: TestClient, task_id: str) -> str:
    response = client.post(
        "/api/v1/permissions/evaluate",
        json=_evaluate_payload(task_id, "allocate_ad_budget"),
    )
    return response.json()["approval_request_id"]


def test_api_list_and_filter_approvals(client: TestClient, sample_task: Task) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)

    response = client.get("/api/v1/approvals")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [approval_id]

    response = client.get("/api/v1/approvals", params={"status": "PENDING"})
    assert len(response.json()) == 1
    response = client.get("/api/v1/approvals", params={"status": "APPROVED"})
    assert response.json() == []


def test_api_get_approval_and_404(client: TestClient, sample_task: Task) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)
    response = client.get(f"/api/v1/approvals/{approval_id}")
    assert response.status_code == 200
    assert response.json()["id"] == approval_id
    assert response.json()["status"] == "PENDING"

    assert client.get("/api/v1/approvals/nope").status_code == 404


def test_api_approve_flow_and_double_approve_409(client: TestClient, sample_task: Task) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)
    response = client.post(
        f"/api/v1/approvals/{approval_id}/approve",
        json={"approver_identity": "atharva", "execution_result": {"ok": True}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["approver_identity"] == "atharva"
    assert body["execution_result"] == {"ok": True}

    second = client.post(
        f"/api/v1/approvals/{approval_id}/approve",
        json={"approver_identity": "atharva"},
    )
    assert second.status_code == 409


def test_api_approve_missing_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/approvals/nope/approve", json={"approver_identity": "atharva"}
    )
    assert response.status_code == 404


def test_api_expired_approval_returns_409(
    client: TestClient, db: Session, sample_task: Task
) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)
    _force_expired(db, approval_id)
    response = client.post(
        f"/api/v1/approvals/{approval_id}/approve",
        json={"approver_identity": "atharva"},
    )
    assert response.status_code == 409
    assert client.get(f"/api/v1/approvals/{approval_id}").json()["status"] == "EXPIRED"


def test_api_reject_flow(client: TestClient, sample_task: Task) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)
    response = client.post(
        f"/api/v1/approvals/{approval_id}/reject",
        json={"approver_identity": "atharva", "reason": "Budget too high"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REJECTED"
    assert body["rejection_reason"] == "Budget too high"

    again = client.post(
        f"/api/v1/approvals/{approval_id}/reject",
        json={"approver_identity": "atharva", "reason": "again"},
    )
    assert again.status_code == 409


def test_api_reject_without_reason_400(client: TestClient, sample_task: Task) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)
    response = client.post(
        f"/api/v1/approvals/{approval_id}/reject",
        json={"approver_identity": "atharva"},
    )
    assert response.status_code == 400


def test_api_expire_stale_endpoint(client: TestClient, db: Session, sample_task: Task) -> None:
    approval_id = _api_raise_approval(client, sample_task.id)
    _force_expired(db, approval_id)

    response = client.post("/api/v1/approvals/expire-stale")
    assert response.status_code == 200
    assert response.json() == {"expired_count": 1}

    response = client.post("/api/v1/approvals/expire-stale")
    assert response.json() == {"expired_count": 0}


# ── HTTP API: audit logs ──────────────────────────────────────────────────


def test_api_audit_logs_and_filters(client: TestClient, sample_task: Task) -> None:
    client.post(
        "/api/v1/permissions/evaluate",
        json=_evaluate_payload(sample_task.id, "research_products"),
    )
    client.post(
        "/api/v1/permissions/evaluate",
        json=_evaluate_payload(sample_task.id, "allocate_ad_budget"),
    )

    response = client.get("/api/v1/audit-logs")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 2
    assert {entry["status"] for entry in logs} == {"ALLOWED_AUTO", "WAITING_HUMAN_APPROVAL"}

    response = client.get("/api/v1/audit-logs", params={"task_id": sample_task.id})
    assert len(response.json()) == 2
    response = client.get(
        "/api/v1/audit-logs", params={"agent_type": AgentType.DELIVERY_OPERATIONS.value}
    )
    assert response.json() == []
