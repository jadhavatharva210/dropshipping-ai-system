"""Milestone 1C suite: agents, tools, memory, orchestrator and ops API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agents import (
    AgentOutput,
    AgentRegistry,
    DeliveryOperationsAgent,
    ProductResearchAgent,
    create_default_agent_registry,
)
from app.core.ai_provider import MockAIProvider, get_ai_provider
from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus
from app.core.tools import ToolDefinition, ToolRegistry, create_default_registry
from app.models.core import ApprovalRequest, Task
from app.services.audit import AuditService
from app.services.memory import SharedMemoryService
from app.services.orchestrator import MainOrchestrator


def _orchestrator() -> MainOrchestrator:
    return MainOrchestrator()


# ── AI provider ───────────────────────────────────────────────────────────


def test_mock_provider_is_default_and_deterministic() -> None:
    provider = get_ai_provider()
    assert isinstance(provider, MockAIProvider)
    assert provider.generate_text("hello") == provider.generate_text("hello")
    plan = provider.generate_structured("plan", __import__("app.agents.specialists", fromlist=["AdPlan"]).AdPlan)
    assert plan.headline and plan.body and plan.cta


def test_unknown_ai_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import config

    monkeypatch.setattr(config.settings, "ai_provider", "openai")
    with pytest.raises(ValueError):
        get_ai_provider()


# ── Agent registry ────────────────────────────────────────────────────────


def test_agent_registry_register_get_list() -> None:
    registry = create_default_agent_registry()
    assert isinstance(registry.get_agent(AgentType.PRODUCT_RESEARCH), ProductResearchAgent)
    assert isinstance(registry.get_agent(AgentType.DELIVERY_OPERATIONS), DeliveryOperationsAgent)
    assert len(registry.list_agents()) == 3
    with pytest.raises(KeyError):
        registry.get_agent(AgentType.MAIN_ORCHESTRATOR)


def test_agent_registry_custom_register() -> None:
    registry = AgentRegistry()
    agent = ProductResearchAgent()
    registry.register(agent)
    assert registry.get_agent(AgentType.PRODUCT_RESEARCH) is agent
    assert registry.list_agents() == [agent]


# ── Tool registry ─────────────────────────────────────────────────────────


def test_tool_level1_executes_and_audits(db: Session, sample_task: Task) -> None:
    registry = create_default_registry()
    outcome = registry.execute_tool(
        db,
        tool_name="mock_product_discovery",
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"niche": "pets"},
    )
    assert outcome["executed"] is True
    assert outcome["result"]["candidates"]
    logs = AuditService.list_logs(db, task_id=sample_task.id)
    assert any(entry.status == "TOOL_EXECUTED" for entry in logs)


def test_tool_unknown_name_raises(db: Session, sample_task: Task) -> None:
    registry = create_default_registry()
    with pytest.raises((KeyError, ValueError)):
        registry.execute_tool(
            db,
            tool_name="does_not_exist",
            task_id=sample_task.id,
            agent_type=AgentType.PRODUCT_RESEARCH,
            parameters={},
        )


def test_tool_unauthorized_agent_raises_and_audits(db: Session, sample_task: Task) -> None:
    registry = create_default_registry()
    with pytest.raises(PermissionError):
        registry.execute_tool(
            db,
            tool_name="mock_ad_copy_generator",
            task_id=sample_task.id,
            agent_type=AgentType.PRODUCT_RESEARCH,  # not in allowed_agents
            parameters={},
        )
    logs = AuditService.list_logs(db, task_id=sample_task.id)
    assert any(entry.status == "TOOL_UNAUTHORIZED_AGENT" for entry in logs)


def test_tool_level2_pauses_without_executing(db: Session, sample_task: Task) -> None:
    called = {"count": 0}

    def spy_handler(parameters):
        called["count"] += 1
        return {}

    registry = create_default_registry()
    registry.register(ToolDefinition(
        name="spy_publish",
        action_name="publish_advertisement",
        description="spy",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=spy_handler,
    ))
    outcome = registry.execute_tool(
        db,
        tool_name="spy_publish",
        task_id=sample_task.id,
        agent_type=AgentType.ADVERTISING_ANALYTICS,
        parameters={"product": "LED collar"},
    )
    assert outcome["executed"] is False
    assert outcome["permission_level"] == "LEVEL_2_APPROVAL"
    assert outcome["task_status"] == "WAITING_APPROVAL"
    assert outcome["approval_request_id"]
    assert called["count"] == 0  # handler never ran before approval
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.WAITING_APPROVAL


def test_tool_level3_blocked(db: Session, sample_task: Task) -> None:
    registry = create_default_registry()
    registry.register(ToolDefinition(
        name="evil_shell",
        action_name="execute_arbitrary_shell",
        description="never runs",
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=lambda p: {"oops": True},
    ))
    outcome = registry.execute_tool(
        db,
        tool_name="evil_shell",
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={},
    )
    assert outcome["executed"] is False
    assert outcome["permission_level"] == "LEVEL_3_BLOCKED"
    db.refresh(sample_task)
    assert sample_task.status == TaskStatus.BLOCKED


# ── Shared memory ─────────────────────────────────────────────────────────


def test_memory_upsert_get_and_list(db: Session) -> None:
    memory = SharedMemoryService()
    first = memory.set_memory(
        db, namespace="ns", key="k", value={"v": 1}, agent_type=AgentType.PRODUCT_RESEARCH
    )
    second = memory.set_memory(
        db, namespace="ns", key="k", value={"v": 2}, agent_type=AgentType.ADVERTISING_ANALYTICS
    )
    assert first.id == second.id  # upsert, not duplicate
    fetched = memory.get_memory(db, namespace="ns", key="k")
    assert fetched.value == {"v": 2}
    assert fetched.updated_by_agent == AgentType.ADVERTISING_ANALYTICS
    assert len(memory.list_namespace(db, namespace="ns")) == 1
    assert memory.get_memory(db, namespace="ns", key="missing") is None


def test_cross_agent_memory_sharing(db: Session) -> None:
    memory = SharedMemoryService()
    memory.set_memory(
        db,
        namespace="product-research",
        key="latest_candidates",
        value={"candidates": [{"name": "LED collar"}]},
        agent_type=AgentType.PRODUCT_RESEARCH,
    )
    read_back = memory.get_memory(db, namespace="product-research", key="latest_candidates")
    assert read_back.value["candidates"][0]["name"] == "LED collar"


# ── Orchestrator: end-to-end workflows ────────────────────────────────────


def test_e2e_level1_workflow_completes(db: Session) -> None:
    report = _orchestrator().run_workflow(
        db, objective="Find promising general products and prepare a marketing plan"
    )
    assert report["status"] == "COMPLETED"
    assert report["workflow_type"] == "product_to_marketing"
    assert len(report["children"]) == 2
    assert all(child["status"] == "COMPLETED" for child in report["children"])
    assert report["approval_request_id"] is None

    parent = db.get(Task, report["parent_task_id"])
    assert parent.status == TaskStatus.COMPLETED
    assert all(child.output_payload for child in parent.children)

    research = SharedMemoryService.get_memory(
        db, namespace="product-research", key="latest_candidates"
    )
    assert research is not None and research.value["candidates"]


def test_level2_gated_workflow_pauses_then_resumes(db: Session) -> None:
    orchestrator = _orchestrator()
    report = orchestrator.run_workflow(
        db,
        objective="Research products and publish a live ad campaign",
        workflow_type="product_to_marketing",
        parameters={"publish": True},
    )
    assert report["status"] == "WAITING_APPROVAL"
    approval_id = report["approval_request_id"]
    assert approval_id is not None

    approval = db.get(ApprovalRequest, approval_id)
    assert approval.status == ApprovalStatus.PENDING
    gated_child = db.get(Task, approval.task_id)
    assert gated_child.status == TaskStatus.WAITING_APPROVAL

    resumed = orchestrator.resume_after_approval(
        db, approval_id=approval_id, approver_identity="atharva"
    )
    assert resumed["status"] == "COMPLETED"

    db.refresh(approval)
    db.refresh(gated_child)
    assert approval.status == ApprovalStatus.APPROVED
    assert approval.approver_identity == "atharva"
    assert approval.execution_result["status"] == "published"  # mock handler ran
    assert gated_child.status == TaskStatus.COMPLETED


def test_operations_check_workflow(db: Session) -> None:
    report = _orchestrator().run_workflow(
        db, objective="Check order delivery status", workflow_type="operations_check"
    )
    assert report["status"] == "COMPLETED"
    assert [child["agent_type"] for child in report["children"]] == ["DELIVERY_OPERATIONS"]


def test_retry_on_transient_agent_error(db: Session, sample_task: Task) -> None:
    calls = {"count": 0}

    class FlakyAgent(ProductResearchAgent):
        def execute(self, db, task, tool_registry, memory, ai_provider) -> AgentOutput:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient boom")
            return super().execute(db, task, tool_registry, memory, ai_provider)

    registry = AgentRegistry()
    registry.register(FlakyAgent())
    orchestrator = MainOrchestrator(agent_registry=registry)
    report = orchestrator.run_workflow(
        db, objective="research only", workflow_type="custom",
        parameters={"tasks": [{"agent_type": "PRODUCT_RESEARCH"}]},
    )
    assert report["status"] == "COMPLETED"
    assert calls["count"] == 2

    child = db.get(Task, report["children"][0]["task_id"])
    assert child.retry_count == 1
    logs = AuditService.list_logs(db, task_id=child.id)
    assert any(entry.status == "AGENT_RETRY" for entry in logs)


def test_permanent_failure_marks_tasks_failed(db: Session) -> None:
    class BrokenAgent(ProductResearchAgent):
        def execute(self, db, task, tool_registry, memory, ai_provider) -> AgentOutput:
            raise RuntimeError("always broken")

    registry = AgentRegistry()
    registry.register(BrokenAgent())
    orchestrator = MainOrchestrator(agent_registry=registry, max_retries=2)
    report = orchestrator.run_workflow(
        db, objective="doomed research", workflow_type="custom",
        parameters={"tasks": [{"agent_type": "PRODUCT_RESEARCH"}]},
    )
    assert report["status"] == "FAILED"
    child = db.get(Task, report["children"][0]["task_id"])
    assert child.status == TaskStatus.FAILED
    assert child.retry_count == 2
    assert "always broken" in child.error_message


def test_unknown_workflow_type_rejected(db: Session) -> None:
    with pytest.raises(ValueError):
        _orchestrator().run_workflow(db, objective="x", workflow_type="not_a_workflow")


# ── API: workflows / resume / tasks / memory / agents ─────────────────────


def test_api_run_workflow_and_inspect_tasks(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orchestrator/workflows",
        json={"objective": "Find products and prepare marketing"},
    )
    assert response.status_code == 200
    report = response.json()
    assert report["status"] == "COMPLETED"
    assert len(report["children"]) == 2

    tasks = client.get("/api/v1/tasks")
    assert tasks.status_code == 200
    assert len(tasks.json()) == 3  # parent + 2 children

    parent_id = report["parent_task_id"]
    single = client.get(f"/api/v1/tasks/{parent_id}")
    assert single.status_code == 200
    assert single.json()["assigned_agent"] == "MAIN_ORCHESTRATOR"
    assert client.get("/api/v1/tasks/nope").status_code == 404


def test_api_gated_workflow_resume_flow(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orchestrator/workflows",
        json={
            "objective": "Research and publish campaign",
            "parameters": {"publish": True},
        },
    )
    report = response.json()
    assert report["status"] == "WAITING_APPROVAL"
    approval_id = report["approval_request_id"]

    resumed = client.post(
        f"/api/v1/orchestrator/approvals/{approval_id}/resume",
        json={"approver_identity": "atharva"},
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "COMPLETED"

    again = client.post(
        f"/api/v1/orchestrator/approvals/{approval_id}/resume",
        json={"approver_identity": "atharva"},
    )
    assert again.status_code == 409

    missing = client.post(
        "/api/v1/orchestrator/approvals/nope/resume",
        json={"approver_identity": "atharva"},
    )
    assert missing.status_code == 404


def test_api_memory_namespace(client: TestClient) -> None:
    client.post(
        "/api/v1/orchestrator/workflows",
        json={"objective": "Find products and prepare marketing"},
    )
    response = client.get("/api/v1/memory/product-research")
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 1
    assert entries[0]["key"] == "latest_candidates"
    assert entries[0]["value"]["candidates"]

    assert client.get("/api/v1/memory/no-such-namespace").json() == []


def test_api_list_agents(client: TestClient) -> None:
    response = client.get("/api/v1/agents")
    assert response.status_code == 200
    agents = response.json()
    assert len(agents) == 3
    assert {a["agent_type"] for a in agents} == {
        "PRODUCT_RESEARCH",
        "ADVERTISING_ANALYTICS",
        "DELIVERY_OPERATIONS",
    }


def test_api_invalid_workflow_type_400(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orchestrator/workflows",
        json={"objective": "x", "workflow_type": "bogus"},
    )
    assert response.status_code == 400


# ── Resume gate regression: handlers never run without a valid approval ──


def _spy_orchestrator(calls: dict) -> MainOrchestrator:
    """Orchestrator whose publish handler counts invocations."""

    def spy_publish(parameters):
        calls["count"] += 1
        return {"campaign_id": "spy-campaign", "status": "published"}

    registry = create_default_registry()
    registry.register(ToolDefinition(
        name="mock_publish_ad_campaign",
        action_name="publish_advertisement",
        description="Publish a live ad campaign (mock, Level 2 approval)",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=spy_publish,
    ))
    return MainOrchestrator(tool_registry=registry)


def _raise_gated_approval(db: Session, orchestrator: MainOrchestrator) -> str:
    report = orchestrator.run_workflow(
        db,
        objective="Research and publish campaign",
        workflow_type="product_to_marketing",
        parameters={"publish": True},
    )
    assert report["status"] == "WAITING_APPROVAL"
    return report["approval_request_id"]


def test_resume_expired_approval_never_runs_handler(db: Session) -> None:
    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    approval = db.get(ApprovalRequest, approval_id)
    approval.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    from app.services.approval import ApprovalExpiredError

    with pytest.raises(ApprovalExpiredError):
        orchestrator.resume_after_approval(
            db, approval_id=approval_id, approver_identity="atharva"
        )
    assert calls["count"] == 0
    db.refresh(approval)
    assert approval.status == ApprovalStatus.EXPIRED


def test_resume_rejected_approval_never_runs_handler(db: Session) -> None:
    from app.services.approval import ApprovalAlreadyDecidedError, ApprovalService

    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    ApprovalService.reject_request(db, approval_id, "atharva", "not now")
    with pytest.raises(ApprovalAlreadyDecidedError):
        orchestrator.resume_after_approval(
            db, approval_id=approval_id, approver_identity="atharva"
        )
    assert calls["count"] == 0


def test_resume_twice_never_replays_handler(db: Session) -> None:
    from app.services.approval import ApprovalAlreadyDecidedError

    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    orchestrator.resume_after_approval(db, approval_id=approval_id, approver_identity="atharva")
    assert calls["count"] == 1

    with pytest.raises(ApprovalAlreadyDecidedError):
        orchestrator.resume_after_approval(
            db, approval_id=approval_id, approver_identity="atharva"
        )
    assert calls["count"] == 1  # no replay


def test_resume_tampered_pending_record_never_runs_handler(db: Session) -> None:
    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    # Tamper: pending record no longer matches the approved action.
    memory = SharedMemoryService()
    pending = memory.get_memory(db, namespace="orchestrator-pending", key=approval_id)
    memory.set_memory(
        db,
        namespace="orchestrator-pending",
        key=approval_id,
        value={**(pending.value or {}), "parameters": {"product": "something-else"}},
        agent_type=AgentType.MAIN_ORCHESTRATOR,
    )

    with pytest.raises(ValueError, match="does not match"):
        orchestrator.resume_after_approval(
            db, approval_id=approval_id, approver_identity="atharva"
        )
    assert calls["count"] == 0
    db.refresh(db.get(ApprovalRequest, approval_id))
    assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.PENDING


# ── Claim-before-execute: atomicity, binding, no-replay on failure ────────


def test_claim_approval_is_atomic_double_claim_fails(db: Session) -> None:
    from app.services.approval import ApprovalAlreadyDecidedError, ApprovalService

    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    ApprovalService.claim_approval(db, approval_id, "atharva")
    with pytest.raises(ApprovalAlreadyDecidedError):
        ApprovalService.claim_approval(db, approval_id, "atharva")
    assert calls["count"] == 0  # claim alone never executes


def test_claim_approval_rejects_empty_identity(db: Session) -> None:
    from app.services.approval import ApprovalService

    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    with pytest.raises(ValueError):
        ApprovalService.claim_approval(db, approval_id, "   ")
    assert calls["count"] == 0
    assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.PENDING


def test_resume_empty_approver_rejected_before_execution(db: Session) -> None:
    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    with pytest.raises(ValueError):
        orchestrator.resume_after_approval(db, approval_id=approval_id, approver_identity="")
    assert calls["count"] == 0
    assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.PENDING


def test_resume_tool_action_mismatch_never_runs_handler(db: Session) -> None:
    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    # Swap the pending record to a different tool of the same agent:
    # draft_ad_copy != the approved publish_advertisement action.
    memory = SharedMemoryService()
    pending = memory.get_memory(db, namespace="orchestrator-pending", key=approval_id)
    memory.set_memory(
        db,
        namespace="orchestrator-pending",
        key=approval_id,
        value={**(pending.value or {}), "tool_name": "mock_ad_copy_generator"},
        agent_type=AgentType.MAIN_ORCHESTRATOR,
    )

    with pytest.raises(ValueError, match="does not match"):
        orchestrator.resume_after_approval(db, approval_id=approval_id, approver_identity="atharva")
    assert calls["count"] == 0
    assert db.get(ApprovalRequest, approval_id).status == ApprovalStatus.PENDING


def test_handler_failure_after_claim_never_replays(db: Session) -> None:
    from app.services.approval import ApprovalAlreadyDecidedError

    def broken_publish(parameters):
        raise RuntimeError("publish backend down")

    registry = create_default_registry()
    registry.register(ToolDefinition(
        name="mock_publish_ad_campaign",
        action_name="publish_advertisement",
        description="Publish a live ad campaign (mock, Level 2 approval)",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=broken_publish,
    ))
    orchestrator = MainOrchestrator(tool_registry=registry)
    approval_id = _raise_gated_approval(db, orchestrator)

    report = orchestrator.resume_after_approval(db, approval_id=approval_id, approver_identity="atharva")
    assert report["status"] == "FAILED"

    approval = db.get(ApprovalRequest, approval_id)
    assert approval.status == ApprovalStatus.APPROVED  # decision recorded
    child = db.get(Task, approval.task_id)
    assert child.status == TaskStatus.FAILED
    assert "publish backend down" in child.error_message

    logs = AuditService.list_logs(db, task_id=child.id)
    assert any(entry.status == "TOOL_EXECUTION_FAILED" for entry in logs)

    # No replay: the approval is decided, the pending record is consumed.
    with pytest.raises(ApprovalAlreadyDecidedError):
        orchestrator.resume_after_approval(db, approval_id=approval_id, approver_identity="atharva")


def test_successful_resume_records_approval_before_execution(db: Session) -> None:
    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    report = orchestrator.resume_after_approval(db, approval_id=approval_id, approver_identity="atharva")
    assert report["status"] == "COMPLETED"
    assert calls["count"] == 1

    approval = db.get(ApprovalRequest, approval_id)
    assert approval.status == ApprovalStatus.APPROVED
    assert approval.decided_at is not None
    # Audit order: APPROVED_BY_HUMAN recorded before the publish TOOL_EXECUTED.
    logs = [
        entry.status
        for entry in AuditService.list_logs(db, task_id=approval.task_id)
        if entry.status in {"APPROVED_BY_HUMAN", "TOOL_EXECUTED"}
    ]
    assert logs[-2:] == ["APPROVED_BY_HUMAN", "TOOL_EXECUTED"]


def test_claim_checks_expiry_atomically_at_claim_time(db: Session) -> None:
    """A PENDING approval past expiry cannot be claimed; it flips to EXPIRED."""
    from app.services.approval import ApprovalExpiredError, ApprovalService

    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    approval = db.get(ApprovalRequest, approval_id)
    approval.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()

    with pytest.raises(ApprovalExpiredError):
        ApprovalService.claim_approval(db, approval_id, "atharva")
    db.refresh(approval)
    assert approval.status == ApprovalStatus.EXPIRED
    assert calls["count"] == 0


def test_claim_succeeds_within_expiry_window(db: Session) -> None:
    from app.services.approval import ApprovalService

    calls = {"count": 0}
    orchestrator = _spy_orchestrator(calls)
    approval_id = _raise_gated_approval(db, orchestrator)

    claimed = ApprovalService.claim_approval(db, approval_id, "atharva")
    assert claimed.status == ApprovalStatus.APPROVED
    assert claimed.decided_at is not None
