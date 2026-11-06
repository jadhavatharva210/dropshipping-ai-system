"""Main orchestrator: decomposes objectives into specialist child tasks.

Runs child tasks in order through the AgentRegistry, persists outputs,
honours Level 2 / Level 3 gates by pausing or blocking cleanly, retries
transient agent failures, and resumes gated workflows after human approval.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents import AgentRegistry, create_default_agent_registry
from app.core.ai_provider import AIProvider, get_ai_provider
from app.core.enums import AgentType, PermissionLevel, TaskStatus
from app.core.tools import PENDING_NAMESPACE, ToolRegistry, create_default_registry
from app.models.core import ApprovalRequest, Task
from app.services.approval import ApprovalService
from app.services.audit import AuditService
from app.services.memory import SharedMemoryService

WORKFLOW_DECOMPOSITIONS: dict[str, list[AgentType]] = {
    "product_to_marketing": [AgentType.PRODUCT_RESEARCH, AgentType.ADVERTISING_ANALYTICS],
    "operations_check": [AgentType.DELIVERY_OPERATIONS],
}


class MainOrchestrator:
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
        memory: SharedMemoryService | None = None,
        ai_provider: AIProvider | None = None,
        max_retries: int = 2,
    ) -> None:
        self.tool_registry = tool_registry or create_default_registry()
        self.agent_registry = agent_registry or create_default_agent_registry()
        self.memory = memory or SharedMemoryService()
        self.ai_provider = ai_provider or get_ai_provider()
        self.max_retries = max_retries

    # ── workflow execution ────────────────────────────────────────────────

    def run_workflow(
        self,
        db: Session,
        *,
        objective: str,
        workflow_type: str = "product_to_marketing",
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parameters = parameters or {}
        agent_sequence = self._decompose(workflow_type, parameters)

        parent = Task(
            assigned_agent=AgentType.MAIN_ORCHESTRATOR,
            title=f"Workflow: {workflow_type}",
            objective=objective,
            status=TaskStatus.IN_PROGRESS,
            permission_level=PermissionLevel.LEVEL_1_AUTO,
            input_payload={"workflow_type": workflow_type, "parameters": parameters},
        )
        db.add(parent)
        db.commit()
        db.refresh(parent)

        children: list[Task] = []
        for index, agent_type in enumerate(agent_sequence, start=1):
            child = Task(
                parent_task_id=parent.id,
                assigned_agent=agent_type,
                title=f"Task {index}: {agent_type.value}",
                objective=objective,
                status=TaskStatus.PENDING,
                permission_level=PermissionLevel.LEVEL_1_AUTO,
                input_payload=parameters,
            )
            db.add(child)
            children.append(child)
        db.commit()
        for child in children:
            db.refresh(child)

        return self._execute_children(db, parent, children)

    def resume_after_approval(
        self,
        db: Session,
        *,
        approval_id: str,
        approver_identity: str,
    ) -> dict[str, Any]:
        """Approve (atomically claim) first, then execute the approved action.

        Order: validate identity and the full binding -> claim PENDING->APPROVED
        atomically -> run the handler -> record result and completion. After the
        claim the approval is never PENDING again, so no resume can replay the
        action, even if the handler fails.
        """
        if not approver_identity or not approver_identity.strip():
            raise ValueError("A non-empty approver_identity is required")

        # Pre-validation for clear errors; claim_approval re-enforces atomically.
        approval = ApprovalService.ensure_decidable(db, approval_id)

        # Resolve and fully bind the pending tool call BEFORE claiming.
        pending = self.memory.get_memory(db, namespace=PENDING_NAMESPACE, key=approval_id)
        call: dict[str, Any] | None = None
        tool = None
        if pending is not None:
            call = dict(pending.value or {})
            if call.get("executed"):
                raise ValueError("Approved action was already executed")
            tool = self.tool_registry.get_tool(call["tool_name"])
            if (
                call.get("task_id") != approval.task_id
                or call.get("agent_type") != approval.requesting_agent.value
                or call.get("parameters") != (approval.requested_parameters or {})
            ):
                raise ValueError(
                    "Pending tool call does not match the approved action"
            )
            gated = [
                entry
                for entry in AuditService.list_logs(db, task_id=approval.task_id)
                if entry.status == "WAITING_HUMAN_APPROVAL"
                and (entry.details or {}).get("approval_request_id") == approval_id
            ]
            if gated and gated[0].action_name != tool.action_name:
                raise ValueError(
                    "Tool action does not match the action that was approved"
                )
            if AgentType(call["agent_type"]) not in tool.allowed_agents:
                raise PermissionError(
                    f"Agent {call['agent_type']} may not use tool {call['tool_name']!r}"
                )

        # Atomic claim: human approval is recorded before anything executes.
        approval = ApprovalService.claim_approval(db, approval_id, approver_identity)
        child = db.get(Task, approval.task_id)

        def _flag_executed() -> None:
            if call is not None:
                self.memory.set_memory(
                    db,
                    namespace=PENDING_NAMESPACE,
                    key=approval_id,
                    value={**call, "executed": True},
                    agent_type=AgentType.MAIN_ORCHESTRATOR,
                )

        try:
            execution_result = tool.handler(call.get("parameters") or {}) if tool else {}
        except Exception as exc:
            # Decision stands; the action is never eligible for replay.
            if child is not None:
                child.status = TaskStatus.FAILED
                child.error_message = str(exc)
            AuditService.log_event(
                db,
                agent_type=approval.requesting_agent,
                action_name=approval.action_description,
                permission_level=approval.task.permission_level,
                status="TOOL_EXECUTION_FAILED",
                task_id=approval.task_id,
                details={"tool_name": call.get("tool_name") if call else None, "error": str(exc)},
            )
            _flag_executed()
            db.commit()
            parent = db.get(Task, child.parent_task_id) if child and child.parent_task_id else None
            if parent is None:
                return {"status": "failed", "task_id": approval.task_id, "error": str(exc)}
            parent.status = TaskStatus.FAILED
            db.commit()
            return self._report(db, parent)

        approval.execution_result = execution_result
        if child is not None and child.status == TaskStatus.IN_PROGRESS:
            child.status = TaskStatus.COMPLETED
        AuditService.log_event(
            db,
            agent_type=approval.requesting_agent,
            action_name=approval.action_description,
            permission_level=approval.task.permission_level,
            status="TOOL_EXECUTED",
            task_id=approval.task_id,
            details={"tool_name": call.get("tool_name") if call else None},
        )
        _flag_executed()
        db.commit()

        parent = db.get(Task, child.parent_task_id) if child and child.parent_task_id else None
        if parent is None:
            return {"status": "completed", "task_id": approval.task_id}

        remaining = [
            sibling
            for sibling in parent.children
            if sibling.status == TaskStatus.PENDING
        ]
        # Children execute in creation order.
        remaining.sort(key=lambda task: task.created_at)
        return self._execute_children(db, parent, remaining, include_completed=True)

    # ── internals ─────────────────────────────────────────────────────────

    def _decompose(
        self, workflow_type: str, parameters: dict[str, Any]
    ) -> list[AgentType]:
        if workflow_type == "custom":
            custom = parameters.get("tasks") or []
            return [AgentType(entry["agent_type"]) for entry in custom]
        if workflow_type not in WORKFLOW_DECOMPOSITIONS:
            raise ValueError(f"Unknown workflow_type {workflow_type!r}")
        return WORKFLOW_DECOMPOSITIONS[workflow_type]

    def _execute_children(
        self,
        db: Session,
        parent: Task,
        children: list[Task],
        *,
        include_completed: bool = False,
    ) -> dict[str, Any]:
        for child in children:
            output = self._run_child_with_retries(db, child)
            if output is None:
                # Exhausted retries: child already FAILED, halt everything.
                parent.status = TaskStatus.FAILED
                db.commit()
                return self._report(db, parent)

            child.status = output.status
            child.output_payload = output.model_dump(mode="json")
            db.commit()
            db.refresh(child)

            if child.status == TaskStatus.WAITING_APPROVAL:
                parent.status = TaskStatus.WAITING_APPROVAL
                db.commit()
                return self._report(db, parent, approval_request_id=output.approval_request_id)
            if child.status == TaskStatus.BLOCKED:
                parent.status = TaskStatus.BLOCKED
                db.commit()
                return self._report(db, parent)

        parent.status = TaskStatus.COMPLETED
        db.commit()
        return self._report(db, parent)

    def _run_child_with_retries(self, db: Session, child: Task):
        agent = self.agent_registry.get_agent(child.assigned_agent)
        attempt = 0
        while True:
            try:
                child.status = TaskStatus.IN_PROGRESS
                db.commit()
                return agent.execute(
                    db, child, self.tool_registry, self.memory, self.ai_provider
                )
            except Exception as exc:  # transient runtime failure -> retry
                AuditService.log_event(
                    db,
                    agent_type=child.assigned_agent,
                    action_name="agent.execute",
                    permission_level=child.permission_level,
                    status="AGENT_RETRY",
                    task_id=child.id,
                    details={"attempt": attempt, "error": str(exc)},
                )
                if attempt >= self.max_retries:
                    child.status = TaskStatus.FAILED
                    child.error_message = str(exc)
                    db.commit()
                    return None
                attempt += 1
                child.retry_count += 1
                db.commit()

    def _report(
        self,
        db: Session,
        parent: Task,
        *,
        approval_request_id: str | None = None,
    ) -> dict[str, Any]:
        db.refresh(parent)
        children = sorted(parent.children, key=lambda task: task.created_at)
        return {
            "parent_task_id": parent.id,
            "workflow_type": (parent.input_payload or {}).get("workflow_type"),
            "status": parent.status.value,
            "approval_request_id": approval_request_id,
            "children": [
                {
                    "task_id": child.id,
                    "agent_type": child.assigned_agent.value,
                    "status": child.status.value,
                    "retry_count": child.retry_count,
                    "summary": (child.output_payload or {}).get("summary"),
                }
                for child in children
            ],
            "summary": f"Workflow {parent.title} finished with status {parent.status.value}",
        }
