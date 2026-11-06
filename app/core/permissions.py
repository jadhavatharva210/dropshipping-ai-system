"""3-tier permission engine: action registry + evaluation manager.

Tier semantics:
- LEVEL_1_AUTO     low-risk: execute immediately, task moves to IN_PROGRESS
- LEVEL_2_APPROVAL medium-risk: task pauses in WAITING_APPROVAL until a human decides
- LEVEL_3_BLOCKED  prohibited: task is BLOCKED, action never runs

Deny by default: any action name missing from the registry is LEVEL_3_BLOCKED.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.enums import PermissionLevel, TaskStatus
from app.models.core import ApprovalRequest, Task
from app.schemas.governance import ActionEvaluationRequest, ActionEvaluationResult
from app.services.audit import AuditService


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskNotFoundError(ValueError):
    """Raised when an action is evaluated against a task that does not exist."""


class PermissionPolicy:
    """Static, extensible registry mapping action names to permission tiers."""

    LEVEL_1_ACTIONS: frozenset[str] = frozenset({
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
    })

    LEVEL_2_ACTIONS: frozenset[str] = frozenset({
        "publish_advertisement",
        "allocate_ad_budget",
        "place_supplier_order",
        "commit_customer_order",
        "issue_refund",
        "cancel_order",
        "update_live_price",
        "send_customer_communication",
        "update_account_settings",
    })

    LEVEL_3_ACTIONS: frozenset[str] = frozenset({
        "delete_audit_logs",
        "bypass_permission_check",
        "export_raw_credentials",
        "execute_arbitrary_shell",
    })

    @classmethod
    def level_for(cls, action_name: str) -> PermissionLevel:
        """Resolve an action's tier; unknown actions are denied by default."""
        if action_name in cls.LEVEL_1_ACTIONS:
            return PermissionLevel.LEVEL_1_AUTO
        if action_name in cls.LEVEL_2_ACTIONS:
            return PermissionLevel.LEVEL_2_APPROVAL
        return PermissionLevel.LEVEL_3_BLOCKED


class PermissionManager:
    """Evaluates agent actions against the policy and drives task state."""

    @staticmethod
    def evaluate_action(
        db: Session, request: ActionEvaluationRequest
    ) -> ActionEvaluationResult:
        task = db.get(Task, request.task_id)
        if task is None:
            raise TaskNotFoundError(f"Task {request.task_id!r} not found")

        level = PermissionPolicy.level_for(request.action_name)

        if level == PermissionLevel.LEVEL_1_AUTO:
            task.permission_level = PermissionLevel.LEVEL_1_AUTO
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
            audit = AuditService.log_event(
                db,
                agent_type=request.agent_type,
                action_name=request.action_name,
                permission_level=level,
                status="ALLOWED_AUTO",
                task_id=task.id,
                details={"action_description": request.action_description},
            )
            db.commit()
            db.refresh(task)
            return ActionEvaluationResult(
                allowed_to_proceed=True,
                permission_level=level,
                task_status=task.status,
                approval_request_id=None,
                audit_log_id=audit.id,
                message="Action auto-approved (Level 1): proceeding immediately.",
            )

        if level == PermissionLevel.LEVEL_2_APPROVAL:
            task.permission_level = PermissionLevel.LEVEL_2_APPROVAL
            task.status = TaskStatus.WAITING_APPROVAL
            approval = ApprovalRequest(
                task_id=task.id,
                requesting_agent=request.agent_type,
                action_description=request.action_description,
                context_summary=request.context_summary,
                requested_parameters=request.parameters,
                expires_at=utc_now() + settings.approval_expiry_timedelta,
            )
            db.add(approval)
            db.flush()
            audit = AuditService.log_event(
                db,
                agent_type=request.agent_type,
                action_name=request.action_name,
                permission_level=level,
                status="WAITING_HUMAN_APPROVAL",
                task_id=task.id,
                details={
                    "action_description": request.action_description,
                    "approval_request_id": approval.id,
                },
            )
            db.commit()
            db.refresh(task)
            return ActionEvaluationResult(
                allowed_to_proceed=False,
                permission_level=level,
                task_status=task.status,
                approval_request_id=approval.id,
                audit_log_id=audit.id,
                message="Action requires human approval (Level 2): task paused.",
            )

        # LEVEL_3_BLOCKED (registered prohibited actions + unknown actions)
        task.permission_level = PermissionLevel.LEVEL_3_BLOCKED
        task.status = TaskStatus.BLOCKED
        audit = AuditService.log_event(
            db,
            agent_type=request.agent_type,
            action_name=request.action_name,
            permission_level=level,
            status="BLOCKED_PROHIBITED_ACTION",
            task_id=task.id,
            details={"action_description": request.action_description},
        )
        db.commit()
        db.refresh(task)
        return ActionEvaluationResult(
            allowed_to_proceed=False,
            permission_level=level,
            task_status=task.status,
            approval_request_id=None,
            audit_log_id=audit.id,
            message="Action prohibited (Level 3): blocked and will not execute.",
        )
