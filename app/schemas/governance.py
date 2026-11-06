"""Pydantic v2 schemas for the Milestone 1B governance layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus


class ActionEvaluationRequest(BaseModel):
    """Incoming request asking the permission engine to evaluate an action."""

    task_id: str
    agent_type: AgentType
    action_name: str
    action_description: str
    context_summary: str
    parameters: dict[str, Any]


class ActionEvaluationResult(BaseModel):
    """Outcome of a permission evaluation."""

    allowed_to_proceed: bool
    permission_level: PermissionLevel
    task_status: TaskStatus
    approval_request_id: str | None
    audit_log_id: str
    message: str


class ApprovalDecisionRequest(BaseModel):
    """Human decision on a pending approval (shared by approve / reject)."""

    approver_identity: str = Field(min_length=1)
    reason: str | None = None


class ApprovalExecutionReport(BaseModel):
    """Optional execution evidence attached to an approval."""

    execution_result: dict[str, Any]


class ApprovalApproveRequest(ApprovalDecisionRequest):
    """Approve a pending request, optionally with an execution result."""

    execution_result: dict[str, Any] | None = None


class ApprovalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_id: str
    requesting_agent: AgentType
    action_description: str
    context_summary: str | None
    requested_parameters: dict[str, Any] | None
    status: ApprovalStatus
    approver_identity: str | None
    rejection_reason: str | None
    created_at: datetime
    expires_at: datetime | None
    decided_at: datetime | None
    execution_result: dict[str, Any] | None


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_id: str | None
    agent_type: AgentType
    action_name: str
    permission_level: PermissionLevel
    status: str
    details: dict[str, Any] | None
    created_at: datetime
