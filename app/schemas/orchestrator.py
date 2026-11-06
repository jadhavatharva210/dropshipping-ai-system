"""Schemas for the orchestrator / operations API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AgentType, PermissionLevel, TaskStatus


class WorkflowRunRequest(BaseModel):
    objective: str = Field(min_length=1)
    workflow_type: str = "product_to_marketing"
    parameters: dict[str, Any] | None = None


class WorkflowChildReport(BaseModel):
    task_id: str
    agent_type: str
    status: str
    retry_count: int
    summary: str | None


class WorkflowReport(BaseModel):
    parent_task_id: str
    workflow_type: str | None
    status: str
    approval_request_id: str | None
    children: list[WorkflowChildReport]
    summary: str


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    parent_task_id: str | None
    assigned_agent: AgentType
    title: str
    objective: str
    status: TaskStatus
    permission_level: PermissionLevel
    input_payload: dict[str, Any] | None
    output_payload: dict[str, Any] | None
    error_message: str | None
    retry_count: int
    created_at: datetime
    updated_at: datetime


class MemoryEntryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    namespace: str
    key: str
    value: dict[str, Any] | None
    updated_by_agent: AgentType
    created_at: datetime
    updated_at: datetime


class AgentInfoResponse(BaseModel):
    agent_type: AgentType
    name: str
    description: str
