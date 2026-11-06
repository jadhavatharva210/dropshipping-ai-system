"""Agent contracts: typed input/output and the BaseAgent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.ai_provider import AIProvider
from app.core.enums import AgentType, TaskStatus
from app.core.tools import ToolRegistry
from app.models.core import Task
from app.services.memory import SharedMemoryService


class AgentInput(BaseModel):
    objective: str
    parameters: dict[str, Any] = {}
    context: dict[str, Any] | None = None


class AgentOutput(BaseModel):
    status: TaskStatus
    agent_type: AgentType
    summary: str
    data: dict[str, Any] = {}
    requires_approval: bool = False
    approval_request_id: str | None = None


class BaseAgent(ABC):
    """All specialist agents implement this interface."""

    agent_type: AgentType
    description: str = ""

    @abstractmethod
    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        """Run the agent against its task and return a validated output."""
