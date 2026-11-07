"""Core domain enumerations for the dropshipping AI system.

These four enums are the shared vocabulary of the whole platform: every agent,
every API payload and every database row speaks in terms of them, so they live
in ``app.core`` with zero dependencies on SQLAlchemy, Pydantic or FastAPI.

Two-way approval model (``PermissionLevel``)
--------------------------------------------
``LEVEL_1_AUTO``
    Low-risk, reversible actions. The agent executes immediately and only
    writes an audit entry.
``LEVEL_2_APPROVAL``
    Medium-risk actions (spending money, contacting suppliers, changing live
    ads). The agent must raise an ``ApprovalRequest`` and block until a human
    approves or rejects it.
``LEVEL_3_BLOCKED``
    High-risk actions (deleting stores, moving funds, bulk price overrides).
    Never executed by an agent under any circumstances.

All enums mix in ``str`` so they serialise cleanly to JSON, render readably in
logs and survive the round-trip through SQLite/PostgreSQL VARCHAR columns.
"""

from __future__ import annotations

import enum

__all__ = [
    "AgentType",
    "ApprovalStatus",
    "PermissionLevel",
    "TaskStatus",
]


class PermissionLevel(str, enum.Enum):
    """How much human supervision an action requires."""

    LEVEL_1_AUTO = "LEVEL_1_AUTO"
    LEVEL_2_APPROVAL = "LEVEL_2_APPROVAL"
    LEVEL_3_BLOCKED = "LEVEL_3_BLOCKED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_autonomous(self) -> bool:
        """True when the agent may act without asking anyone."""
        return self is PermissionLevel.LEVEL_1_AUTO

    @property
    def requires_approval(self) -> bool:
        """True when a human decision is mandatory before execution."""
        return self is PermissionLevel.LEVEL_2_APPROVAL

    @property
    def is_blocked(self) -> bool:
        """True when the action must never be executed by an agent."""
        return self is PermissionLevel.LEVEL_3_BLOCKED


class AgentType(str, enum.Enum):
    """The four agents of the Milestone-1 architecture."""

    MAIN_ORCHESTRATOR = "MAIN_ORCHESTRATOR"
    PRODUCT_RESEARCH = "PRODUCT_RESEARCH"
    ADVERTISING_ANALYTICS = "ADVERTISING_ANALYTICS"
    DELIVERY_OPERATIONS = "DELIVERY_OPERATIONS"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_orchestrator(self) -> bool:
        """True only for the agent allowed to create and delegate sub-tasks."""
        return self is AgentType.MAIN_ORCHESTRATOR


class TaskStatus(str, enum.Enum):
    """Lifecycle of a unit of work."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_terminal(self) -> bool:
        """True when no further work will happen on the task."""
        return self in _TERMINAL_TASK_STATUSES

    @property
    def is_active(self) -> bool:
        """True when the task is still moving through the pipeline."""
        return not self.is_terminal


#: Statuses a task can never leave without an explicit reset/retry.
_TERMINAL_TASK_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED}
)


class ApprovalStatus(str, enum.Enum):
    """Lifecycle of a human approval decision."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value

    @property
    def is_decided(self) -> bool:
        """True once a human (or the expiry sweeper) settled the request."""
        return self is not ApprovalStatus.PENDING

    @property
    def allows_execution(self) -> bool:
        """True only for an explicit human approval."""
        return self is ApprovalStatus.APPROVED
