"""Core domain enumerations for the dropshipping AI system.

Two-way approval model:
- LEVEL_1_AUTO     low-risk actions, execute immediately
- LEVEL_2_APPROVAL medium-risk actions, require human approval first
- LEVEL_3_BLOCKED  high-risk actions, never executed
"""

from __future__ import annotations

import enum


class PermissionLevel(str, enum.Enum):
    LEVEL_1_AUTO = "LEVEL_1_AUTO"
    LEVEL_2_APPROVAL = "LEVEL_2_APPROVAL"
    LEVEL_3_BLOCKED = "LEVEL_3_BLOCKED"


class AgentType(str, enum.Enum):
    MAIN_ORCHESTRATOR = "MAIN_ORCHESTRATOR"
    PRODUCT_RESEARCH = "PRODUCT_RESEARCH"
    ADVERTISING_ANALYTICS = "ADVERTISING_ANALYTICS"
    DELIVERY_OPERATIONS = "DELIVERY_OPERATIONS"


class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ApprovalStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
