"""Controlled tool registry.

Every tool maps to a PermissionPolicy action, so each execution is gated by
the 3-tier permission engine: Level 1 runs immediately, Level 2 pauses the
task for human approval without running the handler, Level 3 is blocked.
Phase 1 registers mock tools only - no real ads, orders, refunds or spend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import AgentType
from app.core.permissions import PermissionManager, PermissionPolicy
from app.schemas.governance import ActionEvaluationRequest
from app.services.audit import AuditService
from app.services.memory import SharedMemoryService

PENDING_NAMESPACE = "orchestrator-pending"


@dataclass
class ToolDefinition:
    name: str
    action_name: str  # mapped into PermissionPolicy by PermissionManager
    description: str
    allowed_agents: set[AgentType]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool {name!r} is not registered") from exc

    def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in sorted(self._tools)]

    def execute_tool(
        self,
        db: Session,
        *,
        tool_name: str,
        task_id: str,
        agent_type: AgentType,
        parameters: dict[str, Any],
        context_summary: str = "",
    ) -> dict[str, Any]:
        tool = self.get_tool(tool_name)

        if agent_type not in tool.allowed_agents:
            AuditService.log_event(
                db,
                agent_type=agent_type,
                action_name=tool.action_name,
                permission_level=PermissionPolicy.level_for(tool.action_name),
                status="TOOL_UNAUTHORIZED_AGENT",
                task_id=task_id,
                details={"tool_name": tool_name},
            )
            db.commit()
            raise PermissionError(
                f"Agent {agent_type.value} may not use tool {tool_name!r}"
            )

        evaluation = PermissionManager.evaluate_action(
            db,
            ActionEvaluationRequest(
                task_id=task_id,
                agent_type=agent_type,
                action_name=tool.action_name,
                action_description=tool.description,
                context_summary=context_summary or tool.description,
                parameters=parameters,
            ),
        )

        if not evaluation.allowed_to_proceed:
            if evaluation.approval_request_id is not None:
                # Stash the pending call so resume_after_approval can run it.
                SharedMemoryService.set_memory(
                    db,
                    namespace=PENDING_NAMESPACE,
                    key=evaluation.approval_request_id,
                    value={
                        "tool_name": tool_name,
                        "task_id": task_id,
                        "agent_type": agent_type.value,
                        "parameters": parameters,
                        "context_summary": context_summary,
                    },
                    agent_type=AgentType.MAIN_ORCHESTRATOR,
                )
            return {
                "executed": False,
                "permission_level": evaluation.permission_level.value,
                "task_status": evaluation.task_status.value,
                "approval_request_id": evaluation.approval_request_id,
                "message": evaluation.message,
            }

        result = tool.handler(parameters)
        AuditService.log_event(
            db,
            agent_type=agent_type,
            action_name=tool.action_name,
            permission_level=evaluation.permission_level,
            status="TOOL_EXECUTED",
            task_id=task_id,
            details={"tool_name": tool_name},
        )
        db.commit()
        return {
            "executed": True,
            "permission_level": evaluation.permission_level.value,
            "task_status": evaluation.task_status.value,
            "approval_request_id": None,
            "result": result,
        }


# ── Phase 1 mock tools (no real-world effects) ───────────────────────────


def _mock_product_discovery(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "niche": parameters.get("niche", "general"),
        "candidates": [
            {"name": "LED dog collar", "est_price_usd": 12.50, "trend_score": 0.81},
            {"name": "Pet water fountain", "est_price_usd": 18.00, "trend_score": 0.77},
            {"name": "Slow feeder bowl", "est_price_usd": 9.99, "trend_score": 0.72},
        ],
    }


def _mock_margin_calculator(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "currency": "USD",
        "avg_margin_pct": 42.0,
        "inputs": parameters,
    }


def _mock_ad_copy_generator(parameters: dict[str, Any]) -> dict[str, Any]:
    product = parameters.get("product", "the product")
    return {
        "headline": f"Meet your new favorite {product}",
        "body": f"Why thousands are switching to {product} this season.",
        "cta": "Shop now",
    }


def _mock_publish_ad_campaign(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": "mock-campaign-001",
        "status": "published",
        "daily_budget_usd": parameters.get("daily_budget_usd", 10.0),
    }


def _mock_order_tracker(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": parameters.get("order_id", "mock-order-1"),
        "status": "in_transit",
        "eta_days": 4,
    }


def _mock_issue_refund(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "refund_id": "mock-refund-001",
        "status": "issued",
        "amount": parameters.get("amount", 0.0),
    }


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="mock_product_discovery",
        action_name="research_products",
        description="Discover candidate products for a niche (mock)",
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_mock_product_discovery,
    ))
    registry.register(ToolDefinition(
        name="mock_margin_calculator",
        action_name="calculate_margins",
        description="Estimate margins for candidate products (mock)",
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_mock_margin_calculator,
    ))
    registry.register(ToolDefinition(
        name="mock_ad_copy_generator",
        action_name="draft_ad_copy",
        description="Draft ad copy for a product (mock)",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_mock_ad_copy_generator,
    ))
    registry.register(ToolDefinition(
        name="mock_publish_ad_campaign",
        action_name="publish_advertisement",
        description="Publish a live ad campaign (mock, Level 2 approval)",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_mock_publish_ad_campaign,
    ))
    registry.register(ToolDefinition(
        name="mock_order_tracker",
        action_name="check_order_status",
        description="Track status of customer orders (mock)",
        allowed_agents={AgentType.DELIVERY_OPERATIONS},
        handler=_mock_order_tracker,
    ))
    registry.register(ToolDefinition(
        name="mock_issue_refund",
        action_name="issue_refund",
        description="Issue a customer refund (mock, Level 2 approval)",
        allowed_agents={AgentType.DELIVERY_OPERATIONS},
        handler=_mock_issue_refund,
    ))
    return registry
