"""Phase 1 specialist agents built on the shared contracts.

Each agent only calls tools through the ToolRegistry (so every action is
permission-gated), shares context via SharedMemoryService, and returns a
validated AgentOutput.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.base import AgentOutput, BaseAgent
from app.core.ai_provider import AIProvider
from app.core.enums import AgentType, TaskStatus
from app.core.tools import ToolRegistry
from app.models.core import Task
from app.services.memory import SharedMemoryService

RESEARCH_NAMESPACE = "product-research"
RESEARCH_KEY = "latest_candidates"


class AdPlan(BaseModel):
    headline: str
    body: str
    cta: str


def _gated_output(
    agent_type: AgentType, task: Task, tool_result: dict[str, Any], summary: str
) -> AgentOutput:
    return AgentOutput(
        status=TaskStatus(tool_result["task_status"]),
        agent_type=agent_type,
        summary=summary,
        data={"permission": tool_result},
        requires_approval=tool_result.get("approval_request_id") is not None,
        approval_request_id=tool_result.get("approval_request_id"),
    )


class ProductResearchAgent(BaseAgent):
    agent_type = AgentType.PRODUCT_RESEARCH
    description = "Discovers candidate products and validates margins."

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})
        discovery = tool_registry.execute_tool(
            db,
            tool_name="mock_product_discovery",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters=params,
            context_summary=task.objective,
        )
        if not discovery["executed"]:
            return _gated_output(self.agent_type, task, discovery, "Research gated")

        margins = tool_registry.execute_tool(
            db,
            tool_name="mock_margin_calculator",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"candidates": discovery["result"]["candidates"]},
            context_summary=task.objective,
        )
        if not margins["executed"]:
            return _gated_output(self.agent_type, task, margins, "Margin check gated")

        memory.set_memory(
            db,
            namespace=RESEARCH_NAMESPACE,
            key=RESEARCH_KEY,
            value={
                "candidates": discovery["result"]["candidates"],
                "margins": margins["result"],
                "source_task_id": task.id,
            },
            agent_type=self.agent_type,
        )
        summary = ai_provider.generate_text(
            f"Summarize product research for: {task.objective}",
            context={"candidates": discovery["result"]["candidates"]},
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data={"discovery": discovery["result"], "margins": margins["result"]},
        )


class AdvertisingAnalyticsAgent(BaseAgent):
    agent_type = AgentType.ADVERTISING_ANALYTICS
    description = "Drafts ad copy and (with approval) publishes campaigns."

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})
        research = memory.get_memory(db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY)
        candidates = (research.value or {}).get("candidates", []) if research else []
        top_product = candidates[0]["name"] if candidates else params.get("product", "the product")

        ad_copy = tool_registry.execute_tool(
            db,
            tool_name="mock_ad_copy_generator",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"product": top_product},
            context_summary=task.objective,
        )
        if not ad_copy["executed"]:
            return _gated_output(self.agent_type, task, ad_copy, "Ad copy gated")

        plan = ai_provider.generate_structured(
            f"Ad plan for {top_product}", AdPlan, context={"product": top_product}
        )

        data: dict[str, Any] = {
            "product": top_product,
            "ad_copy": ad_copy["result"],
            "ad_plan": plan.model_dump(),
            "research_source": RESEARCH_NAMESPACE,
        }

        if params.get("publish"):
            publish = tool_registry.execute_tool(
                db,
                tool_name="mock_publish_ad_campaign",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={"product": top_product, "daily_budget_usd": params.get("daily_budget_usd", 10.0)},
                context_summary=task.objective,
            )
            if not publish["executed"]:
                return AgentOutput(
                    status=TaskStatus(publish["task_status"]),
                    agent_type=self.agent_type,
                    summary="Campaign publish awaiting human approval",
                    data={**data, "permission": publish},
                    requires_approval=publish.get("approval_request_id") is not None,
                    approval_request_id=publish.get("approval_request_id"),
                )
            data["campaign"] = publish["result"]

        summary = ai_provider.generate_text(
            f"Summarize marketing plan for {top_product}", context=data
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data=data,
        )


class DeliveryOperationsAgent(BaseAgent):
    agent_type = AgentType.DELIVERY_OPERATIONS
    description = "Tracks orders and (with approval) issues refunds."

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})
        tracking = tool_registry.execute_tool(
            db,
            tool_name="mock_order_tracker",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"order_id": params.get("order_id", "mock-order-1")},
            context_summary=task.objective,
        )
        if not tracking["executed"]:
            return _gated_output(self.agent_type, task, tracking, "Tracking gated")

        data: dict[str, Any] = {"tracking": tracking["result"]}

        if params.get("issue_refund"):
            refund = tool_registry.execute_tool(
                db,
                tool_name="mock_issue_refund",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={"order_id": params.get("order_id", "mock-order-1"), "amount": params.get("amount", 0.0)},
                context_summary=task.objective,
            )
            if not refund["executed"]:
                return AgentOutput(
                    status=TaskStatus(refund["task_status"]),
                    agent_type=self.agent_type,
                    summary="Refund awaiting human approval",
                    data={**data, "permission": refund},
                    requires_approval=refund.get("approval_request_id") is not None,
                    approval_request_id=refund.get("approval_request_id"),
                )
            data["refund"] = refund["result"]

        memory.set_memory(
            db,
            namespace="delivery-operations",
            key="latest_status",
            value=data,
            agent_type=self.agent_type,
        )
        summary = ai_provider.generate_text(
            "Summarize delivery operations status", context=data
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data=data,
        )
