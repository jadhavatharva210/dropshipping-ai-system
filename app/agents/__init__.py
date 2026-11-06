"""Specialist agents and their registry."""

from app.agents.base import AgentInput, AgentOutput, BaseAgent
from app.agents.registry import AgentRegistry
from app.agents.specialists import (
    AdvertisingAnalyticsAgent,
    DeliveryOperationsAgent,
    ProductResearchAgent,
)

__all__ = [
    "AgentInput",
    "AgentOutput",
    "BaseAgent",
    "AgentRegistry",
    "ProductResearchAgent",
    "AdvertisingAnalyticsAgent",
    "DeliveryOperationsAgent",
    "create_default_agent_registry",
]


def create_default_agent_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(ProductResearchAgent())
    registry.register(AdvertisingAnalyticsAgent())
    registry.register(DeliveryOperationsAgent())
    return registry
