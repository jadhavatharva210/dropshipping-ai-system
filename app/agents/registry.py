"""Agent registry: maps AgentType to specialist implementations."""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.enums import AgentType


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[AgentType, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_type] = agent

    def get_agent(self, agent_type: AgentType) -> BaseAgent:
        try:
            return self._agents[agent_type]
        except KeyError as exc:
            raise KeyError(f"No agent registered for {agent_type.value}") from exc

    def list_agents(self) -> list[BaseAgent]:
        return [self._agents[key] for key in sorted(self._agents, key=lambda a: a.value)]
