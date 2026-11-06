"""Real-provider milestone: OpenAI-compatible backend + facts-driven ad copy.

These tests never touch the network: the provider runs against an
httpx.MockTransport, and agent tests use a recording in-process stub. The
MockAIProvider stays the default everywhere else, so the existing 110-test
suite is untouched.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.specialists import AdPlan
from app.core import config
from app.core.ai_provider import (
    AIProviderError,
    AIRateLimitError,
    MockAIProvider,
    OpenAICompatibleProvider,
    get_ai_provider,
)
from app.core.enums import TaskStatus
from app.services.orchestrator import MainOrchestrator

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"


def _provider(handler) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-key",
        base_url=BASE_URL,
        model=MODEL,
        transport=httpx.MockTransport(handler),
    )


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


# ── provider construction ─────────────────────────────────────────────────


def test_openai_compatible_requires_api_key() -> None:
    with pytest.raises(ValueError, match="AI_API_KEY"):
        OpenAICompatibleProvider(api_key="", base_url=BASE_URL, model=MODEL)
    with pytest.raises(ValueError, match="AI_API_KEY"):
        OpenAICompatibleProvider(api_key="   ", base_url=BASE_URL, model=MODEL)


def test_factory_builds_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config.settings, "ai_provider", "openai_compatible")
    monkeypatch.setattr(config.settings, "ai_api_key", "test-key")
    provider = get_ai_provider()
    assert isinstance(provider, OpenAICompatibleProvider)


def test_factory_still_defaults_to_mock() -> None:
    assert isinstance(get_ai_provider(), MockAIProvider)


# ── provider behaviour ────────────────────────────────────────────────────


def test_generate_text_hits_chat_completions() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        body = json.loads(request.content)
        seen["model"] = body["model"]
        seen["user"] = body["messages"][-1]["content"]
        return _chat_response("Real AI summary")

    provider = _provider(handler)
    result = provider.generate_text("Summarize this", context={"a": 1})
    assert result == "Real AI summary"
    assert seen["url"] == f"{BASE_URL}/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["model"] == MODEL
    assert "Summarize this" in seen["user"] and '"a": 1' in seen["user"]


def test_generate_structured_validates_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response('{"headline": "Hi", "body": "Some copy", "cta": "Buy"}')

    provider = _provider(handler)
    plan = provider.generate_structured("Write ads", AdPlan)
    assert isinstance(plan, AdPlan)
    assert plan.headline == "Hi" and plan.cta == "Buy"


def test_generate_structured_tolerates_markdown_fences() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(
            '```json\n{"headline": "Hi", "body": "Copy", "cta": "Buy"}\n```'
        )

    provider = _provider(handler)
    plan = provider.generate_structured("Write ads", AdPlan)
    assert plan.headline == "Hi"


def test_generate_structured_retries_once_then_raises() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _chat_response("not json at all")

    provider = _provider(handler)
    with pytest.raises(AIProviderError):
        provider.generate_structured("Write ads", AdPlan)
    assert calls["count"] == 2  # one retry with feedback, then a clear error


def test_http_error_becomes_provider_error_without_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    provider = _provider(handler)
    with pytest.raises(AIProviderError, match="HTTP 401"):
        provider.generate_text("hello")


# ── facts-driven ad copy ──────────────────────────────────────────────────


class RecordingProvider:
    """In-process AIProvider stub that captures prompts it receives."""

    def __init__(self) -> None:
        self.structured_prompts: list[str] = []

    def generate_structured(self, prompt, schema, context=None):
        self.structured_prompts.append(prompt)
        return schema(
            headline="AI headline", body="AI body from facts", cta="AI CTA"
        )

    def generate_text(self, prompt, context=None):
        return "recorded summary"


def test_ad_copy_uses_supplied_product_facts(db: Session) -> None:
    recorder = RecordingProvider()
    orchestrator = MainOrchestrator(ai_provider=recorder)
    report = orchestrator.run_workflow(
        db,
        objective="Launch the collar campaign",
        workflow_type="product_to_marketing",
        parameters={
            "product": "LED dog collar",
            "price_usd": 12.5,
            "benefits": ["USB rechargeable", "visible at night"],
            "target_audience": "dog owners who walk after dark",
            "tone": "playful",
        },
    )
    assert report["status"] == TaskStatus.COMPLETED.value

    ad_prompt = next(p for p in recorder.structured_prompts if "Supplied facts" in p)
    assert "LED dog collar" in ad_prompt
    assert "12.5" in ad_prompt
    assert "USB rechargeable" in ad_prompt
    assert "dog owners who walk after dark" in ad_prompt
    assert "Do not invent claims" in ad_prompt

    marketing = next(
        child
        for child in report["children"]
        if child["agent_type"] == "ADVERTISING_ANALYTICS"
    )
    assert marketing["status"] == TaskStatus.COMPLETED.value


def test_ad_copy_output_carries_ai_plan_and_facts(db: Session) -> None:
    recorder = RecordingProvider()
    orchestrator = MainOrchestrator(ai_provider=recorder)
    orchestrator.run_workflow(
        db,
        objective="Launch the collar campaign",
        workflow_type="product_to_marketing",
        parameters={"product": "LED dog collar", "price_usd": 12.5},
    )
    from app.models.core import Task
    from app.core.enums import AgentType
    from sqlalchemy import select

    child = db.scalars(
        select(Task).where(Task.assigned_agent == AgentType.ADVERTISING_ANALYTICS)
    ).one()
    payload = child.output_payload
    assert payload["data"]["ad_plan"] == {
        "headline": "AI headline",
        "body": "AI body from facts",
        "cta": "AI CTA",
    }
    assert payload["data"]["product_facts"]["price_usd"] == 12.5
    assert payload["data"]["ad_copy"]["headline"]  # mock tool trail preserved


# ── rate-limit safety (free-only constraint) ─────────────────────────────


def test_429_raises_rate_limit_error_with_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "20"},
            json={"error": {"message": "rate limited"}},
        )

    provider = _provider(handler)
    with pytest.raises(AIRateLimitError) as exc_info:
        provider.generate_text("hello")
    assert exc_info.value.retry_after == 20.0
    assert "not retried" in str(exc_info.value)


def test_orchestrator_does_not_retry_rate_limited_agent(db: Session) -> None:
    from app.agents import AgentRegistry
    from app.agents.specialists import AdvertisingAnalyticsAgent
    from app.core.enums import AgentType

    calls = {"count": 0}

    class RateLimitedAdsAgent(AdvertisingAnalyticsAgent):
        def execute(self, db, task, tool_registry, memory, ai_provider):
            calls["count"] += 1
            raise AIRateLimitError(
                "AI provider rate limit reached (HTTP 429).", retry_after=30.0
            )

    registry = AgentRegistry()
    registry.register(RateLimitedAdsAgent())
    orchestrator = MainOrchestrator(agent_registry=registry, max_retries=2)
    report = orchestrator.run_workflow(
        db,
        objective="Rate limit check",
        workflow_type="custom",
        parameters={"tasks": [{"agent_type": "ADVERTISING_ANALYTICS"}]},
    )
    assert calls["count"] == 1  # failed fast: no retry despite max_retries=2
    assert report["status"] == TaskStatus.FAILED.value

    from sqlalchemy import select
    from app.models.core import Task

    child = db.scalars(
        select(Task).where(Task.assigned_agent == AgentType.ADVERTISING_ANALYTICS)
    ).one()
    assert child.status == TaskStatus.FAILED
    assert "rate limit" in child.error_message.lower()
    assert child.retry_count == 0
