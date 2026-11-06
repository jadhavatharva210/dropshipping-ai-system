"""Provider-independent AI interface.

Phase 1 ships only the deterministic MockAIProvider so tests and local dev
need no network or API keys. Real providers plug in behind the same
AIProvider interface and the get_ai_provider() factory.
"""

from __future__ import annotations

import types
import typing
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from app.core.config import settings


class AIProvider(ABC):
    """Contract every AI backend must satisfy."""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        context: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Return a validated instance of `schema`."""

    @abstractmethod
    def generate_text(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """Return plain text."""


class MockAIProvider(AIProvider):
    """Deterministic offline provider for tests and local development."""

    def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        context: dict[str, Any] | None = None,
    ) -> BaseModel:
        values = {
            name: self._placeholder(name, field.annotation)
            for name, field in schema.model_fields.items()
        }
        try:
            return schema(**values)
        except Exception:
            # Last resort: keep the mock total for unusual field shapes.
            return schema.model_construct(**values)

    def generate_text(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        context_keys = ",".join(sorted((context or {}).keys())) or "none"
        return f"[mock-ai] {prompt.strip()[:80]} (context keys: {context_keys})"

    def _placeholder(self, field_name: str, annotation: Any) -> Any:
        origin = typing.get_origin(annotation)
        if origin is typing.Union or origin is types.UnionType:
            return None
        if annotation is str:
            return f"mock-{field_name}"
        if annotation is int:
            return 1
        if annotation is float:
            return 1.0
        if annotation is bool:
            return True
        if origin is list:
            return []
        if origin is dict:
            return {}
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return self.generate_structured("nested", annotation)
        return f"mock-{field_name}"


def get_ai_provider() -> AIProvider:
    """Factory: resolve the configured provider. Only 'mock' ships in Phase 1."""
    name = settings.ai_provider.lower()
    if name == "mock":
        return MockAIProvider()
    raise ValueError(
        f"AI provider {settings.ai_provider!r} is not configured; only 'mock' is available"
    )
