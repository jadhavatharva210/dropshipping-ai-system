"""Provider-independent AI interface.

The deterministic MockAIProvider stays the default so tests and local dev
need no network or API keys. A real backend plugs in behind the same
AIProvider interface via OpenAICompatibleProvider, which talks to any
OpenAI-compatible chat-completions API (Groq, Gemini's OpenAI endpoint,
OpenRouter, DeepSeek, ...) selected entirely through environment config.
"""

from __future__ import annotations

import json
import types
import typing
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings


class AIProvider(ABC):
    """Contract every AI backend must satisfy.

    A provider is also a context manager, so callers can guarantee that any
    underlying connection pool is released::

        with get_ai_provider() as ai:
            ai.generate_text("...")

    :meth:`close` is a no-op by default: stateless providers such as
    :class:`MockAIProvider` hold nothing to release. Providers that own an HTTP
    client must override it.
    """

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

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release underlying resources. Must be idempotent."""

    def __enter__(self) -> AIProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.close()


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


class AIProviderError(RuntimeError):
    """Raised when a real AI backend cannot produce a usable answer."""


class AIRateLimitError(AIProviderError):
    """The provider's rate limit was hit (HTTP 429).

    Fail fast, never auto-retry: free plans cap requests per day, so an
    immediate retry just burns quota. `retry_after` carries the provider's
    Retry-After hint (seconds) when it sent one.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OpenAICompatibleProvider(AIProvider):
    """Real provider over any OpenAI-compatible chat-completions endpoint.

    Config comes from the environment (see .env.example):
      AI_PROVIDER=openai_compatible
      AI_API_KEY=<key for the chosen service>   (never logged or committed)
      AI_BASE_URL=https://api.groq.com/openai/v1
      AI_MODEL=openai/gpt-oss-120b

    The default base URL/model target Groq's free plan, but any
    OpenAI-compatible service works by changing AI_BASE_URL and AI_MODEL.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(
                "AI_API_KEY is required when AI_PROVIDER=openai_compatible"
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )
        # The Client is always constructed here (only the *transport* is
        # injectable, for offline tests), so it is always ours to close. The
        # flag keeps that ownership explicit - and makes close() idempotent -
        # if a caller-provided client is ever supported.
        self._owns_client = True

    # ── lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the owned ``httpx.Client``, releasing its connection pool.

        Without this, every ``get_ai_provider()`` call leaks a pool: the
        orchestrator builds a provider per instance, so a long-running process
        would accumulate sockets. Idempotent - a second call is a no-op.
        """
        if self._owns_client:
            self._client.close()
            self._owns_client = False

    # ── public interface ─────────────────────────────────────────────────

    def generate_text(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        content = self._chat(
            [
                {
                    "role": "system",
                    "content": "You are a concise assistant. Answer in plain text.",
                },
                {"role": "user", "content": self._user_message(prompt, context)},
            ]
        )
        return content.strip()

    def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        context: dict[str, Any] | None = None,
    ) -> BaseModel:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You answer with a single JSON object only - no prose, no "
                    "markdown fences. The object must satisfy this JSON Schema:\n"
                    f"{schema_json}"
                ),
            },
            {"role": "user", "content": self._user_message(prompt, context)},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            content = self._chat(messages)
            try:
                return schema.model_validate_json(self._strip_fences(content))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 0:
                    # One retry, telling the model exactly what was wrong.
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "That reply was not valid for the schema "
                                f"({exc}). Reply again with only the JSON object."
                            ),
                        },
                    ]
        raise AIProviderError(
            f"AI provider did not return valid {schema.__name__} JSON: {last_error}"
        )

    # ── internals ────────────────────────────────────────────────────────

    def _user_message(self, prompt: str, context: dict[str, Any] | None) -> str:
        if not context:
            return prompt
        return f"{prompt}\n\nContext (JSON):\n{json.dumps(context, default=str)}"

    def _chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                json={"model": self._model, "messages": messages},
            )
        except httpx.HTTPError as exc:
            raise AIProviderError(f"AI provider request failed: {exc}") from exc
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            hint = f" Retry after ~{retry_after:g}s." if retry_after else ""
            raise AIRateLimitError(
                f"AI provider rate limit reached (HTTP 429).{hint} "
                "Stop here - the request was not retried.",
                retry_after=retry_after,
            )
        if response.status_code != 200:
            # Never echo the request or headers back: they carry the API key.
            raise AIProviderError(
                f"AI provider returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise AIProviderError(
                f"AI provider returned an unexpected response shape: {exc}"
            ) from exc

    @staticmethod
    def _strip_fences(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            # Tolerate models that wrap JSON in markdown fences anyway.
            lines = text.splitlines()
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return text


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def get_ai_provider() -> AIProvider:
    """Factory: resolve the configured provider.

    'mock' (default) needs no key and no network. 'openai_compatible' builds
    a real provider from AI_API_KEY / AI_BASE_URL / AI_MODEL.
    """
    name = settings.ai_provider.lower()
    if name == "mock":
        return MockAIProvider()
    if name == "openai_compatible":
        return OpenAICompatibleProvider(
            api_key=settings.ai_api_key,
            base_url=settings.ai_base_url,
            model=settings.ai_model,
            timeout_seconds=settings.ai_timeout_seconds,
        )
    raise ValueError(
        f"AI provider {settings.ai_provider!r} is not configured; "
        "available: 'mock', 'openai_compatible'"
    )
