"""Routing used by the local demo console.

The console exists to let a human watch the real agent work. Two boundaries are
kept separate from the production code:

**Outbound HTTP is intercepted, never allowed.** ``demo_transport`` answers any
request for a URL under ``demo-shop.example`` with the bundled page from
:mod:`demo.sample_pages`, and raises for every other host. So a click can never
reach the open internet - whether this machine has network access or not - and
the outcome is the same on every run. The pages go through the *real*
:func:`app.services.web_research.extract_product_from_html`, so what the console
shows is what the extractor actually does with that markup.

**Which fixtures are real is decided per request, not per process.** The console
asks for a scenario first; the *host* app calls :func:`install_demo_service`
before dispatching, so a run's parameters say what was asked for, the router
pins the service for that request, and the parameters are restored afterwards.
Nothing here is injected through ``Task.input_payload`` (an ``httpx.Client``
cannot be serialised into a JSON column) - the router injects the service into
the tool registry, which is where the agent reads it from.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.core import tools as tools_module
from app.services.web_research import WebResearchService
from demo.sample_pages import DEMO_CATALOGUE, DEMO_HOST, page_html

__all__ = [
    "DEMO_SCENARIOS",
    "ScenarioSeeder",
    "demo_transport",
    "install_demo_service",
    "seed_request",
    "SCENARIO_PARAM",
]

#: Query parameter that names the scenario, e.g. ``?scenario=happy_path``.
SCENARIO_PARAM = "scenario"

#: The SSRF guard resolves hostnames before fetching. Demo pages are served by
#: an in-process transport, so the resolver reports a public address instead of
#: asking DNS - which also keeps the console hermetic on a machine with no
#: network at all.
_DEMO_RESOLVER = lambda host: ["93.184.216.34"]  # noqa: E731


def demo_transport() -> httpx.MockTransport:
    """A transport that can only ever serve the bundled sample pages."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = urlsplit(str(request.url)).hostname or ""
        if host != DEMO_HOST:
            raise httpx.ConnectError(
                "The demo console is offline by design: only the bundled "
                f"sample pages under {DEMO_HOST} can be fetched.",
                request=request,
            )
        try:
            body = page_html(str(request.url))
        except KeyError:
            return httpx.Response(404, text="<html><body>Not found</body></html>")
        return httpx.Response(
            200, text=body, headers={"content-type": "text/html; charset=utf-8"}
        )

    return httpx.MockTransport(handler)


_EMPTY = (None, "", [], {})


def seed_request(body: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Fill in the fields a scenario owns that ``body`` left empty.

    *Seeding*, not overriding: anything the caller actually sent wins, so a form
    the human edited is posted verbatim, and nothing they typed is silently
    replaced. The scenario only supplies what was left blank - which makes both
    "click a scenario" and "curl with just a product name" behave sensibly.
    """
    spec = DEMO_SCENARIOS.get((scenario or "").strip())
    if spec is None:
        raise KeyError(scenario)
    seeded = dict(body)
    for key, value in spec["request"].items():
        if key not in seeded or seeded[key] in _EMPTY:
            seeded[key] = value
    return seeded


class ScenarioSeeder:
    """ASGI middleware: seed a request body, and scope outbound HTTP to it.

    Written at the raw ASGI layer rather than with ``call_next`` because the
    body has to be *replaced*, and Starlette's ``BaseHTTPMiddleware`` hands the
    downstream app a cached body that a rewrite does not reach.
    """

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        query = scope.get("query_string", b"")
        query_text = query.decode() if isinstance(query, bytes) else str(query or "")
        scenario = scenario_of(str(scope.get("path", "")) + "?" + query_text)
        if scope.get("type") != "http" or scenario is None:
            await self.app(scope, receive, send)
            return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        payload: dict[str, Any] = {}
        if body.strip():
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}

        seeded = json.dumps(seed_request(payload, scenario)).encode()
        headers = [
            (key, value)
            for key, value in scope.get("headers", [])
            if key.lower() not in {b"content-length", b"content-type"}
        ]
        headers += [
            (b"content-length", str(len(seeded)).encode()),
            (b"content-type", b"application/json"),
        ]
        new_scope = dict(scope)
        new_scope["headers"] = headers

        replayed = False

        async def replay() -> dict[str, Any]:
            nonlocal replayed
            if replayed:  # a second read means "no more body"
                return {"type": "http.request", "body": b"", "more_body": False}
            replayed = True
            return {"type": "http.request", "body": seeded, "more_body": False}

        with install_demo_service():
            await self.app(new_scope, replay, send)


def _supplier_urls() -> list[str]:
    return [
        str(entry["url"])
        for entry in DEMO_CATALOGUE
        if entry["role"] == "supplier"
        and entry["declares_price"]
        and entry["fixture"] in {"shopify", "woocommerce"}
    ]


def _competitor_urls() -> list[str]:
    return [
        str(entry["url"])
        for entry in DEMO_CATALOGUE
        if entry["role"] == "competitor"
    ]


#: Each scenario is a *question*, and the pages it points at are what makes the
#: answer interesting. ``request`` is exactly a ``OpportunityReportRequest`` body.
DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    "happy_path": {
        "label": "Live research: two suppliers + one competitor page",
        "blurb": (
            "The agent fetches real product pages, reads their structured prices, "
            "and prices the offer off VERIFIED costs instead of simulated ones."
        ),
        "watch_for": (
            "Every cost is tagged VERIFIED and carries the URL it was read from; "
            "the margin is arithmetic on real page prices, not a placeholder."
        ),
        "request": {
            "objective": "Assess the LED dog collar opportunity from live supplier pages",
            "product_name": "LED dog collar",
            "product_category": "pet accessories",
            "niche": "pet accessories",
            "target_selling_price_usd": 29.99,
            "supplier_urls": _supplier_urls(),
            "competitor_urls": _competitor_urls(),
        },
    },
    "prose_only": {
        "label": "Refusal: the page never declares a price",
        "blurb": (
            "A blog post mentions a price in visible text and nothing in structured "
            "data. The correct answer is NO price - not a guessed one."
        ),
        "watch_for": (
            "The task fails with a 422 that names the URL. Nothing is invented, and "
            "no simulated cost is substituted for the page you actually asked for."
        ),
        "request": {
            "objective": "Assess the LED dog collar opportunity from the supplied page",
            "product_name": "LED dog collar",
            "product_category": "pet accessories",
            "target_selling_price_usd": 29.99,
            "supplier_urls": [
                str(entry["url"])
                for entry in DEMO_CATALOGUE
                if entry["fixture"] == "prose"
            ],
        },
    },
    "offline_benchmark": {
        "label": "Offline baseline: simulated tiers, clearly labelled",
        "blurb": (
            "No URLs supplied, so nothing is fetched. You get the simulated "
            "benchmark, and the report says SIMULATED on every cost it quotes."
        ),
        "watch_for": (
            "Compare this with the live run: same product, same target price, but "
            "LOW confidence and a TEST_CAUTIOUSLY recommendation because the "
            "numbers are placeholders."
        ),
        "request": {
            "objective": "Assess the LED dog collar opportunity offline",
            "product_name": "LED dog collar",
            "product_category": "pet accessories",
            "niche": "pet accessories",
            "target_selling_price_usd": 29.99,
        },
    },
    "blocked_url": {
        "label": "SSRF guard: try to make the agent fetch its own host",
        "blurb": (
            "Points the agent at 127.0.0.1 and 169.254.169.254 - an internal "
            "service and a cloud metadata endpoint."
        ),
        "watch_for": (
            "The task fails with a 422 that names both URLs. They are rejected by "
            "the scheme/host/range rules before any socket is opened, and no "
            "simulated cost is substituted for them."
        ),
        "request": {
            "objective": "Attempt to read an internal address",
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": [
                "http://127.0.0.1:8000/api/v1/health",
                "http://169.254.169.254/latest/meta-data/",
            ],
        },
    },
}


@contextmanager
def install_demo_service() -> Any:
    """Make every live tool call during this request use the demo transport."""
    previous = tools_module._web_research_factory

    def _factory() -> WebResearchService:
        return WebResearchService(
            transport=demo_transport(), resolver=_DEMO_RESOLVER, search_provider="none"
        )

    tools_module._web_research_factory = _factory
    try:
        yield
    finally:
        tools_module._web_research_factory = previous


def scenario_of(path: str) -> str | None:
    """Read ``?scenario=...`` out of a request path (or path + query string)."""
    query = urlsplit(path).query
    if not query:
        return None
    for pair in query.split("&"):
        key, _, value = pair.partition("=")
        if key == SCENARIO_PARAM:
            return value
    return None


def describe_catalogue() -> str:
    """One-line-per-page summary shown in the console."""
    return json.dumps(
        [
            {
                "fixture": entry["fixture"],
                "url": entry["url"],
                "role": entry["role"],
                "expect": entry["expect"],
            }
            for entry in DEMO_CATALOGUE
        ],
        indent=2,
    )
