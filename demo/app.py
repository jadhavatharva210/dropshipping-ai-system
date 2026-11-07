"""Local demo console for the agent system.

Run it with::

    .venv/bin/python -m uvicorn demo.app:app --host 0.0.0.0 --port 8000

Then open the printed URL. The console is a thin host around the *real* FastAPI
app from :mod:`app.main`: it serves a UI, a small ``/demo/meta.json`` describing
the bundled scenarios, and it pins outbound HTTP to the bundled sample pages
while a scenario run is in flight. It adds no API of its own, and it refuses to
import unless it owns no routes beyond the mount - so it cannot drift from the
API it is showing you.

Two details worth knowing:

* Mounting an ASGI app does **not** run its lifespan, so the inner app's
  startup (``init_db``) is driven explicitly here. Without that a fresh clone
  would serve the UI and then fail on the first write with "no such table".
* The scenario parameter selects *which HTTP transport* the agent's tools use,
  never what the request contains. The UI fills the form from the scenario, so
  what you see in the form is exactly what gets posted - and can be copied
  straight into ``curl``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from app.core.config import settings
from app.main import app as production_app
from app.main import create_app
from demo.router import DEMO_SCENARIOS, ScenarioSeeder, scenario_of
from demo.sample_pages import DEMO_CATALOGUE

__all__ = [
    "app",
    "console_only_routes",
    "create_demo_app",
    "mounted_app",
    "route_signature",
]

production_app_version = production_app.version

_STATIC = Path(__file__).with_name("static")
_INDEX = _STATIC / "index.html"

#: Path that serves the UI. Everything else is the production app.
UI_PATH = "/"
META_PATH = "/demo/meta.json"


def route_signature(application: FastAPI) -> str:
    """The documented API surface: ``METHOD /path`` for every OpenAPI path.

    Read from the schema rather than the route table because the schema is what
    a caller sees, and because routers included from ``app.main`` are nested
    objects in the route table rather than flat entries.
    """
    schema = application.openapi()
    rows = [
        f"{method.upper()} {path}"
        for path, operations in schema.get("paths", {}).items()
        for method in operations
    ]
    return "\n".join(sorted(rows))


def console_only_routes(application: FastAPI) -> list[str]:
    """Routes the console owns *besides* the mounted production app.

    The console mounts the real app, so it must define no endpoints of its own.
    Anything listed here would exist only in the demo - precisely the drift this
    check is meant to catch. (The UI and meta endpoints are added by middleware,
    which is why they do not appear here.)
    """
    extra: list[str] = []
    for route in application.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is not None and methods:
            extra.append(f"{sorted(methods)} {path}")
    return extra


def mounted_app(application: FastAPI) -> FastAPI | None:
    """The production app mounted at the root, unwrapping the seeder if present."""
    for route in application.routes:
        mounted = getattr(route, "app", None)
        if mounted is None or getattr(route, "path", None) != "":
            continue
        if isinstance(mounted, ScenarioSeeder):
            mounted = mounted.app
        if isinstance(mounted, FastAPI):
            return mounted
    return None


def _demo_meta() -> dict[str, object]:
    return {
        "scenarios": {
            key: {
                "label": spec["label"],
                "blurb": spec["blurb"],
                "watch_for": spec["watch_for"],
                "request": spec["request"],
            }
            for key, spec in DEMO_SCENARIOS.items()
        },
        "default_scenario": "happy_path",
        "catalogue": [
            {
                "fixture": entry["fixture"],
                "url": entry["url"],
                "role": entry["role"],
                "expect": entry["expect"],
                "label": entry["label"],
            }
            for entry in DEMO_CATALOGUE
        ],
        # The console's own transport never leaves the machine; say so in the
        # UI rather than letting a click look broken.
        "offline": True,
        "runtime": {
            "environment": settings.app_env,
            "architecture_phase": production_app_version,
            "ai_provider": settings.ai_provider,
            "live_pages": sorted({str(e["url"]) for e in DEMO_CATALOGUE}),
        },
    }


def create_demo_app() -> FastAPI:
    """Build the console around a fresh :func:`app.main.create_app`."""
    inner = create_app()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Drive the inner app's lifespan - a mount would not do it for us."""
        async with inner.router.lifespan_context(inner):
            yield

    demo = FastAPI(
        title="Dropshipping AI System - Agent Console",
        version=inner.version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @demo.middleware("http")
    async def _console(request: Request, call_next: object) -> Response:
        path = request.url.path
        if path in {UI_PATH, "/console"}:
            # `no-store`: the preview proxy in front of this server must never
            # hand back a stale copy of the page after an edit.
            return HTMLResponse(
                _INDEX.read_text(encoding="utf-8"),
                headers={"cache-control": "no-store, max-age=0"},
            )
        if path == META_PATH:
            return JSONResponse(_demo_meta(), headers={"cache-control": "no-store"})
        scenario = scenario_of(str(request.url))
        if scenario is not None and scenario not in DEMO_SCENARIOS:
            return JSONResponse(
                {"detail": f"Unknown demo scenario {scenario!r}"}, status_code=404
            )
        return await call_next(request)  # type: ignore[operator]

    # Everything under the mount is the production app, with the scenario's
    # body seeding and transport interception applied in front of it.
    demo.mount("/", ScenarioSeeder(inner))
    return demo


def _assert_no_drift() -> None:
    """Fail loudly if the console exposes anything the real app does not."""
    extra = console_only_routes(app)
    if extra:
        raise RuntimeError(
            "the demo console must not define API routes of its own; found: "
            + ", ".join(extra)
        )
    inner = mounted_app(app)
    if inner is None:
        raise RuntimeError("the demo console is not mounting the production app")
    if route_signature(inner) != route_signature(production_app):
        raise RuntimeError(
            "the mounted app's API surface differs from the production app"
        )


#: ASGI instance for ``uvicorn demo.app:app``.
app = create_demo_app()
_assert_no_drift()
