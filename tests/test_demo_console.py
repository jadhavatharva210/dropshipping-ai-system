"""Tests for the local demo console.

The console is the only user-facing surface in the project, so the things worth
pinning are the things a viewer would be misled by if they broke:

* the console must **not** invent API surface - it mounts the production app,
  and this suite proves the mounted app's documented API equals the real one;
* the demo HTTP transport can only ever serve the bundled pages, so a demo run
  is hermetic and identical on a machine with no internet;
* each scenario really exercises what its blurb claims - a live run prices off
  VERIFIED page data, the prose-only page refuses, the SSRF scenario is blocked;
* seeding only fills fields the caller left empty, so a form the human edited is
  posted verbatim.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.services.web_research import WebResearchService
from demo.app import (
    app as demo_app,
    console_only_routes,
    mounted_app,
    route_signature,
)
from demo.router import (
    DEMO_SCENARIOS,
    demo_transport,
    scenario_of,
    seed_request,
)
from demo.sample_pages import DEMO_CATALOGUE, DEMO_HOST, page_html

REPORT_PATH = "/api/v1/research/opportunity-report"
SUPPLIER_URL = "https://demo-shop.example/products/glowpaw-led-collar"


# ══════════════════════════════════════════════════════════════════════════
# 1. The console cannot drift from the API
# ══════════════════════════════════════════════════════════════════════════


def test_console_defines_no_api_routes_of_its_own() -> None:
    assert console_only_routes(demo_app) == []


def test_mounted_app_is_the_production_app() -> None:
    from app.main import app as production_app

    inner = mounted_app(demo_app)
    assert inner is not None
    assert route_signature(inner) == route_signature(production_app)
    assert "POST " + REPORT_PATH in route_signature(inner)


def test_console_serves_no_second_openapi_schema() -> None:
    """One schema only: the API's. A second would let them disagree."""
    assert demo_app.openapi_url is None
    with TestClient(demo_app) as client:
        assert client.get("/openapi.json").status_code == 200  # the inner app's
        assert client.get("/demo/openapi.json").status_code == 404


# ══════════════════════════════════════════════════════════════════════════
# 2. The demo transport is sealed off from the internet
# ══════════════════════════════════════════════════════════════════════════


def test_demo_transport_serves_bundled_pages() -> None:
    with httpx.Client(transport=demo_transport()) as client:
        response = client.get(SUPPLIER_URL)
    assert response.status_code == 200
    assert "GlowPaw" in response.text


def test_demo_transport_refuses_any_other_host() -> None:
    with httpx.Client(transport=demo_transport()) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("https://example.com/product")


def test_demo_transport_404s_an_unknown_demo_path() -> None:
    with httpx.Client(transport=demo_transport()) as client:
        assert client.get(f"https://{DEMO_HOST}/nope").status_code == 404


def test_every_catalogue_entry_is_servable() -> None:
    for entry in DEMO_CATALOGUE:
        assert page_html(str(entry["url"]))


def test_console_run_makes_no_real_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even the SSRF guard's DNS lookup is stubbed, so a run needs no network."""
    import socket

    def _no_dns(*args: object, **kwargs: object) -> None:
        raise AssertionError("demo console must not perform DNS lookups")

    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)
    with TestClient(demo_app) as client:
        response = client.post(f"{REPORT_PATH}?scenario=happy_path", json={})
    assert response.status_code == 200, response.text


# ══════════════════════════════════════════════════════════════════════════
# 3. Scenarios
# ══════════════════════════════════════════════════════════════════════════


def test_scenario_of_reads_the_query_parameter() -> None:
    assert scenario_of(f"{REPORT_PATH}?scenario=happy_path") == "happy_path"
    assert scenario_of(f"{REPORT_PATH}?a=1&scenario=prose_only") == "prose_only"
    assert scenario_of(REPORT_PATH) is None


def test_seeding_only_fills_blank_fields() -> None:
    seeded = seed_request({"product_name": "my own product"}, "happy_path")
    # The caller's value survives.
    assert seeded["product_name"] == "my own product"
    # The scenario supplies what was missing.
    assert seeded["supplier_urls"] == DEMO_SCENARIOS["happy_path"]["request"]["supplier_urls"]


def test_seeding_replaces_explicitly_empty_values() -> None:
    seeded = seed_request({"supplier_urls": [], "competitor_urls": [SUPPLIER_URL]}, "happy_path")
    assert seeded["supplier_urls"] == DEMO_SCENARIOS["happy_path"]["request"]["supplier_urls"]
    assert seeded["competitor_urls"] == [SUPPLIER_URL]  # a value, so kept


def test_unknown_scenario_is_rejected() -> None:
    with TestClient(demo_app) as client:
        response = client.post(f"{REPORT_PATH}?scenario=does_not_exist", json={})
    assert response.status_code == 404


def test_happy_path_reads_verified_prices_from_the_sample_pages() -> None:
    with TestClient(demo_app) as client:
        response = client.post(f"{REPORT_PATH}?scenario=happy_path", json={})
    assert response.status_code == 200, response.text
    report = response.json()["report"]

    live = report["live_research"]
    # `pages` is the live tool's own receipt; the counts also sit at the top.
    assert live["pages"]["pages_fetched"] == 3
    assert live["pages"]["verified_price_count"] == 3
    assert live["verified_price_count"] == 3
    assert live["blocked_urls"] == []
    assert live["unpriced_urls"] == []

    # Costs are the numbers printed in the bundled HTML, not simulations.
    costs = sorted(s["total_landed_cost_usd"] for s in report["suppliers"])
    assert costs == [8.2, 9.4]
    assert {s["provenance"] for s in report["suppliers"]} == {"VERIFIED"}
    assert {s["source_reference"] for s in report["suppliers"]} == {
        "https://demo-shop.example/products/glowpaw-led-collar",
        "https://demo-shop.example/products/led-collar-usbc",
    }

    # ...and the margin is arithmetic on them.
    margin = report["margin_analysis"]
    assert margin["total_cost_usd"] == 8.2
    assert margin["recommended_selling_price_usd"] == 29.99
    assert round(margin["estimated_gross_margin_pct"], 1) == 72.7
    assert margin["cost_provenance"] == "VERIFIED"
    assert report["research_confidence"] == "HIGH"
    assert report["recommendation"] == "PROCEED"


def test_happy_path_competitor_price_comes_from_the_opengraph_page() -> None:
    with TestClient(demo_app) as client:
        report = client.post(f"{REPORT_PATH}?scenario=happy_path", json={}).json()["report"]
    assert [c["observed_price_usd"] for c in report["competitors"]] == [10.75]
    assert report["competitors"][0]["provenance"] == "VERIFIED"


def test_prose_only_scenario_refuses_instead_of_inventing() -> None:
    with TestClient(demo_app) as client:
        response = client.post(f"{REPORT_PATH}?scenario=prose_only", json={})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "declared a price in structured data" in detail
    assert "blog/led-collars-night-walks" in detail
    # The prose mentioned $9.99; it must not appear anywhere in the answer.
    assert "9.99" not in detail


def test_blocked_url_scenario_reports_the_guard() -> None:
    with TestClient(demo_app) as client:
        response = client.post(f"{REPORT_PATH}?scenario=blocked_url", json={})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert "127.0.0.1" in detail
    assert "169.254.169.254" in detail


def test_offline_scenario_labels_every_cost_as_simulated() -> None:
    with TestClient(demo_app) as client:
        response = client.post(f"{REPORT_PATH}?scenario=offline_benchmark", json={})
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["live_research"] == {}
    assert {s["provenance"] for s in report["suppliers"]} == {"SIMULATED"}
    assert report["margin_analysis"]["cost_provenance"] == "SIMULATED"
    assert report["research_confidence"] == "LOW"


# ══════════════════════════════════════════════════════════════════════════
# 4. The UI itself
# ══════════════════════════════════════════════════════════════════════════


def test_console_serves_the_ui() -> None:
    with TestClient(demo_app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The UI talks only to relative URLs, so it works behind any proxy host.
    assert "localhost" not in response.text
    assert "127.0.0.1" not in response.text
    assert "/api/v1/research/opportunity-report" in response.text


def test_meta_lists_every_scenario_and_sample_page() -> None:
    with TestClient(demo_app) as client:
        meta = client.get("/demo/meta.json").json()
    assert set(meta["scenarios"]) == set(DEMO_SCENARIOS)
    assert meta["default_scenario"] in DEMO_SCENARIOS
    assert len(meta["catalogue"]) == len(DEMO_CATALOGUE)
    for row in meta["catalogue"]:
        assert row["url"].startswith(f"https://{DEMO_HOST}/")


def test_console_lifespan_creates_the_schema() -> None:
    """A mount does not run the inner lifespan; the console must do it itself."""
    with TestClient(demo_app) as client:
        assert client.get("/api/v1/health").status_code == 200
        # A write only succeeds if startup created the tables.
        assert client.post(f"{REPORT_PATH}?scenario=offline_benchmark", json={}).status_code == 200


# ══════════════════════════════════════════════════════════════════════════
# 5. The extractor runs for real against the fixtures
# ══════════════════════════════════════════════════════════════════════════


def test_fixtures_are_parsed_by_the_real_extractor() -> None:
    service = WebResearchService(
        transport=demo_transport(), resolver=lambda host: ["93.184.216.34"]
    )
    try:
        for entry in DEMO_CATALOGUE:
            page = service.fetch_product_page(str(entry["url"]))
            if entry["declares_price"]:
                assert page.price_usd is not None, entry["fixture"]
                assert page.source_reference == entry["url"]
                assert page.provenance.value == "VERIFIED"
            else:
                assert page.price_usd is None, entry["fixture"]
                assert page.escalation_reason
    finally:
        service.close()
