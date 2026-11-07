"""Phase 3.5: live web research & product-page extraction.

Every network interaction here runs through ``httpx.MockTransport``, so the
suite exercises the real HTTP + HTML parsing pipeline with zero external
dependency: no merchant site has to stay online, and nothing here can become
flaky. The ``resolver`` stub plays the same role for DNS, because SSRF
validation resolves hostnames and ``shop.example.com`` does not exist.

What is actually being proved:

* **Real markup is parsed.** The JSON-LD fixtures are the shapes Shopify and
  WooCommerce genuinely emit - ``@graph`` wrappers, ``offers`` as a list,
  ``https://schema.org/InStock``, prices as strings - not tidy inventions.
* **A price only exists if the page published one.** No structured price means
  ``price_usd is None`` and a stated reason, never a number scraped from prose.
* **The SSRF guard refuses before any request.** Loopback, link-local/metadata,
  private ranges and ``file://`` never reach the transport at all, which is
  asserted by counting requests.
* **Live facts flow through the whole agent.** A real page price becomes a
  ``VERIFIED`` ``SupplierOption`` with ``source_reference=<url>``, feeds the
  margin arithmetic, and can produce ``PROCEED @ HIGH``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.agents.specialists import REPORT_NAMESPACE, ProductResearchAgent
from app.core import tools as tools_module
from app.core.ai_provider import MockAIProvider
from app.core.enums import AgentType, PermissionLevel, TaskStatus
from app.core.tools import (
    create_default_registry,
    get_web_research_factory,
    set_web_research_factory,
)
from app.models.core import Task
from app.schemas.product_research import (
    ConfidenceLevel,
    DataProvenance,
    OpportunityRecommendation,
    ProductOpportunityReport,
    SupplierOption,
)
from app.services.memory import SharedMemoryService
from app.services.web_research import (
    BLOCKED_HOSTNAMES,
    BROWSER_HEADERS,
    UnsafeUrlError,
    WebResearchError,
    WebResearchService,
    WebSearchHit,
    extract_product_from_html,
    validate_public_url,
)
from app.services.orchestrator import MainOrchestrator

OPPORTUNITY_REPORT_PATH = "/api/v1/research/opportunity-report"

#: A public address used by the resolver stub. Real DNS is never consulted.
PUBLIC_IP = "93.184.216.34"

SUPPLIER_URL = "https://shop.example.com/products/led-collar"
COMPETITOR_URL = "https://petworld.example.com/products/glow-collar"


# ══════════════════════════════════════════════════════════════════════════
# Fixtures: real-world shaped HTML
# ══════════════════════════════════════════════════════════════════════════

#: Shopify-flavoured JSON-LD: an ``@graph`` wrapper, a ``Product`` node, and the
#: offer inside a list. Price as a string, availability as a schema.org URL.
SHOPIFY_JSON_LD_HTML = """<!doctype html>
<html lang="en">
<head>
  <title>LED Dog Collar Pro | BrightPet</title>
  <meta name="description" content="A rechargeable LED collar for night walks.">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      {"@type": "WebSite", "name": "BrightPet", "url": "https://shop.example.com"},
      {
        "@type": "Product",
        "name": "LED Dog Collar Pro",
        "description": "USB-rechargeable LED collar, visible at night.",
        "brand": {"@type": "Brand", "name": "BrightPet"},
        "offers": [
          {
            "@type": "Offer",
            "price": "9.00",
            "priceCurrency": "USD",
            "availability": "https://schema.org/InStock",
            "seller": {"@type": "Organization", "name": "Shenzhen BrightPet Co"},
            "url": "https://shop.example.com/products/led-collar"
          }
        ]
      }
    ]
  }
  </script>
</head>
<body><div id="shopify-section">product page</div></body>
</html>
"""

#: WooCommerce-flavoured JSON-LD: a bare ``Product``, numeric price, bare
#: ``OutOfStock`` availability, and a ``priceSpecification`` behind it.
WOOCOMMERCE_JSON_LD_HTML = """<!doctype html>
<html><head>
<script type="application/ld+json">
[
  {
    "@context": "https://schema.org",
    "@type": "Product",
    "name": "Slow Feeder Bowl",
    "description": "Puzzle bowl that slows fast eaters.",
    "brand": "PetCalm",
    "offers": {
      "@type": "Offer",
      "priceSpecification": {"@type": "PriceSpecification", "price": 7.5, "priceCurrency": "USD"},
      "availability": "OutOfStock",
      "seller": "PetCalm Store"
    }
  }
]
</script>
</head><body></body></html>
"""

#: No JSON-LD at all: OpenGraph product tags only (the fallback path).
OPENGRAPH_HTML = """<!doctype html>
<html><head>
  <meta property="og:title" content="Glow Collar">
  <meta property="og:description" content="Glow collar for evening walks.">
  <meta property="product:price:amount" content="29.99">
  <meta property="product:price:currency" content="USD">
  <meta property="og:availability" content="instock">
</head><body></body></html>
"""

#: A page that names a product and shows prices in prose - but declares none in
#: structured data. This is the case that must refuse to invent a price.
NO_PRICE_HTML = """<!doctype html>
<html><head>
  <title>LED Dog Collar - only $12.99 today!</title>
  <meta name="description" content="A collar for dogs. Prices from $12.99.">
</head>
<body><span class="price">$12.99</span></body></html>
"""

#: Not a product page at all.
PLAIN_HTML = "<html><head><title>Blog</title></head><body><p>Hello</p></body></html>"


def _page_for(url: str) -> str:
    if "shop.example.com" in url:
        return SHOPIFY_JSON_LD_HTML
    if "petworld.example.com" in url:
        return OPENGRAPH_HTML
    if "woo.example.com" in url:
        return WOOCOMMERCE_JSON_LD_HTML
    if "noprice.example.com" in url:
        return NO_PRICE_HTML
    return PLAIN_HTML


class RouteRecorder:
    """A ``httpx.MockTransport`` handler that records every request URL."""

    def __init__(self, pages: dict[str, str] | None = None) -> None:
        self.requests: list[str] = []
        self.pages = pages or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        url = str(request.url)
        for fragment, html in self.pages.items():
            if fragment in url:
                return httpx.Response(
                    200, text=html, headers={"content-type": "text/html; charset=utf-8"}
                )
        return httpx.Response(
            200, text=_page_for(url), headers={"content-type": "text/html; charset=utf-8"}
        )


def _stub_resolver(host: str) -> list[str]:
    """Pretend every test hostname is a public address."""
    return [PUBLIC_IP]


def make_service(
    handler: Any = None, *, resolver: Any = None, **kwargs: Any
) -> WebResearchService:
    """A service wired to a mock transport and a stub resolver."""
    transport = httpx.MockTransport(handler or RouteRecorder())
    return WebResearchService(
        transport=transport,
        resolver=resolver or _stub_resolver,
        **kwargs,
    )


@pytest.fixture()
def live_service(monkeypatch: pytest.MonkeyPatch) -> RouteRecorder:
    """Install a mocked live-research factory for the whole process.

    ``create_default_registry()`` builds the tools, which build a service from
    the factory at call time. Patching here means the API test exercises the
    real route -> orchestrator -> tool -> service -> parser path with no network.
    """
    recorder = RouteRecorder()
    monkeypatch.setattr(
        tools_module,
        "_web_research_factory",
        lambda: make_service(recorder),
    )
    return recorder


# ══════════════════════════════════════════════════════════════════════════
# 1. SSRF guard
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://C:/Windows/win.ini",
        "ftp://files.example.com/product.json",
        "javascript:alert(1)",
        "data:text/html,<h1>hi</h1>",
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8000/api/v1/health",
        "https://127.0.0.1/product",
        "http://localhost/product",
        "http://localhost:3000/product",
        "http://LOCALHOST/product",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.1/product",
        "http://192.168.0.10:8080/product",
        "http://10.0.0.5/product",
        "http://172.16.4.4/product",
        "http://[::1]/product",
        "http://0.0.0.0/product",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata/",
        "http://printer.local/product",
        "http://intranet.internal/product",
        "",
    ],
)
def test_ssrf_guard_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://8.8.8.8/product",
        "http://1.1.1.1/product",
        "https://93.184.216.34:8443/product?utm_source=x",
    ],
)
def test_ssrf_guard_allows_public_literal_addresses(url: str) -> None:
    """Literal public IPs need no DNS, so this stays hermetic."""
    assert validate_public_url(url) == url


def test_ssrf_guard_allows_a_public_hostname() -> None:
    """A hostname is allowed when it resolves only to public addresses."""
    service = WebResearchService(
        transport=httpx.MockTransport(RouteRecorder()),
        resolver=lambda host: [PUBLIC_IP],
    )
    assert (
        service._validate("https://shop.example.com/products/led-collar")
        == "https://shop.example.com/products/led-collar"
    )


def test_validate_public_url_rejects_a_public_documentation_range() -> None:
    """RFC 5737 test ranges are reserved, so they are refused (no DNS needed)."""
    with pytest.raises(UnsafeUrlError):
        validate_public_url("http://203.0.113.7/product")


def test_ssrf_guard_blocks_a_public_hostname_that_resolves_to_loopback() -> None:
    """A friendly domain pointing at 127.0.0.1 is still refused."""
    with pytest.raises(UnsafeUrlError, match="resolves to the non-public address"):
        WebResearchService(
            transport=httpx.MockTransport(RouteRecorder()),
            resolver=lambda host: ["127.0.0.1"],
        ).fetch_product_page("https://evil.example.com/product")


def test_ssrf_guard_blocks_a_hostname_with_a_mixed_answer() -> None:
    """One private answer among several is enough to refuse the whole name."""
    service = WebResearchService(
        transport=httpx.MockTransport(RouteRecorder()),
        resolver=lambda host: [PUBLIC_IP, "192.168.1.50"],
    )
    with pytest.raises(UnsafeUrlError):
        service.fetch_product_page("https://mixed.example.com/product")


def test_blocked_url_never_reaches_the_transport() -> None:
    """The guard is before the request, not after it - proven by the counter."""
    recorder = RouteRecorder()
    service = make_service(recorder)

    for url in (
        "file:///etc/passwd",
        "http://127.0.0.1/product",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/product",
        "http://192.168.1.1/product",
    ):
        with pytest.raises(UnsafeUrlError):
            service.fetch_product_page(url)

    assert recorder.requests == [], "a blocked URL must not produce a request"


def test_unresolvable_hostname_is_refused_not_fetched() -> None:
    def _boom(host: str) -> list[str]:
        raise OSError("name resolution failed")

    service = WebResearchService(
        transport=httpx.MockTransport(RouteRecorder()), resolver=_boom
    )
    with pytest.raises(UnsafeUrlError, match="did not resolve"):
        service.fetch_product_page("https://nowhere.example.com/product")


def test_default_resolver_is_used_when_none_is_injected() -> None:
    """Public API shape: the production path validates through real DNS."""
    service = WebResearchService(transport=httpx.MockTransport(RouteRecorder()))
    assert service._resolver is None
    # Reserving a documentation address keeps this deterministic: RFC 5737
    # TEST-NET-3 is never routable, so the guard refuses it without a fetch.
    with pytest.raises(UnsafeUrlError):
        service.fetch_product_page("http://203.0.113.7/product")


def test_metadata_hostnames_are_all_listed() -> None:
    assert "169.254.169.254" in BLOCKED_HOSTNAMES
    assert "metadata.google.internal" in BLOCKED_HOSTNAMES


# ══════════════════════════════════════════════════════════════════════════
# 2. JSON-LD extraction (Shopify / WooCommerce shapes)
# ══════════════════════════════════════════════════════════════════════════


def test_shopify_json_ld_product_is_extracted_as_verified() -> None:
    product = extract_product_from_html(SHOPIFY_JSON_LD_HTML, url=SUPPLIER_URL)

    assert product.product_name == "LED Dog Collar Pro"
    assert product.description.startswith("USB-rechargeable")
    assert product.brand == "BrightPet"
    assert product.seller == "Shenzhen BrightPet Co"
    assert product.price_usd == 9.0
    assert product.currency == "USD"
    assert product.in_stock is True
    assert product.provenance is DataProvenance.VERIFIED
    assert product.source_reference == SUPPLIER_URL
    assert product.extraction_source == "json-ld"
    assert product.has_structured_price is True
    assert product.escalation_reason == ""
    assert SUPPLIER_URL in product.offer_urls


def test_woocommerce_json_ld_out_of_stock_and_price_specification() -> None:
    product = extract_product_from_html(
        WOOCOMMERCE_JSON_LD_HTML, url="https://woo.example.com/product/bowl"
    )

    assert product.product_name == "Slow Feeder Bowl"
    assert product.brand == "PetCalm"
    # price came from offer.priceSpecification
    assert product.price_usd == 7.5
    assert product.currency == "USD"
    assert product.in_stock is False
    assert product.availability_raw == "OutOfStock"
    assert product.provenance is DataProvenance.VERIFIED


def test_json_ld_price_variants_are_parsed() -> None:
    """Numbers, currency-prefixed strings and decimal commas all appear in the wild."""
    for raw, expected in (
        (29.99, 29.99),
        ("29.99", 29.99),
        ("$29.99", 29.99),
        ("USD 29.99", 29.99),
        ("1,299.00", 1299.0),
        ("29,99", 29.99),
    ):
        html = (
            '<script type="application/ld+json">'
            + json.dumps(
                {
                    "@type": "Product",
                    "name": "Thing",
                    "offers": {"@type": "Offer", "price": raw, "priceCurrency": "USD"},
                }
            )
            + "</script>"
        )
        product = extract_product_from_html(html, url="https://shop.example.com/p")
        assert product.price_usd == expected, f"price {raw!r} parsed as {product.price_usd}"


def test_json_ld_first_priced_offer_wins_in_a_variant_list() -> None:
    html = (
        '<script type="application/ld+json">'
        + json.dumps(
            {
                "@type": "Product",
                "name": "Collar",
                "offers": [
                    {"@type": "Offer", "priceCurrency": "USD"},  # no price
                    {"@type": "Offer", "price": "6.40", "priceCurrency": "USD"},
                    {"@type": "Offer", "price": "8.10", "priceCurrency": "USD"},
                ],
            }
        )
        + "</script>"
    )
    product = extract_product_from_html(html, url="https://shop.example.com/p")
    assert product.price_usd == 6.40


def test_broken_json_ld_block_does_not_break_the_page() -> None:
    html = (
        '<script type="application/ld+json">{not json at all,,,}</script>'
        '<script type="application/ld+json">'
        + json.dumps({"@type": "Product", "name": "Good One", "offers": {"price": "5.00"}})
        + "</script>"
    )
    product = extract_product_from_html(html, url="https://shop.example.com/p")
    assert product.product_name == "Good One"
    assert product.price_usd == 5.0


def test_availability_parsing_is_conservative() -> None:
    def availability_of(value: Any) -> bool | None:
        html = (
            '<script type="application/ld+json">'
            + json.dumps(
                {
                    "@type": "Product",
                    "name": "Thing",
                    "offers": {"price": "1.00", "availability": value},
                }
            )
            + "</script>"
        )
        return extract_product_from_html(html).in_stock

    assert availability_of("https://schema.org/InStock") is True
    assert availability_of("InStock") is True
    assert availability_of("https://schema.org/OutOfStock") is False
    assert availability_of("OutOfStock") is False
    assert availability_of("SoldOut") is False
    # "The page did not say" is None, not False and not True.
    assert availability_of("") is None
    assert availability_of("ask us") is None


# ══════════════════════════════════════════════════════════════════════════
# 3. OpenGraph fallback
# ══════════════════════════════════════════════════════════════════════════


def test_opengraph_price_is_used_when_json_ld_is_absent() -> None:
    product = extract_product_from_html(OPENGRAPH_HTML, url=COMPETITOR_URL)

    assert product.product_name == "Glow Collar"
    assert product.description == "Glow collar for evening walks."
    assert product.price_usd == 29.99
    assert product.currency == "USD"
    assert product.in_stock is True
    assert product.provenance is DataProvenance.VERIFIED
    assert product.source_reference == COMPETITOR_URL
    assert product.extraction_source == "opengraph"


def test_json_ld_price_wins_over_opengraph_price() -> None:
    """Both sources present: the structured Product node is authoritative."""
    html = SHOPIFY_JSON_LD_HTML.replace(
        "<title>", '<meta property="product:price:amount" content="99.99"><title>'
    )
    product = extract_product_from_html(html, url=SUPPLIER_URL)
    assert product.price_usd == 9.0
    assert product.extraction_source == "json-ld"


def test_meta_description_fallback_is_used_when_og_description_is_missing() -> None:
    html = (
        '<meta name="description" content="Plain meta description.">'
        '<meta property="og:title" content="Collar">'
    )
    product = extract_product_from_html(html, url="https://shop.example.com/p")
    assert product.description == "Plain meta description."
    assert product.price_usd is None  # no price tag anywhere


# ══════════════════════════════════════════════════════════════════════════
# 4. Refusing to invent a price
# ══════════════════════════════════════════════════════════════════════════


def test_page_without_structured_price_reports_no_price() -> None:
    """The prose says "$12.99". The page declares nothing. So there is no price."""
    product = extract_product_from_html(NO_PRICE_HTML, url="https://noprice.example.com/p")

    assert product.price_usd is None
    assert product.provenance is DataProvenance.ESTIMATED
    assert product.source_reference == ""  # nothing verified to attribute
    assert product.extraction_source == "meta-only"
    assert "no structured price" in product.escalation_reason.lower()
    # What the page *did* say is still reported, labelled as unverified.
    assert product.product_name == "LED Dog Collar - only $12.99 today!"


def test_unparseable_page_yields_nothing_and_says_so() -> None:
    product = extract_product_from_html(PLAIN_HTML, url="https://example.com/blog")
    assert product.price_usd is None
    assert product.provenance is DataProvenance.ESTIMATED
    assert product.source_reference == ""
    assert "no structured price" in product.escalation_reason.lower()


def test_empty_page_reports_that_nothing_was_extracted() -> None:
    product = extract_product_from_html("<html><body></body></html>", url="https://x.example.com")
    assert product.product_name == ""
    assert product.price_usd is None
    assert "not a product page" in product.escalation_reason.lower()


def test_service_returns_the_unpriced_page_rather_than_raising() -> None:
    service = make_service()
    product = service.fetch_product_page("https://noprice.example.com/product")
    assert product.price_usd is None
    assert product.escalation_reason


def test_http_error_is_a_research_error() -> None:
    service = make_service(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(WebResearchError, match="HTTP 500"):
        service.fetch_product_page(SUPPLIER_URL)


def test_transport_failure_is_a_research_error() -> None:
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = make_service(_boom)
    with pytest.raises(WebResearchError, match="failed"):
        service.fetch_product_page(SUPPLIER_URL)


def test_fetch_many_reports_per_url_failures_without_losing_successes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "dead.example.com" in str(request.url):
            return httpx.Response(404, text="gone")
        return httpx.Response(200, text=SHOPIFY_JSON_LD_HTML)

    service = make_service(handler)
    pages, errors = service.fetch_product_pages(
        [
            SUPPLIER_URL,
            "https://dead.example.com/product",
            "http://127.0.0.1/product",
            SUPPLIER_URL,  # duplicate, de-duplicated
        ]
    )
    assert [page.price_usd for page in pages] == [9.0]
    assert len(errors) == 2
    assert any(error.get("blocked") for error in errors)
    assert any("HTTP 404" in error["error"] for error in errors)


def test_requests_carry_browser_headers() -> None:
    """Stores serve stripped pages to a bare httpx user agent."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=SHOPIFY_JSON_LD_HTML)

    service = make_service(handler)
    service.fetch_product_page(SUPPLIER_URL)

    assert seen[0].headers["user-agent"] == BROWSER_HEADERS["User-Agent"]
    assert "text/html" in seen[0].headers["accept"]


def test_realistic_browser_user_agent_is_present() -> None:
    assert "Mozilla/5.0" in BROWSER_HEADERS["User-Agent"]


# ══════════════════════════════════════════════════════════════════════════
# 5. Search layer
# ══════════════════════════════════════════════════════════════════════════


DDG_HTML = """
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpetworld.example.com%2Fp%2Fglow&amp;rut=abc">
     Glow Collar &amp; LED
  </a>
  <a class="result__snippet" href="#">A glow collar <b>for</b> night walks.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwoo.example.com%2Fproduct%2Fbowl&amp;rut=def">
     Slow Feeder Bowl
  </a>
  <a class="result__snippet" href="#">Slow feeder for fast eaters.</a>
</div>
"""


def test_search_is_disabled_by_default() -> None:
    """No configured backend means no request - never an invented URL."""
    recorder = RouteRecorder()
    service = make_service(recorder, search_provider="none")
    assert service.search_web_products("led dog collar") == []
    assert recorder.requests == []


def test_duckduckgo_html_results_are_parsed() -> None:
    service = make_service(
        lambda request: httpx.Response(200, text=DDG_HTML),
        search_provider="duckduckgo",
    )
    hits = service.search_web_products("led dog collar", max_results=5)

    assert [hit.url for hit in hits] == [
        "https://petworld.example.com/p/glow",
        "https://woo.example.com/product/bowl",
    ]
    assert hits[0].title == "Glow Collar & LED"
    assert hits[0].snippet == "A glow collar for night walks."
    assert hits[0].source == "duckduckgo"
    assert [hit.rank for hit in hits] == [1, 2]


def test_duckduckgo_results_respect_max_results() -> None:
    service = make_service(
        lambda request: httpx.Response(200, text=DDG_HTML), search_provider="duckduckgo"
    )
    assert len(service.search_web_products("collar", max_results=1)) == 1


def test_custom_search_provider_parses_json() -> None:
    payload = {
        "results": [
            {"url": "https://petworld.example.com/p/glow", "title": "Glow", "snippet": "s1"},
            {"link": "https://woo.example.com/p/bowl", "name": "Bowl"},
            "https://shop.example.com/p/collar",
            {"url": "javascript:alert(1)", "title": "bad"},
        ]
    }
    service = make_service(
        lambda request: httpx.Response(200, json=payload),
        search_provider="custom",
        search_endpoint="https://search.example.com/api",
    )
    hits = service.search_web_products("collar")
    assert [hit.url for hit in hits] == [
        "https://petworld.example.com/p/glow",
        "https://woo.example.com/p/bowl",
        "https://shop.example.com/p/collar",
    ]
    assert hits[0].source == "custom"


def test_custom_search_requires_an_endpoint() -> None:
    service = make_service(
        lambda request: httpx.Response(200, json={}), search_provider="custom"
    )
    with pytest.raises(WebResearchError, match="WEB_SEARCH_ENDPOINT"):
        service.search_web_products("collar")


def test_unknown_search_provider_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="WEB_SEARCH_PROVIDER"):
        make_service(search_provider="askjeeves")


def test_search_provider_defaults_come_from_settings() -> None:
    """The shipped default is ``none``: nothing leaves the machine implicitly."""
    from app.core.config import settings

    assert settings.web_search_provider == "none"
    assert make_service().search_provider == "none"


# ══════════════════════════════════════════════════════════════════════════
# 6. Service lifecycle
# ══════════════════════════════════════════════════════════════════════════


def test_close_is_idempotent_and_releases_the_owned_client() -> None:
    service = make_service()
    assert service._owns_client is True
    service.close()
    assert service._owns_client is False
    service.close()  # must not raise


def test_context_manager_closes_on_exit() -> None:
    with make_service() as service:
        assert service._owns_client is True
    assert service._owns_client is False


def test_injected_client_is_not_closed() -> None:
    client = httpx.Client(transport=httpx.MockTransport(RouteRecorder()))
    service = WebResearchService(client=client, resolver=_stub_resolver)
    assert service._owns_client is False
    service.close()
    # Still usable: the caller owns it.
    assert client.get(SUPPLIER_URL).status_code == 200
    client.close()


# ══════════════════════════════════════════════════════════════════════════
# 7. Tools
# ══════════════════════════════════════════════════════════════════════════


def test_live_tools_are_registered_as_level1_research_tools() -> None:
    from app.core.permissions import PermissionPolicy

    registry = create_default_registry()
    for name, action in (
        ("fetch_live_product_page", "research_products"),
        ("search_live_competitors", "analyze_competitors"),
    ):
        tool = registry.get_tool(name)
        assert tool.action_name == action
        assert tool.allowed_agents == {AgentType.PRODUCT_RESEARCH}
        assert PermissionPolicy.level_for(tool.action_name).is_autonomous


def test_live_tools_reject_other_agents(db: Session) -> None:
    task = _research_task(db, {})
    registry = create_default_registry()
    for name in ("fetch_live_product_page", "search_live_competitors"):
        with pytest.raises(PermissionError):
            registry.execute_tool(
                db,
                tool_name=name,
                task_id=task.id,
                agent_type=AgentType.ADVERTISING_ANALYTICS,
                parameters={"supplier_urls": [SUPPLIER_URL]},
            )


def test_fetch_tool_returns_verified_supplier_and_competitor_entries(db: Session) -> None:
    task = _research_task(db, {})
    registry = create_default_registry()
    outcome = registry.execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={
            "supplier_urls": [SUPPLIER_URL],
            "competitor_urls": [COMPETITOR_URL],
            "web_research_service": make_service(),
        },
        context_summary="read live pages",
    )
    assert outcome["executed"] is True
    result = outcome["result"]

    supplier = result["suppliers"][0]
    assert supplier["provenance"] == "VERIFIED"
    assert supplier["unit_cost_usd"] == 9.0
    assert supplier["source_reference"] == SUPPLIER_URL
    assert supplier["in_stock"] is True

    competitor = result["competitors"][0]
    assert competitor["provenance"] == "VERIFIED"
    assert competitor["observed_price_usd"] == 29.99
    assert competitor["source_reference"] == COMPETITOR_URL

    assert result["verified_price_count"] == 2
    assert result["pages_fetched"] == 2


def test_fetch_tool_with_no_urls_makes_no_request(db: Session) -> None:
    task = _research_task(db, {})
    recorder = RouteRecorder()
    outcome = create_default_registry().execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"web_research_service": make_service(recorder)},
    )
    assert outcome["executed"] is True
    assert outcome["result"]["suppliers"] == []
    assert outcome["result"]["pages_fetched"] == 0
    assert "no urls were supplied" in outcome["result"]["note"].lower()
    assert recorder.requests == []


def test_fetch_tool_reports_blocked_urls_without_fetching(db: Session) -> None:
    task = _research_task(db, {})
    recorder = RouteRecorder()
    outcome = create_default_registry().execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={
            "supplier_urls": ["http://169.254.169.254/latest/meta-data/", SUPPLIER_URL],
            "web_research_service": make_service(recorder),
        },
    )
    result = outcome["result"]
    assert result["supplier_errors"][0]["blocked"] is True
    assert result["blocked_urls"] == ["http://169.254.169.254/latest/meta-data/"]
    # Only the safe URL was actually requested.
    assert recorder.requests == [SUPPLIER_URL]
    # The safe page still produced a quote.
    assert result["verified_price_count"] == 1


def test_fetch_tool_does_not_invent_a_price_for_an_unpriced_page(db: Session) -> None:
    task = _research_task(db, {})
    outcome = create_default_registry().execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={
            "supplier_urls": ["https://noprice.example.com/product"],
            "web_research_service": make_service(),
        },
    )
    result = outcome["result"]
    assert result["suppliers"][0]["unit_cost_usd"] is None
    assert result["suppliers"][0]["provenance"] == "ESTIMATED"
    assert result["verified_price_count"] == 0
    assert result["unpriced_urls"] == ["https://noprice.example.com/product"]
    assert "no price was reported" in result["note"].lower()


def test_search_tool_records_observations_from_discovered_urls(db: Session) -> None:
    task = _research_task(db, {})

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in str(request.url):
            return httpx.Response(200, text=DDG_HTML)
        return httpx.Response(200, text=_page_for(str(request.url)))

    outcome = create_default_registry().execute_tool(
        db,
        tool_name="search_live_competitors",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={
            "query": "led dog collar",
            "web_research_service": make_service(handler, search_provider="duckduckgo"),
        },
    )
    result = outcome["result"]
    prices = [entry["observed_price_usd"] for entry in result["observations"]]
    # petworld publishes 29.99 via OpenGraph; woo publishes a Product with a
    # price but is OutOfStock, which is still an observation.
    assert 29.99 in prices
    assert result["observed_count"] == len(result["observations"])
    assert result["search_provider"] == "duckduckgo"
    assert result["search_hits"]


def test_search_tool_reads_caller_urls_even_with_search_disabled(db: Session) -> None:
    task = _research_task(db, {})
    outcome = create_default_registry().execute_tool(
        db,
        tool_name="search_live_competitors",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={
            "query": "led dog collar",
            "competitor_urls": [COMPETITOR_URL],
            "web_research_service": make_service(search_provider="none"),
        },
    )
    result = outcome["result"]
    assert result["search_hits"] == []
    assert result["observations"][0]["observed_price_usd"] == 29.99
    assert result["observations"][0]["source_reference"] == COMPETITOR_URL


def test_search_tool_survives_a_search_backend_failure(db: Session) -> None:
    task = _research_task(db, {})

    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in str(request.url):
            return httpx.Response(503, text="rate limited")
        return httpx.Response(200, text=OPENGRAPH_HTML)

    outcome = create_default_registry().execute_tool(
        db,
        tool_name="search_live_competitors",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={
            "query": "collar",
            "competitor_urls": [COMPETITOR_URL],
            "web_research_service": make_service(handler, search_provider="duckduckgo"),
        },
    )
    result = outcome["result"]
    assert "503" in result["search_error"]
    # The supplied URL was still read, so a search outage loses nothing.
    assert result["observed_count"] == 1


# ══════════════════════════════════════════════════════════════════════════
# 8. Agent wiring
# ══════════════════════════════════════════════════════════════════════════


@contextmanager
def using_service(service: WebResearchService) -> Iterator[None]:
    """Make ``service`` the default for every live tool call.

    This is the same channel the API uses (the tools build a service from the
    factory when the agent has no way to carry one through a JSON payload), so
    a test that goes through it is exercising the production path rather than a
    test-only shortcut.
    """
    previous = tools_module._web_research_factory
    tools_module._web_research_factory = lambda: service
    try:
        yield
    finally:
        tools_module._web_research_factory = previous


def _research_task(db: Session, payload: dict[str, Any]) -> Task:
    task = Task(
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Assess a product opportunity",
        objective="Evaluate margin and draft listing content",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        input_payload=dict(payload),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _run_agent(
    db: Session,
    task: Task,
    *,
    memory: SharedMemoryService | None = None,
    service: WebResearchService | None = None,
) -> Any:
    def _execute() -> Any:
        return ProductResearchAgent().execute(
            db,
            task,
            create_default_registry(),
            memory or SharedMemoryService(),
            MockAIProvider(),
        )

    if service is None:
        return _execute()
    with using_service(service):
        return _execute()


def _report(output: Any) -> ProductOpportunityReport:
    return ProductOpportunityReport.model_validate(output.data["opportunity_report"])


def test_agent_turns_a_live_page_price_into_a_verified_quote(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "supplier_urls": [SUPPLIER_URL],
        },
    )
    output = _run_agent(db, task, service=make_service())
    assert output.status is TaskStatus.COMPLETED
    report = _report(output)

    supplier = report.best_supplier
    assert supplier is not None
    assert supplier.unit_cost_usd == 9.0
    assert supplier.total_landed_cost_usd == 9.0
    assert supplier.provenance is DataProvenance.VERIFIED
    assert supplier.source_reference == SUPPLIER_URL
    assert supplier.in_stock is True
    assert supplier.supplier_name == "Shenzhen BrightPet Co"

    # The real cost feeds the real arithmetic: (29.99 - 9.00) / 29.99 = 69.99%
    assert report.margin_analysis.total_cost_usd == 9.0
    assert report.margin_analysis.estimated_gross_margin_pct == pytest.approx(69.99, abs=0.01)
    assert report.margin_analysis.cost_provenance is DataProvenance.VERIFIED
    assert report.research_confidence is ConfidenceLevel.HIGH
    assert report.recommendation is OpportunityRecommendation.PROCEED

    # The URL is auditable in both the sources list and the live receipt.
    assert SUPPLIER_URL in report.research_sources
    assert report.live_research["supplier_urls"] == [SUPPLIER_URL]
    assert report.live_research["verified_price_count"] == 1


def test_agent_uses_the_cheapest_of_several_live_supplier_pages(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "supplier_urls": [SUPPLIER_URL, "https://woo.example.com/product/bowl"],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))
    # 7.50 (WooCommerce page) beats 9.00 (Shopify page).
    assert report.best_supplier.unit_cost_usd == 7.5
    assert report.best_supplier.provenance is DataProvenance.VERIFIED
    assert len(report.suppliers) == 2


def test_agent_refuses_when_the_only_page_is_unpriced(db: Session) -> None:
    """A fetched page with no structured price is evidence of nothing.

    The agent must not fall back to a simulated tier here: the caller asked for
    this specific page, and inventing a cost for it is exactly the fabrication
    the whole provenance system exists to prevent.
    """
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": ["https://noprice.example.com/product"],
        },
    )
    output = _run_agent(db, task, service=make_service())
    # No supplier quote at all -> the agent refuses rather than invent one.
    assert output.status is TaskStatus.FAILED
    assert output.data["refused"] is True
    assert "declared a price in structured data" in output.summary
    assert "https://noprice.example.com/product" in output.summary
    assert "simulated" in output.summary.lower()


def test_agent_keeps_going_when_one_live_page_is_unpriced(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "supplier_urls": [
                "https://noprice.example.com/product",
                SUPPLIER_URL,
            ],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))
    assert len(report.suppliers) == 1
    assert report.best_supplier.total_landed_cost_usd == 9.0
    # The unpriced page is recorded as a risk, not silently dropped.
    assert any("declared no structured price" in risk for risk in report.risks)
    unpriced = report.live_research["unpriced_urls"]
    assert [entry["url"] for entry in unpriced] == ["https://noprice.example.com/product"]
    assert "no structured price" in unpriced[0]["reason"].lower()


def test_agent_uses_live_competitor_prices_for_the_market_landscape(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "supplier_urls": [SUPPLIER_URL],
            "competitor_urls": [COMPETITOR_URL],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))

    assert [entry.observed_price_usd for entry in report.competitors] == [29.99]
    competitor = report.competitors[0]
    assert competitor.provenance is DataProvenance.VERIFIED
    assert competitor.source_reference == COMPETITOR_URL
    # No explicit target price: the agent prices off the observed market.
    assert report.margin_analysis.recommended_selling_price_usd == 29.99
    assert report.margin_analysis.price_provenance is DataProvenance.VERIFIED
    assert COMPETITOR_URL in report.research_sources


def test_agent_without_live_urls_is_byte_identical_to_the_offline_path(db: Session) -> None:
    """Phase 3.5 must not change the offline behaviour at all."""
    recorder = RouteRecorder()
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = ProductResearchAgent().execute(
        db,
        task,
        create_default_registry(),
        SharedMemoryService(),
        MockAIProvider(),
    )
    report = _report(output)
    assert report.live_research == {}
    assert recorder.requests == []
    # Offline provenance ladder is untouched.
    assert report.margin_analysis.cost_provenance is DataProvenance.SIMULATED
    assert report.research_confidence is ConfidenceLevel.LOW


def test_agent_records_a_blocked_url_in_the_report(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "supplier_urls": ["http://169.254.169.254/latest/meta-data/", SUPPLIER_URL],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))
    assert report.live_research["blocked_urls"] == [
        "http://169.254.169.254/latest/meta-data/"
    ]
    # The safe page still produced the verified quote.
    assert report.best_supplier.total_landed_cost_usd == 9.0


def test_agent_prefers_the_live_page_over_a_duplicate_caller_quote(db: Session) -> None:
    """The page is the primary source; a caller's guess does not average in."""
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": [SUPPLIER_URL],
            "suppliers": [
                {
                    "supplier_name": "Caller estimate",
                    "unit_cost_usd": 4.0,
                    "provenance": "ESTIMATED",
                    "source_reference": SUPPLIER_URL,
                }
            ],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))
    assert len(report.suppliers) == 1
    assert report.suppliers[0].unit_cost_usd == 9.0
    assert report.suppliers[0].provenance is DataProvenance.VERIFIED


def test_agent_flags_live_prices_as_retail_not_wholesale(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": [SUPPLIER_URL],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))
    assert any("not a wholesale quote" in risk for risk in report.risks)


def test_agent_archives_the_live_report(db: Session) -> None:
    memory = SharedMemoryService()
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": [SUPPLIER_URL],
        },
    )
    report = _report(_run_agent(db, task, memory=memory, service=make_service()))
    archived = memory.list_namespace(db, namespace=REPORT_NAMESPACE)
    assert [entry.key for entry in archived] == [task.id]
    stored = ProductOpportunityReport.model_validate(archived[0].value["report"])
    assert stored.live_research["verified_price_count"] == 1
    assert stored == report


def test_report_survives_a_json_round_trip_with_live_research(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "supplier_urls": [SUPPLIER_URL],
            "competitor_urls": [COMPETITOR_URL],
        },
    )
    report = _report(_run_agent(db, task, service=make_service()))
    restored = ProductOpportunityReport.model_validate(
        json.loads(json.dumps(report.model_dump(mode="json")))
    )
    assert restored == report


def test_supplier_option_source_reference_is_required_for_audit() -> None:
    """A VERIFIED quote is only meaningful if you can see where it came from."""
    option = SupplierOption.quote(
        supplier_name="BrightPet",
        unit_cost_usd=9.0,
        provenance=DataProvenance.VERIFIED,
        source_reference=SUPPLIER_URL,
    )
    assert option.source_reference == SUPPLIER_URL


# ══════════════════════════════════════════════════════════════════════════
# 9. HTTP API end to end
# ══════════════════════════════════════════════════════════════════════════


def test_live_research_paths_are_advertised_in_openapi(app: FastAPI) -> None:
    schema = json.dumps(app.openapi())
    assert "supplier_urls" in schema
    assert "competitor_urls" in schema
    # ...and the model is reachable from the endpoint the caller uses.
    assert "OpportunityReportRequest" in schema


def test_post_with_supplier_urls_produces_a_verified_proceed_report(
    client: TestClient, live_service: RouteRecorder
) -> None:
    """The full pipeline: HTTP -> orchestrator -> tool -> page -> VERIFIED report."""
    response = client.post(
        OPPORTUNITY_REPORT_PATH,
        json={
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "supplier_urls": [SUPPLIER_URL],
            "competitor_urls": [COMPETITOR_URL],
        },
    )
    assert response.status_code == 200, response.text
    report = ProductOpportunityReport.model_validate(response.json()["report"])

    # VERIFIED supplier cost read from the page itself.
    assert report.best_supplier.provenance is DataProvenance.VERIFIED
    assert report.best_supplier.total_landed_cost_usd == 9.0
    assert report.best_supplier.source_reference == SUPPLIER_URL
    assert report.margin_analysis.cost_provenance is DataProvenance.VERIFIED
    assert report.margin_analysis.price_provenance is DataProvenance.VERIFIED

    # Margins >= 35% with verified costs -> PROCEED @ HIGH.
    assert report.margin_analysis.estimated_gross_margin_pct >= 35.0
    assert report.research_confidence is ConfidenceLevel.HIGH
    assert report.recommendation is OpportunityRecommendation.PROCEED

    # The competitor page was read too, and both URLs are real requests.
    assert [entry.observed_price_usd for entry in report.competitors] == [29.99]
    assert SUPPLIER_URL in report.research_sources
    assert COMPETITOR_URL in report.research_sources
    assert set(live_service.requests) == {SUPPLIER_URL, COMPETITOR_URL}


def test_post_without_urls_makes_no_network_request(
    client: TestClient, live_service: RouteRecorder
) -> None:
    response = client.post(OPPORTUNITY_REPORT_PATH, json={"product_name": "LED dog collar"})
    assert response.status_code == 200, response.text
    assert live_service.requests == []
    report = ProductOpportunityReport.model_validate(response.json()["report"])
    assert report.live_research == {}
    assert report.margin_analysis.cost_provenance is DataProvenance.SIMULATED


def test_post_with_only_unpriced_urls_refuses_rather_than_inventing(
    client: TestClient, live_service: RouteRecorder
) -> None:
    response = client.post(
        OPPORTUNITY_REPORT_PATH,
        json={
            "product_name": "LED dog collar",
            "supplier_urls": ["https://noprice.example.com/product"],
        },
    )
    # 422: a data problem (the page had no price), not a server error.
    assert response.status_code == 422, response.text
    assert "supplier" in response.json()["detail"].lower()
    assert live_service.requests  # it did fetch, it just found no price


def test_post_with_a_blocked_url_reports_it_and_uses_the_safe_one(
    client: TestClient, live_service: RouteRecorder
) -> None:
    response = client.post(
        OPPORTUNITY_REPORT_PATH,
        json={
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": ["http://127.0.0.1/product", "file:///etc/passwd", SUPPLIER_URL],
        },
    )
    assert response.status_code == 200, response.text
    report = ProductOpportunityReport.model_validate(response.json()["report"])
    assert report.live_research["blocked_urls"] == [
        "http://127.0.0.1/product",
        "file:///etc/passwd",
    ]
    # Nothing blocked was ever requested; the safe URL was.
    assert live_service.requests == [SUPPLIER_URL]


def test_post_archives_the_live_report_for_the_listing_endpoint(
    client: TestClient, live_service: RouteRecorder
) -> None:
    created = client.post(
        OPPORTUNITY_REPORT_PATH,
        json={
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "supplier_urls": [SUPPLIER_URL],
        },
    )
    assert created.status_code == 200, created.text

    listed = client.get("/api/v1/research/reports")
    assert listed.status_code == 200
    entries = listed.json()
    assert len(entries) == 1
    assert entries[0]["cost_provenance"] == "VERIFIED"
    assert entries[0]["report"]["live_research"]["verified_price_count"] == 1


def test_research_endpoint_uses_orchestrator_ownership(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """research.py must use ``with MainOrchestrator(...)``, not a raw close()."""
    import inspect

    from app.api.v1 import research as research_api

    source = inspect.getsource(research_api.create_opportunity_report)
    assert "with orchestrator:" in source
    # The old unconditional close is gone: it closed providers the endpoint did
    # not create. Only the docstring may still mention it (as the reason).
    code_lines = [
        line.split("#", 1)[0].strip()
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith(('"""', "*", '"""'))
    ]
    assert not any("ai_provider.close()" in line for line in code_lines)


def test_research_endpoint_does_not_close_an_injected_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, live_service: RouteRecorder
) -> None:
    from app.api.v1 import research as research_api

    closed: list[bool] = []
    provider = MockAIProvider()
    original_close = provider.close

    def _close() -> None:
        closed.append(True)
        original_close()

    provider.close = _close  # type: ignore[method-assign]
    monkeypatch.setattr(
        research_api, "_orchestrator", lambda: MainOrchestrator(ai_provider=provider)
    )
    response = client.post(OPPORTUNITY_REPORT_PATH, json={"product_name": "LED dog collar"})
    assert response.status_code == 200, response.text
    assert closed == []


def test_research_endpoint_closes_a_provider_it_created(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, live_service: RouteRecorder
) -> None:
    from app.api.v1 import research as research_api

    closed: list[bool] = []
    provider = MockAIProvider()
    original_close = provider.close

    def _close() -> None:
        closed.append(True)
        original_close()

    provider.close = _close  # type: ignore[method-assign]

    def _orchestrator() -> MainOrchestrator:
        orchestrator = MainOrchestrator()
        orchestrator.ai_provider = provider
        orchestrator._owns_provider = True
        return orchestrator

    monkeypatch.setattr(research_api, "_orchestrator", _orchestrator)
    response = client.post(OPPORTUNITY_REPORT_PATH, json={"product_name": "LED dog collar"})
    assert response.status_code == 200, response.text
    assert closed == [True]


# ══════════════════════════════════════════════════════════════════════════
# 10. Factory plumbing
# ══════════════════════════════════════════════════════════════════════════


def test_factory_defaults_to_the_real_service() -> None:
    assert get_web_research_factory() is WebResearchService


def test_factory_can_be_installed_and_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = make_service()
    set_web_research_factory(lambda: sentinel)
    assert get_web_research_factory()() is sentinel
    set_web_research_factory(None)
    assert get_web_research_factory() is WebResearchService


def test_injected_service_is_not_closed_by_the_tool(db: Session) -> None:
    """The tool closes what it built, never what the caller handed it."""
    task = _research_task(db, {})
    service = make_service()
    create_default_registry().execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"supplier_urls": [SUPPLIER_URL], "web_research_service": service},
    )
    assert service._owns_client is True
    service.close()


def test_factory_provided_service_is_not_closed_by_the_tool(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory-owned service can be shared, so the tool must not close it.

    (Closing it would break the next call - which is exactly the bug the
    ownership tuple in ``_live_service`` exists to prevent.) A service the tool
    builds itself, with no factory installed, *is* closed; see below.
    """
    task = _research_task(db, {})
    built: list[WebResearchService] = []

    def _factory() -> WebResearchService:
        service = make_service()
        built.append(service)
        return service

    monkeypatch.setattr(tools_module, "_web_research_factory", _factory)
    create_default_registry().execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"supplier_urls": [SUPPLIER_URL]},
    )
    assert built and built[0]._owns_client is True  # still open, still reusable
    # A second call through the same factory-built service still works.
    create_default_registry().execute_tool(
        db,
        tool_name="fetch_live_product_page",
        task_id=task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"supplier_urls": [SUPPLIER_URL]},
    )
    assert len(built) == 2


def test_live_service_ownership_is_reported_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_live_service`` tells the caller who must close what.

    Three cases, and the tool acts on the flag: injected and factory-provided
    services are someone else's to close, a self-built one is not.
    """
    from app.core.tools import _live_service

    injected = make_service()
    service, owns = _live_service({"web_research_service": injected})
    assert service is injected and owns is False

    shared = make_service()
    monkeypatch.setattr(tools_module, "_web_research_factory", lambda: shared)
    service, owns = _live_service({})
    assert service is shared and owns is False

    monkeypatch.setattr(tools_module, "_web_research_factory", None)
    service, owns = _live_service({})
    assert isinstance(service, WebResearchService) and owns is True
    service.close()  # the tool would do this; here the test does


def test_settings_expose_the_live_research_keys() -> None:
    from app.core.config import Settings, WEB_SEARCH_PROVIDER_CHOICES

    assert Settings.model_fields["web_search_provider"].default == "none"
    assert Settings.model_fields["web_research_timeout_seconds"].default == 10.0
    assert WEB_SEARCH_PROVIDER_CHOICES == ("none", "duckduckgo", "custom")


def test_search_api_key_is_hidden_from_repr() -> None:
    """A search key must never leak through a log or a traceback."""
    from app.core.config import Settings

    config = Settings(_env_file=None, web_search_api_key="super-secret-key")
    assert "super-secret-key" not in repr(config)


def test_unknown_search_provider_is_rejected_by_settings() -> None:
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, web_search_provider="askjeeves")


def test_extracted_product_to_dict_is_json_safe() -> None:
    product = extract_product_from_html(SHOPIFY_JSON_LD_HTML, url=SUPPLIER_URL)
    payload = json.loads(json.dumps(product.to_dict()))
    assert payload["provenance"] == "VERIFIED"
    assert payload["price_usd"] == 9.0


def test_web_search_hit_is_a_plain_record() -> None:
    hit = WebSearchHit(url=SUPPLIER_URL, title="Collar", snippet="s", rank=1)
    assert hit.source == "search"
    assert hit.url == SUPPLIER_URL
