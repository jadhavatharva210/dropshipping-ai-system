# Phase 3.5 — Live Web Research & Product Page Extraction

Agent 1 could previously only reason about data a caller typed in, or about
clearly-labelled simulated adapters. This phase lets it read **real supplier and
competitor product pages** and turn what those pages declare into `VERIFIED`
facts — without a headless browser, without a new dependency, and without ever
inventing a number.

## What it does

```python
from app.services.web_research import WebResearchService

with WebResearchService() as web:
    page = web.fetch_product_page("https://shop.example.com/products/led-collar")
    page.price_usd            # 9.0  (from JSON-LD Offer)
    page.currency             # "USD"
    page.in_stock             # True (schema.org/InStock)
    page.provenance           # DataProvenance.VERIFIED
    page.source_reference     # the URL the price was read from
```

## Where a price is allowed to come from

| Source | Read as | Provenance |
| --- | --- | --- |
| JSON-LD `<script type="application/ld+json">` → `Product` → `offers` | `price` / `lowPrice` / `priceSpecification.price` | `VERIFIED` |
| OpenGraph / meta → `product:price:amount`, `og:price:amount` | the meta content | `VERIFIED` |
| `<title>`, `meta[name=description]` only | name/description, **no price** | `ESTIMATED` |

`VERIFIED` means *the merchant published this number in machine-readable
markup and it is quoted against its URL* — it is auditable, not infallible. It
is not a supplier quote: a retail page price is what the shop charges, so every
report that uses one says so in `risks`.

If a page declares no structured price, `price_usd` is `None`, the
`escalation_reason` says why, and the page contributes **no** quote. A price
printed in prose is not evidence, and scraping it would be fabrication with
extra steps.

### Shapes it actually handles

* `@graph` wrappers (`WebSite` → `Product`), and products nested in `ItemPage`.
* `offers` as an object *or* a list of variants — the first offer that declares
  a price wins.
* `price` as a number, a string, `"$29.99"`, `"1,299.00"`, `"29,99"` (decimal
  comma), or inside `priceSpecification`.
* `availability` as `https://schema.org/InStock`, bare `InStock`,
  `OutOfStock`, `SoldOut`, `BackOrder`, `PreOrder`… and `None` when the page
  simply did not say (which is not the same as out of stock).
* Broken JSON-LD in one block does not stop the next block from parsing.

## SSRF guard

Refused before any request is built:

* schemes other than `http`/`https` — `file://`, `ftp://`, `javascript:`, `data:`;
* `localhost`, `*.local`, `*.internal`, `*.intranet`, `*.lan`, `metadata`,
  `metadata.google.internal`;
* loopback (`127.0.0.0/8`, `::1`), link-local incl. cloud metadata
  (`169.254.0.0/16`), private (`10/8`, `172.16/12`, `192.168/16`), CGNAT and
  reserved ranges (`is_global` is the real test, not `is_private` alone);
* **a public hostname that resolves to any non-public address** — every answer
  must be public, so `evil.example.com → 127.0.0.1` is refused.

`resolver=` is injectable so tests can validate hostnames that do not exist
without a DNS lookup.

## Search

`WEB_SEARCH_PROVIDER` selects how candidate URLs are discovered:

| Value | Behaviour |
| --- | --- |
| `none` *(default)* | Never searches. No request leaves the machine; the agent falls back to the offline discovery adapter. |
| `duckduckgo` | Parses the keyless public HTML endpoint; every hit is still SSRF-validated. |
| `custom` | `GET WEB_SEARCH_ENDPOINT?q=...&limit=...`, bearer-authenticated with `WEB_SEARCH_API_KEY` when set. |

Both parse `{"results": [{"url", "title", "snippet"}]}`-style JSON.
`WEB_SEARCH_API_KEY` is `repr=False`, like `AI_API_KEY`.

## Tools

| Tool | Action | Tier | Agent |
| --- | --- | --- | --- |
| `fetch_live_product_page` | `research_products` | Level 1 | `PRODUCT_RESEARCH` |
| `search_live_competitors` | `analyze_competitors` | Level 1 | `PRODUCT_RESEARCH` |

Nothing is fetched unless URLs are supplied: with no `supplier_urls` /
`competitor_urls` the tools make no request and say so in `note`.

## Agent wiring

`POST /api/v1/research/opportunity-report` accepts `supplier_urls`,
`competitor_urls`, `search_competitors`, `search_query`. The agent then:

1. fetches the live pages through `fetch_live_product_page` (and
   `search_live_competitors` for competitor URLs);
2. turns each priced page into a `SupplierOption` / `CompetitorEntry` with
   `provenance=VERIFIED` and `source_reference=<url>`;
3. merges them with caller-supplied quotes — a live page **replaces** a caller
   quote for the same URL rather than being averaged with it;
4. runs the existing margin arithmetic and facts-only AI content on the result;
5. records the whole thing in `report.live_research` and adds every source URL
   to `research_sources`.

### Refusal instead of substitution

If the caller supplied `supplier_urls` and **none** of them declared a price,
the task fails with an explanation rather than silently pricing the product off
the simulated tiers. The caller asked about those links; answering with
invented costs would be a different question.

## Ownership

* `WebResearchService` owns its `httpx.Client` and closes it; an injected
  `client=` is left alone.
* The live tools close a service only when they built it themselves
  (`_live_service` returns `(service, tool_owns_it)`). A service from an
  installed factory is shared and stays open; an injected one belongs to the
  caller.
* `tests` install a factory via `tools_module._web_research_factory` /
  `app.core.tools.set_web_research_factory(...)`.
* `app/api/v1/research.py` now uses `with MainOrchestrator(...) as orchestrator:`
  instead of unconditionally calling `orchestrator.ai_provider.close()`, so a
  provider the endpoint did not create is never closed behind its owner's back.

## Settings

```
WEB_RESEARCH_TIMEOUT_SECONDS=10
WEB_SEARCH_PROVIDER=none      # none | duckduckgo | custom
WEB_SEARCH_API_KEY=
WEB_SEARCH_ENDPOINT=
```

## Resource footprint

One `httpx.Client`, at most 10 pages per tool call, a 10 s default timeout, a
2 MB response cap and a 500 000-character parse cap. Parsing is
`html.parser` + `json` + `re` from the stdlib — no browser, no
BeautifulSoup/lxml — which is what keeps it inside the project's 8 GB RAM
budget.
