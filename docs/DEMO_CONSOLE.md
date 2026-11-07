# The Agent Console (local demo UI)

A one-page console for watching the Product Research agent work, so you do not
have to read JSON in a terminal to understand what it does.

```bash
.venv/bin/python -m uvicorn demo.app:app --host 0.0.0.0 --port 8000
```

Then open <http://127.0.0.1:8000/>. Everything else, including
<http://127.0.0.1:8000/docs>, is the normal API.

## What you can do in it

Pick a scenario, look at the form (it is filled in from the scenario), press
**Run the agent**, and read the result. Four scenarios ship with it:

| Scenario | What it shows |
| --- | --- |
| **Live research: two suppliers + one competitor** | Three product pages fetched, structured prices read, costs tagged `VERIFIED` with the URL each came from. Ends in `PROCEED` at `HIGH` confidence on a 72.7% margin. |
| **Refusal: the page never declares a price** | A blog post that mentions `$9.99` in visible text. The agent returns 422 and names the URL; no number is invented and no simulated cost is substituted. |
| **Offline baseline** | No URLs, nothing fetched: the same product priced off clearly-labelled `SIMULATED` tiers, ending in `LOW` confidence. Compare it with the live run. |
| **SSRF guard** | `127.0.0.1` and `169.254.169.254` are refused before any socket is opened. |

After a run the page shows the supplier table (unit + shipping = landed, with
provenance and source URL), the competitor price points, every risk sentence the
agent recorded, the sources it cited, and the listing copy it drafted from those
facts. **Full response** at the bottom has the raw JSON.

## Why the pages are bundled

The console never fetches the open internet. `demo.router.demo_transport` is an
in-process `httpx.MockTransport` that answers URLs under `demo-shop.example`
(RFC 2606 reserved, so they cannot collide with a real shop) with HTML committed
in `demo/sample_pages.py`, and raises `ConnectError` for every other host.

That is deliberate:

* a demo run is identical on every machine, with or without network access;
* a click can never reach a third-party site, so nothing installed here can
  leak outward;
* the pages go through the **real** extractor
  (`app.services.web_research.extract_product_from_html`) and the **real**
  agent pipeline, so the numbers are genuinely read out of the HTML rather than
  hard-coded into the UI.

The four fixtures are shaped like the pages the extractor is built for:

| Fixture | Shape | Expected outcome |
| --- | --- | --- |
| `shopify` | `@graph` JSON-LD → `Product` → `offers.price` | `$9.40`, `VERIFIED` |
| `woocommerce` | `offers` as a **list**; price in `priceSpecification` | `$8.20` (the in-stock variant), `VERIFIED` |
| `opengraph` | no JSON-LD; `product:price:amount` only | `$10.75`, `VERIFIED` |
| `prose` | price only in visible text | **no price**, with an escalation reason |

## Using your own pages

With real internet access the console is unnecessary - post to the API directly:

```bash
curl -s http://127.0.0.1:8000/api/v1/research/opportunity-report \
  -H 'content-type: application/json' \
  -d '{
        "product_name": "LED dog collar",
        "target_selling_price_usd": 29.99,
        "supplier_urls": ["https://your-supplier.example/products/led-collar"],
        "competitor_urls": ["https://a-competitor.example/products/led-collar"]
      }'
```

A page counts only if it declares its price in machine-readable markup (JSON-LD
`Product`/`Offer`, or `product:price:amount`). A page that does not is reported
in `live_research.unpriced_urls` and contributes no cost. See
[PHASE_3_5_LIVE_WEB_RESEARCH.md](PHASE_3_5_LIVE_WEB_RESEARCH.md).

## How the console stays honest

`demo/app.py` mounts the production app from `app.main` and adds **no API routes
of its own**; the UI and `/demo/meta.json` are served by middleware. On import
it asserts that it owns zero routes beyond the mount and that the mounted app's
OpenAPI surface equals the real app's, so the console cannot quietly grow an
endpoint the API does not have. `tests/test_demo_console.py` pins the same
guarantees, plus "seeding only fills blank fields" - a form you edited is posted
verbatim.

The `?scenario=` parameter selects *which HTTP transport the tools use for that
one request*. It never rewrites a field you filled in, and its effect is scoped
to a single request, so one run cannot contaminate the next.

## Files

```
demo/app.py           the ASGI host (mount, UI, meta, lifespan, drift check)
demo/router.py        scenario definitions, seeding, the sealed transport
demo/sample_pages.py  the four bundled pages
demo/static/index.html the single-page UI (no build step, no CDN)
tests/test_demo_console.py  21 tests
```
