# Dropshipping AI System

An autonomous dropshipping agent system with a human-approval model. This slice
(`1A-foundation` + Milestone 2/3 agents + live web research) runs on one machine,
SQLite by default, no Docker.

---

## Run the demo UI (one command)

```bash
python demo/run.py
```

Then open **http://127.0.0.1:8000/**. The launcher sets everything up for you: a
local SQLite file, the mock AI provider, and search disabled (nothing leaves your
machine). It also opens a browser tab when it can — `--no-browser` skips that,
`--port 9000` moves it.

First time only, in the repo root:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Then `python demo/run.py` works from any directory.

### What you can do there

Pick a scenario and press **Run the agent**. The page shows what the Product
Research agent actually did: supplier costs tagged `VERIFIED` with the URL each
came from, competitor price points, every risk it recorded, and the listing copy
it drafted from those facts.

| Scenario | Expected result |
| --- | --- |
| Live research (2 suppliers + 1 competitor) | 3 pages read → `$8.20` / `$9.40` → **72.7% margin → PROCEED @ HIGH** |
| Refusal: page never declares a price | **422**, URL named, nothing invented |
| Offline baseline | `SIMULATED` costs → LOW confidence → `TEST_CAUTIOUSLY` |
| SSRF guard | `127.0.0.1` + `169.254.169.254` refused before any connection |

The console never reaches the open internet: it serves four bundled pages under
`demo-shop.example` through an in-process transport, parsed by the real
extractor. Details: [docs/DEMO_CONSOLE.md](docs/DEMO_CONSOLE.md).

---

## Run the API without the demo UI

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* Interactive API docs: <http://127.0.0.1:8000/docs>
* Health: `GET /api/v1/health` → `{"status":"ok","architecture_phase":"1A-foundation"}`

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

A page only counts when it publishes its price in machine-readable markup
(JSON-LD `Product`/`Offer`, or `product:price:amount`). Anything else is reported
as unpriced and contributes no cost — see
[docs/PHASE_3_5_LIVE_WEB_RESEARCH.md](docs/PHASE_3_5_LIVE_WEB_RESEARCH.md).

---

## Windows note

If `python` is not the venv's interpreter, call it explicitly:

```bat
.venv\Scripts\python.exe demo\run.py
```

If port 8000 is taken, the launcher says so and suggests `--port 8001` rather
than dying with a traceback.

---

## Tests

```bash
.venv/bin/python -m pytest -q        # 577+ tests, no network, no API key
```

Every test is hermetic: `httpx.MockTransport` in place of the network, an
in-memory SQLite database per test, and the mock AI provider.

## Docs

| File | Contents |
| --- | --- |
| [docs/DEMO_CONSOLE.md](docs/DEMO_CONSOLE.md) | the demo UI, its scenarios, and how it stays honest |
| [docs/PHASE_3_5_LIVE_WEB_RESEARCH.md](docs/PHASE_3_5_LIVE_WEB_RESEARCH.md) | page extraction, SSRF guard, evidence rules |
| [docs/PHASE_3_ADVERTISING.md](docs/PHASE_3_ADVERTISING.md) | marketing strategy reports and campaign analysis |
