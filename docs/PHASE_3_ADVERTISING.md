# Phase 3 — Advertising & Analytics Agent (Agent 2)

Phase 3 adds the second specialist: an agent that plans advertising, analyses
campaign performance, and can only spend money with a human's explicit approval.
It is built on the same four rules as Phase 1/2 — permission-gated tools, an
audited trail, grounded content, and deterministic arithmetic.

## What it produces

`POST /api/v1/advertising/strategy-report` returns a validated
`MarketingStrategyReport`:

| Field | Meaning |
| --- | --- |
| `campaign_objective` | `CONVERSIONS` / `TRAFFIC` / `ENGAGEMENT` / `BRAND_AWARENESS` |
| `target_audiences` | Audience hypotheses, with empty lists where no fact was supplied |
| `ad_copy_variants` | Copy angles, each listing the facts it rests on |
| `creative_briefs` | Production briefs (`UGC_VIDEO`, `CAROUSEL`, `STATIC_IMAGE`) + compliance notes |
| `performance_metrics` | Raw metrics plus validated derived rates, or `null` pre-launch |
| `conversion_analysis` | Deterministic diagnosis + optimisations |
| `budget_considerations` | A **recommendation**, `requires_human_approval=True` |
| `facts_used` | The exact facts the copy was allowed to use |
| `content_origin` | Which copy was accepted, and why a fallback was used |

`GET /api/v1/advertising/reports` lists everything archived in the
`marketing_reports` memory namespace.

One endpoint serves two modes: omit the raw metrics for **pre-launch planning**
(`health_status = NO_DATA`), or include them for **post-launch analysis**.

## Derived metrics

`CampaignPerformanceMetrics` stores the raw numbers and the derived rates side
by side, and a validator recomputes every derived field from the raw ones — a
hand-edited `roas` is a validation error, not a nicer number.

```
ctr_pct              = clicks / impressions * 100        (0.0 when impressions = 0)
cpc_usd              = spend / clicks                    (None when clicks = 0)
cpa_usd              = spend / conversions               (None when conversions = 0)
conversion_rate_pct  = conversions / clicks * 100        (0.0 when clicks = 0)
roas                 = revenue / spend                   (None when spend = 0)
net_margin_after_ad_spend_usd = revenue - (conversions * landed_cost) - spend
break_even_roas      = round(selling_price / gross_profit, 2)
```

`None` means "this ratio does not exist yet", `0.0` means "it really is zero".
Unknown unit economics are `None`, never assumed to be zero.

### `break_even_roas` is contribution break-even

`gross_profit = selling_price - landed_cost` for **one unit**, so
`break_even_roas = selling_price / gross_profit`.

At that ROAS the ad spend has exactly consumed the product's contribution
margin: the campaign is not yet losing money on unit economics, but it covers
**no fixed overhead** (salaries, software, payment friction, refunds). It is
deliberately *not* an accounting break-even for the business.

It is `None` when the selling price or landed cost is unknown, or when gross
profit is `<= 0` — in which case no ROAS can make the campaign profitable.

## Deterministic diagnosis

`diagnose_campaign_health` evaluates fixed rules in a fixed order:

| Condition | Status |
| --- | --- |
| no metrics, or `impressions == 0` | `NO_DATA` (pre-launch) |
| `ctr_pct < 1.0` | `UNDERPERFORMING_CREATIVE` |
| `conversion_rate_pct < 1.5` | `UNDERPERFORMING_CONVERSION` |
| `roas < break_even_roas` (or `< 1.2` when break-even is unknown) | `UNPROFITABLE_PAUSE` |
| `roas >= 2.5`, above break-even, `conversions >= 5` | `SCALING_CANDIDATE` |
| otherwise | `HEALTHY` |

The order is deliberate: clicking well but converting badly is a conversion
problem *until* it is proven unprofitable, and profitability is always judged
against the product's own contribution margin when it is known.

## Tools

| Tool | Action | Tier |
| --- | --- | --- |
| `mock_ad_copy_generator` | `draft_ad_copy` | Level 1 (kept from Phase 1) |
| `prepare_creative_briefs` | `prepare_creative_brief` | Level 1 |
| `analyze_campaign_performance` | `analyze_campaign_metrics` | Level 1 |
| `mock_publish_ad_campaign` | `publish_advertisement` | Level 2 (kept from Phase 1) |
| `allocate_ad_budget` | `allocate_ad_budget` | Level 2 |

## Where the money gate sits

Producing a `BudgetRecommendation` inside the report is a **Level 1 analytical
output** — it spends nothing, and it always carries
`requires_human_approval=True`.

`publish_live=True` or `commit_budget=True` invokes the corresponding Level 2
tool. The task moves to `WAITING_APPROVAL`, the tool's handler is **never**
reached, and the API returns the finished report **plus** the
`approval_request_id` so a human can review exactly what they are approving.
Resuming happens through the existing endpoint:

```
POST /api/v1/orchestrator/approvals/{approval_id}/resume
{"approver_identity": "..."}
```

Only the mock handlers can run afterwards, and they move no real money.

## Grounding gate

Ad copy and creative briefs may only restate supplied facts. The AI prompt says
so, and then `find_ungrounded_claims` enforces it: discounts, coupons, urgency,
reviews, star ratings, social proof, guarantees, freebies and **any number not
present in the facts** are rejected. A rejected AI answer is replaced by the
deterministic grounded draft, and the reason is recorded in `content_origin`
(e.g. `deterministic_grounded_draft (ai_output=rejected_ungrounded: ...)`).

The mock provider's stub copy cites no fact at all, so it is always rejected —
which is why the offline path is deterministic and grounded by construction.

## Cross-agent handoff

`product_to_marketing` runs Agent 1 and then Agent 2 in one workflow. Agent 2
reads:

* the `product-research` / `latest_candidates` handoff (as it always has);
* the `product_research_reports` archive it points at, which supplies
  `landed_cost`, `selling_price`, `gross_profit` and provenance.

That is what makes `break_even_roas` knowable inside a workflow. With no
research at all, the unit economics stay unknown (`break_even_roas = None`) and
the strategy says so instead of guessing.

If the caller re-prices the product (`price_usd`, `landed_cost_usd`, …), the
archived gross profit is **discarded** rather than mixed with the new numbers —
a stale figure from a different price would silently corrupt the break-even.

## Memory namespaces

| Namespace | Purpose |
| --- | --- |
| `product-research` | Agent 1 handoff (unchanged) |
| `product_research_reports` | Agent 1 report archive (unchanged) |
| `advertising-analytics` | Agent 2 handoff (hyphenated, unchanged spelling) |
| `marketing_reports` | Agent 2 report archive (new, kept separate) |

## Lifecycle

`MainOrchestrator` now tracks `_owns_provider` and supports `close()` /
`with`:

* a provider the orchestrator created is closed by `close()`;
* an injected provider is **not** touched — it belongs to the caller.

The API endpoints use `with MainOrchestrator() as orchestrator:`, so a real
provider's `httpx.Client` pool is released on success *and* on failure instead
of leaking one pool per request.
