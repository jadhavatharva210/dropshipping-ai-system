"""Phase 1 specialist agents built on the shared contracts.

Each agent only calls tools through the ToolRegistry (so every action is
permission-gated), shares context via SharedMemoryService, and returns a
validated AgentOutput.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.base import AgentOutput, BaseAgent
from app.core.ai_provider import AIProvider
from app.core.enums import AgentType, TaskStatus
from app.core.tools import ToolRegistry
from app.models.core import Task
from app.schemas.advertising import (
    AD_COPY_PROMPT_HEADER,
    AdCopyVariant,
    CampaignObjective,
    CampaignPerformanceMetrics,
    ConversionAnalysis,
    CreativeBrief,
    MarketingStrategyReport,
    TargetAudienceSegment,
    build_conversion_analysis,
    diagnose_campaign_health,
    find_ungrounded_claims,
    grounding_tokens,
    recommend_budget,
)
from app.schemas.product_research import (
    MIN_VIABLE_MARGIN_PCT,
    CompetitorEntry,
    DataProvenance,
    GeneratedProductContent,
    MarginAnalysis,
    ProductOpportunityReport,
    SupplierOption,
    assess_opportunity,
    describe_provenance,
)
from app.services.memory import SharedMemoryService
from app.services.web_research import WebResearchService

#: Cross-agent handoff namespace. `AdvertisingAnalyticsAgent` reads this exact
#: namespace/key pair, so it must not change: the downstream marketing step
#: depends on `candidates` being present here.
RESEARCH_NAMESPACE = "product-research"
RESEARCH_KEY = "latest_candidates"

#: Archive of validated opportunity reports, served by
#: `GET /api/v1/research/reports`. Deliberately a *separate* namespace: the
#: handoff above stays a single record, and report history does not pollute it.
REPORT_NAMESPACE = "product_research_reports"

DISCOVERY_SOURCE = "mock_product_discovery adapter (offline, simulated candidates)"

#: Instructions for the AI copy step. Section 3 of the build spec forbids
#: inventing supplier prices, availability or market demand, so the prompt says
#: so explicitly instead of relying on the model's judgement.
CONTENT_PROMPT_HEADER = (
    "Write product listing copy for the product described by the facts below, "
    "using ONLY those facts.\n"
    "Hard rules:\n"
    "- Never invent supplier prices, stock availability, shipping times, ratings, "
    "reviews, sales volumes, market demand or statistics.\n"
    "- Never state a number that does not appear in the facts.\n"
    "- Facts labelled SIMULATED or ESTIMATED must be presented as approximate; "
    "never as a confirmed quote or verified market data.\n"
    "- If a fact is missing, leave it out instead of filling the gap.\n\n"
    "Facts (JSON):\n"
)


class AdPlan(BaseModel):
    headline: str
    body: str
    cta: str


def _gated_output(
    agent_type: AgentType, task: Task, tool_result: dict[str, Any], summary: str
) -> AgentOutput:
    return AgentOutput(
        status=TaskStatus(tool_result["task_status"]),
        agent_type=agent_type,
        summary=summary,
        data={"permission": tool_result},
        requires_approval=tool_result.get("approval_request_id") is not None,
        approval_request_id=tool_result.get("approval_request_id"),
    )


def _first_float(*values: Any) -> float | None:
    """First value that parses as a positive float, else ``None``."""
    for value in values:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _provenance_from_param(raw: Any, default: DataProvenance) -> DataProvenance:
    """Read a caller-declared provenance; missing/unrecognised keeps `default`.

    The default is never ``VERIFIED`` unless the caller says so explicitly.
    """
    if isinstance(raw, DataProvenance):
        return raw
    if raw is None:
        return default
    try:
        return DataProvenance(str(raw).strip().upper())
    except ValueError:
        return default


_CONTENT_FIELDS = (
    "product_titles",
    "product_description",
    "key_bullet_points",
    "marketing_angles",
)


def _grounding_tokens(facts: dict[str, Any]) -> set[str]:
    """Fact-derived strings that grounded copy is expected to contain.

    Only the product/supplier names and the supplied numbers count. Category is
    excluded on purpose: values like ``general`` are ordinary English words and
    would make almost any text look grounded.
    """
    tokens: set[str] = set()
    for key in ("product_name", "supplier_name"):
        value = facts.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip().lower()
        tokens.add(text)
        tokens.update(word for word in re.split(r"[^a-z0-9]+", text) if len(word) >= 4)
    for value in facts.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number <= 0:
            continue
        tokens.add(f"{number:.2f}")
        tokens.add(f"{number:.1f}")
        tokens.add(f"{number:.0f}")
        if number == int(number):
            tokens.add(str(int(number)))
    return tokens


def _ai_output_is_grounded(ai_content: GeneratedProductContent, facts: dict[str, Any]) -> bool:
    """True when the model's copy actually references the supplied facts.

    Copy that mentions neither the product nor a single supplied figure is not
    usable: accepting it would put words in the report that no fact supports,
    which is the failure mode this milestone exists to prevent. The stub output
    of the offline mock provider is the obvious case, but an off-brief response
    from a real model is caught the same way. The deterministic draft is
    grounded by construction, so falling back to it is always safe.
    """
    tokens = _grounding_tokens(facts)
    if not tokens:
        return False
    haystack = " ".join(
        [
            ai_content.product_description,
            *ai_content.product_titles,
            *ai_content.key_bullet_points,
            *ai_content.marketing_angles,
        ]
    ).lower()
    return any(token in haystack for token in tokens)


def _draft_content(draft: dict[str, Any], facts: dict[str, Any]) -> GeneratedProductContent:
    """The deterministic facts-only draft as a validated content model."""
    return GeneratedProductContent(
        product_titles=[str(item) for item in (draft.get("product_titles") or [])],
        product_description=str(draft.get("product_description") or ""),
        key_bullet_points=[str(item) for item in (draft.get("key_bullet_points") or [])],
        marketing_angles=[str(item) for item in (draft.get("marketing_angles") or [])],
        facts_used=facts,
    )


def _merge_content(
    draft: dict[str, Any], ai_content: GeneratedProductContent, facts: dict[str, Any]
) -> tuple[GeneratedProductContent, dict[str, str]]:
    """Prefer AI-written copy, fall back to the deterministic draft per field.

    ``facts_used`` always comes from the agent-assembled bundle and never from
    model output, so the receipt of what the copy was allowed to use cannot be
    rewritten by the model itself. The returned origin map records which fields
    came from the AI, so the offline path stays auditable.
    """
    if not _ai_output_is_grounded(ai_content, facts):
        origin = {field: "deterministic_draft" for field in _CONTENT_FIELDS}
        origin["ai_output"] = "rejected_ungrounded"
        origin["facts_used"] = "agent_assembled"
        return _draft_content(draft, facts), origin

    origin: dict[str, str] = {"ai_output": "accepted"}

    def pick_list(field: str, ai_value: list[str], draft_value: Any) -> list[str]:
        cleaned = [str(item).strip() for item in ai_value if str(item).strip()]
        origin[field] = "ai" if cleaned else "deterministic_draft"
        return cleaned or [str(item) for item in (draft_value or [])]

    def pick_text(field: str, ai_value: str, draft_value: Any) -> str:
        text = ai_value.strip()
        origin[field] = "ai" if text else "deterministic_draft"
        return text or str(draft_value or "")

    content = GeneratedProductContent(
        product_titles=pick_list(
            "product_titles", ai_content.product_titles, draft.get("product_titles")
        ),
        product_description=pick_text(
            "product_description",
            ai_content.product_description,
            draft.get("product_description"),
        ),
        key_bullet_points=pick_list(
            "key_bullet_points", ai_content.key_bullet_points, draft.get("key_bullet_points")
        ),
        marketing_angles=pick_list(
            "marketing_angles", ai_content.marketing_angles, draft.get("marketing_angles")
        ),
        facts_used=facts,
    )
    origin["facts_used"] = "agent_assembled"
    return content, origin


def _build_risks(
    *,
    margin: MarginAnalysis,
    best_supplier: SupplierOption,
    demand_provenance: DataProvenance,
    unpriced_urls: list[dict[str, Any]] | None = None,
    live_quotes_used: bool = False,
) -> list[str]:
    """Deterministic risk list: every caveat the provenance actually implies.

    ``unpriced_urls`` carries the pages that were fetched but declared no
    structured price. That is a *fact* about the research (the page was read and
    did not publish a price), so it belongs in the report rather than being
    silently dropped.
    """
    risks: list[str] = []

    if margin.cost_provenance is not DataProvenance.VERIFIED:
        risks.append(
            f"Supplier costs are {describe_provenance(margin.cost_provenance)}: confirm with a "
            f"live quote before committing inventory or ad spend."
        )
    if margin.price_provenance is not DataProvenance.VERIFIED:
        risks.append(
            f"Selling price is {describe_provenance(margin.price_provenance)}: validate against "
            f"live listings before pricing the offer."
        )
    if not margin.is_viable:
        risks.append(
            f"Estimated gross margin {margin.estimated_gross_margin_pct:.1f}% is below the "
            f"{MIN_VIABLE_MARGIN_PCT:.0f}% viability floor once ad spend, fees and refunds "
            f"are accounted for."
        )
    if best_supplier.in_stock is None:
        risks.append(
            "Stock availability was not confirmed by the supplier - verify before ordering."
        )
    elif best_supplier.in_stock is False:
        risks.append("The cheapest supplier reported the product out of stock.")
    if best_supplier.shipping_days_max == 0:
        risks.append("No shipping window was quoted, so delivery promises cannot be made.")
    elif best_supplier.shipping_days_max > 20:
        risks.append(
            f"Long shipping window ({best_supplier.shipping_days_min}-"
            f"{best_supplier.shipping_days_max} days) raises refund and chargeback risk."
        )
    if demand_provenance is not DataProvenance.VERIFIED:
        risks.append(
            "Demand signal is not from a verified source; no real search volume or sales "
            "data was consulted."
        )
    if unpriced_urls:
        listed = ", ".join(str(entry.get("url")) for entry in unpriced_urls[:3])
        risks.append(
            f"{len(unpriced_urls)} fetched page(s) declared no structured price and were "
            f"not used as a quote ({listed}). A page has to publish a price in schema "
            f"markup or a price meta tag before it counts as evidence."
        )
    if live_quotes_used:
        risks.append(
            "At least one cost is the price published on a live supplier page, not a "
            "wholesale quote: confirm per-unit and shipping terms with the supplier "
            "before ordering."
        )

    if not risks:
        risks.append(
            "No material risk detected from the supplied facts. Advertising cost, payment "
            "fees, returns and refunds are not modelled here."
        )
    return risks


def _url_list(raw: Any) -> list[str]:
    """Coerce a URL input into a de-duplicated list of trimmed strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return []
    ordered: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def live_service_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Carry an injected :class:`WebResearchService` into the live tools.

    The orchestrator runs agents through a shared ``ToolRegistry``, so the only
    channel from a caller to a tool handler is the parameter dict. Tests inject
    a service built on an ``httpx.MockTransport`` here; production passes
    nothing and the tool builds one from ``Settings``.
    """
    service = params.get("web_research_service")
    if isinstance(service, WebResearchService):
        return {"web_research_service": service}
    return {}


def _live_raw_snapshot(tool_result: dict[str, Any]) -> dict[str, Any]:
    """The parts of a live-tool result worth keeping in the report receipt."""
    keys = (
        "suppliers",
        "competitors",
        "pages_fetched",
        "verified_price_count",
        "unpriced_urls",
        "blocked_urls",
        "source",
        "note",
    )
    return {key: tool_result[key] for key in keys if key in tool_result}


def _live_product_urls(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-page records a live tool reported (suppliers and competitors)."""
    records: list[dict[str, Any]] = []
    for key in ("suppliers", "competitors", "observations"):
        for entry in tool_result.get(key) or []:
            if isinstance(entry, dict):
                records.append(entry)
    return records


def _merge_live_supplier_quotes(
    *,
    live_result: dict[str, Any],
    caller_quotes: list[Any],
    live_urls: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fold live extractions and caller quotes into one supplier-quote list.

    Two rules keep this honest:

    * A caller-supplied quote for a URL that was also fetched live is
      **replaced** by the page's own claim, not averaged with it. The page is
      the primary source; mixing an estimate into a verified price would make
      the blended figure unverifiable while still calling it VERIFIED.
    * A live page that yielded no structured price contributes **no quote**.
      It is returned separately in ``unpriced`` so the reason travels with the
      result instead of vanishing.

    Landing a page's retail price as a supplier cost is a real assumption, so
    every live quote is labelled with what was actually read
    (``quoted_from_live_page``) rather than being presented as a wholesale quote.
    """
    quotes: list[dict[str, Any]] = [
        dict(entry) for entry in (caller_quotes or []) if isinstance(entry, dict)
    ]
    fetched = {
        str(entry.get("source_reference") or "").strip()
        for entry in live_result.get("suppliers") or []
        if isinstance(entry, dict)
    }
    # Caller quotes that duplicate a fetched URL give way to the page itself.
    quotes = [
        quote
        for quote in quotes
        if str(quote.get("source_reference") or "").strip() not in fetched
    ]

    unpriced: list[dict[str, Any]] = []
    for entry in live_result.get("suppliers") or []:
        if not isinstance(entry, dict):
            continue
        price = _first_float(entry.get("unit_cost_usd"))
        source = str(entry.get("source_reference") or "").strip()
        if price is None:
            unpriced.append(
                {
                    "url": source or str(entry.get("url") or ""),
                    "reason": entry.get("escalation_reason")
                    or "the page declared no structured price",
                }
            )
            continue
        quotes.append(
            {
                "supplier_name": str(
                    entry.get("supplier_name") or entry.get("product_name") or source
                ),
                "unit_cost_usd": price,
                "shipping_cost_usd": 0.0,
                "shipping_days_min": 0,
                "shipping_days_max": 0,
                "in_stock": entry.get("in_stock"),
                "provenance": entry.get("provenance") or DataProvenance.ESTIMATED.value,
                "source_reference": source or "live product page",
                "read_from": "quoted_from_live_page",
                "currency": entry.get("currency") or "",
                "extraction_source": entry.get("extraction_source") or "",
            }
        )
    return quotes, unpriced


def _merge_live_competitor_observations(
    *,
    search_result: dict[str, Any],
    caller_observations: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fold live competitor observations and caller observations into one list.

    Observations are additive here (unlike supplier quotes): a caller's observed
    price and a fetched page's price can both be real, and the landscape tool
    prices off the median of everything observed. Live observations win only
    when they duplicate the same URL.
    """
    observations: list[dict[str, Any]] = [
        dict(entry) for entry in (caller_observations or []) if isinstance(entry, dict)
    ]
    live_urls = {
        str(entry.get("source_reference") or "").strip()
        for entry in search_result.get("observations") or []
        if isinstance(entry, dict)
    }
    observations = [
        entry
        for entry in observations
        if str(entry.get("source_reference") or "").strip() not in live_urls
    ]

    unpriced = list(search_result.get("unpriced_urls") or [])
    for entry in search_result.get("observations") or []:
        if not isinstance(entry, dict):
            continue
        price = _first_float(entry.get("observed_price_usd"))
        source = str(entry.get("source_reference") or "").strip()
        if price is None:
            continue
        observations.append(
            {
                "competitor_name": str(
                    entry.get("competitor_name") or entry.get("product_name") or source
                ),
                "observed_price_usd": price,
                "market_positioning": "",
                "provenance": entry.get("provenance") or DataProvenance.ESTIMATED.value,
                "source_reference": source or "live competitor page",
                "read_from": "observed_on_live_page",
                "currency": entry.get("currency") or "",
            }
        )
    return observations, unpriced


class ProductResearchAgent(BaseAgent):
    """Agent 1: product research, opportunity assessment and listing content.

    Runs the full pipeline through the ToolRegistry so every step stays
    permission-gated and audited:

    discovery -> supplier comparison -> competitor landscape -> margin
    calculation -> facts-driven AI content -> validated opportunity report.

    The report never presents simulated numbers as verified ones: provenance
    travels with every figure, confidence is capped when costs are unverified,
    and the agent fails the task rather than invent a price or a supplier cost
    it could not source.
    """

    agent_type = AgentType.PRODUCT_RESEARCH
    description = (
        "Researches products, compares suppliers and competitors, validates margins "
        "and drafts listing content with sourced provenance."
    )

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})

        # ── 1. Discovery ───────────────────────────────────────────────────
        discovery = tool_registry.execute_tool(
            db,
            tool_name="mock_product_discovery",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters=params,
            context_summary=task.objective,
        )
        gated = self._gate(task, discovery, "Research gated")
        if gated is not None:
            return gated

        discovery_result: dict[str, Any] = discovery["result"]
        candidates = list(discovery_result.get("candidates") or [])
        niche = str(discovery_result.get("niche") or params.get("niche") or "general")

        # An explicit product_name wins; otherwise take the top-ranked candidate.
        requested = str(params.get("product_name") or "").strip()
        chosen: dict[str, Any] | None = None
        if requested:
            chosen = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.get("name", "")).strip().lower() == requested.lower()
                ),
                None,
            )
        if chosen is None and candidates:
            chosen = dict(candidates[0])
        product_name = requested or str((chosen or {}).get("name") or "unspecified product")
        product_category = str(params.get("product_category") or niche or "general")

        reference_price = _first_float(
            params.get("reference_price_usd"),
            (chosen or {}).get("est_price_usd"),
            params.get("target_selling_price_usd"),
        )

        # ── 1b. Live web extraction (only when URLs were supplied) ─────────
        # No URLs means no request: the offline path is unchanged, so nothing
        # in this milestone silently starts touching the network.
        supplier_urls = _url_list(params.get("supplier_urls"))
        competitor_urls = _url_list(params.get("competitor_urls"))
        live_result: dict[str, Any] = {}
        search_result: dict[str, Any] = {}
        live_raw: dict[str, Any] = {}
        if supplier_urls:
            live_call = tool_registry.execute_tool(
                db,
                tool_name="fetch_live_product_page",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={
                    "supplier_urls": supplier_urls,
                    "competitor_urls": competitor_urls,
                    **live_service_parameters(params),
                },
                context_summary=task.objective,
            )
            gated = self._gate(task, live_call, "Live product page fetch gated")
            if gated is not None:
                return gated
            live_result = dict(live_call["result"] or {})
            live_raw = _live_raw_snapshot(live_result)
        if competitor_urls or params.get("search_competitors"):
            search_call = tool_registry.execute_tool(
                db,
                tool_name="search_live_competitors",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={
                    "query": params.get("search_query") or product_name,
                    "competitor_urls": competitor_urls,
                    "max_results": params.get("max_results") or 5,
                    **live_service_parameters(params),
                },
                context_summary=task.objective,
            )
            gated = self._gate(task, search_call, "Live competitor search gated")
            if gated is not None:
                return gated
            search_result = dict(search_call["result"] or {})
            if not live_result:
                # The search tool reads the competitor pages itself; keep what
                # it saw available for the report receipt.
                live_raw = {}

        supplier_quotes, unpriced_supplier_urls = _merge_live_supplier_quotes(
            live_result=live_result,
            caller_quotes=list(params.get("suppliers") or []),
            live_urls=supplier_urls,
        )
        competitor_observations, unpriced_competitor_urls = (
            _merge_live_competitor_observations(
                search_result=search_result,
                caller_observations=list(params.get("competitors") or []),
            )
        )

        # A caller who supplied supplier URLs asked for *those* pages. If none
        # of them published a structured price, falling back to the simulated
        # tiers would answer a different question than the one asked - and would
        # report simulated costs as if they described the links. Refuse instead.
        live_supplier_quotes = [
            quote
            for quote in supplier_quotes
            if quote.get("read_from") == "quoted_from_live_page"
        ]
        if supplier_urls and not live_supplier_quotes and not params.get("suppliers"):
            unpriced = ", ".join(
                str(entry.get("url")) for entry in unpriced_supplier_urls[:3]
            ) or ", ".join(supplier_urls[:3])
            return self._refusal(
                f"None of the supplied supplier URLs declared a price in structured data "
                f"({unpriced}), so no landed cost could be read from them. Refusing to "
                f"substitute a simulated cost for pages that were actually requested: "
                f"point the task at a product page that publishes schema.org/OpenGraph "
                f"pricing, or supply the quote explicitly.",
                discovery_result,
            )

        # ── 2. Supplier comparison (landed cost) ───────────────────────────
        supplier_call = tool_registry.execute_tool(
            db,
            tool_name="compare_supplier_options",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={
                "suppliers": supplier_quotes,
                "reference_price_usd": reference_price,
            },
            context_summary=task.objective,
        )
        gated = self._gate(task, supplier_call, "Supplier comparison gated")
        if gated is not None:
            return gated
        supplier_result: dict[str, Any] = supplier_call["result"]
        suppliers = [
            SupplierOption.model_validate(option) for option in supplier_result.get("options") or []
        ]

        # ── 3. Competitor landscape ────────────────────────────────────────
        competitor_call = tool_registry.execute_tool(
            db,
            tool_name="analyze_competitor_landscape",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={
                "competitors": competitor_observations,
                "reference_price_usd": reference_price,
            },
            context_summary=task.objective,
        )
        gated = self._gate(task, competitor_call, "Competitor analysis gated")
        if gated is not None:
            return gated
        competitor_result: dict[str, Any] = competitor_call["result"]
        competitors = [
            CompetitorEntry.model_validate(entry)
            for entry in competitor_result.get("competitors") or []
        ]

        # ── 4. Selling price: explicit target beats observed median ────────
        explicit_price = _first_float(params.get("target_selling_price_usd"))
        observed_median = _first_float(competitor_result.get("recommended_selling_price_usd"))
        if explicit_price is not None:
            selling_price = round(explicit_price, 2)
            price_provenance = _provenance_from_param(
                params.get("price_provenance"), DataProvenance.ESTIMATED
            )
        elif observed_median is not None:
            selling_price = observed_median
            price_provenance = _provenance_from_param(
                competitor_result.get("price_provenance"), DataProvenance.SIMULATED
            )
        else:
            selling_price = 0.0
            price_provenance = DataProvenance.SIMULATED

        # Refuse to invent the two numbers the whole verdict rests on.
        if not suppliers:
            return self._refusal(
                "No supplier option could be sourced: no live quotes were supplied and no "
                "reference price was available to simulate against. Refusing to assume a "
                "zero landed cost, which would fabricate a 100% margin.",
                discovery_result,
            )
        if selling_price <= 0:
            return self._refusal(
                "No selling price could be sourced: discovery returned no candidate price, "
                "no competitor observations were supplied and no target price was given. "
                "Refusing to invent one.",
                discovery_result,
            )

        best = min(suppliers, key=lambda option: option.total_landed_cost_usd)
        margin = MarginAnalysis.compute(
            unit_cost_usd=best.unit_cost_usd,
            shipping_cost_usd=best.shipping_cost_usd,
            recommended_selling_price_usd=selling_price,
            cost_provenance=best.provenance,
            price_provenance=price_provenance,
        )

        # The mock calculator is retained so the `calculate_margins` action keeps
        # its permission gate and audit trail, and so its recorded inputs carry
        # the real figures. Its hardcoded `avg_margin_pct` is a placeholder: the
        # authoritative margin is the validated MarginAnalysis above.
        margins = tool_registry.execute_tool(
            db,
            tool_name="mock_margin_calculator",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={
                "product": product_name,
                "unit_cost_usd": margin.unit_cost_usd,
                "shipping_cost_usd": margin.shipping_cost_usd,
                "total_cost_usd": margin.total_cost_usd,
                "selling_price_usd": margin.recommended_selling_price_usd,
                "gross_margin_pct": margin.estimated_gross_margin_pct,
                "cost_provenance": margin.cost_provenance.value,
                "price_provenance": margin.price_provenance.value,
            },
            context_summary=task.objective,
        )
        gated = self._gate(task, margins, "Margin check gated")
        if gated is not None:
            return gated

        # ── 5. Deterministic verdict ───────────────────────────────────────
        confidence, recommendation = assess_opportunity(
            gross_margin_pct=margin.estimated_gross_margin_pct,
            cost_provenance=margin.cost_provenance,
        )

        demand_provenance = _provenance_from_param(
            params.get("demand_provenance"), DataProvenance.SIMULATED
        )
        explicit_demand = params.get("demand_summary")
        if explicit_demand:
            demand_summary = str(explicit_demand)
        else:
            trend = (chosen or {}).get("trend_score")
            if trend is not None:
                demand_summary = (
                    f"The offline discovery adapter ranked {product_name} in the {niche!r} "
                    f"niche with a trend score of {trend}. No live search volume, sales or "
                    f"marketplace demand data was consulted, so this is not evidence of real "
                    f"demand."
                )
            else:
                demand_summary = (
                    "No demand signal was available: the discovery adapter returned no trend "
                    "score and no demand facts were supplied."
                )

        sources = [
            DISCOVERY_SOURCE,
            str(supplier_result.get("source") or ""),
            str(competitor_result.get("source") or ""),
        ]
        if live_result:
            sources.append(str(live_result.get("source") or ""))
        if search_result:
            sources.append(str(search_result.get("source") or ""))
        # The exact URLs a figure came from - that is what makes a VERIFIED
        # provenance auditable after the fact.
        sources.extend(
            option.source_reference
            for option in suppliers
            if option.provenance is DataProvenance.VERIFIED and option.source_reference
        )
        sources.extend(
            entry.source_reference
            for entry in competitors
            if entry.provenance is DataProvenance.VERIFIED and entry.source_reference
        )
        sources.extend(str(item) for item in (params.get("research_sources") or []))
        research_sources = list(dict.fromkeys(source for source in sources if source))

        risks = _build_risks(
            margin=margin,
            best_supplier=best,
            demand_provenance=demand_provenance,
            unpriced_urls=unpriced_supplier_urls + unpriced_competitor_urls,
            live_quotes_used=any(
                quote.get("read_from") == "quoted_from_live_page"
                for quote in supplier_quotes
            ),
        )

        # ── 6. Content: deterministic draft, then facts-only AI rewrite ────
        facts: dict[str, Any] = {
            "product_name": product_name,
            "product_category": product_category,
            "supplier_name": best.supplier_name,
            "supplier_options_compared": len(suppliers),
            "unit_cost_usd": margin.unit_cost_usd,
            "shipping_cost_usd": margin.shipping_cost_usd,
            "total_landed_cost_usd": margin.total_cost_usd,
            "recommended_selling_price_usd": margin.recommended_selling_price_usd,
            "estimated_gross_profit_usd": margin.estimated_gross_profit_usd,
            "estimated_gross_margin_pct": margin.estimated_gross_margin_pct,
            "shipping_days_min": best.shipping_days_min,
            "shipping_days_max": best.shipping_days_max,
            "in_stock": best.in_stock,
            "cost_provenance": margin.cost_provenance.value,
            "price_provenance": margin.price_provenance.value,
            "competitor_price_floor_usd": competitor_result.get("observed_price_floor_usd"),
            "competitor_price_ceiling_usd": competitor_result.get("observed_price_ceiling_usd"),
            "demand_provenance": demand_provenance.value,
        }

        draft_call = tool_registry.execute_tool(
            db,
            tool_name="generate_product_content",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"facts": facts, "product_name": product_name},
            context_summary=task.objective,
        )
        gated = self._gate(task, draft_call, "Content draft gated")
        if gated is not None:
            return gated
        draft: dict[str, Any] = draft_call["result"]

        # The prompt carries only sourced facts, and states the fabrication
        # rules explicitly - see CONTENT_PROMPT_HEADER. Provider errors are
        # deliberately NOT swallowed: AIRateLimitError has to reach the
        # orchestrator so it can fail fast without burning more quota.
        prompt = CONTENT_PROMPT_HEADER + json.dumps(facts, sort_keys=True, default=str)
        ai_content = GeneratedProductContent.model_validate(
            ai_provider.generate_structured(
                prompt, GeneratedProductContent, context={"product_facts": facts}
            ).model_dump()
        )
        content, content_origin = _merge_content(draft, ai_content, facts)

        # ── 7. Validate, persist, report ───────────────────────────────────
        live_receipt: dict[str, Any] = {}
        if supplier_urls or competitor_urls or search_result:
            live_receipt = {
                "supplier_urls": supplier_urls,
                "competitor_urls": competitor_urls,
                "pages": live_raw,
                "search": {
                    key: search_result[key]
                    for key in (
                        "search_provider",
                        "search_hits",
                        "search_error",
                        "observed_count",
                    )
                    if key in search_result
                },
                "unpriced_urls": unpriced_supplier_urls + unpriced_competitor_urls,
                "verified_price_count": int(live_result.get("verified_price_count") or 0),
                "blocked_urls": list(live_result.get("blocked_urls") or []),
            }

        report = ProductOpportunityReport(
            product_name=product_name,
            product_category=product_category,
            research_sources=research_sources,
            demand_summary=demand_summary,
            demand_provenance=demand_provenance,
            suppliers=suppliers,
            competitors=competitors,
            margin_analysis=margin,
            risks=risks,
            research_confidence=confidence,
            recommendation=recommendation,
            generated_content=content,
            live_research=live_receipt,
        )
        # Re-validate through a JSON round trip: that is how the report is
        # stored and served, so it must survive it unchanged.
        report = ProductOpportunityReport.model_validate(report.model_dump(mode="json"))

        # Handoff for AdvertisingAnalyticsAgent: same namespace and key as
        # before, and `candidates` stays present so downstream keeps working.
        memory.set_memory(
            db,
            namespace=RESEARCH_NAMESPACE,
            key=RESEARCH_KEY,
            value={
                "candidates": candidates,
                "margins": margins["result"],
                "source_task_id": task.id,
                "product_name": product_name,
                "recommendation": report.recommendation.value,
                "research_confidence": report.research_confidence.value,
            },
            agent_type=self.agent_type,
        )
        # Archive the report for GET /api/v1/research/reports, keyed by task so
        # every run is retrievable and traceable to its audit trail.
        memory.set_memory(
            db,
            namespace=REPORT_NAMESPACE,
            key=task.id,
            value={
                "report": report.model_dump(mode="json"),
                "product_name": report.product_name,
                "product_category": report.product_category,
                "recommendation": report.recommendation.value,
                "research_confidence": report.research_confidence.value,
                "estimated_gross_margin_pct": report.margin_analysis.estimated_gross_margin_pct,
                "cost_provenance": report.margin_analysis.cost_provenance.value,
                "source_task_id": task.id,
            },
            agent_type=self.agent_type,
        )

        summary = (
            f"{product_name}: {margin.estimated_gross_margin_pct:.1f}% estimated gross margin "
            f"on a ${margin.recommended_selling_price_usd:.2f} price against "
            f"${margin.total_cost_usd:.2f} landed cost "
            f"({describe_provenance(margin.cost_provenance)} costs) -> "
            f"{recommendation.value} at {confidence.value} confidence."
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data={
                "discovery": discovery_result,
                "margins": margins["result"],
                "opportunity_report": report.model_dump(mode="json"),
                "content_origin": content_origin,
                # Same receipt the report carries, so the task payload and the
                # report can never disagree about what was read.
                "live_research": live_receipt,
            },
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _gate(
        self, task: Task, tool_result: dict[str, Any], summary: str
    ) -> AgentOutput | None:
        """Return a gated output when a tool did not execute, else ``None``."""
        if tool_result["executed"]:
            return None
        return _gated_output(self.agent_type, task, tool_result, summary)

    def _refusal(self, reason: str, discovery_result: dict[str, Any]) -> AgentOutput:
        """Fail loudly instead of fabricating the missing input."""
        return AgentOutput(
            status=TaskStatus.FAILED,
            agent_type=self.agent_type,
            summary=reason,
            data={"discovery": discovery_result, "refused": True, "reason": reason},
        )


#: Cross-agent handoff namespace for Agent 2. The existing contract: Agent 2
#: publishes its latest strategy here. The name is hyphenated to match
#: `product-research` and must not change.
ADVERTISING_NAMESPACE = "advertising-analytics"
ADVERTISING_KEY = "latest_campaign_plan"

#: Archive of validated strategy reports, served by
#: `GET /api/v1/advertising/reports`. A separate namespace, exactly like Agent
#: 1's `product_research_reports`, so report history never pollutes the single
#: handoff record above.
AD_REPORT_NAMESPACE = "marketing_reports"


class _AiAdPlan(BaseModel):
    """Schema handed to the AI for the ad-copy step.

    ``headline``/``body``/``cta`` are the shape the ad-copy step has used since
    Milestone 2A, so provider stubs written against it keep working. The extra
    fields are additive and optional.

    Deliberately narrow: there is no field for a discount, a rating or a review
    count, so a model that tries to invent one has nowhere to put it. The
    grounding gate still checks the text it does return.
    """

    headline: str
    body: str = ""
    cta: str = ""
    hook: str = ""
    ad_copy_angles: list[str] = []
    marketing_strategy: str = ""

    @property
    def primary_text(self) -> str:
        """The main copy line: ``body``, falling back to ``hook``."""
        return self.body.strip() or self.hook.strip()


#: Provenance values a *fact bundle* may legitimately carry as text. Excluded
#: from grounding tokens so "SIMULATED" never counts as citing a product fact.
_PROVENANCE_WORDS = frozenset({"simulated", "estimated", "verified"})


def _campaign_objective(raw: Any) -> CampaignObjective:
    """Resolve a campaign objective; anything unrecognised is CONVERSIONS."""
    try:
        return CampaignObjective(str(raw).strip().upper())
    except (ValueError, AttributeError):
        return CampaignObjective.CONVERSIONS


def _assemble_product_facts(
    params: dict[str, Any], handoff: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    """Merge task input and Agent 1's research into one auditable facts bundle.

    Precedence: explicit task input beats the archived research report, which
    beats the lightweight handoff. Nothing is invented - a key that no source
    supplied is simply absent, and every caller downstream treats a missing key
    as unknown rather than as zero.

    Unit economics are derived here (never guessed): landed cost from the report's
    margin analysis when the caller did not supply one, and gross profit per unit
    from whichever price/cost pair is actually known. Those two numbers are what
    make a contribution `break_even_roas` possible.
    """
    facts: dict[str, Any] = {}

    # ── lowest precedence: the handoff record ─────────────────────────────
    if handoff.get("product_name"):
        facts["product_name"] = str(handoff["product_name"])
    margins = handoff.get("margins")
    if isinstance(margins, dict):
        for key in (
            "unit_cost_usd",
            "shipping_cost_usd",
            "total_cost_usd",
            "selling_price_usd",
            "gross_margin_pct",
        ):
            if margins.get(key) is not None:
                facts[f"handoff_{key}"] = margins[key]

    # ── middle precedence: Agent 1's validated report ─────────────────────
    if report:
        facts.setdefault("product_name", report.get("product_name"))
        if report.get("product_category"):
            facts["product_category"] = report["product_category"]
        margin_analysis = report.get("margin_analysis") or {}
        if isinstance(margin_analysis, dict) and margin_analysis:
            mapping = {
                "unit_cost_usd": "unit_cost_usd",
                "shipping_cost_usd": "shipping_cost_usd",
                "total_cost_usd": "total_landed_cost_usd",
                "recommended_selling_price_usd": "recommended_selling_price_usd",
                "estimated_gross_profit_usd": "estimated_gross_profit_usd",
                "estimated_gross_margin_pct": "estimated_gross_margin_pct",
                "cost_provenance": "cost_provenance",
                "price_provenance": "price_provenance",
            }
            for source_key, fact_key in mapping.items():
                if margin_analysis.get(source_key) is not None:
                    facts[fact_key] = margin_analysis[source_key]
        best = (report.get("suppliers") or [])
        if best:
            cheapest = min(
                (option for option in best if isinstance(option, dict)),
                key=lambda option: float(option.get("total_landed_cost_usd") or 0.0),
                default=None,
            )
            if cheapest is not None:
                facts.setdefault("supplier_name", cheapest.get("supplier_name"))
                if facts.get("shipping_days_min") is None:
                    facts["shipping_days_min"] = cheapest.get("shipping_days_min")
                    facts["shipping_days_max"] = cheapest.get("shipping_days_max")
                if facts.get("in_stock") is None:
                    facts["in_stock"] = cheapest.get("in_stock")
        if report.get("recommendation"):
            facts["research_recommendation"] = report["recommendation"]
        if report.get("research_confidence"):
            facts["research_confidence"] = report["research_confidence"]
        generated = report.get("generated_content") or {}
        if isinstance(generated, dict) and generated.get("marketing_angles"):
            facts["research_marketing_angles"] = list(generated["marketing_angles"])
        demand_provenance = report.get("demand_provenance")
        if demand_provenance:
            facts["demand_provenance"] = demand_provenance

    # ── highest precedence: what the caller supplied explicitly ───────────
    for key in (
        "product",
        "product_name",
        "product_category",
        "price_usd",
        "benefits",
        "target_audience",
        "age_range",
        "interests",
        "pain_points",
        "platforms",
        "platform",
        "tone",
        "shipping_days_min",
        "shipping_days_max",
        "daily_budget_usd",
        "test_duration_days",
        "campaign_objective",
        "price_provenance",
        "cost_provenance",
        "in_stock",
        "supplier_name",
    ):
        if params.get(key) is not None:
            facts[key] = params[key]

    selling_price = _first_float(
        params.get("target_selling_price_usd"),
        params.get("price_usd"),
        facts.get("recommended_selling_price_usd"),
        (margins or {}).get("selling_price_usd") if isinstance(margins, dict) else None,
    )
    if selling_price is not None:
        facts["recommended_selling_price_usd"] = round(selling_price, 2)

    landed_cost = _first_float(
        params.get("landed_cost_usd"),
        facts.get("total_landed_cost_usd"),
    )
    unit_cost = _first_float(params.get("unit_cost_usd"), facts.get("unit_cost_usd"))
    shipping_cost = _first_float(
        params.get("shipping_cost_usd"), facts.get("shipping_cost_usd")
    )
    if landed_cost is None and unit_cost is not None:
        landed_cost = round(unit_cost + (shipping_cost or 0.0), 2)
    if landed_cost is not None:
        if unit_cost is not None:
            facts["unit_cost_usd"] = round(unit_cost, 2)
        if shipping_cost is not None:
            facts["shipping_cost_usd"] = round(shipping_cost, 2)
        facts["total_landed_cost_usd"] = round(landed_cost, 2)

    # Gross profit per unit: only from a real price/cost pair, never assumed.
    #
    # Precedence matters here. The archived report carries its own gross profit
    # computed from *its* price and cost; if the caller re-priced the product
    # (or supplied a landed cost of their own), that archived figure no longer
    # describes this campaign, so it is discarded rather than averaged in.
    caller_priced = any(
        params.get(key) is not None
        for key in (
            "target_selling_price_usd",
            "price_usd",
            "landed_cost_usd",
            "unit_cost_usd",
            "shipping_cost_usd",
        )
    )
    gross_profit = _first_float(params.get("gross_profit_per_unit_usd"))
    if gross_profit is None and not caller_priced:
        gross_profit = _first_float(facts.get("estimated_gross_profit_usd"))
    if gross_profit is None:
        selling_price_fact = facts.get("recommended_selling_price_usd")
        if selling_price_fact is not None and landed_cost is not None:
            gross_profit = round(float(selling_price_fact) - float(landed_cost), 2)
    if gross_profit is not None:
        facts["gross_profit_per_unit_usd"] = round(gross_profit, 2)

    if facts.get("product_name") is None:
        facts.pop("product_name", None)
    return facts


def _gross_profit_per_unit(facts: dict[str, Any]) -> float | None:
    """Contribution margin per unit, or ``None`` when it is genuinely unknown."""
    value = _first_float(facts.get("gross_profit_per_unit_usd"))
    if value is not None:
        return value
    price = _first_float(facts.get("recommended_selling_price_usd"))
    cost = _first_float(facts.get("total_landed_cost_usd"))
    if price is None or cost is None:
        return None
    return round(price - cost, 2)


def _platforms(
    facts: dict[str, Any], audiences: list[TargetAudienceSegment]
) -> list[str]:
    """Explicit platforms first, then the audience's, then the defaults."""
    explicit = facts.get("platforms") or facts.get("platform")
    if isinstance(explicit, str):
        explicit = [explicit]
    found = [str(item).strip() for item in (explicit or []) if str(item).strip()]
    for audience in audiences:
        found.extend(audience.platforms)
    return list(dict.fromkeys(name for name in found if name))


def _metrics_from(params: dict[str, Any]) -> dict[str, Any]:
    """Raw campaign metrics from the task payload, or ``{}`` when pre-launch.

    All five raw numbers must be present to be treated as a measurement:
    a partial row (say impressions only) is *not* a campaign result, and
    analysing it would mean inventing the missing half.
    """
    keys = ("impressions", "clicks", "spend_usd", "conversions", "revenue_usd")
    present = [key for key in keys if params.get(key) is not None]
    if not present:
        return {}
    if len(present) != len(keys):
        # Fill the gaps with zero rather than refusing: a caller that reports
        # clicks but no revenue has told us revenue was zero, and the analysis
        # says which stage that leaves the funnel in.
        return {key: params.get(key) if params.get(key) is not None else 0 for key in keys} | {
            "provenance": params.get("metrics_provenance") or DataProvenance.SIMULATED.value
        }
    return {key: params[key] for key in keys} | {
        "provenance": _provenance_from_param(
            params.get("metrics_provenance"), DataProvenance.SIMULATED
        ).value
    }


def _draft_variants(
    *,
    product_facts: dict[str, Any],
    product_name: str,
    briefs_result: dict[str, Any],
    conversion_analysis: ConversionAnalysis,
    legacy_ad_plan: dict[str, Any],
) -> list[AdCopyVariant]:
    """Deterministic, grounded ad copy - the baseline the AI must beat.

    One variant per supplied benefit, plus a price variant when a price is
    known. Every sentence restates a fact and names it in ``facts_referenced``.
    Nothing here can mention a discount, a review or a rating: there is no
    code path that would produce one.
    """
    benefits = [
        str(item).strip()
        for item in (product_facts.get("benefits") or [])
        if str(item).strip()
    ]
    price = _first_float(product_facts.get("recommended_selling_price_usd"))
    cost_provenance = _provenance_from_param(
        product_facts.get("cost_provenance"), DataProvenance.SIMULATED
    )
    price_provenance = _provenance_from_param(
        product_facts.get("price_provenance"), DataProvenance.SIMULATED
    )
    days_min = product_facts.get("shipping_days_min") or 0
    days_max = product_facts.get("shipping_days_max") or 0
    # The AI's call to action when the provider supplied one, else the mock
    # tool's placeholder, else a neutral default. A CTA carries no product
    # claim, so reusing it cannot smuggle an invented fact into the copy.
    cta = (
        str(product_facts.get("call_to_action") or "").strip()
        or str(legacy_ad_plan.get("cta") or "").strip()
        or "Shop now"
    )

    variants: list[AdCopyVariant] = []
    for index, benefit in enumerate(benefits[:3], start=1):
        variants.append(
            AdCopyVariant(
                angle_name=f"Benefit angle {index}",
                hook=benefit,
                primary_text=(
                    f"{product_name}: {benefit}. Every claim here comes from the supplied "
                    f"product facts; no rating, review or discount is claimed."
                ),
                headline=benefit[:90],
                call_to_action=cta,
                facts_referenced=["product_name", "benefits"],
            )
        )

    if price is not None:
        shipping_note = ""
        if days_min and days_max:
            shipping_note = f" Quoted shipping window: {days_min}-{days_max} days."
        variants.append(
            AdCopyVariant(
                angle_name="Price clarity angle",
                hook=f"{product_name} at ${price:.2f}",
                primary_text=(
                    f"{product_name} at a suggested price of ${price:.2f} "
                    f"({describe_provenance(price_provenance)}).{shipping_note} "
                    f"Costs are {describe_provenance(cost_provenance)}, so margins are "
                    f"not presented as confirmed."
                ),
                headline=f"{product_name} - ${price:.2f}",
                call_to_action=cta,
                facts_referenced=[
                    "product_name",
                    "recommended_selling_price_usd",
                    "price_provenance",
                    "cost_provenance",
                ]
                + (["shipping_days_min", "shipping_days_max"] if shipping_note else []),
            )
        )

    if not variants:
        # Nothing but the name was supplied: say so instead of inventing an angle.
        variants.append(
            AdCopyVariant(
                angle_name="Product-led angle",
                hook=f"Meet {product_name}",
                primary_text=(
                    f"{product_name}. No benefit, price or shipping claim was supplied, so "
                    f"this draft states only the product name. "
                    f"Current health: {conversion_analysis.health_status.value}."
                ),
                headline=product_name[:90],
                call_to_action=cta,
                facts_referenced=["product_name"],
            )
        )
    return variants


def _draft_strategy(
    *,
    product_facts: dict[str, Any],
    product_name: str,
    campaign_objective: CampaignObjective,
    conversion_analysis: ConversionAnalysis,
) -> str:
    """Deterministic strategy paragraph, built from facts and measured rules."""
    parts = [
        f"Objective: {campaign_objective.value} for {product_name}.",
        f"Current diagnosis: {conversion_analysis.health_status.value} "
        f"(bottleneck: {conversion_analysis.bottleneck_stage}).",
    ]
    break_even = None
    price = _first_float(product_facts.get("recommended_selling_price_usd"))
    gross_profit = _gross_profit_per_unit(product_facts)
    if price is not None and gross_profit is not None and gross_profit > 0:
        break_even = round(price / gross_profit, 2)
        parts.append(
            f"Contribution break-even ROAS is {break_even:.2f} "
            f"(${price:.2f} price / ${gross_profit:.2f} gross profit per unit), before "
            f"any fixed overhead."
        )
    else:
        parts.append(
            "Break-even ROAS is unknown because the unit economics were not supplied, so "
            "no profitability claim is made and the fallback ROAS floor applies."
        )
    parts.append(
        "Plan: test the grounded copy variants against one audience, keep spend inside "
        "the recommended daily budget, and judge the result with the deterministic "
        "diagnosis above rather than intuition. Committing budget or publishing live "
        "requires human approval."
    )
    return " ".join(parts)


def _references_a_fact(plan: _AiAdPlan, facts: dict[str, Any]) -> bool:
    """True when the AI output cites at least one supplied fact.

    Copy that mentions neither the product nor a single supplied figure is not
    usable: accepting it would put words in the report that no fact supports.
    The mock provider's stub output is the obvious case; an off-brief real model
    is caught the same way.
    """
    tokens = grounding_tokens(facts) - _PROVENANCE_WORDS
    if not tokens:
        return False
    haystack = " ".join(
        [
            plan.headline,
            plan.hook,
            plan.primary_text,
            plan.cta,
            plan.marketing_strategy,
            *plan.ad_copy_angles,
        ]
    ).lower()
    return any(token in haystack for token in tokens)


def _variants_from_ai(
    plan: _AiAdPlan, facts: dict[str, Any], draft: list[AdCopyVariant]
) -> list[AdCopyVariant]:
    """Wrap accepted AI copy as variants, keeping the draft's fact receipts.

    A single accepted plan becomes one variant; the deterministic drafts are kept
    behind it so a report never becomes emptier than the offline path.
    """
    variants = [
        AdCopyVariant(
            angle_name="AI angle 1",
            hook=plan.hook.strip() or draft[0].hook,
            primary_text=plan.primary_text or draft[0].primary_text,
            headline=plan.headline.strip() or draft[0].headline,
            call_to_action=plan.cta.strip() or draft[0].call_to_action,
            facts_referenced=["product_name"],
        )
    ]
    for extra in plan.ad_copy_angles[:2]:
        text = str(extra).strip()
        if not text:
            continue
        variants.append(
            AdCopyVariant(
                angle_name=f"AI angle {len(variants) + 1}",
                hook=text[:120],
                primary_text=text,
                headline=plan.headline.strip() or draft[0].headline,
                call_to_action=plan.cta.strip() or draft[0].call_to_action,
                facts_referenced=["product_name"],
            )
        )
    return variants + draft


class AdvertisingAnalyticsAgent(BaseAgent):
    """Agent 2: advertising strategy, creative briefs, campaign analysis.

    Runs every step through the ToolRegistry so it stays permission-gated and
    audited, and it is split into two clear halves:

    **Level 1 (analytical, runs immediately)**

    ``mock_ad_copy_generator`` -> ``prepare_creative_briefs`` ->
    ``analyze_campaign_performance`` -> validated
    :class:`MarketingStrategyReport`.

    Producing that report spends nothing: the budget inside it is a
    recommendation carrying ``requires_human_approval=True``.

    **Level 2 (money, gated)**

    ``publish_live`` -> ``mock_publish_ad_campaign`` and ``commit_budget`` ->
    ``allocate_ad_budget``. Requesting either pauses the task in
    ``WAITING_APPROVAL`` and the handler never runs until a human approves it -
    so a report can never be a back door to spend.

    Grounding: product facts come from the task payload *and* from Agent 1's
    ``product-research`` handoff / ``product_research_reports`` archive, which is
    where the landed cost and gross profit that make ``break_even_roas``
    possible come from. AI-written copy is accepted only after it passes
    :func:`app.schemas.advertising.find_ungrounded_claims`; otherwise the
    deterministic grounded draft is used and the reason is recorded in
    ``content_origin``.
    """

    agent_type = AgentType.ADVERTISING_ANALYTICS
    description = (
        "Plans advertising: audience segments, ad copy, creative briefs, deterministic "
        "campaign analysis and a budget recommendation; publishes or commits budget "
        "only with human approval."
    )

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})

        # ── 1. Product facts: task input first, then Agent 1's handoff ─────
        research = memory.get_memory(db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY)
        handoff: dict[str, Any] = dict(research.value or {}) if research else {}
        candidates = list(handoff.get("candidates") or [])

        report = self._research_report(db, memory, params)
        product_facts = _assemble_product_facts(params, handoff, report)

        # Precedence: what the caller named explicitly, then Agent 1's
        # top-ranked discovered candidate (the 2A contract), then the name the
        # research report/ handoff carries. All three are sourced - nothing is
        # invented - and the bundle is kept in sync with the winner so the
        # grounding tokens always describe the product being advertised.
        product_name = str(
            (str(params.get("product_name") or "").strip() or None)
            or (str(params.get("product") or "").strip() or None)
            or (candidates[0].get("name") if candidates else None)
            or product_facts.get("product_name")
            or "the product"
        )
        if product_name != "the product":
            product_facts["product_name"] = product_name

        objective_raw = product_facts.get("campaign_objective")
        campaign_objective = _campaign_objective(objective_raw)

        # ── 2. Level 1: ad copy (deterministic mock, kept for compat) ──────
        ad_copy = tool_registry.execute_tool(
            db,
            tool_name="mock_ad_copy_generator",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"product": product_name},
            context_summary=task.objective,
        )
        gated = self._gate(task, ad_copy, "Ad copy gated")
        if gated is not None:
            return gated

        # ── 3. Level 1: creative briefs + audience segments ───────────────
        briefs_call = tool_registry.execute_tool(
            db,
            tool_name="prepare_creative_briefs",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={
                "facts": product_facts,
                "product_name": product_name,
                "benefits": product_facts.get("benefits"),
                "platforms": product_facts.get("platforms"),
                "target_audience": product_facts.get("target_audience"),
                "age_range": product_facts.get("age_range"),
                "interests": product_facts.get("interests"),
                "pain_points": product_facts.get("pain_points"),
            },
            context_summary=task.objective,
        )
        gated = self._gate(task, briefs_call, "Creative briefs gated")
        if gated is not None:
            return gated
        briefs_result: dict[str, Any] = briefs_call["result"]
        target_audiences = [
            TargetAudienceSegment.model_validate(segment)
            for segment in briefs_result.get("target_audiences") or []
        ]
        creative_briefs = [
            CreativeBrief.model_validate(brief)
            for brief in briefs_result.get("creative_briefs") or []
        ]

        # ── 4. Level 1: deterministic campaign analysis ───────────────────
        metrics_input = _metrics_from(params)
        analysis_call = tool_registry.execute_tool(
            db,
            tool_name="analyze_campaign_performance",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={
                **({"metrics": metrics_input} if metrics_input else {}),
                "campaign_objective": campaign_objective.value,
                "selling_price_usd": product_facts.get("recommended_selling_price_usd"),
                "landed_cost_usd": product_facts.get("total_landed_cost_usd"),
            },
            context_summary=task.objective,
        )
        gated = self._gate(task, analysis_call, "Campaign analysis gated")
        if gated is not None:
            return gated
        analysis_result: dict[str, Any] = analysis_call["result"]

        raw_metrics = analysis_result.get("metrics")
        performance_metrics = (
            CampaignPerformanceMetrics.model_validate(raw_metrics)
            if raw_metrics
            else None
        )
        health = diagnose_campaign_health(performance_metrics)
        if performance_metrics is None:
            conversion_analysis = build_conversion_analysis(
                health, None, objective=campaign_objective
            )
        else:
            # Re-derive through the schema so the report's analysis and its
            # metrics cannot drift apart, even if the tool is swapped out.
            conversion_analysis = build_conversion_analysis(
                health, performance_metrics, objective=campaign_objective
            )

        # ── 5. Grounded copy + strategy (AI gated, draft as the baseline) ──
        variants, strategy, content_origin, ai_plan = self._grounded_marketing_content(
            ai_provider=ai_provider,
            product_facts=product_facts,
            product_name=product_name,
            briefs_result=briefs_result,
            campaign_objective=campaign_objective,
            conversion_analysis=conversion_analysis,
            legacy_ad_plan=ad_copy["result"],
        )

        budget = recommend_budget(
            gross_profit_per_unit_usd=_gross_profit_per_unit(product_facts),
            platforms=_platforms(product_facts, target_audiences),
            daily_budget_usd=_first_float(
                params.get("daily_budget_usd"), product_facts.get("daily_budget_usd")
            ),
            test_duration_days=int(_first_float(params.get("test_duration_days")) or 7),
        )

        marketing_report = MarketingStrategyReport(
            product_name=product_name,
            campaign_objective=campaign_objective,
            target_audiences=target_audiences,
            marketing_strategy=strategy,
            ad_copy_variants=variants,
            creative_briefs=creative_briefs,
            performance_metrics=performance_metrics,
            conversion_analysis=conversion_analysis,
            recommendations=list(conversion_analysis.recommended_optimizations),
            budget_considerations=budget,
            facts_used=product_facts,
            content_origin=content_origin,
        )
        # Re-validate through a JSON round trip: that is how it is stored and
        # served, so it must survive the trip unchanged.
        marketing_report = MarketingStrategyReport.model_validate(
            marketing_report.model_dump(mode="json")
        )

        data: dict[str, Any] = {
            "product": product_name,
            # Legacy keys kept alongside `marketing_report` so the existing
            # orchestrator/AI-copy contract keeps working unchanged.
            "product_facts": product_facts,
            "ad_copy": ad_copy["result"],
            "ad_plan": self._legacy_ad_plan(ai_plan, variants, ad_copy["result"]),
            "research_source": RESEARCH_NAMESPACE,
            "marketing_report": marketing_report.model_dump(mode="json"),
            "content_origin": content_origin,
            "campaign_analysis": analysis_result,
        }

        # The analysis is finished and archived *before* any Level 2 request:
        # if a budget commitment pauses for approval, the completed report must
        # still be retrievable rather than stranded in a waiting task.
        self._archive(db, memory, task, marketing_report)

        # ── 6. Level 2: the two money actions, both gated ─────────────────
        if params.get("publish_live") or params.get("publish"):
            publish = tool_registry.execute_tool(
                db,
                tool_name="mock_publish_ad_campaign",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={
                    "product": product_name,
                    "daily_budget_usd": budget.recommended_daily_budget_usd,
                },
                context_summary=task.objective,
            )
            if not publish["executed"]:
                return AgentOutput(
                    status=TaskStatus(publish["task_status"]),
                    agent_type=self.agent_type,
                    summary=(
                        "Marketing strategy report complete; publishing the live campaign "
                        "is awaiting human approval"
                    ),
                    data={**data, "permission": publish},
                    requires_approval=publish.get("approval_request_id") is not None,
                    approval_request_id=publish.get("approval_request_id"),
                )
            data["campaign"] = publish["result"]

        if params.get("commit_budget"):
            allocation = tool_registry.execute_tool(
                db,
                tool_name="allocate_ad_budget",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={
                    "product": product_name,
                    "daily_budget_usd": budget.recommended_daily_budget_usd,
                    "test_duration_days": budget.test_duration_days,
                    "allocation_by_platform": budget.allocation_by_platform,
                },
                context_summary=task.objective,
            )
            if not allocation["executed"]:
                return AgentOutput(
                    status=TaskStatus(allocation["task_status"]),
                    agent_type=self.agent_type,
                    summary=(
                        "Marketing strategy report complete; committing the ad budget is "
                        "awaiting human approval"
                    ),
                    data={**data, "permission": allocation},
                    requires_approval=allocation.get("approval_request_id") is not None,
                    approval_request_id=allocation.get("approval_request_id"),
                )
            data["budget_allocation"] = allocation["result"]

        # ── 7. Summary ────────────────────────────────────────────────────
        approval_note = ""
        if data.get("campaign") or data.get("budget_allocation"):
            approval_note = " Level 2 actions executed after human approval."
        summary = (
            f"{product_name}: {campaign_objective.value} plan with "
            f"{len(variants)} ad copy variant(s) and {len(creative_briefs)} creative "
            f"brief(s). Health {marketing_report.conversion_analysis.health_status.value}"
            f" ({marketing_report.conversion_analysis.bottleneck_stage}); recommended test "
            f"budget ${budget.recommended_daily_budget_usd:.2f}/day for "
            f"{budget.test_duration_days} days, approval required before any spend."
            + approval_note
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data=data,
        )

    # ── grounding ─────────────────────────────────────────────────────────

    def _grounded_marketing_content(
        self,
        *,
        ai_provider: AIProvider,
        product_facts: dict[str, Any],
        product_name: str,
        briefs_result: dict[str, Any],
        campaign_objective: CampaignObjective,
        conversion_analysis: ConversionAnalysis,
        legacy_ad_plan: dict[str, Any],
    ) -> tuple[list[AdCopyVariant], str, str, "_AiAdPlan | None"]:
        """Ad copy + strategy, AI-written only if it passes the grounding gate.

        Returns ``(variants, strategy, content_origin, ai_plan)``. The deterministic
        draft is always built first: it is grounded by construction (it only
        restates supplied facts), so it is a safe fallback and a fixed baseline
        for judging the model. The AI prompt is an explicit instruction set and
        the answer is re-checked by :func:`find_ungrounded_claims` afterwards -
        a prompt is a request, the gate is the rule.
        """
        draft_variants = _draft_variants(
            product_facts=product_facts,
            product_name=product_name,
            briefs_result=briefs_result,
            conversion_analysis=conversion_analysis,
            legacy_ad_plan=legacy_ad_plan,
        )
        draft_strategy = _draft_strategy(
            product_facts=product_facts,
            product_name=product_name,
            campaign_objective=campaign_objective,
            conversion_analysis=conversion_analysis,
        )

        prompt = AD_COPY_PROMPT_HEADER + json.dumps(
            product_facts, sort_keys=True, default=str
        )
        # Provider errors (notably AIRateLimitError) are deliberately not
        # swallowed: the orchestrator must fail fast without burning quota.
        ai_draft = ai_provider.generate_structured(
            prompt, AdPlan, context={"product_facts": product_facts}
        )
        # `AdPlan` is the 2A schema handed to the provider; the wider _AiAdPlan
        # view lets an optional richer payload through without breaking a stub
        # that only knows headline/body/cta.
        ai_plan = _AiAdPlan.model_validate(ai_draft.model_dump())

        reasons: list[str] = list(
            find_ungrounded_claims(
                " ".join(
                    [
                        ai_plan.headline,
                        ai_plan.primary_text,
                        ai_plan.hook,
                        ai_plan.marketing_strategy,
                        *ai_plan.ad_copy_angles,
                    ]
                ),
                product_facts,
            )
        )
        if not _references_a_fact(ai_plan, product_facts):
            reasons.append("copy references no supplied product fact")

        if reasons:
            return (
                draft_variants,
                draft_strategy,
                "deterministic_grounded_draft "
                f"(ai_output=rejected_ungrounded: {'; '.join(reasons[:3])})",
                ai_plan,
            )

        variants = _variants_from_ai(ai_plan, product_facts, draft_variants)
        strategy = ai_plan.marketing_strategy.strip() or draft_strategy
        return (
            variants,
            strategy,
            "ai_ad_copy_accepted (grounding gate passed)",
            ai_plan,
        )

    # ── inputs ────────────────────────────────────────────────────────────

    def _research_report(
        self, db: Session, memory: SharedMemoryService, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Agent 1's archived report, if one is available.

        This is where unit economics live: ``margin_analysis`` gives the landed
        cost, selling price and gross profit that ``break_even_roas`` is
        derived from. A missing archive simply means less is known - the agent
        degrades to ``break_even_roas = None`` instead of guessing.
        """
        source_task = str(params.get("research_task_id") or "").strip()
        candidates: list[Any] = []
        if source_task:
            candidates.append(source_task)
        handoff = memory.get_memory(db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY)
        if handoff is not None:
            handoff_source = (handoff.value or {}).get("source_task_id")
            if handoff_source:
                candidates.append(str(handoff_source))

        seen: set[str] = set()
        for key in candidates:
            if not key or key in seen:
                continue
            seen.add(key)
            record = memory.get_memory(db, namespace=REPORT_NAMESPACE, key=key)
            if record is None:
                continue
            report = (record.value or {}).get("report")
            if isinstance(report, dict):
                return dict(report)
        return {}

    # ── outputs ───────────────────────────────────────────────────────────

    def _archive(
        self,
        db: Session,
        memory: SharedMemoryService,
        task: Task,
        report: MarketingStrategyReport,
    ) -> None:
        """Archive the report + publish the handoff.

        Two namespaces on purpose: ``marketing_reports`` is the archived history
        served by ``GET /api/v1/advertising/reports``, and
        ``advertising-analytics`` stays the single cross-agent handoff record
        (its name is part of the existing contract and does not change).
        """
        memory.set_memory(
            db,
            namespace=AD_REPORT_NAMESPACE,
            key=task.id,
            value={
                "report": report.model_dump(mode="json"),
                "product_name": report.product_name,
                "campaign_objective": report.campaign_objective.value,
                "health_status": report.conversion_analysis.health_status.value,
                "roas": (
                    report.performance_metrics.roas
                    if report.performance_metrics is not None
                    else None
                ),
                "source_task_id": task.id,
            },
            agent_type=self.agent_type,
        )
        # Handoff keeps its existing namespace and gains the report; existing
        # consumers only read `candidates`/`product_name`, so they are unaffected.
        memory.set_memory(
            db,
            namespace=ADVERTISING_NAMESPACE,
            key=ADVERTISING_KEY,
            value={
                "product_name": report.product_name,
                "campaign_objective": report.campaign_objective.value,
                "health_status": report.conversion_analysis.health_status.value,
                "recommended_daily_budget_usd": report.budget_considerations.recommended_daily_budget_usd,
                "requires_human_approval": report.budget_considerations.requires_human_approval,
                "marketing_report": report.model_dump(mode="json"),
                "source_task_id": task.id,
            },
            agent_type=self.agent_type,
        )

    def _gate(
        self, task: Task, tool_result: dict[str, Any], summary: str
    ) -> AgentOutput | None:
        """Return a gated output when a tool did not execute, else ``None``."""
        if tool_result["executed"]:
            return None
        return _gated_output(self.agent_type, task, tool_result, summary)

    @staticmethod
    def _legacy_ad_plan(
        ai_plan: "_AiAdPlan | None",
        variants: list[AdCopyVariant],
        mock_plan: dict[str, Any],
    ) -> dict[str, Any]:
        """``ad_plan`` keeps the headline/body/cta shape it has had since 2A.

        Two consumers depend on it: the orchestrator/AI-copy contract and the
        mock ad-copy tool's trail. So the order is: the provider's own plan when
        it returned one (even a rejected one - the gate's verdict is recorded
        separately in ``content_origin``, and silently swapping the model's
        words here would make the two disagree), then grounded copy, then the
        mock placeholder. The key is therefore always present and always the
        same shape.
        """
        primary = variants[0] if variants else None
        from_ai = {
            "headline": ai_plan.headline.strip() if ai_plan else "",
            "body": ai_plan.primary_text if ai_plan else "",
            "cta": ai_plan.cta.strip() if ai_plan else "",
        }
        grounded = {
            "headline": primary.headline if primary else "",
            "body": primary.primary_text if primary else "",
            "cta": primary.call_to_action if primary else "",
        }
        return {
            key: from_ai[key] or grounded[key] or str(mock_plan.get(key) or "")
            for key in ("headline", "body", "cta")
        }


class DeliveryOperationsAgent(BaseAgent):
    agent_type = AgentType.DELIVERY_OPERATIONS
    description = "Tracks orders and (with approval) issues refunds."

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})
        tracking = tool_registry.execute_tool(
            db,
            tool_name="mock_order_tracker",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"order_id": params.get("order_id", "mock-order-1")},
            context_summary=task.objective,
        )
        if not tracking["executed"]:
            return _gated_output(self.agent_type, task, tracking, "Tracking gated")

        data: dict[str, Any] = {"tracking": tracking["result"]}

        if params.get("issue_refund"):
            refund = tool_registry.execute_tool(
                db,
                tool_name="mock_issue_refund",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={"order_id": params.get("order_id", "mock-order-1"), "amount": params.get("amount", 0.0)},
                context_summary=task.objective,
            )
            if not refund["executed"]:
                return AgentOutput(
                    status=TaskStatus(refund["task_status"]),
                    agent_type=self.agent_type,
                    summary="Refund awaiting human approval",
                    data={**data, "permission": refund},
                    requires_approval=refund.get("approval_request_id") is not None,
                    approval_request_id=refund.get("approval_request_id"),
                )
            data["refund"] = refund["result"]

        memory.set_memory(
            db,
            namespace="delivery-operations",
            key="latest_status",
            value=data,
            agent_type=self.agent_type,
        )
        summary = ai_provider.generate_text(
            "Summarize delivery operations status", context=data
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data=data,
        )
