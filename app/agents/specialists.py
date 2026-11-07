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
) -> list[str]:
    """Deterministic risk list: every caveat the provenance actually implies."""
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

    if not risks:
        risks.append(
            "No material risk detected from the supplied facts. Advertising cost, payment "
            "fees, returns and refunds are not modelled here."
        )
    return risks


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

        # ── 2. Supplier comparison (landed cost) ───────────────────────────
        supplier_call = tool_registry.execute_tool(
            db,
            tool_name="compare_supplier_options",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={
                "suppliers": params.get("suppliers") or [],
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
                "competitors": params.get("competitors") or [],
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
        sources.extend(str(item) for item in (params.get("research_sources") or []))
        research_sources = list(dict.fromkeys(source for source in sources if source))

        risks = _build_risks(
            margin=margin, best_supplier=best, demand_provenance=demand_provenance
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


class AdvertisingAnalyticsAgent(BaseAgent):
    agent_type = AgentType.ADVERTISING_ANALYTICS
    description = "Drafts ad copy and (with approval) publishes campaigns."

    def execute(
        self,
        db: Session,
        task: Task,
        tool_registry: ToolRegistry,
        memory: SharedMemoryService,
        ai_provider: AIProvider,
    ) -> AgentOutput:
        params: dict[str, Any] = dict(task.input_payload or {})
        research = memory.get_memory(db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY)
        candidates = (research.value or {}).get("candidates", []) if research else []
        top_product = candidates[0]["name"] if candidates else params.get("product", "the product")

        ad_copy = tool_registry.execute_tool(
            db,
            tool_name="mock_ad_copy_generator",
            task_id=task.id,
            agent_type=self.agent_type,
            parameters={"product": top_product},
            context_summary=task.objective,
        )
        if not ad_copy["executed"]:
            return _gated_output(self.agent_type, task, ad_copy, "Ad copy gated")

        # AI-written ad copy grounded ONLY in supplied product facts - never
        # invented claims. Facts come from the task input (price_usd, benefits,
        # target_audience, tone, platform are all optional); with the mock
        # provider this stays deterministic placeholder output for tests.
        product_facts: dict[str, Any] = {"product": top_product}
        for fact_key in ("price_usd", "benefits", "target_audience", "tone", "platform"):
            if params.get(fact_key) is not None:
                product_facts[fact_key] = params[fact_key]
        plan_prompt = (
            "Write ad copy for this product using ONLY the supplied facts "
            "below. Do not invent claims, statistics, reviews or discounts "
            "that are not in the facts.\n\n"
            f"Supplied facts (JSON):\n{json.dumps(product_facts, default=str)}"
        )
        plan = ai_provider.generate_structured(
            plan_prompt, AdPlan, context={"product_facts": product_facts}
        )

        data: dict[str, Any] = {
            "product": top_product,
            "product_facts": product_facts,
            "ad_copy": ad_copy["result"],
            "ad_plan": plan.model_dump(),
            "research_source": RESEARCH_NAMESPACE,
        }

        if params.get("publish"):
            publish = tool_registry.execute_tool(
                db,
                tool_name="mock_publish_ad_campaign",
                task_id=task.id,
                agent_type=self.agent_type,
                parameters={"product": top_product, "daily_budget_usd": params.get("daily_budget_usd", 10.0)},
                context_summary=task.objective,
            )
            if not publish["executed"]:
                return AgentOutput(
                    status=TaskStatus(publish["task_status"]),
                    agent_type=self.agent_type,
                    summary="Campaign publish awaiting human approval",
                    data={**data, "permission": publish},
                    requires_approval=publish.get("approval_request_id") is not None,
                    approval_request_id=publish.get("approval_request_id"),
                )
            data["campaign"] = publish["result"]

        summary = ai_provider.generate_text(
            f"Summarize marketing plan for {top_product}", context=data
        )
        return AgentOutput(
            status=TaskStatus.COMPLETED,
            agent_type=self.agent_type,
            summary=summary,
            data=data,
        )


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
