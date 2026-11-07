"""Controlled tool registry.

Every tool maps to a PermissionPolicy action, so each execution is gated by
the 3-tier permission engine: Level 1 runs immediately, Level 2 pauses the
task for human approval without running the handler, Level 3 is blocked.
Phase 1 registers mock tools only - no real ads, orders, refunds or spend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.enums import AgentType
from app.core.permissions import PermissionManager, PermissionPolicy
from app.schemas.governance import ActionEvaluationRequest
from app.schemas.product_research import (
    STRONG_MARGIN_PCT,
    CompetitorEntry,
    DataProvenance,
    SupplierOption,
    describe_provenance,
)
from app.services.audit import AuditService
from app.services.memory import SharedMemoryService

PENDING_NAMESPACE = "orchestrator-pending"


@dataclass
class ToolDefinition:
    name: str
    action_name: str  # mapped into PermissionPolicy by PermissionManager
    description: str
    allowed_agents: set[AgentType]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool {name!r} is not registered") from exc

    def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in sorted(self._tools)]

    def execute_tool(
        self,
        db: Session,
        *,
        tool_name: str,
        task_id: str,
        agent_type: AgentType,
        parameters: dict[str, Any],
        context_summary: str = "",
    ) -> dict[str, Any]:
        tool = self.get_tool(tool_name)

        if agent_type not in tool.allowed_agents:
            AuditService.log_event(
                db,
                agent_type=agent_type,
                action_name=tool.action_name,
                permission_level=PermissionPolicy.level_for(tool.action_name),
                status="TOOL_UNAUTHORIZED_AGENT",
                task_id=task_id,
                details={"tool_name": tool_name},
            )
            db.commit()
            raise PermissionError(
                f"Agent {agent_type.value} may not use tool {tool_name!r}"
            )

        evaluation = PermissionManager.evaluate_action(
            db,
            ActionEvaluationRequest(
                task_id=task_id,
                agent_type=agent_type,
                action_name=tool.action_name,
                action_description=tool.description,
                context_summary=context_summary or tool.description,
                parameters=parameters,
            ),
        )

        if not evaluation.allowed_to_proceed:
            if evaluation.approval_request_id is not None:
                # Stash the pending call so resume_after_approval can run it.
                SharedMemoryService.set_memory(
                    db,
                    namespace=PENDING_NAMESPACE,
                    key=evaluation.approval_request_id,
                    value={
                        "tool_name": tool_name,
                        "task_id": task_id,
                        "agent_type": agent_type.value,
                        "parameters": parameters,
                        "context_summary": context_summary,
                    },
                    agent_type=AgentType.MAIN_ORCHESTRATOR,
                )
            return {
                "executed": False,
                "permission_level": evaluation.permission_level.value,
                "task_status": evaluation.task_status.value,
                "approval_request_id": evaluation.approval_request_id,
                "message": evaluation.message,
            }

        result = tool.handler(parameters)
        AuditService.log_event(
            db,
            agent_type=agent_type,
            action_name=tool.action_name,
            permission_level=evaluation.permission_level,
            status="TOOL_EXECUTED",
            task_id=task_id,
            details={"tool_name": tool_name},
        )
        db.commit()
        return {
            "executed": True,
            "permission_level": evaluation.permission_level.value,
            "task_status": evaluation.task_status.value,
            "approval_request_id": None,
            "result": result,
        }


# ── Phase 1 mock tools (no real-world effects) ───────────────────────────


def _mock_product_discovery(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "niche": parameters.get("niche", "general"),
        "candidates": [
            {"name": "LED dog collar", "est_price_usd": 12.50, "trend_score": 0.81},
            {"name": "Pet water fountain", "est_price_usd": 18.00, "trend_score": 0.77},
            {"name": "Slow feeder bowl", "est_price_usd": 9.99, "trend_score": 0.72},
        ],
    }


def _mock_margin_calculator(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "currency": "USD",
        "avg_margin_pct": 42.0,
        "inputs": parameters,
    }


def _mock_ad_copy_generator(parameters: dict[str, Any]) -> dict[str, Any]:
    product = parameters.get("product", "the product")
    return {
        "headline": f"Meet your new favorite {product}",
        "body": f"Why thousands are switching to {product} this season.",
        "cta": "Shop now",
    }


def _mock_publish_ad_campaign(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": "mock-campaign-001",
        "status": "published",
        "daily_budget_usd": parameters.get("daily_budget_usd", 10.0),
    }


def _mock_order_tracker(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": parameters.get("order_id", "mock-order-1"),
        "status": "in_transit",
        "eta_days": 4,
    }


def _mock_issue_refund(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "refund_id": "mock-refund-001",
        "status": "issued",
        "amount": parameters.get("amount", 0.0),
    }


# ── Milestone 2B: product research tools (Agent 1) ────────────────────────
#
# Rule these adapters never break: they do not fabricate verified data. When a
# caller supplies real quotes or observed prices, those are used as given (and
# only then may provenance be VERIFIED). When nothing is supplied, the adapter
# generates clearly-labelled SIMULATED figures so the pipeline still runs
# offline with no network and no spend - and says so in the result.

SIMULATED_SUPPLIER_ADAPTER = "simulated_supplier_adapter (no live quotes supplied)"
SIMULATED_COMPETITOR_ADAPTER = "simulated_competitor_adapter (no live listings observed)"

#: Landed-cost shape of the simulated supplier tiers, as fractions of a
#: reference retail price. Fixed so demos and tests are reproducible.
_SIMULATED_SUPPLIER_TIERS: tuple[dict[str, Any], ...] = (
    {"label": "Simulated budget-tier supplier", "unit": 0.30, "shipping": 0.12, "days": (18, 30)},
    {"label": "Simulated standard-tier supplier", "unit": 0.38, "shipping": 0.08, "days": (10, 18)},
    {"label": "Simulated express-tier supplier", "unit": 0.45, "shipping": 0.06, "days": (5, 9)},
)

#: Simulated competitor price points relative to the same reference price.
_SIMULATED_COMPETITOR_TIERS: tuple[dict[str, Any], ...] = (
    {"label": "Simulated competitor (budget)", "factor": 0.90, "positioning": "budget"},
    {"label": "Simulated competitor (mid-market)", "factor": 1.00, "positioning": "mid"},
    {"label": "Simulated competitor (premium)", "factor": 1.25, "positioning": "premium"},
)

#: Weakest-first ordering, used so an aggregate never claims more trust than
#: its least trustworthy input.
_PROVENANCE_STRENGTH: tuple[DataProvenance, ...] = (
    DataProvenance.SIMULATED,
    DataProvenance.ESTIMATED,
    DataProvenance.VERIFIED,
)


def _weakest_provenance(values: list[DataProvenance]) -> DataProvenance:
    """Aggregate provenance is only as strong as its weakest contributor."""
    if not values:
        return DataProvenance.SIMULATED
    return min(values, key=_PROVENANCE_STRENGTH.index)


def _declared_provenance(raw: Any) -> DataProvenance:
    """Read a caller-declared provenance; anything unrecognised is ESTIMATED.

    The fallback is deliberately *not* VERIFIED: a typo or a missing field must
    never silently promote untrusted data to verified.
    """
    if isinstance(raw, DataProvenance):
        return raw
    try:
        return DataProvenance(str(raw).strip().upper())
    except ValueError:
        return DataProvenance.ESTIMATED


def _as_optional_bool(raw: Any) -> bool | None:
    """Coerce a stock claim; anything ambiguous becomes ``None`` (unknown).

    Guessing ``True`` here would be exactly the fabricated availability the
    spec forbids, so unrecognised values degrade to "not confirmed".
    """
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"true", "yes", "y", "1", "in_stock", "available"}:
            return True
        if text in {"false", "no", "n", "0", "out_of_stock", "unavailable"}:
            return False
    return None


def _as_optional_float(raw: Any) -> float | None:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _position_against(price: float, median: float) -> str:
    """Descriptive price position. Not a claim about share or revenue."""
    if median <= 0:
        return ""
    if price < median * 0.97:
        return "budget"
    if price > median * 1.03:
        return "premium"
    return "mid"


def _compare_supplier_options(parameters: dict[str, Any]) -> dict[str, Any]:
    """Compare supplier quotes by landed cost (unit + shipping).

    Caller-supplied quotes are used as given. With none supplied, three
    clearly-labelled SIMULATED tiers are derived from ``reference_price_usd``
    so the pipeline stays runnable offline - stock is left unknown rather than
    asserted, because nothing was actually checked.
    """
    options: list[SupplierOption] = []
    for entry in parameters.get("suppliers") or []:
        if not isinstance(entry, dict):
            continue
        unit_cost = _as_optional_float(entry.get("unit_cost_usd"))
        if unit_cost is None or unit_cost < 0:
            # A "quote" with no cost carries no information; skipping it is
            # safer than inventing a number.
            continue
        days_min = int(entry.get("shipping_days_min") or 0)
        days_max = int(entry.get("shipping_days_max") or 0)
        options.append(
            SupplierOption.quote(
                supplier_name=str(entry.get("supplier_name") or "Unnamed supplier"),
                unit_cost_usd=unit_cost,
                shipping_cost_usd=_as_optional_float(entry.get("shipping_cost_usd")) or 0.0,
                shipping_days_min=min(days_min, days_max) if days_max else days_min,
                shipping_days_max=max(days_min, days_max),
                in_stock=_as_optional_bool(entry.get("in_stock")),
                provenance=_declared_provenance(
                    entry.get("provenance", DataProvenance.ESTIMATED.value)
                ),
                source_reference=str(entry.get("source_reference") or "caller-supplied quote"),
            )
        )

    note = ""
    if options:
        source = "caller-supplied supplier quotes"
    else:
        reference = _as_optional_float(parameters.get("reference_price_usd")) or 0.0
        source = SIMULATED_SUPPLIER_ADAPTER
        if reference <= 0:
            note = (
                "No live supplier quotes were supplied and no reference price was "
                "available, so no supplier options were produced. Nothing was invented."
            )
        else:
            for tier in _SIMULATED_SUPPLIER_TIERS:
                options.append(
                    SupplierOption.quote(
                        supplier_name=str(tier["label"]),
                        unit_cost_usd=round(reference * float(tier["unit"]), 2),
                        shipping_cost_usd=round(reference * float(tier["shipping"]), 2),
                        shipping_days_min=int(tier["days"][0]),
                        shipping_days_max=int(tier["days"][1]),
                        in_stock=None,
                        provenance=DataProvenance.SIMULATED,
                        source_reference=SIMULATED_SUPPLIER_ADAPTER,
                    )
                )
            note = (
                "Supplier costs are simulated from the reference retail price because "
                "no live quotes were supplied. They are not real prices and must not be "
                "used to place an order."
            )

    options.sort(key=lambda option: (option.total_landed_cost_usd, option.supplier_name))
    best = options[0] if options else None
    return {
        "options": [option.model_dump(mode="json") for option in options],
        "best_value": best.model_dump(mode="json") if best else None,
        "best_value_provenance": best.provenance.value if best else DataProvenance.SIMULATED.value,
        "compared_count": len(options),
        "source": source,
        "note": note,
    }


def _analyze_competitor_landscape(parameters: dict[str, Any]) -> dict[str, Any]:
    """Summarise observed competitor prices and suggest a market-aligned price.

    The suggested price is the median of what was actually observed, and it
    inherits the weakest provenance among those observations - a median built
    from simulated listings is itself simulated.
    """
    entries: list[CompetitorEntry] = []
    for entry in parameters.get("competitors") or []:
        if not isinstance(entry, dict):
            continue
        price = _as_optional_float(entry.get("observed_price_usd"))
        if price is None or price <= 0:
            continue
        entries.append(
            CompetitorEntry(
                competitor_name=str(entry.get("competitor_name") or "Unnamed competitor"),
                observed_price_usd=round(price, 2),
                market_positioning=str(entry.get("market_positioning") or ""),
                provenance=_declared_provenance(
                    entry.get("provenance", DataProvenance.ESTIMATED.value)
                ),
                source_reference=str(
                    entry.get("source_reference") or "caller-supplied observation"
                ),
            )
        )

    note = ""
    if entries:
        source = "caller-supplied competitor observations"
    else:
        reference = _as_optional_float(parameters.get("reference_price_usd")) or 0.0
        source = SIMULATED_COMPETITOR_ADAPTER
        if reference <= 0:
            note = (
                "No competitor prices were supplied and no reference price was available, "
                "so no market landscape was produced. Nothing was invented."
            )
        else:
            for tier in _SIMULATED_COMPETITOR_TIERS:
                entries.append(
                    CompetitorEntry(
                        competitor_name=str(tier["label"]),
                        observed_price_usd=round(reference * float(tier["factor"]), 2),
                        market_positioning=str(tier["positioning"]),
                        provenance=DataProvenance.SIMULATED,
                        source_reference=SIMULATED_COMPETITOR_ADAPTER,
                    )
                )
            note = (
                "Competitor prices are simulated around the reference retail price because "
                "no live listings were observed. They are not real market prices."
            )

    if not entries:
        return {
            "competitors": [],
            "observed_count": 0,
            "observed_price_floor_usd": None,
            "observed_price_median_usd": None,
            "observed_price_ceiling_usd": None,
            "recommended_selling_price_usd": None,
            "price_provenance": DataProvenance.SIMULATED.value,
            "source": source,
            "note": note,
        }

    prices = sorted(entry.observed_price_usd for entry in entries)
    median = round(_median(prices), 2)
    provenance = _weakest_provenance([entry.provenance for entry in entries])

    # Fill in positioning only where the caller left it blank: it is a
    # descriptive label relative to the observed median, not a new fact.
    landscape = [
        CompetitorEntry(
            competitor_name=entry.competitor_name,
            observed_price_usd=entry.observed_price_usd,
            market_positioning=entry.market_positioning
            or _position_against(entry.observed_price_usd, median),
            provenance=entry.provenance,
            source_reference=entry.source_reference,
        ).model_dump(mode="json")
        for entry in sorted(entries, key=lambda item: item.observed_price_usd)
    ]
    return {
        "competitors": landscape,
        "observed_count": len(landscape),
        "observed_price_floor_usd": prices[0],
        "observed_price_median_usd": median,
        "observed_price_ceiling_usd": prices[-1],
        "recommended_selling_price_usd": median,
        "price_provenance": provenance.value,
        "source": source,
        "note": note,
    }


def _median(sorted_values: list[float]) -> float:
    count = len(sorted_values)
    middle = count // 2
    if count % 2:
        return float(sorted_values[middle])
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def _generate_product_content(parameters: dict[str, Any]) -> dict[str, Any]:
    """Draft listing copy from supplied facts only - deterministically.

    This is the offline drafter: every sentence restates a fact that was passed
    in, labelled with its provenance. It exists so the pipeline produces usable
    copy with the mock provider, and so the AI step always has a facts-only
    baseline to be measured against. Nothing here extrapolates.
    """
    facts: dict[str, Any] = dict(parameters.get("facts") or {})
    name = str(facts.get("product_name") or parameters.get("product_name") or "the product")
    category = str(facts.get("product_category") or "general")

    price = _as_optional_float(facts.get("recommended_selling_price_usd"))
    landed = _as_optional_float(facts.get("total_landed_cost_usd"))
    unit_cost = _as_optional_float(facts.get("unit_cost_usd"))
    shipping_cost = _as_optional_float(facts.get("shipping_cost_usd"))
    margin = _as_optional_float(facts.get("estimated_gross_margin_pct"))
    days_min = facts.get("shipping_days_min") or 0
    days_max = facts.get("shipping_days_max") or 0
    in_stock = _as_optional_bool(facts.get("in_stock"))
    cost_provenance = _declared_provenance(
        facts.get("cost_provenance", DataProvenance.SIMULATED.value)
    )
    price_provenance = _declared_provenance(
        facts.get("price_provenance", DataProvenance.SIMULATED.value)
    )

    titles = [name]
    if category and category != "general":
        titles.append(f"{name} | {category.replace('_', ' ')}")
    if days_min and days_max:
        titles.append(f"{name} - ships in {days_min}-{days_max} days")

    bullets: list[str] = []
    if landed is not None:
        bullets.append(
            f"Landed cost ${landed:.2f} per unit ({describe_provenance(cost_provenance)})."
        )
    if unit_cost is not None and shipping_cost is not None:
        bullets.append(
            f"Unit cost ${unit_cost:.2f} plus ${shipping_cost:.2f} shipping "
            f"({describe_provenance(cost_provenance)})."
        )
    if price is not None:
        bullets.append(
            f"Suggested retail ${price:.2f} ({describe_provenance(price_provenance)})."
        )
    if margin is not None:
        bullets.append(
            f"Estimated gross margin {margin:.1f}%, before ad spend, fees and refunds."
        )
    if days_min and days_max:
        bullets.append(f"Quoted shipping window {days_min}-{days_max} days.")
    if in_stock is True:
        bullets.append("Supplier confirmed stock availability.")
    elif in_stock is None:
        bullets.append("Stock availability not confirmed - verify before ordering.")
    else:
        bullets.append("Reported out of stock at time of research.")

    angles: list[str] = []
    if margin is not None and margin >= STRONG_MARGIN_PCT:
        angles.append(
            f"Margin headroom: an estimated {margin:.1f}% gross margin leaves room for "
            f"paid acquisition - {describe_provenance(cost_provenance)}."
        )
    elif margin is not None:
        angles.append(
            f"Tight economics: an estimated {margin:.1f}% gross margin means ad spend must "
            f"stay disciplined."
        )
    if days_max and days_max <= 10:
        angles.append("Speed angle: a shorter-than-typical shipping window for this niche.")
    if in_stock is None:
        angles.append("Risk-first angle: confirm supplier stock before scaling spend.")

    sentences = [f"{name} ({category.replace('_', ' ')})"]
    if price is not None and landed is not None and margin is not None:
        sentences.append(
            f"a suggested retail price of ${price:.2f} against a landed cost of "
            f"${landed:.2f} per unit, an estimated gross margin of {margin:.1f}% before "
            f"advertising and fees"
        )
    if days_min and days_max:
        sentences.append(f"the quoted shipping window is {days_min}-{days_max} days")
    sentences.append(
        f"cost figures are {describe_provenance(cost_provenance)} and price figures are "
        f"{describe_provenance(price_provenance)}"
    )
    description = (
        ". ".join(sentence.capitalize() if index else sentence for index, sentence in enumerate(sentences))
        + ". No supplier prices, stock levels or demand statistics beyond these facts were used."
    )

    return {
        "product_titles": titles[:5],
        "product_description": description,
        "key_bullet_points": bullets,
        "marketing_angles": angles,
        "facts_used": facts,
        "source": "deterministic facts-only drafter",
        "note": (
            "Draft assembled only from the supplied facts; every figure carries its "
            "provenance label."
        ),
    }


# ── Phase 3: advertising & analytics tools (Agent 2) ──────────────────────
#
# Same rule as Agent 1's adapters: never fabricate. Every string these produce
# restates a supplied fact, and a fact that was not supplied is left out (with
# a note saying so) rather than filled in with a plausible invention. An
# invented discount or a made-up review is exactly what the spec forbids, so a
# discount benchmark only appears when the caller supplied one.

#: Campaign-flow benchmark used to size a budget when the caller supplied no
#: daily budget. Deliberately a plain, documented default and not a claim about
#: this product's performance.
DEFAULT_PLATFORMS: tuple[str, ...] = ("meta", "tiktok")

#: Creative formats the brief generator can produce, in preference order.
_CREATIVE_FORMATS: tuple[tuple[str, str], ...] = (
    ("UGC_VIDEO", "Shot on a phone, first-person, unpolished."),
    ("STATIC_IMAGE", "Single product hero image with a short overlay hook."),
    ("CAROUSEL", "Sequential frames that walk through one benefit each."),
)


def _facts_bundle(parameters: dict[str, Any]) -> dict[str, Any]:
    """The facts a brief/copy tool is allowed to use, from the caller's bundle."""
    facts = parameters.get("facts")
    return dict(facts) if isinstance(facts, dict) else {}


def _optional_str_list(raw: Any) -> list[str]:
    """Coerce a list-ish input into stripped, non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _prepare_creative_briefs(parameters: dict[str, Any]) -> dict[str, Any]:
    """Structured creative briefs + audience segments, from supplied facts only.

    Deterministic: the same facts always produce the same briefs. Each brief
    names the facts it rests on, and every brief carries the compliance notes
    that apply to *this* product - including an explicit "do not advertise a
    discount" note, because none was supplied.
    """
    facts = _facts_bundle(parameters)
    name = str(
        facts.get("product_name") or parameters.get("product_name") or "the product"
    ).strip() or "the product"

    price = _as_optional_float(facts.get("recommended_selling_price_usd"))
    margin = _as_optional_float(facts.get("estimated_gross_margin_pct"))
    days_min = facts.get("shipping_days_min") or 0
    days_max = facts.get("shipping_days_max") or 0
    in_stock = _as_optional_bool(facts.get("in_stock"))
    cost_provenance = _declared_provenance(
        facts.get("cost_provenance", DataProvenance.SIMULATED.value)
    )
    price_provenance = _declared_provenance(
        facts.get("price_provenance", DataProvenance.SIMULATED.value)
    )
    benefits = _optional_str_list(parameters.get("benefits") or facts.get("benefits"))
    platforms = _optional_str_list(
        parameters.get("platforms") or facts.get("platforms") or facts.get("platform")
    ) or list(DEFAULT_PLATFORMS)
    pain_points = _optional_str_list(
        parameters.get("pain_points") or facts.get("pain_points")
    )
    interests = _optional_str_list(parameters.get("interests") or facts.get("interests"))

    # ── compliance notes: only the caveats these facts actually imply ─────
    compliance: list[str] = []
    if cost_provenance is not DataProvenance.VERIFIED:
        compliance.append(
            f"Costs are {describe_provenance(cost_provenance)}: do not state a landed "
            f"cost or a profit figure as fact in the ad."
        )
    if price_provenance is not DataProvenance.VERIFIED:
        compliance.append(
            f"Price is {describe_provenance(price_provenance)}: show it as the current "
            f"offer only if the store is actually selling at that price."
        )
    if in_stock is None:
        compliance.append(
            "Stock availability is unconfirmed: do not claim 'in stock' or promise a "
            "delivery date that the supplier has not committed to."
        )
    elif in_stock is False:
        compliance.append(
            "Supplier reported out of stock: do not launch this concept until "
            "availability is confirmed."
        )
    compliance.append(
        "No discount, coupon, free-shipping offer, star rating, review, testimonial or "
        "guarantee was supplied - none may appear in the creative."
    )

    # ── audience segment: a hypothesis built from supplied facts ──────────
    segment_pain_points = pain_points
    segments = [
        {
            "segment_name": str(
                parameters.get("target_audience")
                or facts.get("target_audience")
                or f"Broad cold audience for {name}"
            ).strip(),
            "age_range": str(parameters.get("age_range") or facts.get("age_range") or "").strip(),
            "interests": interests,
            "pain_points": segment_pain_points,
            "platforms": platforms,
        }
    ]

    # ── briefs: one per supplied benefit (or one generic if none) ─────────
    messages: list[tuple[str, str]] = []
    for index, benefit in enumerate(benefits[:3], start=1):
        messages.append((f"Benefit-led concept {index}", benefit))
    if margin is not None and margin >= STRONG_MARGIN_PCT:
        messages.append(
            (
                "Value concept",
                f"An estimated {margin:.1f}% gross margin leaves room for a paid-offer "
                f"test; costs are {describe_provenance(cost_provenance)}.",
            )
        )
    if not messages:
        messages.append(
            (
                "Product-led concept",
                f"Introduce {name} using only the supplied facts; no benefit claims were "
                f"supplied.",
            )
        )

    briefs: list[dict[str, Any]] = []
    for index, (concept_name, key_message) in enumerate(messages):
        format_type, format_note = _CREATIVE_FORMATS[index % len(_CREATIVE_FORMATS)]
        facts_referenced = ["product_name"]
        if price is not None:
            facts_referenced.append("recommended_selling_price_usd")
        if days_min and days_max:
            facts_referenced.append("shipping_days_min/shipping_days_max")
        if benefits:
            facts_referenced.append("benefits")
        shot_list = [
            f"Open on the product: {name}.",
            f"Hook: {key_message}",
            "Show the product in use, exactly as the supplied facts describe it.",
        ]
        if days_min and days_max:
            shot_list.append(
                f"Overlay the quoted shipping window, {days_min}-{days_max} days, as a "
                f"stated range - never as a promise."
            )
        # Deliberately phrased without naming forbidden claims: this list is
        # scanned by the same grounding gate as ad copy, so a negated mention
        # ("no discount overlay") would itself be flagged. The prohibitions live
        # in `compliance_notes`, which is the field meant for them.
        shot_list.append(
            "Close on the call to action with only the supplied facts on screen."
        )
        briefs.append(
            {
                "concept_name": concept_name,
                "format_type": format_type,
                "visual_hook": key_message,
                "key_message": key_message,
                "shot_list": shot_list,
                "compliance_notes": compliance + [format_note],
                "facts_referenced": facts_referenced,
            }
        )

    return {
        "product_name": name,
        "target_audiences": segments,
        "creative_briefs": briefs,
        "facts_used": facts,
        "source": "deterministic grounded brief builder",
        "note": (
            "Briefs restate supplied facts only. Interests, pain points and platforms "
            "are empty where the caller supplied none; nothing was invented to fill them."
        ),
    }


def _analyze_campaign_performance(parameters: dict[str, Any]) -> dict[str, Any]:
    """Compute the derived metrics and diagnose the campaign - deterministically.

    Accepts an explicit ``metrics`` mapping (an already-built row) or the raw
    numbers at the top level. Returns ``health_status = NO_DATA`` when nothing
    was supplied, which is the pre-launch case, not a failure.
    """
    from app.schemas.advertising import (  # local import: keeps cycles impossible
        CampaignObjective,
        CampaignPerformanceMetrics,
        build_conversion_analysis,
        diagnose_campaign_health,
    )

    raw = parameters.get("metrics")
    metrics_payload: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

    def pick(key: str, default: Any = None) -> Any:
        if metrics_payload.get(key) is not None:
            return metrics_payload[key]
        return parameters.get(key, default)

    impressions = _as_optional_float(pick("impressions"))
    clicks = _as_optional_float(pick("clicks"))
    spend = _as_optional_float(pick("spend_usd"))
    conversions = _as_optional_float(pick("conversions"))
    revenue = _as_optional_float(pick("revenue_usd"))

    supplied = [impressions, clicks, spend, conversions, revenue]
    provenance = _declared_provenance(
        pick("provenance", parameters.get("metrics_provenance"))
        or DataProvenance.SIMULATED.value
    )

    if all(value is None for value in supplied):
        health = diagnose_campaign_health(None)
        analysis = build_conversion_analysis(health, None)
        return {
            "metrics": None,
            "health_status": health.value,
            "bottleneck_stage": analysis.bottleneck_stage,
            "summary": analysis.summary,
            "recommended_optimizations": analysis.recommended_optimizations,
            "source": "deterministic campaign analyser (no metrics supplied)",
            "note": (
                "No campaign metrics were supplied, so this is pre-launch mode. No "
                "performance was inferred."
            ),
        }

    selling_price = _as_optional_float(
        pick("selling_price_usd", parameters.get("selling_price_usd"))
    )
    landed_cost = _as_optional_float(
        pick("landed_cost_usd", parameters.get("landed_cost_usd"))
    )

    metrics = CampaignPerformanceMetrics.compute(
        impressions=int(impressions or 0),
        clicks=int(clicks or 0),
        spend_usd=float(spend or 0.0),
        conversions=int(conversions or 0),
        revenue_usd=float(revenue or 0.0),
        provenance=provenance,
        selling_price_usd=selling_price if selling_price and selling_price > 0 else None,
        landed_cost_usd=landed_cost if landed_cost is not None and landed_cost >= 0 else None,
    )

    health = diagnose_campaign_health(metrics)
    objective_raw = pick("campaign_objective", parameters.get("campaign_objective"))
    try:
        objective = CampaignObjective(str(objective_raw).strip().upper())
    except (ValueError, AttributeError):
        objective = CampaignObjective.CONVERSIONS
    analysis = build_conversion_analysis(health, metrics, objective=objective)

    return {
        "metrics": metrics.model_dump(mode="json"),
        "health_status": health.value,
        "bottleneck_stage": analysis.bottleneck_stage,
        "summary": analysis.summary,
        "recommended_optimizations": analysis.recommended_optimizations,
        "break_even_roas_basis": (
            "contribution break-even: selling_price / gross_profit per unit, "
            "before fixed overhead"
        ),
        "source": "deterministic campaign analyser",
        "note": (
            "Every rate is derived from the supplied raw metrics; nothing was "
            "estimated. break_even_roas is contribution break-even, not an "
            "accounting break-even for the business."
        ),
    }


def _allocate_ad_budget(parameters: dict[str, Any]) -> dict[str, Any]:
    """Level 2 gated budget allocation (mock - only runs after approval).

    The handler is never reached without an approved ``allocate_ad_budget``
    action; when it is reached the allocation is recorded as a mock so no real
    money moves.
    """
    daily = _as_optional_float(parameters.get("daily_budget_usd")) or 0.0
    days = int(_as_optional_float(parameters.get("test_duration_days")) or 7)
    allocation_raw = parameters.get("allocation_by_platform")
    allocation = (
        {str(key): round(float(value), 2) for key, value in allocation_raw.items()}
        if isinstance(allocation_raw, dict)
        else {platform: round(daily * days / len(DEFAULT_PLATFORMS), 2) for platform in DEFAULT_PLATFORMS}
    )
    return {
        "allocation_id": "mock-allocation-001",
        "status": "allocated",
        "daily_budget_usd": round(daily, 2),
        "test_duration_days": days,
        "total_planned_spend_usd": round(daily * days, 2),
        "allocation_by_platform": allocation,
        "mock": True,
        "note": (
            "Mock allocation executed after explicit human approval. No real ad "
            "account was touched and no funds were moved."
        ),
    }


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="mock_product_discovery",
        action_name="research_products",
        description="Discover candidate products for a niche (mock)",
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_mock_product_discovery,
    ))
    registry.register(ToolDefinition(
        name="mock_margin_calculator",
        action_name="calculate_margins",
        description="Estimate margins for candidate products (mock)",
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_mock_margin_calculator,
    ))
    registry.register(ToolDefinition(
        name="compare_supplier_options",
        action_name="compare_suppliers",
        description=(
            "Compare supplier quotes by landed cost; simulates clearly-labelled "
            "options when no live quotes are supplied"
        ),
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_compare_supplier_options,
    ))
    registry.register(ToolDefinition(
        name="analyze_competitor_landscape",
        action_name="analyze_competitors",
        description=(
            "Summarise observed competitor prices and suggest a market-aligned "
            "selling price"
        ),
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_analyze_competitor_landscape,
    ))
    registry.register(ToolDefinition(
        name="generate_product_content",
        action_name="draft_product_description",
        description="Draft product listing copy from supplied facts only",
        allowed_agents={AgentType.PRODUCT_RESEARCH},
        handler=_generate_product_content,
    ))
    registry.register(ToolDefinition(
        name="mock_ad_copy_generator",
        action_name="draft_ad_copy",
        description="Draft ad copy for a product (mock)",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_mock_ad_copy_generator,
    ))
    registry.register(ToolDefinition(
        name="mock_publish_ad_campaign",
        action_name="publish_advertisement",
        description="Publish a live ad campaign (mock, Level 2 approval)",
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_mock_publish_ad_campaign,
    ))
    registry.register(ToolDefinition(
        name="prepare_creative_briefs",
        action_name="prepare_creative_brief",
        description=(
            "Build structured creative briefs and audience segments grounded strictly "
            "in supplied product facts"
        ),
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_prepare_creative_briefs,
    ))
    registry.register(ToolDefinition(
        name="analyze_campaign_performance",
        action_name="analyze_campaign_metrics",
        description=(
            "Derive CTR/CPC/CPA/CVR/ROAS and contribution break-even from raw campaign "
            "metrics, then diagnose campaign health deterministically"
        ),
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_analyze_campaign_performance,
    ))
    registry.register(ToolDefinition(
        name="allocate_ad_budget",
        action_name="allocate_ad_budget",
        description=(
            "Allocate an approved ad budget across platforms (mock, Level 2 approval)"
        ),
        allowed_agents={AgentType.ADVERTISING_ANALYTICS},
        handler=_allocate_ad_budget,
    ))
    registry.register(ToolDefinition(
        name="mock_order_tracker",
        action_name="check_order_status",
        description="Track status of customer orders (mock)",
        allowed_agents={AgentType.DELIVERY_OPERATIONS},
        handler=_mock_order_tracker,
    ))
    registry.register(ToolDefinition(
        name="mock_issue_refund",
        action_name="issue_refund",
        description="Issue a customer refund (mock, Level 2 approval)",
        allowed_agents={AgentType.DELIVERY_OPERATIONS},
        handler=_mock_issue_refund,
    ))
    return registry
