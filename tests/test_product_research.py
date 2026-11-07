"""Milestone 2B: Product Research & Content Agent (Agent 1).

Everything here runs offline. The AI is either the deterministic
``MockAIProvider`` or an in-process recording stub, and supplier/competitor
data is either supplied by the test or produced by the clearly-labelled
simulated adapters - so the suite proves the *provenance discipline* rather
than any claim about the real market.

The central property under test is the one Section 3 of the build spec demands:
the agent never invents supplier prices, availability or market demand, and
never lets an unverified figure masquerade as a verified one.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.agents.base import AgentOutput
from app.agents.specialists import (
    CONTENT_PROMPT_HEADER,
    REPORT_NAMESPACE,
    RESEARCH_KEY,
    RESEARCH_NAMESPACE,
    ProductResearchAgent,
)
from app.core import config
from app.core.ai_provider import (
    AIProvider,
    MockAIProvider,
    OpenAICompatibleProvider,
    get_ai_provider,
)
from app.core.enums import AgentType, PermissionLevel, TaskStatus
from app.core.permissions import PermissionPolicy
from app.core.tools import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
    _analyze_competitor_landscape,
    _compare_supplier_options,
    _generate_product_content,
)
from app.models.core import Task
from app.schemas.product_research import (
    MIN_VIABLE_MARGIN_PCT,
    STRONG_MARGIN_PCT,
    CompetitorEntry,
    ConfidenceLevel,
    DataProvenance,
    GeneratedProductContent,
    MarginAnalysis,
    OpportunityRecommendation,
    ProductOpportunityReport,
    SupplierOption,
    assess_opportunity,
)
from app.services.memory import SharedMemoryService
from app.services.orchestrator import MainOrchestrator

OPPORTUNITY_REPORT_PATH = "/api/v1/research/opportunity-report"
REPORTS_PATH = "/api/v1/research/reports"

#: A real-looking verified quote used by several tests: 4.10 + 1.90 = 6.00
#: landed against a 29.99 price -> 23.99 profit -> 79.99% margin.
VERIFIED_QUOTE: dict[str, Any] = {
    "supplier_name": "Shenzhen BrightPet Co",
    "unit_cost_usd": 4.10,
    "shipping_cost_usd": 1.90,
    "shipping_days_min": 8,
    "shipping_days_max": 14,
    "in_stock": True,
    "provenance": "VERIFIED",
    "source_reference": "quote-2026-11-04",
}


# ── helpers ───────────────────────────────────────────────────────────────


def _research_task(db: Session, payload: dict[str, Any] | None = None) -> Task:
    task = Task(
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Assess a product opportunity",
        objective="Evaluate margin and draft listing content",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        input_payload=payload or {},
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _run_agent(
    db: Session,
    task: Task,
    *,
    provider: Any = None,
    registry: ToolRegistry | None = None,
    memory: SharedMemoryService | None = None,
) -> AgentOutput:
    return ProductResearchAgent().execute(
        db,
        task,
        registry or create_default_registry(),
        memory or SharedMemoryService(),
        provider if provider is not None else MockAIProvider(),
    )


def _report_from(output: AgentOutput) -> ProductOpportunityReport:
    """Validate the agent's payload back into the report model."""
    raw = output.data["opportunity_report"]
    return ProductOpportunityReport.model_validate(raw)


def _registry_with_discovery(
    candidates: list[dict[str, Any]], niche: str = "pets"
) -> ToolRegistry:
    """A default registry whose discovery adapter returns `candidates`.

    Used to drive the paths where discovery yields nothing usable, so the agent
    has to refuse rather than invent a price.
    """
    registry = create_default_registry()
    existing = registry.get_tool("mock_product_discovery")
    registry.register(
        ToolDefinition(
            name=existing.name,
            action_name=existing.action_name,
            description=existing.description,
            allowed_agents=existing.allowed_agents,
            handler=lambda parameters: {"niche": niche, "candidates": candidates},
        )
    )
    return registry


class ContentRecorder(AIProvider):
    """Captures the content prompt/schema/context and returns scripted copy."""

    def __init__(self, content: GeneratedProductContent | None = None) -> None:
        self.prompts: list[str] = []
        self.schemas: list[type[BaseModel]] = []
        self.contexts: list[dict[str, Any] | None] = []
        self.text_prompts: list[str] = []
        self.closed = False
        self._content = content

    def generate_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        context: dict[str, Any] | None = None,
    ) -> BaseModel:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        self.contexts.append(context)
        if self._content is not None and schema is GeneratedProductContent:
            return self._content
        return schema()  # type: ignore[call-arg]

    def generate_text(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        self.text_prompts.append(prompt)
        return "recorded summary"

    def close(self) -> None:
        self.closed = True


def _grounded_content(product_name: str) -> GeneratedProductContent:
    """Copy that clearly derives from the supplied facts."""
    return GeneratedProductContent(
        product_titles=[f"{product_name} - fast shipping"],
        product_description=(
            f"{product_name} with a quoted shipping window and a margin that "
            f"leaves room for ads."
        ),
        key_bullet_points=[f"{product_name} ships in the quoted window"],
        marketing_angles=["Margin headroom for paid acquisition"],
        facts_used={"hallucinated_by_model": "must never reach the report"},
    )


def _numeric_tokens(facts: dict[str, Any]) -> set[str]:
    """Every string form a supplied number may legitimately appear as."""
    tokens: set[str] = set()
    for value in facts.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        tokens.add(str(value))
        tokens.add(f"{number:.0f}")
        tokens.add(f"{number:.1f}")
        tokens.add(f"{number:.2f}")
        if number == int(number):
            tokens.add(str(int(number)))
    return tokens


def _digits_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


# ══════════════════════════════════════════════════════════════════════════
# 1. Schema validation and provenance enforcement
# ══════════════════════════════════════════════════════════════════════════


def test_provenance_vocabulary_is_exact() -> None:
    assert [member.value for member in DataProvenance] == [
        "VERIFIED",
        "ESTIMATED",
        "SIMULATED",
    ]


def test_confidence_vocabulary_is_exact() -> None:
    assert [member.value for member in ConfidenceLevel] == ["LOW", "MEDIUM", "HIGH"]


def test_recommendation_vocabulary_is_exact() -> None:
    assert [member.value for member in OpportunityRecommendation] == [
        "PROCEED",
        "TEST_CAUTIOUSLY",
        "REJECT",
    ]


def test_supplier_option_requires_provenance() -> None:
    """A sourced number without provenance is not representable."""
    with pytest.raises(ValidationError) as excinfo:
        SupplierOption(
            supplier_name="Acme",
            unit_cost_usd=4.10,
            shipping_cost_usd=1.90,
            total_landed_cost_usd=6.00,
            shipping_days_min=8,
            shipping_days_max=14,
        )
    assert "provenance" in str(excinfo.value)


def test_unknown_provenance_is_rejected() -> None:
    """No fourth state: 'GUESSED' cannot be laundered into the report."""
    with pytest.raises(ValidationError):
        SupplierOption(
            supplier_name="Acme",
            unit_cost_usd=1.0,
            shipping_cost_usd=0.0,
            total_landed_cost_usd=1.0,
            provenance="GUESSED",  # type: ignore[arg-type]
        )


def test_supplier_landed_cost_must_add_up() -> None:
    with pytest.raises(ValidationError, match="total_landed_cost_usd"):
        SupplierOption(
            supplier_name="Acme",
            unit_cost_usd=4.10,
            shipping_cost_usd=1.90,
            total_landed_cost_usd=99.0,
            provenance=DataProvenance.VERIFIED,
        )


def test_supplier_shipping_window_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="shipping_days_min"):
        SupplierOption.quote(
            supplier_name="Acme",
            unit_cost_usd=4.10,
            shipping_days_min=30,
            shipping_days_max=8,
            provenance=DataProvenance.VERIFIED,
        )


def test_supplier_quote_derives_landed_cost() -> None:
    option = SupplierOption.quote(
        supplier_name="Acme",
        unit_cost_usd=4.104,
        shipping_cost_usd=1.901,
        provenance=DataProvenance.ESTIMATED,
    )
    assert option.unit_cost_usd == 4.10
    assert option.shipping_cost_usd == 1.90
    assert option.total_landed_cost_usd == 6.00
    assert option.in_stock is None, "availability must default to unknown"


def test_margin_arithmetic_is_validated() -> None:
    with pytest.raises(ValidationError, match="total_cost_usd"):
        MarginAnalysis(
            unit_cost_usd=6.0,
            shipping_cost_usd=1.0,
            total_cost_usd=12.0,
            recommended_selling_price_usd=20.0,
            estimated_gross_profit_usd=8.0,
            estimated_gross_margin_pct=40.0,
            cost_provenance=DataProvenance.VERIFIED,
            price_provenance=DataProvenance.VERIFIED,
        )


def test_margin_profit_and_percentage_are_validated() -> None:
    with pytest.raises(ValidationError, match="estimated_gross_profit_usd"):
        MarginAnalysis(
            unit_cost_usd=6.0,
            shipping_cost_usd=1.0,
            total_cost_usd=7.0,
            recommended_selling_price_usd=20.0,
            estimated_gross_profit_usd=99.0,
            estimated_gross_margin_pct=65.0,
            cost_provenance=DataProvenance.VERIFIED,
            price_provenance=DataProvenance.VERIFIED,
        )

    with pytest.raises(ValidationError, match="estimated_gross_margin_pct"):
        MarginAnalysis(
            unit_cost_usd=6.0,
            shipping_cost_usd=1.0,
            total_cost_usd=7.0,
            recommended_selling_price_usd=20.0,
            estimated_gross_profit_usd=13.0,
            estimated_gross_margin_pct=99.0,
            cost_provenance=DataProvenance.VERIFIED,
            price_provenance=DataProvenance.VERIFIED,
        )


def test_margin_compute_derives_every_figure() -> None:
    margin = MarginAnalysis.compute(
        unit_cost_usd=4.10,
        shipping_cost_usd=1.90,
        recommended_selling_price_usd=29.99,
        cost_provenance=DataProvenance.VERIFIED,
        price_provenance=DataProvenance.ESTIMATED,
    )
    assert margin.total_cost_usd == 6.00
    assert margin.estimated_gross_profit_usd == 23.99
    assert margin.estimated_gross_margin_pct == 79.99
    assert margin.is_viable is True
    # The two provenance fields stay independent: a verified cost against an
    # estimated price is still an unverified margin.
    assert (margin.cost_provenance, margin.price_provenance) == (
        DataProvenance.VERIFIED,
        DataProvenance.ESTIMATED,
    )


def test_margin_compute_handles_negative_margin() -> None:
    margin = MarginAnalysis.compute(
        unit_cost_usd=10.0,
        shipping_cost_usd=5.0,
        recommended_selling_price_usd=12.0,
        cost_provenance=DataProvenance.VERIFIED,
        price_provenance=DataProvenance.VERIFIED,
    )
    assert margin.estimated_gross_profit_usd == -3.0
    assert margin.estimated_gross_margin_pct == -25.0
    assert margin.is_viable is False


def test_selling_price_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MarginAnalysis.compute(
            unit_cost_usd=1.0,
            shipping_cost_usd=1.0,
            recommended_selling_price_usd=0.0,
            cost_provenance=DataProvenance.VERIFIED,
            price_provenance=DataProvenance.VERIFIED,
        )


def test_report_requires_product_name_and_margin() -> None:
    with pytest.raises(ValidationError):
        ProductOpportunityReport(
            product_name="",
            research_confidence=ConfidenceLevel.LOW,
            recommendation=OpportunityRecommendation.REJECT,
            margin_analysis=MarginAnalysis.compute(
                unit_cost_usd=1.0,
                shipping_cost_usd=0.0,
                recommended_selling_price_usd=10.0,
                cost_provenance=DataProvenance.SIMULATED,
                price_provenance=DataProvenance.SIMULATED,
            ),
        )
    with pytest.raises(ValidationError):
        ProductOpportunityReport(  # type: ignore[call-arg]
            product_name="X",
            research_confidence=ConfidenceLevel.LOW,
            recommendation=OpportunityRecommendation.REJECT,
        )


def test_competitor_entry_requires_a_positive_observed_price() -> None:
    with pytest.raises(ValidationError):
        CompetitorEntry(
            competitor_name="Rival",
            observed_price_usd=0.0,
            provenance=DataProvenance.VERIFIED,
        )


# ══════════════════════════════════════════════════════════════════════════
# 2. The assessment rule: deterministic, and never over-confident
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("provenance", list(DataProvenance))
@pytest.mark.parametrize(
    "margin_pct",
    [0.0, -25.0, 5.0, MIN_VIABLE_MARGIN_PCT - 0.01, MIN_VIABLE_MARGIN_PCT,
     25.0, STRONG_MARGIN_PCT - 0.01, STRONG_MARGIN_PCT, 58.0, 95.0],
)
def test_assessment_is_deterministic(
    margin_pct: float, provenance: DataProvenance
) -> None:
    first = assess_opportunity(gross_margin_pct=margin_pct, cost_provenance=provenance)
    second = assess_opportunity(gross_margin_pct=margin_pct, cost_provenance=provenance)
    assert first == second
    assert first[0] in ConfidenceLevel and first[1] in OpportunityRecommendation


@pytest.mark.parametrize(
    "margin_pct", [-25.0, 0.0, 5.0, 19.99, MIN_VIABLE_MARGIN_PCT - 0.01]
)
@pytest.mark.parametrize("provenance", list(DataProvenance))
def test_thin_margin_always_rejects(
    margin_pct: float, provenance: DataProvenance
) -> None:
    _, recommendation = assess_opportunity(
        gross_margin_pct=margin_pct, cost_provenance=provenance
    )
    assert recommendation is OpportunityRecommendation.REJECT


@pytest.mark.parametrize(
    "margin_pct", [MIN_VIABLE_MARGIN_PCT, 25.0, 34.99, STRONG_MARGIN_PCT, 58.0, 95.0]
)
@pytest.mark.parametrize(
    "provenance", [DataProvenance.ESTIMATED, DataProvenance.SIMULATED]
)
def test_confidence_never_high_with_unverified_costs(
    margin_pct: float, provenance: DataProvenance
) -> None:
    """The core anti-hallucination guarantee, across the whole margin range."""
    confidence, recommendation = assess_opportunity(
        gross_margin_pct=margin_pct, cost_provenance=provenance
    )
    assert confidence is not ConfidenceLevel.HIGH
    assert confidence in {ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM}
    # A big margin on unverified costs is a reason to test, not to commit.
    assert recommendation is OpportunityRecommendation.TEST_CAUTIOUSLY


@pytest.mark.parametrize(
    "margin_pct", [MIN_VIABLE_MARGIN_PCT, 25.0, 34.99, STRONG_MARGIN_PCT, 58.0]
)
def test_simulated_costs_cap_confidence_at_low(margin_pct: float) -> None:
    """Nothing real was consulted, so nothing is even medium-confidence."""
    confidence, _ = assess_opportunity(
        gross_margin_pct=margin_pct, cost_provenance=DataProvenance.SIMULATED
    )
    assert confidence is ConfidenceLevel.LOW


def test_verified_strong_margin_allows_proceed() -> None:
    confidence, recommendation = assess_opportunity(
        gross_margin_pct=STRONG_MARGIN_PCT, cost_provenance=DataProvenance.VERIFIED
    )
    assert (confidence, recommendation) == (
        ConfidenceLevel.HIGH,
        OpportunityRecommendation.PROCEED,
    )


def test_verified_adequate_margin_is_medium_and_cautious() -> None:
    confidence, recommendation = assess_opportunity(
        gross_margin_pct=MIN_VIABLE_MARGIN_PCT, cost_provenance=DataProvenance.VERIFIED
    )
    assert (confidence, recommendation) == (
        ConfidenceLevel.MEDIUM,
        OpportunityRecommendation.TEST_CAUTIOUSLY,
    )


def test_verified_thin_margin_rejects_with_high_confidence() -> None:
    """A verified rejection is a trustworthy rejection."""
    confidence, recommendation = assess_opportunity(
        gross_margin_pct=9.09, cost_provenance=DataProvenance.VERIFIED
    )
    assert (confidence, recommendation) == (
        ConfidenceLevel.HIGH,
        OpportunityRecommendation.REJECT,
    )


def test_thresholds_are_the_documented_constants() -> None:
    assert MIN_VIABLE_MARGIN_PCT == 20.0
    assert STRONG_MARGIN_PCT == 35.0


# ══════════════════════════════════════════════════════════════════════════
# 3. Tool adapters: registered, Level 1, and never fabricating VERIFIED
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("tool_name", "action_name"),
    [
        ("compare_supplier_options", "compare_suppliers"),
        ("analyze_competitor_landscape", "analyze_competitors"),
        ("generate_product_content", "draft_product_description"),
    ],
)
def test_new_tools_are_registered_level1_research_only(
    tool_name: str, action_name: str
) -> None:
    registry = create_default_registry()
    tool = registry.get_tool(tool_name)
    assert tool.action_name == action_name
    assert tool.allowed_agents == {AgentType.PRODUCT_RESEARCH}
    assert PermissionPolicy.level_for(action_name) is PermissionLevel.LEVEL_1_AUTO


@pytest.mark.parametrize(
    "tool_name",
    ["compare_supplier_options", "analyze_competitor_landscape", "generate_product_content"],
)
def test_research_tools_reject_other_agents(
    db: Session, sample_task: Task, tool_name: str
) -> None:
    registry = create_default_registry()
    with pytest.raises(PermissionError):
        registry.execute_tool(
            db,
            tool_name=tool_name,
            task_id=sample_task.id,
            agent_type=AgentType.ADVERTISING_ANALYTICS,
            parameters={},
        )


def test_legacy_research_tools_still_work(db: Session, sample_task: Task) -> None:
    """Backward compatibility: the original mock tools are untouched."""
    registry = create_default_registry()
    discovery = registry.execute_tool(
        db,
        tool_name="mock_product_discovery",
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"niche": "pets"},
    )
    assert discovery["executed"] is True
    assert discovery["result"]["niche"] == "pets"
    assert [c["name"] for c in discovery["result"]["candidates"]] == [
        "LED dog collar",
        "Pet water fountain",
        "Slow feeder bowl",
    ]

    margins = registry.execute_tool(
        db,
        tool_name="mock_margin_calculator",
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        parameters={"candidates": discovery["result"]["candidates"]},
    )
    assert margins["result"]["currency"] == "USD"
    assert margins["result"]["avg_margin_pct"] == 42.0
    assert margins["result"]["inputs"]["candidates"]


def test_supplier_comparison_without_quotes_is_simulated() -> None:
    result = _compare_supplier_options({"reference_price_usd": 25.0})
    options = result["options"]
    assert result["compared_count"] == 3
    assert result["source"].startswith("simulated_supplier_adapter")
    assert "not real prices" in result["note"]

    for option in options:
        assert option["provenance"] == "SIMULATED"
        assert option["in_stock"] is None, "availability was never checked"
        assert option["total_landed_cost_usd"] == round(
            option["unit_cost_usd"] + option["shipping_cost_usd"], 2
        )
        assert "simulated" in option["supplier_name"].lower()

    # Sorted by landed cost, and best_value is the cheapest.
    landed = [option["total_landed_cost_usd"] for option in options]
    assert landed == sorted(landed)
    assert result["best_value"]["total_landed_cost_usd"] == landed[0]
    assert result["best_value_provenance"] == "SIMULATED"


def test_supplier_comparison_honours_explicit_verified_quotes() -> None:
    result = _compare_supplier_options({"suppliers": [VERIFIED_QUOTE]})
    assert result["compared_count"] == 1
    assert result["source"] == "caller-supplied supplier quotes"
    assert result["note"] == ""
    best = result["best_value"]
    assert best["provenance"] == "VERIFIED"
    assert best["supplier_name"] == "Shenzhen BrightPet Co"
    assert best["unit_cost_usd"] == 4.10
    assert best["shipping_cost_usd"] == 1.90
    assert best["total_landed_cost_usd"] == 6.00
    assert best["in_stock"] is True
    assert (best["shipping_days_min"], best["shipping_days_max"]) == (8, 14)


@pytest.mark.parametrize("declared", [None, "", "GUESSED", "verified-ish", 123])
def test_unlabelled_or_invalid_quotes_are_never_verified(declared: Any) -> None:
    """Omission or a typo must not promote untrusted data to VERIFIED."""
    quote = {"supplier_name": "Acme", "unit_cost_usd": 5.0, "shipping_cost_usd": 1.0}
    if declared is not None:
        quote["provenance"] = declared
    result = _compare_supplier_options({"suppliers": [quote]})
    assert result["best_value"]["provenance"] == "ESTIMATED"


def test_supplier_comparison_picks_cheapest_landed_cost() -> None:
    """Cheapest *unit* price is not cheapest landed price - shipping decides."""
    result = _compare_supplier_options(
        {
            "suppliers": [
                {
                    "supplier_name": "Cheap unit, dear shipping",
                    "unit_cost_usd": 2.00,
                    "shipping_cost_usd": 9.00,
                    "provenance": "VERIFIED",
                },
                {
                    "supplier_name": "Balanced",
                    "unit_cost_usd": 5.00,
                    "shipping_cost_usd": 1.00,
                    "provenance": "VERIFIED",
                },
            ]
        }
    )
    assert result["best_value"]["supplier_name"] == "Balanced"
    assert result["best_value"]["total_landed_cost_usd"] == 6.00


def test_supplier_comparison_without_any_input_fabricates_nothing() -> None:
    result = _compare_supplier_options({})
    assert result["options"] == []
    assert result["best_value"] is None
    assert result["compared_count"] == 0
    assert "Nothing was invented" in result["note"]


def test_supplier_quotes_without_a_cost_are_skipped() -> None:
    result = _compare_supplier_options(
        {"suppliers": [{"supplier_name": "No price given"}, VERIFIED_QUOTE]}
    )
    assert result["compared_count"] == 1
    assert result["best_value"]["supplier_name"] == VERIFIED_QUOTE["supplier_name"]


@pytest.mark.parametrize(
    ("raw_stock", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("OUT_OF_STOCK", False),
        ("maybe", None),
        (None, None),
        ("", None),
    ],
)
def test_stock_claims_degrade_to_unknown(raw_stock: Any, expected: bool | None) -> None:
    """Ambiguous availability becomes None rather than a fabricated 'in stock'."""
    quote = dict(VERIFIED_QUOTE, in_stock=raw_stock)
    result = _compare_supplier_options({"suppliers": [quote]})
    assert result["best_value"]["in_stock"] is expected


def test_competitor_landscape_without_observations_is_simulated() -> None:
    result = _analyze_competitor_landscape({"reference_price_usd": 20.0})
    assert result["observed_count"] == 3
    assert result["source"].startswith("simulated_competitor_adapter")
    assert "not real market prices" in result["note"]
    for entry in result["competitors"]:
        assert entry["provenance"] == "SIMULATED"
    assert result["observed_price_floor_usd"] == 18.0
    assert result["observed_price_median_usd"] == 20.0
    assert result["observed_price_ceiling_usd"] == 25.0
    assert result["recommended_selling_price_usd"] == 20.0
    assert result["price_provenance"] == "SIMULATED"


def test_competitor_landscape_uses_observed_median() -> None:
    result = _analyze_competitor_landscape(
        {
            "competitors": [
                {"competitor_name": "A", "observed_price_usd": 10.0, "provenance": "VERIFIED"},
                {"competitor_name": "B", "observed_price_usd": 30.0, "provenance": "VERIFIED"},
                {"competitor_name": "C", "observed_price_usd": 20.0, "provenance": "VERIFIED"},
            ]
        }
    )
    assert result["recommended_selling_price_usd"] == 20.0
    assert result["observed_price_median_usd"] == 20.0
    assert result["price_provenance"] == "VERIFIED"
    # Positioned relative to the observed median.
    positions = {e["competitor_name"]: e["market_positioning"] for e in result["competitors"]}
    assert positions == {"A": "budget", "B": "premium", "C": "mid"}


def test_competitor_median_inherits_the_weakest_provenance() -> None:
    """One simulated listing makes the whole median simulated."""
    result = _analyze_competitor_landscape(
        {
            "competitors": [
                {"competitor_name": "Real", "observed_price_usd": 20.0, "provenance": "VERIFIED"},
                {"competitor_name": "Made up", "observed_price_usd": 22.0},
            ]
        }
    )
    assert result["price_provenance"] == "ESTIMATED"

    mixed = _analyze_competitor_landscape(
        {
            "competitors": [
                {"competitor_name": "Real", "observed_price_usd": 20.0, "provenance": "VERIFIED"},
            ],
            "reference_price_usd": 20.0,
        }
    )
    assert mixed["price_provenance"] == "VERIFIED"


def test_competitor_landscape_without_any_input_fabricates_nothing() -> None:
    result = _analyze_competitor_landscape({})
    assert result["competitors"] == []
    assert result["recommended_selling_price_usd"] is None
    assert result["observed_price_median_usd"] is None
    assert "Nothing was invented" in result["note"]


def test_content_draft_restates_only_supplied_facts() -> None:
    """No number may appear in the copy unless it came from the facts."""
    facts = {
        "product_name": "LED dog collar",
        "product_category": "pet_accessories",
        "unit_cost_usd": 4.10,
        "shipping_cost_usd": 1.90,
        "total_landed_cost_usd": 6.00,
        "recommended_selling_price_usd": 29.99,
        "estimated_gross_margin_pct": 79.99,
        "shipping_days_min": 8,
        "shipping_days_max": 14,
        "in_stock": True,
        "cost_provenance": "VERIFIED",
        "price_provenance": "VERIFIED",
    }
    draft = _generate_product_content({"facts": facts})
    text = " ".join(
        [
            draft["product_description"],
            *draft["product_titles"],
            *draft["key_bullet_points"],
            *draft["marketing_angles"],
        ]
    )
    allowed = _numeric_tokens(facts)
    invented = _digits_in(text) - allowed
    assert not invented, f"copy cites figures absent from the facts: {invented}"
    assert draft["facts_used"] == facts
    assert "verified" in draft["product_description"]


def test_content_draft_labels_unverified_figures() -> None:
    facts = {
        "product_name": "Pet water fountain",
        "product_category": "general",
        "unit_cost_usd": 5.00,
        "shipping_cost_usd": 2.00,
        "total_landed_cost_usd": 7.00,
        "recommended_selling_price_usd": 18.00,
        "estimated_gross_margin_pct": 61.11,
        "shipping_days_min": 0,
        "shipping_days_max": 0,
        "in_stock": None,
        "cost_provenance": "SIMULATED",
        "price_provenance": "ESTIMATED",
    }
    draft = _generate_product_content({"facts": facts})
    bullets = " ".join(draft["key_bullet_points"])
    assert "simulated offline - not a live quote" in bullets
    assert "estimated - not a live quote" in bullets
    assert "Stock availability not confirmed" in bullets
    assert "No shipping window was quoted" not in bullets  # that is a risk, not a bullet
    assert not any("confirmed stock" in b for b in draft["key_bullet_points"])


def test_content_draft_reports_out_of_stock_honestly() -> None:
    draft = _generate_product_content(
        {
            "facts": {
                "product_name": "Slow feeder bowl",
                "total_landed_cost_usd": 4.0,
                "recommended_selling_price_usd": 9.99,
                "estimated_gross_margin_pct": 60.0,
                "in_stock": False,
                "cost_provenance": "VERIFIED",
                "price_provenance": "VERIFIED",
            }
        }
    )
    assert any("out of stock" in b for b in draft["key_bullet_points"])


# ══════════════════════════════════════════════════════════════════════════
# 4. The agent pipeline: provenance survives end to end
# ══════════════════════════════════════════════════════════════════════════


def test_agent_returns_a_validated_report(db: Session) -> None:
    task = _research_task(db)
    output = _run_agent(db, task)
    assert output.status is TaskStatus.COMPLETED
    assert output.agent_type is AgentType.PRODUCT_RESEARCH

    report = _report_from(output)
    assert isinstance(report, ProductOpportunityReport)
    # Survives a JSON round trip unchanged - that is how it is stored and served.
    assert ProductOpportunityReport.model_validate(
        json.loads(report.model_dump_json())
    ) == report


def test_default_run_is_simulated_low_confidence_and_cautious(db: Session) -> None:
    """With no real inputs, the report says so and refuses to be confident."""
    task = _research_task(db, {"niche": "pet accessories"})
    report = _report_from(_run_agent(db, task))

    assert report.product_name == "LED dog collar"
    assert report.demand_provenance is DataProvenance.SIMULATED
    assert report.margin_analysis.cost_provenance is DataProvenance.SIMULATED
    assert report.margin_analysis.price_provenance is DataProvenance.SIMULATED
    assert all(s.provenance is DataProvenance.SIMULATED for s in report.suppliers)
    assert all(c.provenance is DataProvenance.SIMULATED for c in report.competitors)
    assert all(s.in_stock is None for s in report.suppliers)
    assert report.research_confidence is ConfidenceLevel.LOW
    assert report.recommendation is OpportunityRecommendation.TEST_CAUTIOUSLY
    assert any("simulated" in source.lower() for source in report.research_sources)
    assert any("not a live quote" in risk for risk in report.risks)
    assert any("Stock availability was not confirmed" in risk for risk in report.risks)
    assert "not evidence of real demand" in report.demand_summary


def test_agent_margin_is_accurate_against_its_own_inputs(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "product_category": "pet_accessories",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "suppliers": [VERIFIED_QUOTE],
        },
    )
    report = _report_from(_run_agent(db, task))
    margin = report.margin_analysis

    assert margin.unit_cost_usd == 4.10
    assert margin.shipping_cost_usd == 1.90
    assert margin.total_cost_usd == pytest.approx(4.10 + 1.90)
    assert margin.recommended_selling_price_usd == 29.99
    assert margin.estimated_gross_profit_usd == pytest.approx(29.99 - 6.00)
    assert margin.estimated_gross_margin_pct == pytest.approx((29.99 - 6.00) / 29.99 * 100, abs=0.01)
    assert margin.cost_provenance is DataProvenance.VERIFIED
    assert margin.price_provenance is DataProvenance.VERIFIED
    assert report.research_confidence is ConfidenceLevel.HIGH
    assert report.recommendation is OpportunityRecommendation.PROCEED


def test_agent_uses_the_cheapest_landed_supplier(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 20.0,
            "price_provenance": "VERIFIED",
            "suppliers": [
                {"supplier_name": "Dear shipping", "unit_cost_usd": 2.0,
                 "shipping_cost_usd": 9.0, "provenance": "VERIFIED"},
                {"supplier_name": "Balanced", "unit_cost_usd": 5.0,
                 "shipping_cost_usd": 1.0, "provenance": "VERIFIED"},
            ],
        },
    )
    report = _report_from(_run_agent(db, task))
    assert report.best_supplier is not None
    assert report.best_supplier.supplier_name == "Balanced"
    assert report.margin_analysis.unit_cost_usd == 5.0
    assert report.margin_analysis.total_cost_usd == 6.0
    # The dear-shipping quote is still listed, not silently dropped.
    assert {s.supplier_name for s in report.suppliers} == {"Dear shipping", "Balanced"}


@pytest.mark.parametrize(
    ("suppliers", "price", "expected_recommendation", "expected_confidence"),
    [
        # 10.00 landed against 11.00 -> 9.09% -> reject, and we are sure of it.
        ([{"supplier_name": "S", "unit_cost_usd": 9.0, "shipping_cost_usd": 1.0,
           "provenance": "VERIFIED"}], 11.0,
         OpportunityRecommendation.REJECT, ConfidenceLevel.HIGH),
        # 7.50 landed against 10.00 -> 25% -> verified but merely adequate.
        ([{"supplier_name": "S", "unit_cost_usd": 6.0, "shipping_cost_usd": 1.5,
           "provenance": "VERIFIED"}], 10.0,
         OpportunityRecommendation.TEST_CAUTIOUSLY, ConfidenceLevel.MEDIUM),
        # 6.00 landed against 29.99 -> 79.99% -> strong and verified.
        ([VERIFIED_QUOTE], 29.99,
         OpportunityRecommendation.PROCEED, ConfidenceLevel.HIGH),
    ],
)
def test_all_three_recommendation_thresholds(
    db: Session,
    suppliers: list[dict[str, Any]],
    price: float,
    expected_recommendation: OpportunityRecommendation,
    expected_confidence: ConfidenceLevel,
) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": price,
            "price_provenance": "VERIFIED",
            "suppliers": suppliers,
        },
    )
    report = _report_from(_run_agent(db, task))
    assert report.recommendation is expected_recommendation
    assert report.research_confidence is expected_confidence


def test_huge_margin_on_simulated_costs_stays_cautious(db: Session) -> None:
    """The default run yields ~58% margin - still not a PROCEED."""
    task = _research_task(db)
    report = _report_from(_run_agent(db, task))
    assert report.margin_analysis.estimated_gross_margin_pct > STRONG_MARGIN_PCT
    assert report.recommendation is OpportunityRecommendation.TEST_CAUTIOUSLY
    assert report.research_confidence is not ConfidenceLevel.HIGH


def test_explicitly_simulated_quotes_are_never_upgraded(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 99.0,
            "price_provenance": "VERIFIED",
            "suppliers": [
                {"supplier_name": "Back-of-envelope", "unit_cost_usd": 1.0,
                 "shipping_cost_usd": 0.5, "provenance": "SIMULATED"}
            ],
        },
    )
    report = _report_from(_run_agent(db, task))
    assert report.margin_analysis.cost_provenance is DataProvenance.SIMULATED
    assert report.margin_analysis.estimated_gross_margin_pct > 90.0
    assert report.research_confidence is ConfidenceLevel.LOW
    assert report.recommendation is OpportunityRecommendation.TEST_CAUTIOUSLY


def test_unverified_demand_is_not_reported_as_verified(db: Session) -> None:
    task = _research_task(db)
    report = _report_from(_run_agent(db, task))
    assert report.demand_provenance is DataProvenance.SIMULATED
    assert any("Demand signal is not from a verified source" in r for r in report.risks)


def test_caller_supplied_demand_facts_are_honoured(db: Session) -> None:
    task = _research_task(
        db,
        {
            "demand_summary": "Google Trends rising 12 months; 40 orders/week observed.",
            "demand_provenance": "VERIFIED",
            "research_sources": ["https://trends.google.com/pet-collars"],
        },
    )
    report = _report_from(_run_agent(db, task))
    assert report.demand_provenance is DataProvenance.VERIFIED
    assert report.demand_summary.startswith("Google Trends rising")
    assert "https://trends.google.com/pet-collars" in report.research_sources
    assert not any("Demand signal is not from a verified source" in r for r in report.risks)


def test_invalid_demand_provenance_falls_back_to_simulated(db: Session) -> None:
    task = _research_task(db, {"demand_provenance": "TOTALLY_REAL"})
    report = _report_from(_run_agent(db, task))
    assert report.demand_provenance is DataProvenance.SIMULATED


def test_thin_margin_risk_is_explained(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 11.0,
            "price_provenance": "VERIFIED",
            "suppliers": [
                {"supplier_name": "S", "unit_cost_usd": 9.0, "shipping_cost_usd": 1.0,
                 "provenance": "VERIFIED"}
            ],
        },
    )
    report = _report_from(_run_agent(db, task))
    assert report.recommendation is OpportunityRecommendation.REJECT
    assert any("below the 20% viability floor" in risk for risk in report.risks)


def test_long_shipping_window_is_flagged(db: Session) -> None:
    task = _research_task(db)  # simulated budget tier quotes 18-30 days
    report = _report_from(_run_agent(db, task))
    assert any("refund and chargeback risk" in risk for risk in report.risks)


def test_verified_facts_produce_no_provenance_risks(db: Session) -> None:
    task = _research_task(
        db,
        {
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "suppliers": [VERIFIED_QUOTE],
            "demand_summary": "Verified demand from marketplace data.",
            "demand_provenance": "VERIFIED",
        },
    )
    report = _report_from(_run_agent(db, task))
    assert report.risks == [
        "No material risk detected from the supplied facts. Advertising cost, payment "
        "fees, returns and refunds are not modelled here."
    ]


def test_agent_refuses_to_invent_a_supplier_cost(db: Session) -> None:
    """No candidates -> no reference price -> no quotes: fail, don't guess."""
    registry = _registry_with_discovery(candidates=[])
    task = _research_task(db, {"suppliers": []})
    output = _run_agent(db, task, registry=registry)

    assert output.status is TaskStatus.FAILED
    assert output.data["refused"] is True
    assert "Refusing to assume a zero landed cost" in output.data["reason"]
    assert "opportunity_report" not in output.data


def test_agent_refuses_to_invent_a_selling_price(db: Session) -> None:
    """A real quote exists, but nothing to sell at: fail rather than pick a price."""
    registry = _registry_with_discovery(
        candidates=[{"name": "Mystery gadget", "trend_score": 0.5}]
    )
    task = _research_task(db, {"suppliers": [VERIFIED_QUOTE]})
    output = _run_agent(db, task, registry=registry)

    assert output.status is TaskStatus.FAILED
    assert output.data["refused"] is True
    assert "Refusing to invent one" in output.data["reason"]


def test_agent_honours_an_explicit_product_choice(db: Session) -> None:
    task = _research_task(db, {"product_name": "Slow feeder bowl", "product_category": "pet_feeders"})
    report = _report_from(_run_agent(db, task))
    assert report.product_name == "Slow feeder bowl"
    assert report.product_category == "pet_feeders"
    # 9.99 reference price -> simulated landed cost follows that product.
    assert report.margin_analysis.recommended_selling_price_usd == pytest.approx(9.99)


def test_agent_survives_an_unknown_product_name(db: Session) -> None:
    task = _research_task(db, {"product_name": "Solar garden gnome"})
    report = _report_from(_run_agent(db, task))
    assert report.product_name == "Solar garden gnome"
    assert report.recommendation in OpportunityRecommendation


# ══════════════════════════════════════════════════════════════════════════
# 5. Shared memory: handoff preserved, reports archived separately
# ══════════════════════════════════════════════════════════════════════════


def test_handoff_preserves_latest_candidates(db: Session) -> None:
    """`AdvertisingAnalyticsAgent` reads this exact namespace/key - unchanged."""
    memory = SharedMemoryService()
    task = _research_task(db)
    _run_agent(db, task, memory=memory)

    entries = memory.list_namespace(db, namespace=RESEARCH_NAMESPACE)
    assert [entry.key for entry in entries] == [RESEARCH_KEY], (
        "the research namespace must stay a single record so downstream reads "
        "and GET /api/v1/memory/product-research keep working"
    )
    value = entries[0].value
    assert value["candidates"], "downstream marketing depends on candidates"
    assert value["source_task_id"] == task.id
    assert value["recommendation"] in {m.value for m in OpportunityRecommendation}
    assert entries[0].updated_by_agent is AgentType.PRODUCT_RESEARCH


def test_report_is_archived_in_its_own_namespace(db: Session) -> None:
    memory = SharedMemoryService()
    task = _research_task(db, {"product_name": "LED dog collar"})
    report = _report_from(_run_agent(db, task, memory=memory))

    archived = memory.list_namespace(db, namespace=REPORT_NAMESPACE)
    assert [entry.key for entry in archived] == [task.id]
    stored = ProductOpportunityReport.model_validate(archived[0].value["report"])
    assert stored == report
    assert archived[0].value["recommendation"] == report.recommendation.value
    assert archived[0].value["source_task_id"] == task.id
    # The archive never leaks into the cross-agent handoff namespace.
    assert len(memory.list_namespace(db, namespace=RESEARCH_NAMESPACE)) == 1


def test_each_run_archives_its_own_report(db: Session) -> None:
    memory = SharedMemoryService()
    first = _research_task(db, {"product_name": "LED dog collar"})
    second = _research_task(db, {"product_name": "Pet water fountain"})
    _run_agent(db, first, memory=memory)
    _run_agent(db, second, memory=memory)

    archived = memory.list_namespace(db, namespace=REPORT_NAMESPACE)
    assert {entry.key for entry in archived} == {first.id, second.id}
    assert [entry.value["product_name"] for entry in archived] == [
        "LED dog collar",
        "Pet water fountain",
    ]
    # The handoff still holds only the most recent run.
    handoff = memory.get_memory(db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY)
    assert handoff is not None and handoff.value["source_task_id"] == second.id


def test_downstream_advertising_agent_consumes_the_handoff(db: Session) -> None:
    """The cross-agent contract, verified from the consumer's side."""
    from app.agents.specialists import AdvertisingAnalyticsAgent

    memory = SharedMemoryService()
    research_task = _research_task(db, {"product_name": "Pet water fountain"})
    _run_agent(db, research_task, memory=memory)

    ad_task = Task(
        assigned_agent=AgentType.ADVERTISING_ANALYTICS,
        title="Prepare marketing",
        objective="Draft ad copy for the researched product",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        input_payload={},
    )
    db.add(ad_task)
    db.commit()
    db.refresh(ad_task)

    output = AdvertisingAnalyticsAgent().execute(
        db, ad_task, create_default_registry(), memory, MockAIProvider()
    )
    assert output.status is TaskStatus.COMPLETED
    assert output.data["product"] == "LED dog collar"  # top discovered candidate
    assert output.data["research_source"] == RESEARCH_NAMESPACE


# ══════════════════════════════════════════════════════════════════════════
# 6. Facts-driven AI content
# ══════════════════════════════════════════════════════════════════════════


def test_content_prompt_forbids_invention(db: Session) -> None:
    recorder = ContentRecorder()
    task = _research_task(db, {"product_name": "LED dog collar", "suppliers": [VERIFIED_QUOTE],
                               "target_selling_price_usd": 29.99, "price_provenance": "VERIFIED"})
    _run_agent(db, task, provider=recorder)

    assert len(recorder.prompts) == 1
    prompt = recorder.prompts[0]
    assert prompt.startswith(CONTENT_PROMPT_HEADER)
    lowered = prompt.lower()
    for forbidden in (
        "never invent supplier prices",
        "stock availability",
        "statistics",
        "never state a number that does not appear in the facts",
        "using only those facts",
    ):
        assert forbidden in lowered
    assert recorder.schemas == [GeneratedProductContent]


def test_content_prompt_carries_only_supplied_facts(db: Session) -> None:
    """Every figure in the prompt must exist in the facts bundle it declares."""
    recorder = ContentRecorder()
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = _run_agent(db, task, provider=recorder)

    prompt = recorder.prompts[0]
    facts = json.loads(prompt.split("Facts (JSON):\n", 1)[1])
    assert facts == recorder.contexts[0]["product_facts"]
    # The agent's own receipt matches what the model was shown.
    report = _report_from(output)
    assert report.generated_content.facts_used == facts

    invented = _digits_in(prompt) - _numeric_tokens(facts)
    assert not invented, f"prompt cites numbers absent from the facts: {invented}"


def test_facts_bundle_exposes_provenance_to_the_model(db: Session) -> None:
    recorder = ContentRecorder()
    task = _research_task(db)
    _run_agent(db, task, provider=recorder)
    facts = recorder.contexts[0]["product_facts"]
    assert facts["cost_provenance"] == "SIMULATED"
    assert facts["price_provenance"] == "SIMULATED"
    assert facts["demand_provenance"] == "SIMULATED"
    assert facts["in_stock"] is None
    assert facts["product_name"] == "LED dog collar"


def test_grounded_ai_copy_is_used(db: Session) -> None:
    recorder = ContentRecorder(content=_grounded_content("LED dog collar"))
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = _run_agent(db, task, provider=recorder)

    report = _report_from(output)
    assert report.generated_content.product_titles == ["LED dog collar - fast shipping"]
    assert "LED dog collar" in report.generated_content.product_description
    assert output.data["content_origin"]["ai_output"] == "accepted"
    assert output.data["content_origin"]["product_titles"] == "ai"


def test_facts_used_never_comes_from_the_model(db: Session) -> None:
    """The provenance receipt is agent-assembled; the model cannot rewrite it."""
    recorder = ContentRecorder(content=_grounded_content("LED dog collar"))
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = _run_agent(db, task, provider=recorder)

    report = _report_from(output)
    assert "hallucinated_by_model" not in report.generated_content.facts_used
    assert report.generated_content.facts_used["product_name"] == "LED dog collar"
    assert output.data["content_origin"]["facts_used"] == "agent_assembled"


def test_ungrounded_ai_copy_falls_back_to_the_draft(db: Session) -> None:
    """Copy that references no supplied fact is rejected, not published."""
    recorder = ContentRecorder(
        content=GeneratedProductContent(
            product_titles=["Best thing ever"],
            product_description="Loved by millions, sells out instantly.",
            key_bullet_points=["#1 bestseller"],
            marketing_angles=["Viral demand"],
            facts_used={},
        )
    )
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = _run_agent(db, task, provider=recorder)

    report = _report_from(output)
    assert output.data["content_origin"]["ai_output"] == "rejected_ungrounded"
    assert "#1 bestseller" not in report.generated_content.key_bullet_points
    assert "Loved by millions" not in report.generated_content.product_description
    # The replacement is the deterministic, facts-only draft.
    assert all(
        output.data["content_origin"][field] == "deterministic_draft"
        for field in (
            "product_titles",
            "product_description",
            "key_bullet_points",
            "marketing_angles",
        )
    )
    assert report.generated_content.product_titles
    assert not _digits_in(report.generated_content.product_description) - _numeric_tokens(
        report.generated_content.facts_used
    )


def test_stub_provider_output_falls_back_to_the_draft(db: Session) -> None:
    """Offline default: the mock provider's placeholder must not reach a report."""
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = _run_agent(db, task, provider=MockAIProvider())

    report = _report_from(output)
    assert "mock-" not in report.generated_content.product_description
    assert output.data["content_origin"]["ai_output"] == "rejected_ungrounded"
    assert report.generated_content.product_description
    assert report.generated_content.key_bullet_points


def test_empty_ai_content_falls_back_field_by_field(db: Session) -> None:
    """A grounded description with empty lists still yields full copy."""
    recorder = ContentRecorder(
        content=GeneratedProductContent(
            product_titles=[],
            product_description="LED dog collar at the quoted landed cost.",
            key_bullet_points=[],
            marketing_angles=[],
        )
    )
    task = _research_task(db, {"product_name": "LED dog collar"})
    output = _run_agent(db, task, provider=recorder)

    origin = output.data["content_origin"]
    assert origin["ai_output"] == "accepted"
    assert origin["product_description"] == "ai"
    assert origin["product_titles"] == "deterministic_draft"
    report = _report_from(output)
    assert report.generated_content.product_titles
    assert report.generated_content.product_description.startswith("LED dog collar")


# ══════════════════════════════════════════════════════════════════════════
# 7. Cross-agent workflow still runs end to end
# ══════════════════════════════════════════════════════════════════════════


def test_product_to_marketing_workflow_still_completes(db: Session) -> None:
    report = MainOrchestrator().run_workflow(
        db, objective="Find promising products and prepare a marketing plan"
    )
    assert report["status"] == "COMPLETED"
    assert report["workflow_type"] == "product_to_marketing"
    assert [child["agent_type"] for child in report["children"]] == [
        "PRODUCT_RESEARCH",
        "ADVERTISING_ANALYTICS",
    ]
    assert all(child["status"] == "COMPLETED" for child in report["children"])

    research_child = db.get(Task, report["children"][0]["task_id"])
    opportunity = ProductOpportunityReport.model_validate(
        research_child.output_payload["data"]["opportunity_report"]
    )
    assert opportunity.recommendation in OpportunityRecommendation

    handoff = SharedMemoryService.get_memory(
        db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY
    )
    assert handoff is not None and handoff.value["candidates"]


def test_level2_publish_workflow_still_gates(db: Session) -> None:
    """2B must not disturb the approval path that follows research."""
    orchestrator = MainOrchestrator()
    report = orchestrator.run_workflow(
        db,
        objective="Research products and publish a live ad campaign",
        parameters={"publish": True},
    )
    assert report["status"] == "WAITING_APPROVAL"
    assert report["approval_request_id"]

    resumed = orchestrator.resume_after_approval(
        db, approval_id=report["approval_request_id"], approver_identity="atharva"
    )
    assert resumed["status"] == "COMPLETED"
    assert all(child["status"] == "COMPLETED" for child in resumed["children"])


def test_research_agent_rate_limit_halts_the_workflow(db: Session) -> None:
    """2A's fail-fast behaviour applies to the upgraded agent too."""

    class RateLimitedProvider(MockAIProvider):
        def generate_structured(self, prompt, schema, context=None):
            from app.core.ai_provider import AIRateLimitError

            raise AIRateLimitError("free tier exhausted", retry_after=3600)

    orchestrator = MainOrchestrator(ai_provider=RateLimitedProvider())
    report = orchestrator.run_workflow(
        db,
        objective="Research products",
        workflow_type="custom",
        parameters={"tasks": [{"agent_type": "PRODUCT_RESEARCH"}]},
    )
    assert report["status"] == "FAILED"
    child = db.get(Task, report["children"][0]["task_id"])
    assert child.status is TaskStatus.FAILED
    assert child.retry_count == 0, "rate limits must not be retried"
    assert "rate limit" in (child.error_message or "").lower()


# ══════════════════════════════════════════════════════════════════════════
# 8. API endpoints
# ══════════════════════════════════════════════════════════════════════════


def test_research_paths_are_registered(app: FastAPI) -> None:
    paths = app.openapi()["paths"]
    assert OPPORTUNITY_REPORT_PATH in paths
    assert REPORTS_PATH in paths
    assert "post" in paths[OPPORTUNITY_REPORT_PATH]
    assert "get" in paths[REPORTS_PATH]


def test_reports_list_starts_empty(client: TestClient) -> None:
    response = client.get(REPORTS_PATH)
    assert response.status_code == 200
    assert response.json() == []


def test_post_opportunity_report_returns_a_validated_report(client: TestClient) -> None:
    response = client.post(OPPORTUNITY_REPORT_PATH, json={"objective": "Assess pet niche"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["task_status"] == "COMPLETED"
    assert body["parent_task_id"] and body["task_id"]
    assert body["summary"]

    report = ProductOpportunityReport.model_validate(body["report"])
    assert report.recommendation is OpportunityRecommendation.TEST_CAUTIOUSLY
    assert report.research_confidence is ConfidenceLevel.LOW
    assert report.margin_analysis.cost_provenance is DataProvenance.SIMULATED

    # The tracked task carries the same report.
    task = client.get(f"/api/v1/tasks/{body['task_id']}").json()
    assert task["status"] == "COMPLETED"
    assert task["assigned_agent"] == "PRODUCT_RESEARCH"
    assert task["output_payload"]["data"]["opportunity_report"]["product_name"] == (
        report.product_name
    )


def test_post_opportunity_report_with_verified_quotes(client: TestClient) -> None:
    response = client.post(
        OPPORTUNITY_REPORT_PATH,
        json={
            "objective": "Assess LED dog collar with real quotes",
            "product_name": "LED dog collar",
            "product_category": "pet_accessories",
            "target_selling_price_usd": 29.99,
            "price_provenance": "VERIFIED",
            "suppliers": [VERIFIED_QUOTE],
            "competitors": [
                {"competitor_name": "PetSmart", "observed_price_usd": 32.5,
                 "provenance": "VERIFIED"},
                {"competitor_name": "Amazon", "observed_price_usd": 27.99,
                 "provenance": "VERIFIED"},
            ],
            "demand_summary": "Marketplace sales observed directly.",
            "demand_provenance": "VERIFIED",
            "research_sources": ["https://example.com/quote-2026-11-04"],
        },
    )
    assert response.status_code == 200, response.text
    report = ProductOpportunityReport.model_validate(response.json()["report"])

    assert report.product_name == "LED dog collar"
    assert report.margin_analysis.cost_provenance is DataProvenance.VERIFIED
    assert report.margin_analysis.total_cost_usd == 6.00
    assert report.recommendation is OpportunityRecommendation.PROCEED
    assert report.research_confidence is ConfidenceLevel.HIGH
    assert report.demand_provenance is DataProvenance.VERIFIED
    assert "https://example.com/quote-2026-11-04" in report.research_sources
    assert {c.competitor_name for c in report.competitors} == {"PetSmart", "Amazon"}


def test_post_opportunity_report_archives_the_report(client: TestClient) -> None:
    created = client.post(
        OPPORTUNITY_REPORT_PATH, json={"product_name": "Pet water fountain"}
    )
    assert created.status_code == 200
    task_id = created.json()["task_id"]

    listing = client.get(REPORTS_PATH)
    assert listing.status_code == 200
    entries = listing.json()
    assert len(entries) == 1

    entry = entries[0]
    assert entry["report_id"] == task_id
    assert entry["source_task_id"] == task_id
    assert entry["product_name"] == "Pet water fountain"
    assert entry["recommendation"] == "TEST_CAUTIOUSLY"
    assert entry["research_confidence"] == "LOW"
    assert entry["cost_provenance"] == "SIMULATED"
    assert isinstance(entry["estimated_gross_margin_pct"], float)
    assert entry["created_at"] and entry["updated_at"]
    assert ProductOpportunityReport.model_validate(entry["report"])


def test_reports_listing_accumulates(client: TestClient) -> None:
    for product in ("LED dog collar", "Slow feeder bowl"):
        assert client.post(OPPORTUNITY_REPORT_PATH, json={"product_name": product}).status_code == 200

    entries = client.get(REPORTS_PATH).json()
    assert [entry["product_name"] for entry in entries] == [
        "LED dog collar",
        "Slow feeder bowl",
    ]


def test_report_archive_does_not_pollute_the_handoff_namespace(
    client: TestClient,
) -> None:
    """Guards the invariant existing 1C tests depend on."""
    client.post(OPPORTUNITY_REPORT_PATH, json={"objective": "Assess the niche"})
    entries = client.get("/api/v1/memory/product-research").json()
    assert len(entries) == 1
    assert entries[0]["key"] == RESEARCH_KEY
    assert entries[0]["value"]["candidates"]


@pytest.mark.parametrize(
    "payload",
    [
        {"target_selling_price_usd": 0},
        {"target_selling_price_usd": -5.0},
        {"product_name": ""},
        {"objective": ""},
        {"suppliers": [{"unit_cost_usd": 1.0}]},  # supplier_name required
        {"suppliers": [{"supplier_name": "S", "unit_cost_usd": -1.0}]},
        {"competitors": [{"competitor_name": "C", "observed_price_usd": 0}]},
        {"price_provenance": "GUESSED"},
        {"demand_provenance": "TRUST_ME"},
    ],
)
def test_post_opportunity_report_rejects_bad_input(
    client: TestClient, payload: dict[str, Any]
) -> None:
    response = client.post(OPPORTUNITY_REPORT_PATH, json=payload)
    assert response.status_code == 422, response.text


def test_supplier_provenance_defaults_to_estimated_at_the_api(
    client: TestClient,
) -> None:
    """Omitting provenance in a request must not yield VERIFIED costs."""
    response = client.post(
        OPPORTUNITY_REPORT_PATH,
        json={
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "suppliers": [
                {"supplier_name": "Unlabelled", "unit_cost_usd": 4.10, "shipping_cost_usd": 1.90}
            ],
        },
    )
    assert response.status_code == 200, response.text
    report = ProductOpportunityReport.model_validate(response.json()["report"])
    assert report.margin_analysis.cost_provenance is DataProvenance.ESTIMATED
    assert report.research_confidence is not ConfidenceLevel.HIGH
    assert report.recommendation is OpportunityRecommendation.TEST_CAUTIOUSLY


def test_endpoint_closes_the_provider_it_created(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request-scoped provider must not leak its connection pool."""
    from app.api.v1 import research as research_api

    recorder = ContentRecorder(content=_grounded_content("LED dog collar"))
    # An orchestrator that owns its provider, as the default one does: close()
    # releases the pool the endpoint caused to be created.
    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = recorder
    orchestrator._owns_provider = True
    monkeypatch.setattr(research_api, "_orchestrator", lambda: orchestrator)

    response = client.post(OPPORTUNITY_REPORT_PATH, json={"product_name": "LED dog collar"})
    assert response.status_code == 200, response.text
    assert recorder.closed is True


def test_endpoint_never_closes_a_provider_it_did_not_create(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 3.5 ownership rule: an injected provider belongs to its caller.

    The previous endpoint called ``orchestrator.ai_provider.close()``
    unconditionally, so an injected (shared) provider was closed behind the
    caller's back. ``with MainOrchestrator(...)`` fixes that.
    """
    from app.api.v1 import research as research_api

    recorder = ContentRecorder(content=_grounded_content("LED dog collar"))
    monkeypatch.setattr(
        research_api, "_orchestrator", lambda: MainOrchestrator(ai_provider=recorder)
    )

    assert recorder.closed is False
    response = client.post(OPPORTUNITY_REPORT_PATH, json={"product_name": "LED dog collar"})
    assert response.status_code == 200, response.text
    assert recorder.closed is False

    # ...and the provider is still usable afterwards, which is the point.
    assert recorder.generate_text("still alive") == "recorded summary"


def test_endpoint_closes_an_owned_provider_even_when_the_run_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.v1 import research as research_api

    class ExplodingProvider(MockAIProvider):
        def generate_structured(self, prompt, schema, context=None):
            raise RuntimeError("boom")

    provider = ExplodingProvider()
    closed: list[bool] = []
    original_close = provider.close

    def _close() -> None:
        closed.append(True)
        original_close()

    provider.close = _close  # type: ignore[method-assign]
    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = provider
    orchestrator._owns_provider = True
    monkeypatch.setattr(research_api, "_orchestrator", lambda: orchestrator)

    response = client.post(OPPORTUNITY_REPORT_PATH, json={})
    assert response.status_code == 502
    assert closed == [True]


def test_endpoint_leaves_an_injected_provider_open_when_the_run_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.v1 import research as research_api

    class ExplodingProvider(MockAIProvider):
        def generate_structured(self, prompt, schema, context=None):
            raise RuntimeError("boom")

    provider = ExplodingProvider()
    orchestrator = MainOrchestrator(ai_provider=provider)
    monkeypatch.setattr(research_api, "_orchestrator", lambda: orchestrator)

    response = client.post(OPPORTUNITY_REPORT_PATH, json={})
    assert response.status_code == 502
    # Never closed - but the provider is still alive and usable.
    assert provider.generate_text("alive") == "[mock-ai] alive (context keys: none)"


# ══════════════════════════════════════════════════════════════════════════
# 9. AIProvider lifecycle (the httpx.Client leak from Milestone 2A)
# ══════════════════════════════════════════════════════════════════════════


def _mock_transport_provider() -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    return OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        transport=httpx.MockTransport(handler),
    )


def test_mock_provider_close_is_a_noop() -> None:
    provider = MockAIProvider()
    provider.close()  # must not raise
    assert provider.generate_text("still usable") == "[mock-ai] still usable (context keys: none)"


def test_close_releases_the_httpx_client() -> None:
    provider = _mock_transport_provider()
    assert provider.generate_text("warm the pool") == "ok"
    assert provider._owns_client is True

    provider.close()
    assert provider._owns_client is False

    with pytest.raises(RuntimeError):
        provider.generate_text("after close")


def test_close_is_idempotent() -> None:
    provider = _mock_transport_provider()
    provider.close()
    provider.close()
    provider.close()


def test_context_manager_closes_on_exit() -> None:
    with _mock_transport_provider() as provider:
        assert provider.generate_text("inside the block") == "ok"
        assert provider._owns_client is True
    assert provider._owns_client is False
    with pytest.raises(RuntimeError):
        provider.generate_text("after the block")


def test_context_manager_closes_on_exception() -> None:
    provider = _mock_transport_provider()
    with pytest.raises(ValueError):
        with provider:
            raise ValueError("mid-flight failure")
    assert provider._owns_client is False


def test_context_manager_returns_the_same_instance() -> None:
    provider = _mock_transport_provider()
    with provider as entered:
        assert entered is provider


def test_abc_close_default_is_a_noop() -> None:
    """A provider that holds nothing needs no lifecycle implementation."""

    class MinimalProvider(AIProvider):
        def generate_structured(self, prompt, schema, context=None):
            return schema()  # type: ignore[call-arg]

        def generate_text(self, prompt, context=None):
            return "minimal"

    provider = MinimalProvider()
    provider.close()
    with provider as entered:
        assert entered is provider
        assert entered.generate_text("x") == "minimal"


def test_factory_provider_is_closeable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_ai_provider()` results can now be released by callers."""
    monkeypatch.setattr(config.settings, "ai_provider", "openai_compatible")
    monkeypatch.setattr(config.settings, "ai_api_key", "test-key")

    provider = get_ai_provider()
    assert isinstance(provider, OpenAICompatibleProvider)
    provider.close()
    assert provider._owns_client is False
