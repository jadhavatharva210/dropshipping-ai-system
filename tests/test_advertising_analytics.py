"""Phase 3: Advertising & Analytics Agent (Agent 2).

Everything here runs offline. The AI is either the deterministic
``MockAIProvider`` or an in-process recording stub, and campaign metrics are
supplied by the test - so the suite proves the *discipline*, not anything about
a real ad account.

The three properties under test are the ones that make an advertising agent
safe to run unattended:

1. **The arithmetic is derived, never asserted.** Every rate on
   :class:`CampaignPerformanceMetrics` is recomputed by the model from the raw
   numbers, divisions are guarded, and a mismatched derived value is a
   validation error.
2. **The diagnosis is a deterministic rule set.** All six
   :class:`CampaignHealthStatus` verdicts come from fixed thresholds, so the
   same metrics always produce the same advice.
3. **Money never moves without a human.** A
   :class:`BudgetRecommendation` is a Level 1 artifact; ``commit_budget`` and
   ``publish_live`` are Level 2 and pause the task in ``WAITING_APPROVAL``
   with the handler never invoked.

Plus the cross-agent contract: Agent 1's opportunity report is what makes
``break_even_roas`` knowable in ``product_to_marketing``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agents.base import AgentOutput
from app.agents.specialists import (
    AD_REPORT_NAMESPACE,
    ADVERTISING_KEY,
    ADVERTISING_NAMESPACE,
    REPORT_NAMESPACE,
    RESEARCH_KEY,
    RESEARCH_NAMESPACE,
    AdvertisingAnalyticsAgent,
)
from app.core.ai_provider import AIProvider, MockAIProvider
from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus
from app.core.tools import (
    ToolDefinition,
    ToolRegistry,
    create_default_registry,
    _allocate_ad_budget,
    _analyze_campaign_performance,
    _prepare_creative_briefs,
)
from app.models.core import ApprovalRequest, Task
from app.schemas.advertising import (
    AD_COPY_PROMPT_HEADER,
    FALLBACK_BREAK_EVEN_ROAS,
    MIN_HEALTHY_CONVERSION_RATE_PCT,
    MIN_HEALTHY_CTR_PCT,
    MIN_SCALING_CONVERSIONS,
    SCALING_ROAS,
    AdCopyVariant,
    BudgetRecommendation,
    CampaignHealthStatus,
    CampaignObjective,
    CampaignPerformanceMetrics,
    ConversionAnalysis,
    CreativeBrief,
    MarketingStrategyReport,
    TargetAudienceSegment,
    build_conversion_analysis,
    derive_campaign_metrics,
    diagnose_campaign_health,
    find_ungrounded_claims,
    grounding_tokens,
    recommend_budget,
)
from app.schemas.product_research import DataProvenance
from app.services.memory import SharedMemoryService
from app.services.orchestrator import MainOrchestrator

STRATEGY_PATH = "/api/v1/advertising/strategy-report"
REPORTS_PATH = "/api/v1/advertising/reports"

#: 29.99 price against 9.00 landed -> 20.99 contribution ->
#: break-even ROAS 29.99 / 20.99 = 1.43.
UNIT_ECONOMICS: dict[str, Any] = {
    "product_name": "LED dog collar",
    "price_usd": 29.99,
    "landed_cost_usd": 9.00,
}


# ── helpers ───────────────────────────────────────────────────────────────


def _ad_task(db: Session, payload: dict[str, Any] | None = None) -> Task:
    task = Task(
        assigned_agent=AgentType.ADVERTISING_ANALYTICS,
        title="Plan and analyse the campaign",
        objective="Plan a grounded campaign and analyse its performance",
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
    return AdvertisingAnalyticsAgent().execute(
        db,
        task,
        registry or create_default_registry(),
        memory or SharedMemoryService(),
        provider if provider is not None else MockAIProvider(),
    )


def _report(output: AgentOutput) -> MarketingStrategyReport:
    assert output.status is TaskStatus.COMPLETED, output.summary
    return MarketingStrategyReport.model_validate(output.data["marketing_report"])


def _metrics(**overrides: Any) -> CampaignPerformanceMetrics:
    """A campaign row with sane defaults; overrides change one input at a time."""
    payload: dict[str, Any] = {
        "impressions": 10000,
        "clicks": 200,  # CTR 2.00%
        "spend_usd": 100.0,
        "conversions": 10,  # CVR 5.0%
        "revenue_usd": 300.0,  # ROAS 3.00
        "provenance": DataProvenance.ESTIMATED,
        "selling_price_usd": 30.0,
        "landed_cost_usd": 10.0,  # contribution 20.00 -> break-even 1.50
    }
    payload.update(overrides)
    return CampaignPerformanceMetrics.compute(**payload)


class RecordingAdsProvider:
    """AIProvider stub that records prompts and returns a scripted plan."""

    def __init__(self, plan: Any = None) -> None:
        self.prompts: list[str] = []
        self.contexts: list[dict[str, Any] | None] = []
        self.schemas: list[type] = []
        self.plan = plan

    def generate_structured(self, prompt, schema, context=None):
        self.prompts.append(prompt)
        self.contexts.append(context)
        self.schemas.append(schema)
        if self.plan is not None:
            return self.plan
        return schema(
            headline="mock-headline", body="mock-body", cta="mock-cta"
        )

    def generate_text(self, prompt, context=None):
        return "recorded summary"


def _plan(**overrides: Any) -> Any:
    """A grounded AI ad plan for the LED dog collar at $29.99."""
    from app.agents.specialists import AdPlan

    values = {
        "headline": "LED dog collar - visible at night",
        "body": "LED dog collar with USB charging, visible at night. $29.99.",
        "cta": "Shop now",
    }
    values.update(overrides)
    return AdPlan(**values)


# ══════════════════════════════════════════════════════════════════════════
# 1. Deterministic metrics arithmetic & zero-division safety
# ══════════════════════════════════════════════════════════════════════════


def test_derived_metrics_are_computed_from_the_raw_inputs() -> None:
    metrics = _metrics()

    assert metrics.ctr_pct == 2.0  # 200 / 10000
    assert metrics.cpc_usd == 0.5  # 100 / 200
    assert metrics.cpa_usd == 10.0  # 100 / 10
    assert metrics.conversion_rate_pct == 5.0  # 10 / 200
    assert metrics.roas == 3.0  # 300 / 100
    # Contribution break-even: 30.00 / (30.00 - 10.00)
    assert metrics.break_even_roas == 1.5
    # Revenue - COGS - ad spend: 300 - (10 * 10) - 100
    assert metrics.net_margin_after_ad_spend_usd == 100.0


def test_contribution_break_even_uses_selling_price_over_gross_profit() -> None:
    """Documented formula: round(selling_price / gross_profit, 2)."""
    metrics = _metrics(selling_price_usd=29.99, landed_cost_usd=9.00)
    assert metrics.gross_profit_per_unit_usd == 20.99
    assert metrics.break_even_roas == round(29.99 / 20.99, 2) == 1.43


def test_break_even_roas_is_none_without_unit_economics() -> None:
    """Unknown landed cost must not be treated as zero (which fakes 1.00)."""
    metrics = _metrics(selling_price_usd=30.0, landed_cost_usd=None)
    assert metrics.break_even_roas is None
    assert metrics.net_margin_after_ad_spend_usd is None
    assert metrics.gross_profit_per_unit_usd is None


def test_break_even_roas_is_none_when_contribution_is_not_positive() -> None:
    """Selling at or below landed cost: no ROAS can make the campaign work."""
    at_cost = _metrics(selling_price_usd=10.0, landed_cost_usd=10.0)
    assert at_cost.break_even_roas is None

    underwater = _metrics(selling_price_usd=8.0, landed_cost_usd=10.0)
    assert underwater.gross_profit_per_unit_usd == -2.0
    assert underwater.break_even_roas is None


def test_zero_division_is_safe_everywhere() -> None:
    """No impressions, no clicks, no spend, no conversions -> no exceptions."""
    empty = CampaignPerformanceMetrics.compute(
        impressions=0,
        clicks=0,
        spend_usd=0.0,
        conversions=0,
        revenue_usd=0.0,
        provenance=DataProvenance.SIMULATED,
        selling_price_usd=29.99,
        landed_cost_usd=9.0,
    )
    assert empty.ctr_pct == 0.0
    assert empty.conversion_rate_pct == 0.0
    # Ratios with no denominator genuinely do not exist yet -> None, not 0.
    assert empty.cpc_usd is None
    assert empty.cpa_usd is None
    assert empty.roas is None
    assert empty.net_margin_after_ad_spend_usd is None
    # Unit economics are independent of campaign delivery.
    assert empty.break_even_roas == 1.43
    assert empty.has_data is False


def test_clicks_without_spend_or_conversions_is_safe() -> None:
    metrics = _metrics(spend_usd=0.0, conversions=0, revenue_usd=0.0)
    assert metrics.roas is None
    assert metrics.cpa_usd is None
    assert metrics.net_margin_after_ad_spend_usd is None
    assert metrics.cpc_usd == 0.0
    assert metrics.conversion_rate_pct == 0.0


def test_tampered_derived_metric_is_rejected() -> None:
    """A hand-written ROAS that its own inputs contradict cannot be built."""
    honest = _metrics()
    payload = honest.model_dump(mode="json")
    payload["roas"] = 9.99

    with pytest.raises(ValidationError, match="roas is inconsistent"):
        CampaignPerformanceMetrics.model_validate(payload)


def test_tampered_break_even_roas_is_rejected() -> None:
    payload = _metrics().model_dump(mode="json")
    payload["break_even_roas"] = 1.0  # flattering, and wrong
    with pytest.raises(ValidationError, match="break_even_roas is inconsistent"):
        CampaignPerformanceMetrics.model_validate(payload)


def test_none_must_stay_none_on_a_metric_that_cannot_be_derived() -> None:
    """Claiming a CPC when nothing was clicked is a fabricated number."""
    payload = _metrics(clicks=0).model_dump(mode="json")
    assert payload["cpc_usd"] is None
    payload["cpc_usd"] = 0.5  # a number the raw metrics cannot support
    with pytest.raises(ValidationError):
        CampaignPerformanceMetrics.model_validate(payload)


def test_negative_raw_inputs_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CampaignPerformanceMetrics.compute(
            impressions=-1,
            clicks=0,
            spend_usd=0.0,
            conversions=0,
            revenue_usd=0.0,
        )


def test_derive_campaign_metrics_matches_the_model() -> None:
    derived = derive_campaign_metrics(
        impressions=10000,
        clicks=200,
        spend_usd=100.0,
        conversions=10,
        revenue_usd=300.0,
        selling_price_usd=30.0,
        landed_cost_usd=10.0,
    )
    assert derived["gross_profit_per_unit_usd"] == 20.0
    assert derived["ctr_pct"] == 2.0
    assert derived["break_even_roas"] == 1.5


# ══════════════════════════════════════════════════════════════════════════
# 2. All six deterministic health diagnoses
# ══════════════════════════════════════════════════════════════════════════


def test_health_no_data_for_a_missing_row() -> None:
    assert diagnose_campaign_health(None) is CampaignHealthStatus.NO_DATA
    analysis = build_conversion_analysis(CampaignHealthStatus.NO_DATA, None)
    assert analysis.bottleneck_stage == "PRE_LAUNCH"
    assert "pre-launch" in analysis.summary.lower()
    assert analysis.recommended_optimizations  # never an empty plan


def test_health_no_data_when_no_impressions_were_served() -> None:
    metrics = _metrics(impressions=0, clicks=0, conversions=0, revenue_usd=0.0)
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.NO_DATA


def test_health_underperforming_creative() -> None:
    """CTR below 1.0% -> the creative is the bottleneck."""
    metrics = _metrics(impressions=10000, clicks=50)  # CTR 0.50%
    assert metrics.ctr_pct < MIN_HEALTHY_CTR_PCT
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.UNDERPERFORMING_CREATIVE

    analysis = build_conversion_analysis(
        CampaignHealthStatus.UNDERPERFORMING_CREATIVE, metrics
    )
    assert analysis.bottleneck_stage == "CREATIVE"
    assert "0.50%" in analysis.summary


def test_health_underperforming_conversion() -> None:
    """Good CTR but CVR below 1.5% -> the offer/landing page is the bottleneck."""
    metrics = _metrics(impressions=10000, clicks=200, conversions=2, revenue_usd=60.0)
    assert metrics.ctr_pct == 2.0
    assert metrics.conversion_rate_pct == 1.0 < MIN_HEALTHY_CONVERSION_RATE_PCT
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.UNDERPERFORMING_CONVERSION

    analysis = build_conversion_analysis(
        CampaignHealthStatus.UNDERPERFORMING_CONVERSION, metrics
    )
    assert analysis.bottleneck_stage == "CONVERSION"
    assert "landing page" in analysis.summary.lower()


def test_health_unprofitable_pause_against_break_even() -> None:
    """ROAS below contribution break-even -> pause, even though CTR/CVR are fine."""
    metrics = _metrics(revenue_usd=120.0)  # ROAS 1.20 against break-even 1.50
    assert metrics.roas == 1.2 < metrics.break_even_roas
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.UNPROFITABLE_PAUSE

    analysis = build_conversion_analysis(CampaignHealthStatus.UNPROFITABLE_PAUSE, metrics)
    assert analysis.bottleneck_stage == "PROFITABILITY"
    assert "1.50" in analysis.summary
    assert "pause" in analysis.summary.lower()


def test_health_unprofitable_pause_uses_fallback_when_break_even_unknown() -> None:
    """Unknown unit economics drop back to the documented 1.2 floor."""
    metrics = _metrics(
        revenue_usd=100.0, selling_price_usd=None, landed_cost_usd=None
    )  # ROAS 1.00
    assert metrics.break_even_roas is None
    assert metrics.roas == 1.0 < FALLBACK_BREAK_EVEN_ROAS
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.UNPROFITABLE_PAUSE


def test_health_healthy() -> None:
    """Profitable, converting, but only 4 conversions -> not a scaling case yet."""
    metrics = _metrics(conversions=4, revenue_usd=300.0, clicks=200)
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.HEALTHY
    analysis = build_conversion_analysis(CampaignHealthStatus.HEALTHY, metrics)
    assert analysis.bottleneck_stage == "NONE"


def test_health_healthy_between_break_even_and_scaling() -> None:
    """Profitable but not yet a scale-up: ROAS 2.0 < 2.5."""
    metrics = _metrics(revenue_usd=200.0, conversions=10)
    assert metrics.roas == 2.0
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.HEALTHY


def test_health_scaling_candidate() -> None:
    metrics = _metrics(revenue_usd=300.0, conversions=5)  # ROAS 3.0, 5 conversions
    assert metrics.roas >= SCALING_ROAS
    assert metrics.conversions >= MIN_SCALING_CONVERSIONS
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.SCALING_CANDIDATE

    analysis = build_conversion_analysis(CampaignHealthStatus.SCALING_CANDIDATE, metrics)
    assert analysis.bottleneck_stage == "SCALING"
    assert "20-30%" in " ".join(analysis.recommended_optimizations)


def test_health_scaling_needs_enough_conversions() -> None:
    """A high ROAS on three conversions is luck, not a trend."""
    metrics = _metrics(revenue_usd=300.0, conversions=3)
    assert metrics.roas == 3.0
    assert metrics.conversions < MIN_SCALING_CONVERSIONS
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.HEALTHY


def test_health_scaling_must_clear_break_even_too() -> None:
    """ROAS 2.6 clears the 2.5 threshold but not a 3.0 break-even."""
    metrics = _metrics(
        revenue_usd=260.0,
        conversions=6,
        selling_price_usd=30.0,
        landed_cost_usd=20.0,  # contribution 10 -> break-even 3.00
    )
    assert metrics.break_even_roas == 3.0
    assert metrics.roas == 2.6 < metrics.break_even_roas
    assert diagnose_campaign_health(metrics) is CampaignHealthStatus.UNPROFITABLE_PAUSE


def test_every_category_is_reachable() -> None:
    """The six statuses are exhaustive and all six are produced by the rules."""
    reached = {
        diagnose_campaign_health(None),
        diagnose_campaign_health(_metrics(impressions=0, clicks=0)),
        diagnose_campaign_health(_metrics(clicks=50)),
        diagnose_campaign_health(_metrics(conversions=1, revenue_usd=30.0)),
        diagnose_campaign_health(_metrics(revenue_usd=50.0)),
        diagnose_campaign_health(_metrics(revenue_usd=300.0, conversions=5)),
        diagnose_campaign_health(_metrics(conversions=4, revenue_usd=200.0)),
    }
    assert reached == set(CampaignHealthStatus)
    assert len(reached) == 6


def test_health_is_order_independent_of_repeat_calls() -> None:
    metrics = _metrics(revenue_usd=120.0)
    verdicts = {diagnose_campaign_health(metrics) for _ in range(5)}
    assert verdicts == {CampaignHealthStatus.UNPROFITABLE_PAUSE}


# ══════════════════════════════════════════════════════════════════════════
# 3. Grounding gate on ad copy and creative briefs
# ══════════════════════════════════════════════════════════════════════════


def test_forbidden_claims_are_detected() -> None:
    facts = dict(UNIT_ECONOMICS)
    violations = find_ungrounded_claims("20% off today only", facts)
    # Both fabrications, and the invented number behind the discount.
    assert violations[:2] == [
        "invented discount: '20% off'",
        "invented urgency: 'today only'",
    ]
    assert any("number not present" in reason for reason in violations)
    assert any(
        "fabricated review" in reason
        for reason in find_ungrounded_claims("Rated 4.8 by our customers", facts)
    )
    assert any(
        "fabricated social proof" in reason
        for reason in find_ungrounded_claims("Trusted by thousands of dog owners", facts)
    )
    assert any(
        "invented freebie" in reason
        for reason in find_ungrounded_claims("Free shipping on every order", facts)
    )
    assert any(
        "unverified guarantee" in reason
        for reason in find_ungrounded_claims("100% money-back guarantee", facts)
    )


def test_grounded_copy_passes_the_gate() -> None:
    facts = dict(UNIT_ECONOMICS, product_name="LED dog collar")
    grounded = "LED dog collar: visible at night. Suggested price $29.99."
    assert find_ungrounded_claims(grounded, facts) == []


def test_invented_number_is_detected() -> None:
    facts = dict(UNIT_ECONOMICS)
    violations = find_ungrounded_claims("LED dog collar at $19.99", facts)
    assert any("number not present" in reason for reason in violations)


def test_supplied_numbers_are_recognised_in_any_rendering() -> None:
    facts = {"product_name": "LED dog collar", "recommended_selling_price_usd": 29.99}
    assert find_ungrounded_claims("Yours for 29.99 today", facts) == []
    assert find_ungrounded_claims("Yours for $29.99 today", facts) == []


def test_ungrounded_ai_copy_is_rejected_and_the_draft_is_used(db: Session) -> None:
    """A hard-coded discount never reaches the report."""
    provider = RecordingAdsProvider(
        plan=_plan(
            headline="LED dog collar - 50% off",
            body="LED dog collar, now 50% off, rated 4.9 stars.",
        )
    )
    task = _ad_task(db, dict(UNIT_ECONOMICS))
    report = _report(_run_agent(db, task, provider=provider))

    assert report.content_origin.startswith("deterministic_grounded_draft")
    assert "rejected_ungrounded" in report.content_origin
    rendered = " ".join(
        f"{variant.headline} {variant.primary_text} {variant.hook}"
        for variant in report.ad_copy_variants
    ).lower()
    assert "50% off" not in rendered
    assert "4.9 stars" not in rendered
    assert "rated" not in rendered


def test_fabricated_reviews_are_rejected(db: Session) -> None:
    provider = RecordingAdsProvider(
        plan=_plan(
            headline="LED dog collar",
            body="LED dog collar - loved by 10,000 happy customers with 5-star reviews.",
        )
    )
    report = _report(_run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS)), provider=provider))

    assert "rejected_ungrounded" in report.content_origin
    assert find_ungrounded_claims(
        " ".join(variant.primary_text for variant in report.ad_copy_variants),
        report.facts_used,
    ) == []


def test_grounded_ai_copy_is_accepted(db: Session) -> None:
    grounded = _plan(
        headline="LED dog collar - visible at night",
        body="LED dog collar, visible at night. Suggested price $29.99.",
    )
    provider = RecordingAdsProvider(plan=grounded)
    report = _report(_run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS)), provider=provider))

    assert report.content_origin.startswith("ai_ad_copy_accepted")
    assert report.ad_copy_variants[0].headline == "LED dog collar - visible at night"
    # The deterministic drafts are kept behind the accepted AI variant.
    assert len(report.ad_copy_variants) > 1


def test_mock_provider_copy_is_rejected_as_ungrounded(db: Session) -> None:
    """The offline stub cites no fact, so the grounded draft wins."""
    report = _report(_run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS))))
    assert "rejected_ungrounded" in report.content_origin
    assert "mock-headline" not in json.dumps(report.model_dump(mode="json"))


def test_copy_prompt_states_the_fabrication_rules(db: Session) -> None:
    provider = RecordingAdsProvider()
    _run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS)), provider=provider)

    prompt = provider.prompts[0]
    assert prompt.startswith(AD_COPY_PROMPT_HEADER)
    lowered = prompt.lower()
    for forbidden in (
        "do not invent claims, statistics, reviews or discounts",
        "never invent discounts",
        "customer reviews",
        "star ratings",
        "never state a number that does not appear in the facts",
    ):
        assert forbidden in lowered
    # The facts handed to the model are the same receipt the report publishes.
    facts_in_prompt = json.loads(prompt.split("Supplied facts (JSON):\n", 1)[1])
    assert facts_in_prompt["product_name"] == "LED dog collar"
    assert facts_in_prompt["recommended_selling_price_usd"] == 29.99


def test_report_facts_used_match_the_facts_shown_to_the_model(db: Session) -> None:
    provider = RecordingAdsProvider()
    report = _report(_run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS)), provider=provider))
    assert provider.contexts[0]["product_facts"] == report.facts_used


def test_creative_briefs_are_grounded_in_supplied_facts() -> None:
    result = _prepare_creative_briefs(
        {
            "facts": {
                "product_name": "LED dog collar",
                "recommended_selling_price_usd": 29.99,
                "shipping_days_min": 8,
                "shipping_days_max": 14,
                "estimated_gross_margin_pct": 40.0,
            },
            "benefits": ["USB rechargeable"],
            "platforms": ["meta"],
        }
    )

    briefs = [CreativeBrief.model_validate(item) for item in result["creative_briefs"]]
    assert briefs
    for brief in briefs:
        # Every brief's *claim-bearing* text is grounded: no discount, review or
        # guarantee language, and only supplied numbers. (`compliance_notes` is
        # excluded on purpose - it exists to name the forbidden claims, e.g.
        # "no discount... may appear", so scanning it would flag the warning
        # itself rather than the copy.)
        assert find_ungrounded_claims(
            " ".join([brief.visual_hook, brief.key_message, *brief.shot_list]),
            result["facts_used"],
        ) == []
        assert any(
            "no discount" in note.lower() for note in brief.compliance_notes
        )

    audiences = [
        TargetAudienceSegment.model_validate(item)
        for item in result["target_audiences"]
    ]
    assert audiences[0].platforms == ["meta"]


def test_creative_briefs_do_not_invent_audience_details() -> None:
    """No interests/pain points were supplied, so none may appear."""
    result = _prepare_creative_briefs({"facts": {"product_name": "Pet water fountain"}})
    segment = TargetAudienceSegment.model_validate(result["target_audiences"][0])
    assert segment.interests == []
    assert segment.pain_points == []
    assert segment.age_range == ""
    assert "nothing was invented" in result["note"].lower()


def test_creative_briefs_flag_unverified_costs_in_compliance() -> None:
    """SIMULATED costs must be flagged, not presented as fact."""
    result = _prepare_creative_briefs(
        {
            "facts": {
                "product_name": "LED dog collar",
                "recommended_selling_price_usd": 29.99,
                "cost_provenance": "SIMULATED",
                "price_provenance": "ESTIMATED",
            }
        }
    )
    notes = " ".join(result["creative_briefs"][0]["compliance_notes"]).lower()
    assert "simulated offline" in notes
    assert "estimated - not a live quote" in notes


def test_ai_brief_generation_is_not_asked_to_invent_claims() -> None:
    """The copy prompt is the only AI step, and it carries the hard rules."""
    lowered = AD_COPY_PROMPT_HEADER.lower()
    assert "never invent discounts, coupons, urgency" in lowered
    assert "social proof" in lowered


def test_grounding_tokens_cover_names_and_numbers() -> None:
    tokens = grounding_tokens(
        {"product_name": "LED dog collar", "recommended_selling_price_usd": 29.99}
    )
    assert "led dog collar" in tokens
    assert "collar" in tokens
    assert "29.99" in tokens


# ══════════════════════════════════════════════════════════════════════════
# 4. Cross-agent handoff: Agent 1 -> Agent 2 unit economics
# ══════════════════════════════════════════════════════════════════════════


def _run_research(db: Session, memory: SharedMemoryService, payload: dict[str, Any]) -> Task:
    from app.agents.specialists import ProductResearchAgent

    task = Task(
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Assess the product",
        objective="Assess margin and draft listing content",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        input_payload=payload,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    ProductResearchAgent().execute(
        db, task, create_default_registry(), memory, MockAIProvider()
    )
    db.refresh(task)
    return task


def test_product_to_marketing_workflow_computes_break_even_roas(
    db: Session,
) -> None:
    """Agent 1's report is what makes break_even_roas knowable for Agent 2."""
    orchestrator = MainOrchestrator()
    workflow = orchestrator.run_workflow(
        db,
        objective="Find promising products and prepare a marketing plan",
        workflow_type="product_to_marketing",
        parameters={
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "suppliers": [
                {
                    "supplier_name": "Shenzhen BrightPet Co",
                    "unit_cost_usd": 4.10,
                    "shipping_cost_usd": 1.90,
                    "shipping_days_min": 8,
                    "shipping_days_max": 14,
                    "in_stock": True,
                    "provenance": "VERIFIED",
                }
            ],
        },
    )
    assert workflow["status"] == "COMPLETED"

    ad_child = db.get(Task, workflow["children"][1]["task_id"])
    assert ad_child.assigned_agent is AgentType.ADVERTISING_ANALYTICS
    report = MarketingStrategyReport.model_validate(
        ad_child.output_payload["data"]["marketing_report"]
    )

    # 29.99 price against 6.00 landed cost -> 23.99 contribution -> 1.25.
    assert report.facts_used["total_landed_cost_usd"] == 6.0
    assert report.facts_used["gross_profit_per_unit_usd"] == 23.99
    assert report.facts_used["cost_provenance"] == "VERIFIED"
    assert report.facts_used["supplier_name"] == "Shenzhen BrightPet Co"

    # Pre-launch: the report is planning, not a performance claim.
    assert report.performance_metrics is None
    assert report.conversion_analysis.health_status is CampaignHealthStatus.NO_DATA
    assert "1.25" in report.marketing_strategy
    assert "Contribution break-even ROAS is 1.25" in report.marketing_strategy


def test_metrics_analysis_uses_the_handoff_unit_economics_in_a_workflow(
    db: Session,
) -> None:
    """Post-launch metrics + Agent 1's costs produce a real break-even verdict."""
    orchestrator = MainOrchestrator()
    workflow = orchestrator.run_workflow(
        db,
        objective="Analyse the live campaign",
        workflow_type="product_to_marketing",
        parameters={
            "product_name": "LED dog collar",
            "target_selling_price_usd": 29.99,
            "suppliers": [
                {
                    "supplier_name": "Shenzhen BrightPet Co",
                    "unit_cost_usd": 4.10,
                    "shipping_cost_usd": 1.90,
                    "provenance": "VERIFIED",
                }
            ],
            # ROAS 1.00, below the 1.25 contribution break-even.
            "impressions": 20000,
            "clicks": 400,
            "spend_usd": 200.0,
            "conversions": 10,
            "revenue_usd": 200.0,
        },
    )
    ad_child = db.get(Task, workflow["children"][1]["task_id"])
    report = MarketingStrategyReport.model_validate(
        ad_child.output_payload["data"]["marketing_report"]
    )

    assert report.performance_metrics is not None
    assert report.performance_metrics.roas == 1.0
    assert report.performance_metrics.break_even_roas == 1.25
    assert report.conversion_analysis.health_status is CampaignHealthStatus.UNPROFITABLE_PAUSE
    assert report.facts_used["total_landed_cost_usd"] == 6.0


def test_agent_reads_the_research_handoff_and_archive_directly(db: Session) -> None:
    memory = SharedMemoryService()
    research_task = _run_research(
        db,
        memory,
        {
            "target_selling_price_usd": 39.99,
            "suppliers": [
                {
                    "supplier_name": "Verified Collar Co",
                    "unit_cost_usd": 9.00,
                    "shipping_cost_usd": 3.00,
                    "in_stock": True,
                    "provenance": "VERIFIED",
                }
            ],
        },
    )
    handoff = memory.get_memory(db, namespace=RESEARCH_NAMESPACE, key=RESEARCH_KEY)
    assert handoff is not None and handoff.value["source_task_id"] == research_task.id

    report = _report(_run_agent(db, _ad_task(db, {}), memory=memory))

    # The name comes from Agent 1's discovered candidates (the 2A contract);
    # the unit economics come from the archived report it points at.
    assert report.product_name == "LED dog collar"
    assert report.facts_used["supplier_name"] == "Verified Collar Co"
    assert report.facts_used["unit_cost_usd"] == 9.0
    assert report.facts_used["total_landed_cost_usd"] == 12.0
    assert report.facts_used["gross_profit_per_unit_usd"] == 27.99
    assert report.facts_used["cost_provenance"] == "VERIFIED"
    assert report.facts_used["in_stock"] is True


def test_explicit_task_input_outranks_the_research_archive(db: Session) -> None:
    memory = SharedMemoryService()
    _run_research(
        db,
        memory,
        {
            "product_name": "Pet water fountain",
            "target_selling_price_usd": 39.99,
            "suppliers": [
                {
                    "supplier_name": "Verified Fountain Co",
                    "unit_cost_usd": 9.00,
                    "shipping_cost_usd": 3.00,
                    "provenance": "VERIFIED",
                }
            ],
        },
    )
    report = _report(
        _run_agent(
            db,
            _ad_task(
                db,
                {
                    "product_name": "Slow feeder bowl",
                    "price_usd": 19.99,
                    "landed_cost_usd": 4.99,
                },
            ),
        memory=memory,
        )
    )
    assert report.product_name == "Slow feeder bowl"
    assert report.facts_used["recommended_selling_price_usd"] == 19.99
    assert report.facts_used["total_landed_cost_usd"] == 4.99
    assert report.facts_used["gross_profit_per_unit_usd"] == 15.0


def test_without_research_the_unit_economics_are_unknown_not_zero(db: Session) -> None:
    report = _report(_run_agent(db, _ad_task(db, {"product_name": "LED dog collar"})))
    assert "total_landed_cost_usd" not in report.facts_used
    assert "gross_profit_per_unit_usd" not in report.facts_used
    assert "Break-even ROAS is unknown" in report.marketing_strategy
    # Budget therefore falls back to the documented floor, with a note saying so.
    assert report.budget_considerations.recommended_daily_budget_usd == 10.0
    assert "not supplied" in report.budget_considerations.safety_note


def test_report_is_archived_and_handoff_is_published(db: Session) -> None:
    memory = SharedMemoryService()
    task = _ad_task(db, dict(UNIT_ECONOMICS))
    report = _report(_run_agent(db, task, memory=memory))

    archived = memory.list_namespace(db, namespace=AD_REPORT_NAMESPACE)
    assert [entry.key for entry in archived] == [task.id]
    assert MarketingStrategyReport.model_validate(archived[0].value["report"]) == report
    assert archived[0].value["source_task_id"] == task.id
    assert archived[0].value["health_status"] == "NO_DATA"

    handoff = memory.get_memory(db, namespace=ADVERTISING_NAMESPACE, key=ADVERTISING_KEY)
    assert handoff is not None
    assert handoff.value["product_name"] == "LED dog collar"
    assert handoff.value["requires_human_approval"] is True
    # The two namespaces are distinct: history never pollutes the handoff.
    assert AD_REPORT_NAMESPACE != ADVERTISING_NAMESPACE


def test_existing_hyphenated_namespaces_are_preserved() -> None:
    """Locked-in: the handoff namespace spelling must not change."""
    assert RESEARCH_NAMESPACE == "product-research"
    assert ADVERTISING_NAMESPACE == "advertising-analytics"
    assert REPORT_NAMESPACE == "product_research_reports"
    assert AD_REPORT_NAMESPACE == "marketing_reports"


def test_legacy_keys_survive_alongside_the_report(db: Session) -> None:
    output = _run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS)))
    assert output.status is TaskStatus.COMPLETED
    for key in ("ad_plan", "product_facts", "ad_copy", "product", "research_source"):
        assert key in output.data, f"legacy key {key!r} disappeared"
    assert set(output.data["ad_plan"]) == {"headline", "body", "cta"}
    assert output.data["product_facts"]["price_usd"] == 29.99
    assert MarketingStrategyReport.model_validate(output.data["marketing_report"])


# ══════════════════════════════════════════════════════════════════════════
# 5. Level 1 recommendation vs Level 2 approval gate
# ══════════════════════════════════════════════════════════════════════════


def test_budget_recommendation_is_a_level_1_output(db: Session) -> None:
    """Producing a budget spends nothing and needs no approval."""
    task = _ad_task(db, dict(UNIT_ECONOMICS))
    output = _run_agent(db, task)

    assert output.status is TaskStatus.COMPLETED
    assert output.requires_approval is False
    assert output.approval_request_id is None
    db.refresh(task)
    # A Level 1 tool moves a PENDING task to IN_PROGRESS; the orchestrator owns
    # the COMPLETED transition (asserted by the workflow/API tests below).
    assert task.status is TaskStatus.IN_PROGRESS

    budget = _report(output).budget_considerations
    assert isinstance(budget, BudgetRecommendation)
    assert budget.requires_human_approval is True  # a recommendation, always
    assert budget.total_planned_spend_usd == round(
        budget.recommended_daily_budget_usd * budget.test_duration_days, 2
    )
    assert "nothing has been spent" in budget.safety_note.lower()
    # No approval request was created by a Level 1 run.
    assert db.query(ApprovalRequest).count() == 0


def test_commit_budget_pauses_and_commits_nothing(db: Session) -> None:
    calls = {"count": 0}

    def spy_allocation(parameters):
        calls["count"] += 1
        return {"allocation_id": "spy", "status": "allocated"}

    registry = create_default_registry()
    registry.register(
        ToolDefinition(
            name="allocate_ad_budget",
            action_name="allocate_ad_budget",
            description="spy allocation",
            allowed_agents={AgentType.ADVERTISING_ANALYTICS},
            handler=spy_allocation,
        )
    )

    task = _ad_task(db, {**UNIT_ECONOMICS, "commit_budget": True})
    output = _run_agent(db, task, registry=registry)

    assert output.status is TaskStatus.WAITING_APPROVAL
    assert output.requires_approval is True
    assert output.approval_request_id
    assert calls["count"] == 0, "handler must not run before approval"
    assert "budget_allocation" not in output.data

    db.refresh(task)
    assert task.status is TaskStatus.WAITING_APPROVAL
    approval = db.get(ApprovalRequest, output.approval_request_id)
    assert approval is not None and approval.status is ApprovalStatus.PENDING
    assert approval.action_description  # the queue shows what is being asked

    # The report is still delivered: the analysis is what a human reviews.
    assert MarketingStrategyReport.model_validate(output.data["marketing_report"])


def test_publish_live_pauses_and_makes_no_change(db: Session) -> None:
    task = _ad_task(db, {**UNIT_ECONOMICS, "publish_live": True})
    output = _run_agent(db, task)

    assert output.status is TaskStatus.WAITING_APPROVAL
    assert output.approval_request_id
    assert "campaign" not in output.data
    db.refresh(task)
    assert task.status is TaskStatus.WAITING_APPROVAL


def test_both_level2_flags_gate_on_the_first_one(db: Session) -> None:
    """publish_live is requested first, so it is what a human is asked about."""
    task = _ad_task(db, {**UNIT_ECONOMICS, "publish_live": True, "commit_budget": True})
    output = _run_agent(db, task)
    assert output.status is TaskStatus.WAITING_APPROVAL
    assert db.query(ApprovalRequest).count() == 1


def test_commit_budget_resumes_cleanly_after_approval(db: Session) -> None:
    orchestrator = MainOrchestrator()
    workflow = orchestrator.run_workflow(
        db,
        objective="Commit the tested ad budget",
        workflow_type="custom",
        parameters={
            "tasks": [{"agent_type": AgentType.ADVERTISING_ANALYTICS.value}],
            **UNIT_ECONOMICS,
            "commit_budget": True,
        },
    )
    assert workflow["status"] == "WAITING_APPROVAL"
    approval_id = workflow["approval_request_id"]
    assert approval_id

    resumed = orchestrator.resume_after_approval(
        db, approval_id=approval_id, approver_identity="atharva"
    )
    assert resumed["status"] == "COMPLETED"
    assert all(child["status"] == "COMPLETED" for child in resumed["children"])

    approval = db.get(ApprovalRequest, approval_id)
    assert approval.status is ApprovalStatus.APPROVED
    assert approval.execution_result["status"] == "allocated"
    assert approval.execution_result["mock"] is True
    assert approval.execution_result["daily_budget_usd"] > 0


def test_publish_live_resumes_cleanly_after_approval(db: Session) -> None:
    from app.core.tools import _mock_publish_ad_campaign

    orchestrator = MainOrchestrator()
    workflow = orchestrator.run_workflow(
        db,
        objective="Publish the campaign",
        workflow_type="custom",
        parameters={
            "tasks": [{"agent_type": AgentType.ADVERTISING_ANALYTICS.value}],
            **UNIT_ECONOMICS,
            "publish_live": True,
        },
    )
    resumed = orchestrator.resume_after_approval(
        db, approval_id=workflow["approval_request_id"], approver_identity="atharva"
    )
    assert resumed["status"] == "COMPLETED"
    approval = db.get(ApprovalRequest, workflow["approval_request_id"])
    assert approval.execution_result["status"] == _mock_publish_ad_campaign({})["status"]


def test_rejected_approval_never_allocates(db: Session) -> None:
    from app.services.approval import ApprovalService

    task = _ad_task(db, {**UNIT_ECONOMICS, "commit_budget": True})
    output = _run_agent(db, task)
    approval_id = output.approval_request_id

    ApprovalService.reject_request(db, approval_id, "atharva", "budget too high")
    approval = db.get(ApprovalRequest, approval_id)
    assert approval.status is ApprovalStatus.REJECTED
    assert approval.execution_result is None


def test_allocation_handler_only_runs_when_actually_reached() -> None:
    """Direct handler test: the mock never moves real money."""
    result = _allocate_ad_budget(
        {
            "daily_budget_usd": 20.0,
            "test_duration_days": 5,
            "allocation_by_platform": {"meta": 60.0, "tiktok": 40.0},
        }
    )
    assert result["mock"] is True
    assert result["total_planned_spend_usd"] == 100.0
    assert result["allocation_by_platform"] == {"meta": 60.0, "tiktok": 40.0}


def test_budget_recommendation_uses_unit_economics() -> None:
    budget = recommend_budget(gross_profit_per_unit_usd=23.99, platforms=["meta"])
    assert budget.recommended_daily_budget_usd == 23.99
    assert budget.total_planned_spend_usd == round(23.99 * 7, 2)
    assert budget.allocation_by_platform == {"meta": budget.total_planned_spend_usd}
    assert budget.requires_human_approval is True


def test_budget_allocation_sums_exactly_to_the_planned_total() -> None:
    budget = recommend_budget(
        gross_profit_per_unit_usd=10.0, platforms=["meta", "tiktok", "google"], test_duration_days=3
    )
    assert sum(budget.allocation_by_platform.values()) == pytest.approx(
        budget.total_planned_spend_usd
    )


def test_explicit_daily_budget_wins_but_stays_a_recommendation() -> None:
    budget = recommend_budget(daily_budget_usd=25.0, test_duration_days=4)
    assert budget.recommended_daily_budget_usd == 25.0
    assert budget.total_planned_spend_usd == 100.0
    assert "caller-supplied daily budget" in budget.safety_note


# ══════════════════════════════════════════════════════════════════════════
# 6. HTTPX API tests
# ══════════════════════════════════════════════════════════════════════════


def test_advertising_paths_are_registered(app: FastAPI) -> None:
    paths = app.openapi()["paths"]
    assert STRATEGY_PATH in paths
    assert REPORTS_PATH in paths
    assert "post" in paths[STRATEGY_PATH]
    assert "get" in paths[REPORTS_PATH]


def test_reports_list_starts_empty(client: TestClient) -> None:
    response = client.get(REPORTS_PATH)
    assert response.status_code == 200
    assert response.json() == []


def test_post_pre_launch_strategy_report(client: TestClient) -> None:
    response = client.post(
        STRATEGY_PATH,
        json={
            "objective": "Plan the collar campaign",
            **UNIT_ECONOMICS,
            "benefits": ["USB rechargeable", "visible at night"],
            "platforms": ["meta", "tiktok"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["task_status"] == "COMPLETED"
    assert body["approval_request_id"] is None
    report = MarketingStrategyReport.model_validate(body["report"])
    assert report.product_name == "LED dog collar"
    assert report.target_audiences and report.creative_briefs
    assert report.ad_copy_variants[0].facts_referenced
    assert report.conversion_analysis.health_status is CampaignHealthStatus.NO_DATA
    assert report.budget_considerations.requires_human_approval is True
    assert report.performance_metrics is None


def test_post_post_launch_metrics_analysis(client: TestClient) -> None:
    response = client.post(
        STRATEGY_PATH,
        json={
            "objective": "Analyse the live campaign",
            **UNIT_ECONOMICS,
            "impressions": 10000,
            "clicks": 200,
            "spend_usd": 100.0,
            "conversions": 5,
            "revenue_usd": 300.0,
            "metrics_provenance": "ESTIMATED",
        },
    )
    assert response.status_code == 200, response.text
    report = MarketingStrategyReport.model_validate(response.json()["report"])

    assert report.performance_metrics is not None
    assert report.performance_metrics.ctr_pct == 2.0
    assert report.performance_metrics.roas == 3.0
    assert report.performance_metrics.break_even_roas == 1.43
    assert report.conversion_analysis.health_status is CampaignHealthStatus.SCALING_CANDIDATE


def test_post_commit_budget_returns_waiting_approval(client: TestClient) -> None:
    response = client.post(
        STRATEGY_PATH,
        json={**UNIT_ECONOMICS, "commit_budget": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_status"] == "WAITING_APPROVAL"
    assert body["approval_request_id"]
    # The report is still returned, so a human can review what they approve.
    assert MarketingStrategyReport.model_validate(body["report"])


def test_post_publish_live_returns_waiting_approval(client: TestClient) -> None:
    response = client.post(STRATEGY_PATH, json={**UNIT_ECONOMICS, "publish_live": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["task_status"] == "WAITING_APPROVAL"
    assert body["approval_request_id"]


def test_approved_budget_resume_through_the_api(client: TestClient) -> None:
    created = client.post(STRATEGY_PATH, json={**UNIT_ECONOMICS, "commit_budget": True})
    approval_id = created.json()["approval_request_id"]

    resume = client.post(
        f"/api/v1/orchestrator/approvals/{approval_id}/resume",
        json={"approver_identity": "atharva"},
    )
    assert resume.status_code == 200, resume.text
    assert resume.json()["status"] == "COMPLETED"


def test_reports_list_returns_the_archived_reports(client: TestClient) -> None:
    first = client.post(STRATEGY_PATH, json={**UNIT_ECONOMICS, "product_name": "LED dog collar"})
    second = client.post(
        STRATEGY_PATH,
        json={
            "product_name": "Pet water fountain",
            "price_usd": 39.99,
            "landed_cost_usd": 12.0,
            "impressions": 10000,
            "clicks": 200,
            "spend_usd": 100.0,
            "conversions": 5,
            "revenue_usd": 300.0,
        },
    )
    assert first.status_code == 200 and second.status_code == 200

    listed = client.get(REPORTS_PATH)
    assert listed.status_code == 200
    entries = listed.json()
    assert [entry["product_name"] for entry in entries] == [
        "LED dog collar",
        "Pet water fountain",
    ]
    assert entries[0]["health_status"] == "NO_DATA"
    assert entries[0]["roas"] is None
    assert entries[1]["health_status"] == "SCALING_CANDIDATE"
    assert entries[1]["roas"] == 3.0
    assert entries[0]["report_id"] == first.json()["task_id"]
    assert entries[1]["report"]["budget_considerations"]["requires_human_approval"] is True


def test_reports_list_ignores_unrelated_records(
    client: TestClient, db: Session
) -> None:
    SharedMemoryService.set_memory(
        db,
        namespace=AD_REPORT_NAMESPACE,
        key="not-a-report",
        value={"something": "else"},
        agent_type=AgentType.ADVERTISING_ANALYTICS,
    )
    response = client.get(REPORTS_PATH)
    assert response.status_code == 200
    assert response.json() == []


def test_report_endpoint_rejects_a_negative_metric(client: TestClient) -> None:
    response = client.post(STRATEGY_PATH, json={**UNIT_ECONOMICS, "impressions": -5})
    assert response.status_code == 422


def test_strategy_report_endpoint_rejects_an_empty_objective(client: TestClient) -> None:
    response = client.post(STRATEGY_PATH, json={"objective": ""})
    assert response.status_code == 422


def test_endpoint_closes_the_provider_it_created(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.api.v1 import advertising as advertising_api

    closed: list[bool] = []

    class RecordingProvider(MockAIProvider):
        def close(self) -> None:  # type: ignore[override]
            closed.append(True)

    provider = RecordingProvider()
    monkeypatch.setattr(
        advertising_api, "_orchestrator", lambda: MainOrchestrator(ai_provider=provider)
    )
    client.post(STRATEGY_PATH, json=dict(UNIT_ECONOMICS))
    # Injected provider: the endpoint's `with orchestrator` must NOT close it.
    assert closed == []


def test_endpoint_closes_an_owned_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request-scoped provider must not leak its connection pool."""
    from app.api.v1 import advertising as advertising_api

    closed: list[bool] = []

    class OwnedProvider(MockAIProvider):
        def close(self) -> None:  # type: ignore[override]
            closed.append(True)

    def _orchestrator() -> MainOrchestrator:
        orchestrator = MainOrchestrator()
        orchestrator.ai_provider = OwnedProvider()
        orchestrator._owns_provider = True
        return orchestrator

    monkeypatch.setattr(advertising_api, "_orchestrator", _orchestrator)
    response = client.post(STRATEGY_PATH, json=dict(UNIT_ECONOMICS))
    assert response.status_code == 200, response.text
    assert closed == [True]


# ══════════════════════════════════════════════════════════════════════════
# 7. MainOrchestrator provider lifecycle
# ══════════════════════════════════════════════════════════════════════════


class LifecycleProvider(AIProvider):
    """Minimal provider that records close() calls."""

    def __init__(self) -> None:
        self.close_calls = 0

    def generate_structured(self, prompt, schema, context=None):
        return MockAIProvider().generate_structured(prompt, schema, context)

    def generate_text(self, prompt, context=None):
        return "lifecycle summary"

    def close(self) -> None:
        self.close_calls += 1


def test_orchestrator_owns_a_provider_it_created() -> None:
    orchestrator = MainOrchestrator()
    assert orchestrator.owns_provider is True
    orchestrator.close()
    assert orchestrator.owns_provider is True  # ownership flag is permanent


def test_orchestrator_does_not_own_an_injected_provider() -> None:
    provider = LifecycleProvider()
    orchestrator = MainOrchestrator(ai_provider=provider)
    assert orchestrator.owns_provider is False

    orchestrator.close()
    assert provider.close_calls == 0, "an injected provider belongs to the caller"


def test_orchestrator_close_releases_an_owned_provider() -> None:
    provider = LifecycleProvider()
    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = provider
    orchestrator._owns_provider = True

    orchestrator.close()
    assert provider.close_calls == 1


def test_orchestrator_close_is_idempotent() -> None:
    provider = LifecycleProvider()
    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = provider
    orchestrator._owns_provider = True

    orchestrator.close()
    orchestrator.close()
    orchestrator.close()
    assert provider.close_calls == 1


def test_orchestrator_context_manager_closes_on_exit() -> None:
    provider = LifecycleProvider()
    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = provider
    orchestrator._owns_provider = True

    with orchestrator as entered:
        assert entered is orchestrator
        assert provider.close_calls == 0
    assert provider.close_calls == 1


def test_orchestrator_context_manager_closes_on_exception() -> None:
    provider = LifecycleProvider()
    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = provider
    orchestrator._owns_provider = True

    with pytest.raises(RuntimeError):
        with orchestrator:
            raise RuntimeError("boom")
    assert provider.close_calls == 1


def test_orchestrator_context_manager_leaves_an_injected_provider_open() -> None:
    provider = LifecycleProvider()
    with MainOrchestrator(ai_provider=provider):
        pass
    assert provider.close_calls == 0


def test_orchestrator_workflow_runs_inside_the_context_manager(db: Session) -> None:
    provider = LifecycleProvider()
    orchestrator = MainOrchestrator(ai_provider=provider)
    with orchestrator:
        workflow = orchestrator.run_workflow(
            db,
            objective="Plan a campaign",
            workflow_type="custom",
            parameters={
                "tasks": [{"agent_type": AgentType.ADVERTISING_ANALYTICS.value}],
                **UNIT_ECONOMICS,
            },
        )
    assert workflow["status"] == "COMPLETED"
    assert provider.close_calls == 0  # injected, so untouched


def test_owned_provider_is_closed_by_the_documented_usage_pattern() -> None:
    """The pattern the API endpoints use: create, `with`, then forget."""
    closed: list[bool] = []

    class OwnedProvider(MockAIProvider):
        def close(self) -> None:  # type: ignore[override]
            closed.append(True)

    orchestrator = MainOrchestrator()
    orchestrator.ai_provider = OwnedProvider()
    assert orchestrator.owns_provider is True

    with orchestrator:
        pass
    assert closed == [True]


# ══════════════════════════════════════════════════════════════════════════
# 8. Schemas & assembly invariants
# ══════════════════════════════════════════════════════════════════════════


def test_enums_are_exhaustive() -> None:
    assert [status.value for status in CampaignHealthStatus] == [
        "NO_DATA",
        "SCALING_CANDIDATE",
        "HEALTHY",
        "UNDERPERFORMING_CREATIVE",
        "UNDERPERFORMING_CONVERSION",
        "UNPROFITABLE_PAUSE",
    ]
    assert [objective.value for objective in CampaignObjective] == [
        "CONVERSIONS",
        "TRAFFIC",
        "ENGAGEMENT",
        "BRAND_AWARENESS",
    ]


def test_report_survives_a_json_round_trip(db: Session) -> None:
    report = _report(_run_agent(db, _ad_task(db, dict(UNIT_ECONOMICS))))
    restored = MarketingStrategyReport.model_validate(
        json.loads(json.dumps(report.model_dump(mode="json")))
    )
    assert restored == report


def test_report_requires_a_conversion_analysis() -> None:
    with pytest.raises(ValidationError):
        MarketingStrategyReport(
            product_name="LED dog collar",
            campaign_objective=CampaignObjective.CONVERSIONS,
            budget_considerations=recommend_budget(),
        )


def test_campaign_objective_is_carried_into_the_report(db: Session) -> None:
    report = _report(
        _run_agent(
            db,
            _ad_task(db, {**UNIT_ECONOMICS, "campaign_objective": "TRAFFIC"}),
        )
    )
    assert report.campaign_objective is CampaignObjective.TRAFFIC


def test_unknown_campaign_objective_falls_back_to_conversions(db: Session) -> None:
    report = _report(
        _run_agent(db, _ad_task(db, {**UNIT_ECONOMICS, "campaign_objective": "nonsense"}))
    )
    assert report.campaign_objective is CampaignObjective.CONVERSIONS


def test_analyze_tool_is_deterministic_and_returns_no_data_without_metrics() -> None:
    result = _analyze_campaign_performance({})
    assert result["metrics"] is None
    assert result["health_status"] == "NO_DATA"
    assert "pre-launch" in result["note"].lower()


def test_analyze_tool_explains_the_break_even_basis() -> None:
    result = _analyze_campaign_performance(
        {
            "impressions": 10000,
            "clicks": 200,
            "spend_usd": 100.0,
            "conversions": 5,
            "revenue_usd": 300.0,
            "selling_price_usd": 30.0,
            "landed_cost_usd": 10.0,
        }
    )
    assert result["health_status"] == "SCALING_CANDIDATE"
    assert "contribution break-even" in result["break_even_roas_basis"].lower()
    assert "before fixed overhead" in result["break_even_roas_basis"].lower()
    assert "not an accounting break-even" in result["note"].lower()


def test_conversion_analysis_validates() -> None:
    analysis = ConversionAnalysis(
        health_status=CampaignHealthStatus.HEALTHY,
        bottleneck_stage="NONE",
        summary="fine",
        recommended_optimizations=["keep observing"],
    )
    assert ConversionAnalysis.model_validate(analysis.model_dump(mode="json")) == analysis


def test_ad_variant_requires_its_copy_fields() -> None:
    with pytest.raises(ValidationError):
        AdCopyVariant(angle_name="x", hook="", primary_text="p", headline="h", call_to_action="c")


def test_tool_actions_map_to_the_expected_permission_tiers() -> None:
    from app.core.permissions import PermissionPolicy

    assert (
        PermissionPolicy.level_for("prepare_creative_brief").is_autonomous is True
    )
    assert PermissionPolicy.level_for("analyze_campaign_metrics").is_autonomous is True
    assert (
        PermissionPolicy.level_for("allocate_ad_budget").requires_approval is True
    )


def test_invalid_objective_is_not_silently_accepted_as_zero_data(db: Session) -> None:
    """A metrics row for a real campaign never degrades to NO_DATA silently."""
    report = _report(
        _run_agent(
            db,
            _ad_task(
                db,
                {
                    **UNIT_ECONOMICS,
                    "impressions": 1000,
                    "clicks": 20,
                    "spend_usd": 20.0,
                    "conversions": 1,
                    "revenue_usd": 29.99,
                },
            ),
        )
    )
    assert report.performance_metrics is not None
    assert report.conversion_analysis.health_status is not CampaignHealthStatus.NO_DATA


def test_task_level1_permission_is_recorded(db: Session) -> None:
    task = _ad_task(db, dict(UNIT_ECONOMICS))
    output = _run_agent(db, task)
    db.refresh(task)
    assert output.status is TaskStatus.COMPLETED
    assert task.permission_level is PermissionLevel.LEVEL_1_AUTO
    assert task.status is TaskStatus.IN_PROGRESS
