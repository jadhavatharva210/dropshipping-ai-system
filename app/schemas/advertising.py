"""Advertising & analytics schemas (Phase 3 - Agent 2).

Section 3 of the build spec gives Agent 2 the same anti-hallucination contract as
Agent 1, applied to marketing: ad copy, creative briefs and audience segments
may only restate facts that were actually supplied, and every performance number
must be derived from raw metrics rather than asserted.

Three properties are enforced here rather than left to a prompt:

* **Derived metrics cannot disagree with their raw inputs.**
  :class:`CampaignPerformanceMetrics` stores the raw platform numbers and the
  derived rates side by side, and a validator recomputes every derived field
  from the raw ones - so ``ctr_pct`` can never be hand-written as a nicer
  number than ``clicks / impressions`` supports. Every division is guarded, so
  a zero-impression or zero-spend account yields ``0.0``/``None`` instead of a
  ``ZeroDivisionError``.
* **Campaign health is a deterministic function of those metrics.**
  :func:`diagnose_campaign_health` is the single rule set; it always returns the
  same verdict for the same inputs and never claims health it cannot measure
  (no metrics at all -> ``NO_DATA``, i.e. pre-launch mode).
* **Break-even ROAS is contribution break-even, not a revenue target.**
  ``break_even_roas = selling_price / gross_profit`` where gross profit is
  ``selling_price - landed_cost`` for one unit. At that ROAS the ad spend has
  exactly consumed the product's contribution margin: the campaign is not yet
  losing money on unit economics, but it also covers *no* fixed overhead
  (salaries, software, payment friction, refunds). It is deliberately **not**
  an accounting break-even for the business, and it is ``None`` whenever the
  unit economics are unknown or the contribution margin is non-positive (in
  which case no amount of ad efficiency can make the campaign profitable).

Marketing money is the point where a wrong number becomes real spend, so the
budget model carries ``requires_human_approval=True`` by default: a
:class:`BudgetRecommendation` is a *recommendation*, and turning it into live
spend is the Level 2 gated action ``allocate_ad_budget``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.core.enums import TaskStatus
from app.schemas.product_research import ARITHMETIC_TOLERANCE, DataProvenance

__all__ = [
    "AD_COPY_PROMPT_HEADER",
    "CREATIVE_PROMPT_HEADER",
    "FALLBACK_BREAK_EVEN_ROAS",
    "FORBIDDEN_CLAIM_PATTERNS",
    "MIN_HEALTHY_CONVERSION_RATE_PCT",
    "MIN_HEALTHY_CTR_PCT",
    "MIN_SCALING_CONVERSIONS",
    "SCALING_ROAS",
    "AdCopyVariant",
    "BudgetRecommendation",
    "CampaignHealthStatus",
    "CampaignObjective",
    "CampaignPerformanceMetrics",
    "ConversionAnalysis",
    "CreativeBrief",
    "MarketingStrategyReport",
    "StoredMarketingReport",
    "StrategyReportRequest",
    "StrategyReportResponse",
    "TargetAudienceSegment",
    "build_conversion_analysis",
    "derive_campaign_metrics",
    "diagnose_campaign_health",
    "find_ungrounded_claims",
    "grounding_tokens",
    "recommend_budget",
]

# ── deterministic thresholds ──────────────────────────────────────────────
#: Below this click-through rate the creative is what is failing: too few of
#: the people who saw the ad were interested enough to click.
MIN_HEALTHY_CTR_PCT: float = 1.0

#: Below this click-to-conversion rate the *offer/landing page* is what is
#: failing: people clicked, then did not buy.
MIN_HEALTHY_CONVERSION_RATE_PCT: float = 1.5

#: ROAS floor used only when break-even ROAS is unknown (no unit economics).
FALLBACK_BREAK_EVEN_ROAS: float = 1.2

#: At/above this ROAS, with enough conversions to be statistically meaningful,
#: a campaign is a scaling candidate.
SCALING_ROAS: float = 2.5

#: Fewer conversions than this is not enough evidence to call for a scale-up.
MIN_SCALING_CONVERSIONS: int = 5


class CampaignObjective(str, Enum):
    """What the campaign is being optimised for."""

    CONVERSIONS = "CONVERSIONS"
    TRAFFIC = "TRAFFIC"
    ENGAGEMENT = "ENGAGEMENT"
    BRAND_AWARENESS = "BRAND_AWARENESS"


class CampaignHealthStatus(str, Enum):
    """Deterministic verdict on a campaign's measured performance."""

    #: Pre-launch, or a metrics row with no impressions: nothing to judge.
    NO_DATA = "NO_DATA"
    #: Profitable and efficient enough to increase spend.
    SCALING_CANDIDATE = "SCALING_CANDIDATE"
    #: No rule fired: measurable, and not (yet) a problem.
    HEALTHY = "HEALTHY"
    #: Impressions happen, clicks do not -> the creative is the bottleneck.
    UNDERPERFORMING_CREATIVE = "UNDERPERFORMING_CREATIVE"
    #: Clicks happen, purchases do not -> the offer/landing page is the bottleneck.
    UNDERPERFORMING_CONVERSION = "UNDERPERFORMING_CONVERSION"
    #: Below contribution break-even: every extra sale digs the hole deeper.
    UNPROFITABLE_PAUSE = "UNPROFITABLE_PAUSE"


class TargetAudienceSegment(BaseModel):
    """One audience hypothesis, stated only from supplied facts.

    Empty lists are meaningful: they record that the fact was *not* supplied
    rather than silently inventing interests or platforms.
    """

    segment_name: str = Field(min_length=1)
    age_range: str = ""
    interests: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)


class AdCopyVariant(BaseModel):
    """One ad copy angle, plus the facts it is allowed to cite."""

    angle_name: str = Field(min_length=1)
    hook: str = Field(min_length=1)
    primary_text: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    call_to_action: str = Field(min_length=1)
    #: The anti-hallucination receipt: which supplied facts this variant rests on.
    facts_referenced: list[str] = Field(default_factory=list)


class CreativeBrief(BaseModel):
    """A production brief for one creative concept, grounded in product facts."""

    concept_name: str = Field(min_length=1)
    #: e.g. ``UGC_VIDEO``, ``CAROUSEL``, ``STATIC_IMAGE``.
    format_type: str = Field(min_length=1)
    visual_hook: str = Field(min_length=1)
    key_message: str = Field(min_length=1)
    shot_list: list[str] = Field(default_factory=list)
    compliance_notes: list[str] = Field(default_factory=list)


# ── metrics ───────────────────────────────────────────────────────────────


def derive_campaign_metrics(
    *,
    impressions: int,
    clicks: int,
    spend_usd: float,
    conversions: int,
    revenue_usd: float,
    selling_price_usd: float | None = None,
    landed_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Every derived metric, computed once, from the raw inputs only.

    All divisions are guarded: a campaign with no impressions or no clicks
    produces ``0.0`` (a rate that genuinely is zero) or ``None`` (a ratio that
    simply does not exist yet) instead of raising.

    ``break_even_roas`` is **contribution** break-even
    (``selling_price / gross_profit``): the ROAS at which ad spend exactly
    equals the unit contribution margin, before any fixed overhead. It is
    ``None`` when the selling price or the landed cost is unknown, or when the
    contribution margin is ``<= 0`` - in that case no ROAS can be profitable.
    """
    impressions = max(int(impressions), 0)
    clicks = max(int(clicks), 0)
    conversions = max(int(conversions), 0)
    spend = max(float(spend_usd), 0.0)
    revenue = max(float(revenue_usd), 0.0)

    ctr_pct = round((clicks / impressions) * 100.0, 2) if impressions > 0 else 0.0
    cpc_usd = round(spend / clicks, 2) if clicks > 0 else None
    cpa_usd = round(spend / conversions, 2) if conversions > 0 else None
    conversion_rate_pct = (
        round((conversions / clicks) * 100.0, 2) if clicks > 0 else 0.0
    )
    roas = round(revenue / spend, 2) if spend > 0 else None

    gross_profit_per_unit: float | None = None
    if selling_price_usd is not None and landed_cost_usd is not None:
        gross_profit_per_unit = round(
            float(selling_price_usd) - float(landed_cost_usd), 2
        )

    break_even_roas: float | None = None
    if (
        gross_profit_per_unit is not None
        and gross_profit_per_unit > 0
        and selling_price_usd is not None
        and float(selling_price_usd) > 0
    ):
        # Contribution break-even: not an accounting break-even for the business.
        break_even_roas = round(float(selling_price_usd) / gross_profit_per_unit, 2)

    net_margin_after_ad_spend_usd: float | None = None
    if landed_cost_usd is not None and spend > 0:
        cost_of_goods = round(conversions * float(landed_cost_usd), 2)
        net_margin_after_ad_spend_usd = round(revenue - cost_of_goods - spend, 2)

    return {
        "ctr_pct": ctr_pct,
        "cpc_usd": cpc_usd,
        "cpa_usd": cpa_usd,
        "conversion_rate_pct": conversion_rate_pct,
        "roas": roas,
        "break_even_roas": break_even_roas,
        "net_margin_after_ad_spend_usd": net_margin_after_ad_spend_usd,
        "gross_profit_per_unit_usd": gross_profit_per_unit,
    }


class CampaignPerformanceMetrics(BaseModel):
    """Raw platform metrics plus the derived rates they imply.

    The derived fields are required, but :meth:`compute` is the intended
    constructor: it derives them so they cannot contradict the raw inputs. A
    validator re-derives and rejects any mismatch, which makes a hand-edited
    ``roas`` a validation error rather than a silent lie.

    ``provenance`` is mandatory for the same reason as in Agent 1's report: an
    ESTIMATED or SIMULATED metrics row must never be presented as a pull from a
    live ad account.
    """

    # ── raw inputs ────────────────────────────────────────────────────────
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    spend_usd: float = Field(ge=0)
    conversions: int = Field(ge=0)
    revenue_usd: float = Field(ge=0)
    provenance: DataProvenance = DataProvenance.SIMULATED
    #: Unit economics, used for contribution break-even and true net margin.
    #: ``None`` means "not known", never "zero".
    selling_price_usd: float | None = Field(default=None, gt=0)
    landed_cost_usd: float | None = Field(default=None, ge=0)

    # ── derived metrics (validated, never trusted from input) ─────────────
    ctr_pct: float
    cpc_usd: float | None
    cpa_usd: float | None
    conversion_rate_pct: float
    roas: float | None
    #: Contribution break-even ROAS (``selling_price / gross_profit``) before
    #: fixed overhead. ``None`` when unit economics are unknown or non-positive.
    break_even_roas: float | None
    #: Revenue - cost of goods sold - ad spend. ``None`` without a landed cost.
    net_margin_after_ad_spend_usd: float | None

    @classmethod
    def compute(
        cls,
        *,
        impressions: int,
        clicks: int,
        spend_usd: float,
        conversions: int,
        revenue_usd: float,
        provenance: DataProvenance = DataProvenance.SIMULATED,
        selling_price_usd: float | None = None,
        landed_cost_usd: float | None = None,
    ) -> CampaignPerformanceMetrics:
        """Build metrics with every derived field computed from the raw ones.

        The money inputs are rounded *before* deriving, so the stored raw values
        and the stored derived values are always computed from the same numbers
        (otherwise ``spend_usd=100.004`` could store ``100.00`` while ``roas``
        was derived from ``100.004``, and the consistency validator would reject
        a value the class itself produced).
        """
        spend = round(float(spend_usd), 2)
        revenue = round(float(revenue_usd), 2)
        derived = derive_campaign_metrics(
            impressions=impressions,
            clicks=clicks,
            spend_usd=spend,
            conversions=conversions,
            revenue_usd=revenue,
            selling_price_usd=selling_price_usd,
            landed_cost_usd=landed_cost_usd,
        )
        derived.pop("gross_profit_per_unit_usd")
        return cls(
            impressions=impressions,
            clicks=clicks,
            spend_usd=spend,
            conversions=conversions,
            revenue_usd=revenue,
            provenance=provenance,
            selling_price_usd=selling_price_usd,
            landed_cost_usd=landed_cost_usd,
            **derived,
        )

    @model_validator(mode="after")
    def _derived_metrics_must_match_raw_inputs(self) -> CampaignPerformanceMetrics:
        expected = derive_campaign_metrics(
            impressions=self.impressions,
            clicks=self.clicks,
            spend_usd=self.spend_usd,
            conversions=self.conversions,
            revenue_usd=self.revenue_usd,
            selling_price_usd=self.selling_price_usd,
            landed_cost_usd=self.landed_cost_usd,
        )
        expected.pop("gross_profit_per_unit_usd")
        for field_name, expected_value in expected.items():
            actual_value = getattr(self, field_name)
            if expected_value is None or actual_value is None:
                if expected_value is not actual_value:
                    raise ValueError(
                        f"{field_name} must be {expected_value!r} for these raw "
                        f"metrics, got {actual_value!r}"
                    )
                continue
            if not math.isclose(
                float(actual_value), float(expected_value), abs_tol=ARITHMETIC_TOLERANCE
            ):
                raise ValueError(
                    f"{field_name} is inconsistent with the raw metrics: expected "
                    f"{expected_value}, got {actual_value}"
                )
        return self

    @property
    def gross_profit_per_unit_usd(self) -> float | None:
        """Contribution margin per unit, or ``None`` when costs are unknown."""
        if self.selling_price_usd is None or self.landed_cost_usd is None:
            return None
        return round(self.selling_price_usd - self.landed_cost_usd, 2)

    @property
    def has_data(self) -> bool:
        """True when at least one impression was actually served."""
        return self.impressions > 0


class ConversionAnalysis(BaseModel):
    """The deterministic diagnosis of a metrics row."""

    health_status: CampaignHealthStatus
    #: Where the funnel leaks: ``PRE_LAUNCH``/``CREATIVE``/``CONVERSION``/...
    bottleneck_stage: str
    summary: str
    recommended_optimizations: list[str] = Field(default_factory=list)


def diagnose_campaign_health(
    metrics: CampaignPerformanceMetrics | None,
) -> CampaignHealthStatus:
    """Deterministic campaign health, evaluated in a fixed rule order.

    1. no metrics, or ``impressions == 0``            -> ``NO_DATA`` (pre-launch)
    2. ``ctr_pct < 1.0``                              -> ``UNDERPERFORMING_CREATIVE``
    3. ``conversion_rate_pct < 1.5``                  -> ``UNDERPERFORMING_CONVERSION``
    4. ``roas`` below contribution ``break_even_roas``
       (or below 1.2 when break-even is unknown)      -> ``UNPROFITABLE_PAUSE``
    5. ``roas >= 2.5`` above break-even, with
       ``conversions >= 5``                           -> ``SCALING_CANDIDATE``
    6. otherwise                                      -> ``HEALTHY``

    The order matters: a campaign that clicks well but converts badly is a
    conversion problem *until* it is proven unprofitable, and profitability is
    judged against the product's own contribution margin whenever that is known.
    """
    if metrics is None or metrics.impressions == 0:
        return CampaignHealthStatus.NO_DATA

    if metrics.ctr_pct < MIN_HEALTHY_CTR_PCT:
        return CampaignHealthStatus.UNDERPERFORMING_CREATIVE

    if metrics.conversion_rate_pct < MIN_HEALTHY_CONVERSION_RATE_PCT:
        return CampaignHealthStatus.UNDERPERFORMING_CONVERSION

    roas = metrics.roas
    if roas is None:
        # Spent nothing yet: there is no return to judge, and no evidence of a
        # problem. Anything else would be an invented verdict.
        return CampaignHealthStatus.HEALTHY

    break_even = metrics.break_even_roas
    floor = break_even if break_even is not None else FALLBACK_BREAK_EVEN_ROAS
    if roas < floor:
        return CampaignHealthStatus.UNPROFITABLE_PAUSE

    above_break_even = break_even is None or roas > break_even
    if roas >= SCALING_ROAS and metrics.conversions >= MIN_SCALING_CONVERSIONS and above_break_even:
        return CampaignHealthStatus.SCALING_CANDIDATE

    return CampaignHealthStatus.HEALTHY


_HEALTH_BOTTLENECK: dict[CampaignHealthStatus, str] = {
    CampaignHealthStatus.NO_DATA: "PRE_LAUNCH",
    CampaignHealthStatus.UNDERPERFORMING_CREATIVE: "CREATIVE",
    CampaignHealthStatus.UNDERPERFORMING_CONVERSION: "CONVERSION",
    CampaignHealthStatus.UNPROFITABLE_PAUSE: "PROFITABILITY",
    CampaignHealthStatus.SCALING_CANDIDATE: "SCALING",
    CampaignHealthStatus.HEALTHY: "NONE",
}


def build_conversion_analysis(
    health: CampaignHealthStatus,
    metrics: CampaignPerformanceMetrics | None,
    *,
    objective: CampaignObjective = CampaignObjective.CONVERSIONS,
) -> ConversionAnalysis:
    """Turn a health verdict into the diagnosis the report shows.

    Text and optimisations are derived from the *measured* numbers only, so the
    explanation can never describe a metric that was not computed.
    """
    bottleneck = _HEALTH_BOTTLENECK[health]

    if metrics is None or health is CampaignHealthStatus.NO_DATA:
        return ConversionAnalysis(
            health_status=health,
            bottleneck_stage=bottleneck,
            summary=(
                "No campaign metrics are available, so this is a pre-launch plan rather "
                "than a performance review: the budget is a recommendation and no spend "
                "has been committed."
            ),
            recommended_optimizations=[
                "Launch with the smallest test budget the unit economics support and "
                "collect at least 1,000 impressions before judging the creative.",
                "Confirm the landed cost and selling price so break-even ROAS is known "
                "before the first dollar is spent.",
                "Keep a control copy variant so a weak result can be attributed to the "
                "creative rather than to the audience.",
            ],
        )

    spent = f"${metrics.spend_usd:.2f} spent"
    if health is CampaignHealthStatus.UNDERPERFORMING_CREATIVE:
        summary = (
            f"Click-through rate is {metrics.ctr_pct:.2f}% against a "
            f"{MIN_HEALTHY_CTR_PCT:.1f}% floor: {spent} is reaching people who do not "
            f"stop for the creative. The audience is receiving the ad; the hook is not "
            f"earning the click."
        )
        optimizations = [
            "Test a new hook and first frame; keep the audience and offer constant so "
            "the creative is the only variable.",
            "Rebuild the brief around the strongest supplied product fact instead of a "
            "generic benefit claim.",
            "Pause the weakest copy variants once each has enough impressions to compare.",
        ]
    elif health is CampaignHealthStatus.UNDERPERFORMING_CONVERSION:
        summary = (
            f"Click-through is {metrics.ctr_pct:.2f}% (above the "
            f"{MIN_HEALTHY_CTR_PCT:.1f}% floor) but only "
            f"{metrics.conversion_rate_pct:.2f}% of clicks convert, below the "
            f"{MIN_HEALTHY_CONVERSION_RATE_PCT:.1f}% floor. The creative is working; the "
            f"offer or landing page is not."
        )
        optimizations = [
            "Audit the landing page: price clarity, shipping cost and delivery window "
            "must match the ad exactly.",
            "Reduce friction at checkout and surface the delivery estimate before "
            "payment details are requested.",
            "Retarget clickers with the same promise instead of spending more on new "
            "traffic that will also bounce.",
        ]
    elif health is CampaignHealthStatus.UNPROFITABLE_PAUSE:
        break_even_text = (
            f"break-even ROAS of {metrics.break_even_roas:.2f} "
            f"(contribution break-even, before fixed overhead)"
            if metrics.break_even_roas is not None
            else f"the {FALLBACK_BREAK_EVEN_ROAS:.2f} fallback floor (unit economics unknown)"
        )
        summary = (
            f"ROAS is {metrics.roas:.2f}, below {break_even_text}. Every additional sale "
            f"at this efficiency reduces contribution margin, so the campaign should be "
            f"paused rather than scaled."
        )
        optimizations = [
            "Pause or sharply reduce spend until the offer can support a higher ROAS.",
            "Raise contribution margin through a lower landed cost or a higher price "
            "before buying more traffic.",
            "Keep the best-performing creative warm at a minimal budget so the learning "
            "is not lost.",
        ]
    elif health is CampaignHealthStatus.SCALING_CANDIDATE:
        summary = (
            f"ROAS is {metrics.roas:.2f} with {metrics.conversions} conversions at "
            f"{metrics.cpa_usd if metrics.cpa_usd is not None else 0.0:.2f} CPA, above "
            f"the {SCALING_ROAS:.1f} scaling threshold and above contribution "
            f"break-even. There is measurable room to scale."
        )
        optimizations = [
            "Increase budget in steps of no more than 20-30% per day so the learning "
            "phase is not reset.",
            "Duplicate the winning creative into the next-best audience instead of "
            "editing the proven one.",
            "Re-check break-even ROAS after any landed-cost or price change before the "
            "next increase.",
        ]
    else:  # HEALTHY
        summary = (
            f"Campaign is measurable and within bounds: CTR {metrics.ctr_pct:.2f}%, "
            f"conversion rate {metrics.conversion_rate_pct:.2f}%, ROAS "
            f"{metrics.roas if metrics.roas is not None else 0.0:.2f}. No rule fired, "
            f"so the honest verdict is 'keep observing', not 'scale'."
        )
        optimizations = [
            "Keep collecting data until there is enough volume to judge the conversion "
            "stage confidently.",
            "Test one variable at a time; do not change budget and creative together.",
            "Re-run this analysis once more conversions have accumulated.",
        ]

    if objective is not CampaignObjective.CONVERSIONS:
        optimizations.append(
            f"Objective is {objective.value}: judge success against that goal as well as "
            f"ROAS, and never scale spend that cannot be attributed to it."
        )

    return ConversionAnalysis(
        health_status=health,
        bottleneck_stage=bottleneck,
        summary=summary,
        recommended_optimizations=optimizations,
    )


# ── budget ────────────────────────────────────────────────────────────────

#: Hard floor/ceiling for a recommended *test* budget. The ceiling exists
#: because an unreviewed recommendation must never be a large number by
#: accident; raising it is a human decision.
MIN_RECOMMENDED_DAILY_BUDGET_USD: float = 10.0
MAX_RECOMMENDED_DAILY_BUDGET_USD: float = 50.0


class BudgetRecommendation(BaseModel):
    """A budget *recommendation*. Committing it is a Level 2 action.

    ``requires_human_approval`` is ``True`` by default and is the reason this
    model can safely live inside a Level 1 report: producing it spends nothing.
    """

    recommended_daily_budget_usd: float = Field(gt=0)
    test_duration_days: int = Field(ge=1)
    total_planned_spend_usd: float = Field(ge=0)
    allocation_by_platform: dict[str, float] = Field(default_factory=dict)
    requires_human_approval: bool = True
    safety_note: str = ""


def recommend_budget(
    *,
    gross_profit_per_unit_usd: float | None = None,
    platforms: list[str] | None = None,
    daily_budget_usd: float | None = None,
    test_duration_days: int = 7,
) -> BudgetRecommendation:
    """Derive a defensible test budget from the unit economics.

    The rule: a day's spend must be covered by a single sale, so the daily
    budget is capped at one unit's contribution margin, clamped to
    ``[10, 50]``/day. With no unit economics the floor applies and the safety
    note says the number is a placeholder. An explicit ``daily_budget_usd`` from
    the caller always wins - the human asked for it - but is still recorded as a
    recommendation requiring approval.
    """
    days = max(int(test_duration_days), 1)
    if daily_budget_usd is not None and daily_budget_usd > 0:
        daily = round(float(daily_budget_usd), 2)
        basis = "caller-supplied daily budget"
    elif gross_profit_per_unit_usd is not None and gross_profit_per_unit_usd > 0:
        daily = round(
            min(
                MAX_RECOMMENDED_DAILY_BUDGET_USD,
                max(MIN_RECOMMENDED_DAILY_BUDGET_USD, gross_profit_per_unit_usd),
            ),
            2,
        )
        basis = (
            f"1x the ${gross_profit_per_unit_usd:.2f} contribution margin per unit (one "
            f"sale a day covers the day's spend), clamped to "
            f"${MIN_RECOMMENDED_DAILY_BUDGET_USD:.0f}-"
            f"${MAX_RECOMMENDED_DAILY_BUDGET_USD:.0f}/day"
        )
    else:
        daily = MIN_RECOMMENDED_DAILY_BUDGET_USD
        basis = "default floor (unit economics were not supplied)"

    total = round(daily * days, 2)

    names = [str(platform).strip() for platform in (platforms or []) if str(platform).strip()]
    names = list(dict.fromkeys(names)) or ["meta"]
    share = round(total / len(names), 2)
    allocation = {name: share for name in names}
    # Absorb the rounding remainder into the first platform so the allocation
    # always sums exactly to the planned total.
    remainder = round(total - sum(allocation.values()), 2)
    allocation[names[0]] = round(allocation[names[0]] + remainder, 2)

    return BudgetRecommendation(
        recommended_daily_budget_usd=daily,
        test_duration_days=days,
        total_planned_spend_usd=total,
        allocation_by_platform=allocation,
        requires_human_approval=True,
        safety_note=(
            f"Recommendation only - nothing has been spent. Derived from {basis}, over "
            f"a {days}-day test. Committing this budget is a Level 2 action: it pauses "
            f"for explicit human approval before any money is allocated."
        ),
    )


# ── the report ────────────────────────────────────────────────────────────


class MarketingStrategyReport(BaseModel):
    """The validated output of Agent 2 (Advertising & Analytics).

    A Level 1 analytical artifact: no platform is touched and no budget is
    committed to produce it. ``performance_metrics`` is ``None`` when the run
    had no campaign data (pre-launch planning); in that case
    ``conversion_analysis.health_status`` is ``NO_DATA``.
    """

    product_name: str = Field(min_length=1)
    campaign_objective: CampaignObjective
    target_audiences: list[TargetAudienceSegment] = Field(default_factory=list)
    marketing_strategy: str = ""
    ad_copy_variants: list[AdCopyVariant] = Field(default_factory=list)
    creative_briefs: list[CreativeBrief] = Field(default_factory=list)
    performance_metrics: CampaignPerformanceMetrics | None = None
    conversion_analysis: ConversionAnalysis
    recommendations: list[str] = Field(default_factory=list)
    budget_considerations: BudgetRecommendation
    #: The exact facts the copy and briefs were allowed to use.
    facts_used: dict[str, Any] = Field(default_factory=dict)
    #: Where the copy came from, e.g. ``ai_ad_copy_accepted`` or
    #: ``deterministic_grounded_draft (ai_output=rejected_ungrounded)``.
    content_origin: str = "deterministic_grounded_draft"

    @property
    def is_pre_launch(self) -> bool:
        return self.performance_metrics is None


# ── grounding gate ────────────────────────────────────────────────────────

#: Claims that no amount of supplied product data can justify: they are always
#: fabricated unless a human supplied them verbatim. Each entry is
#: ``(label, compiled pattern)`` and the label is what the rejection reports.
FORBIDDEN_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("invented discount", re.compile(r"\b\d{1,3}\s?(?:%|percent)\s?off\b")),
    ("invented discount", re.compile(r"\b(?:discount|coupon|promo\s?code|clearance)\b")),
    ("invented urgency", re.compile(r"\b(?:today only|limited time|while stocks? last|ends tonight)\b")),
    ("fabricated review", re.compile(r"\b(?:reviews?|testimonials?|ratings?|rated)\b")),
    ("fabricated social proof", re.compile(r"\b(?:thousands of|millions of|best[- ]?sell(?:er|ing)|trusted by|customers love)\b")),
    ("fabricated social proof", re.compile(r"#1\b")),
    ("invented star rating", re.compile(r"\b\d(?:\.\d)?\s?[- ]stars?\b")),
    ("unverified guarantee", re.compile(r"\bguarantee[ds]?\b")),
    ("unverified guarantee", re.compile(r"\b(?:risk[- ]free|money[- ]back)\b")),
    ("unverified claim", re.compile(r"\b(?:clinically|scientifically)\s(?:proven|tested|backed)\b")),
    ("unverified claim", re.compile(r"\b(?:fda|ce)\s(?:approved|certified)\b")),
    ("unverified claim", re.compile(r"\bcures?\b")),
    ("invented freebie", re.compile(r"\bfree\s(?:shipping|returns?|gift|sample)\b")),
)


def _numeric_tokens(facts: dict[str, Any]) -> set[str]:
    """Every rendering of every supplied number, so text can be matched to facts."""
    tokens: set[str] = set()

    def add(value: float) -> None:
        tokens.add(f"{value:.2f}")
        tokens.add(f"{value:.1f}")
        tokens.add(f"{value:.0f}")
        tokens.add(str(value))
        if value == int(value):
            tokens.add(str(int(value)))

    def walk(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            add(float(value))
            return
        if isinstance(value, str):
            # Numbers mentioned inside supplied strings ("ships in 8-14 days").
            for match in re.findall(r"\d+(?:\.\d+)?", value):
                add(float(match))
            return
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)

    walk(facts)
    return tokens


def grounding_tokens(facts: dict[str, Any]) -> set[str]:
    """Fact-derived strings a grounded piece of copy is expected to reference."""
    tokens: set[str] = set()
    for key in ("product_name", "product", "supplier_name", "product_category"):
        value = facts.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip().lower()
        tokens.add(text)
        tokens.update(word for word in re.split(r"[^a-z0-9]+", text) if len(word) >= 4)
    for benefit in facts.get("benefits") or []:
        if isinstance(benefit, str):
            tokens.update(word for word in re.split(r"[^a-z0-9]+", benefit.lower()) if len(word) >= 4)
    tokens |= _numeric_tokens(facts)
    return tokens


def find_ungrounded_claims(text: str, facts: dict[str, Any]) -> list[str]:
    """Return every fabrication in ``text``, as human-readable reasons.

    An empty list means the text is acceptable: it cites only supplied numbers
    and contains none of :data:`FORBIDDEN_CLAIM_PATTERNS` (discounts, reviews,
    ratings, guarantees, urgency, freebies, social proof). This is the gate the
    AI output has to pass before it can appear in a report.
    """
    violations: list[str] = []
    lowered = text.lower()

    for label, pattern in FORBIDDEN_CLAIM_PATTERNS:
        match = pattern.search(lowered)
        if match is not None:
            violations.append(f"{label}: {match.group(0)!r}")

    grounded_numbers = _numeric_tokens(facts)
    for raw in re.findall(r"\d+(?:[.,]\d+)?", text):
        normalised = raw.replace(",", "")
        candidates = {normalised}
        try:
            value = float(normalised)
        except ValueError:  # pragma: no cover - the regex guarantees a number
            continue
        candidates |= {f"{value:.2f}", f"{value:.1f}", f"{value:.0f}", str(value)}
        if value == int(value):
            candidates.add(str(int(value)))
        if candidates.isdisjoint(grounded_numbers):
            violations.append(f"number not present in the supplied facts: {raw!r}")

    return violations


#: Instructions for the AI ad-copy step. Stated in the prompt *and* enforced
#: afterwards by :func:`find_ungrounded_claims`, because a prompt is a request
#: and the gate is a rule.
AD_COPY_PROMPT_HEADER = (
    "Write advertising copy for the product described by the facts below, "
    "using ONLY those facts.\n"
    "Hard rules:\n"
    "- Do not invent claims, statistics, reviews or discounts.\n"
    "- Never invent discounts, coupons, urgency, customer reviews, star ratings, "
    "social proof, guarantees or free shipping.\n"
    "- Never state a number that does not appear in the facts.\n"
    "- Facts labelled SIMULATED or ESTIMATED must be presented as approximate; "
    "never as verified market data.\n"
    "- If a fact is missing, leave it out instead of filling the gap.\n\n"
    "Supplied facts (JSON):\n"
)

#: Instructions for the AI creative-brief step (used by the tool, kept next to
#: the copy rules so both prompts cannot drift apart).
CREATIVE_PROMPT_HEADER = (
    "Write creative briefs for the product described by the facts below, using "
    "ONLY those facts.\n"
    "Hard rules:\n"
    "- Do not invent discounts, reviews, ratings, guarantees or statistics.\n"
    "- Never state a number that does not appear in the facts.\n"
    "- Describe only what the supplied facts support.\n\n"
    "Supplied facts (JSON):\n"
)


# ── API schemas ───────────────────────────────────────────────────────────


class StrategyReportRequest(BaseModel):
    """Input for ``POST /api/v1/advertising/strategy-report``.

    Everything is optional except the objective, so the same endpoint serves
    both modes:

    * **pre-launch planning** - no campaign metrics supplied, so the run
      produces a plan, a creative brief set and a budget *recommendation*
      (``health_status = NO_DATA``);
    * **post-launch analysis** - raw metrics supplied, so the metrics are
      derived, validated and diagnosed.

    ``publish_live`` and ``commit_budget`` are the two Level 2 actions. Both
    default to ``False``: without them the endpoint is purely analytical.
    """

    objective: str = Field(
        default="Plan and analyse the advertising campaign", min_length=1
    )
    use_research_handoff: bool = True

    # ── product facts ─────────────────────────────────────────────────────
    product_name: str | None = Field(default=None, min_length=1)
    product_category: str | None = None
    campaign_objective: CampaignObjective = CampaignObjective.CONVERSIONS
    price_usd: float | None = Field(default=None, gt=0)
    target_selling_price_usd: float | None = Field(default=None, gt=0)
    landed_cost_usd: float | None = Field(default=None, ge=0)
    unit_cost_usd: float | None = Field(default=None, ge=0)
    shipping_cost_usd: float | None = Field(default=None, ge=0)
    gross_profit_per_unit_usd: float | None = None
    benefits: list[str] | None = None
    target_audience: str | None = None
    age_range: str | None = None
    interests: list[str] | None = None
    pain_points: list[str] | None = None
    platforms: list[str] | None = None
    platform: str | None = None
    tone: str | None = None
    shipping_days_min: int | None = Field(default=None, ge=0)
    shipping_days_max: int | None = Field(default=None, ge=0)

    # ── campaign metrics (post-launch mode) ───────────────────────────────
    impressions: int | None = Field(default=None, ge=0)
    clicks: int | None = Field(default=None, ge=0)
    spend_usd: float | None = Field(default=None, ge=0)
    conversions: int | None = Field(default=None, ge=0)
    revenue_usd: float | None = Field(default=None, ge=0)
    metrics_provenance: DataProvenance | None = None

    # ── budget / Level 2 flags ────────────────────────────────────────────
    daily_budget_usd: float | None = Field(default=None, gt=0)
    test_duration_days: int = Field(default=7, ge=1, le=90)
    publish_live: bool = False
    commit_budget: bool = False


class StrategyReportResponse(BaseModel):
    """A validated strategy report plus the tracked task that produced it.

    ``task_status``/``approval_request_id`` are the Level 2 story: a
    ``WAITING_APPROVAL`` status with a non-empty ``approval_request_id`` means
    the analysis completed but the requested spend has *not* happened.
    """

    parent_task_id: str
    task_id: str
    task_status: TaskStatus
    approval_request_id: str | None = None
    summary: str | None = None
    report: MarketingStrategyReport


class StoredMarketingReport(BaseModel):
    """One archived report as returned by ``GET /api/v1/advertising/reports``.

    The summary fields are derived from the stored report, so the list view can
    never disagree with the report it links to.
    """

    report_id: str
    product_name: str
    campaign_objective: CampaignObjective
    health_status: CampaignHealthStatus
    roas: float | None = None
    source_task_id: str
    created_at: datetime
    updated_at: datetime
    report: MarketingStrategyReport
