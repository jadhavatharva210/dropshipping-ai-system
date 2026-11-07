"""Product research & content schemas (Milestone 2B - Agent 1).

Anti-hallucination is modelled here, not just requested in a prompt. Section 3
of the build spec is explicit: never invent supplier prices, availability or
market demand, and always distinguish verified information from estimates.
These models make that enforceable rather than aspirational:

* every sourced number carries a required :class:`DataProvenance`, so a report
  cannot present a simulated figure as a verified one;
* the arithmetic connecting those numbers (landed cost, gross profit, margin)
  is validated, so a report cannot state a margin its own inputs contradict;
* confidence and recommendation come from one deterministic rule
  (:func:`assess_opportunity`) that can never return ``HIGH`` while supplier
  costs are unverified.

Provenance vocabulary:

``VERIFIED``
    Observed from a real source that was actually consulted - a live supplier
    quote, a scraped competitor listing, a real order. Only caller-supplied
    facts may claim this; no adapter in this codebase ever promotes data to it.
``ESTIMATED``
    A reasoned approximation from real inputs (e.g. a caller's own guess at a
    retail price). Directionally useful, not a quote.
``SIMULATED``
    Generated offline by a mock adapter. Clearly labelled as such, used only so
    the pipeline runs end-to-end with no network access and no spend.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.enums import TaskStatus

#: Below this gross margin a product is not viable once ad spend, payment fees
#: and refunds are accounted for.
MIN_VIABLE_MARGIN_PCT: float = 20.0

#: At or above this margin - with verified supplier costs - the opportunity is
#: strong enough to recommend proceeding rather than testing.
STRONG_MARGIN_PCT: float = 35.0

#: Tolerance for float comparisons on money/percentage arithmetic.
ARITHMETIC_TOLERANCE: float = 0.01


class DataProvenance(str, Enum):
    """How much trust a figure deserves. Never inferred - always declared."""

    VERIFIED = "VERIFIED"
    ESTIMATED = "ESTIMATED"
    SIMULATED = "SIMULATED"


class ConfidenceLevel(str, Enum):
    """Confidence in the research as a whole."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class OpportunityRecommendation(str, Enum):
    """The decision the research supports."""

    PROCEED = "PROCEED"
    TEST_CAUTIOUSLY = "TEST_CAUTIOUSLY"
    REJECT = "REJECT"


class SupplierOption(BaseModel):
    """One sourced supplier quote, with its landed cost already computed.

    ``source_reference`` is what makes provenance auditable: for a live page it
    is the URL the price was read from, for a caller quote it is whatever
    reference the caller gave, and for the offline adapter it names the
    simulated adapter so a simulated figure can never be mistaken for a quote.
    """

    supplier_name: str = Field(min_length=1)
    unit_cost_usd: float = Field(ge=0)
    shipping_cost_usd: float = Field(ge=0)
    total_landed_cost_usd: float = Field(ge=0)
    #: ``0``/``0`` means the supplier did not quote a window - that is an
    #: absence of information, not a promise of same-day delivery.
    shipping_days_min: int = Field(default=0, ge=0)
    shipping_days_max: int = Field(default=0, ge=0)
    #: ``None`` = availability was not confirmed. Never default to ``True``:
    #: claiming stock that was not checked is exactly the fabrication the spec
    #: forbids.
    in_stock: bool | None = None
    provenance: DataProvenance
    source_reference: str = ""

    @classmethod
    def quote(
        cls,
        *,
        supplier_name: str,
        unit_cost_usd: float,
        shipping_cost_usd: float = 0.0,
        shipping_days_min: int = 0,
        shipping_days_max: int = 0,
        in_stock: bool | None = None,
        provenance: DataProvenance,
        source_reference: str = "",
    ) -> SupplierOption:
        """Build an option, deriving landed cost so it cannot drift."""
        unit = round(float(unit_cost_usd), 2)
        shipping = round(float(shipping_cost_usd), 2)
        return cls(
            supplier_name=supplier_name,
            unit_cost_usd=unit,
            shipping_cost_usd=shipping,
            total_landed_cost_usd=round(unit + shipping, 2),
            shipping_days_min=shipping_days_min,
            shipping_days_max=shipping_days_max,
            in_stock=in_stock,
            provenance=provenance,
            source_reference=source_reference,
        )

    @model_validator(mode="after")
    def _landed_cost_must_add_up(self) -> SupplierOption:
        expected = self.unit_cost_usd + self.shipping_cost_usd
        if not math.isclose(self.total_landed_cost_usd, expected, abs_tol=ARITHMETIC_TOLERANCE):
            raise ValueError(
                "total_landed_cost_usd must equal unit_cost_usd + shipping_cost_usd "
                f"({self.unit_cost_usd} + {self.shipping_cost_usd} = {round(expected, 2)}, "
                f"got {self.total_landed_cost_usd})"
            )
        return self

    @model_validator(mode="after")
    def _shipping_window_must_be_ordered(self) -> SupplierOption:
        if self.shipping_days_min > self.shipping_days_max:
            raise ValueError(
                "shipping_days_min cannot exceed shipping_days_max "
                f"({self.shipping_days_min} > {self.shipping_days_max})"
            )
        return self


class CompetitorEntry(BaseModel):
    """One observed competitor price point."""

    competitor_name: str = Field(min_length=1)
    observed_price_usd: float = Field(gt=0)
    #: Relative market position (e.g. ``budget`` / ``mid`` / ``premium``).
    #: Descriptive, never a claim about the competitor's revenue or share.
    market_positioning: str = ""
    provenance: DataProvenance
    source_reference: str = ""


class MarginAnalysis(BaseModel):
    """Margin arithmetic over a chosen supplier and a chosen selling price.

    The two provenance fields are separate on purpose: a verified cost against
    a simulated price is still an unverified margin, and the report has to be
    able to say which half is weak.
    """

    unit_cost_usd: float = Field(ge=0)
    shipping_cost_usd: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    recommended_selling_price_usd: float = Field(gt=0)
    estimated_gross_profit_usd: float
    estimated_gross_margin_pct: float
    cost_provenance: DataProvenance
    price_provenance: DataProvenance

    @classmethod
    def compute(
        cls,
        *,
        unit_cost_usd: float,
        shipping_cost_usd: float,
        recommended_selling_price_usd: float,
        cost_provenance: DataProvenance,
        price_provenance: DataProvenance,
    ) -> MarginAnalysis:
        """Derive every figure from cost + price so they cannot disagree."""
        unit = round(float(unit_cost_usd), 2)
        shipping = round(float(shipping_cost_usd), 2)
        price = round(float(recommended_selling_price_usd), 2)
        total = round(unit + shipping, 2)
        profit = round(price - total, 2)
        margin_pct = round((profit / price) * 100.0, 2) if price > 0 else 0.0
        return cls(
            unit_cost_usd=unit,
            shipping_cost_usd=shipping,
            total_cost_usd=total,
            recommended_selling_price_usd=price,
            estimated_gross_profit_usd=profit,
            estimated_gross_margin_pct=margin_pct,
            cost_provenance=cost_provenance,
            price_provenance=price_provenance,
        )

    @model_validator(mode="after")
    def _arithmetic_must_be_consistent(self) -> MarginAnalysis:
        total = self.unit_cost_usd + self.shipping_cost_usd
        if not math.isclose(self.total_cost_usd, total, abs_tol=ARITHMETIC_TOLERANCE):
            raise ValueError(
                "total_cost_usd must equal unit_cost_usd + shipping_cost_usd "
                f"({self.unit_cost_usd} + {self.shipping_cost_usd} = {round(total, 2)}, "
                f"got {self.total_cost_usd})"
            )
        profit = self.recommended_selling_price_usd - self.total_cost_usd
        if not math.isclose(self.estimated_gross_profit_usd, profit, abs_tol=ARITHMETIC_TOLERANCE):
            raise ValueError(
                "estimated_gross_profit_usd must equal recommended_selling_price_usd "
                f"- total_cost_usd ({round(profit, 2)} expected, got "
                f"{self.estimated_gross_profit_usd})"
            )
        expected_pct = (
            (profit / self.recommended_selling_price_usd) * 100.0
            if self.recommended_selling_price_usd > 0
            else 0.0
        )
        if not math.isclose(
            self.estimated_gross_margin_pct, expected_pct, abs_tol=ARITHMETIC_TOLERANCE
        ):
            raise ValueError(
                "estimated_gross_margin_pct must equal gross profit / selling price "
                f"({round(expected_pct, 2)}% expected, got {self.estimated_gross_margin_pct}%)"
            )
        return self

    @property
    def is_viable(self) -> bool:
        return self.estimated_gross_margin_pct >= MIN_VIABLE_MARGIN_PCT


class GeneratedProductContent(BaseModel):
    """Listing copy, plus the exact facts it was allowed to use.

    ``facts_used`` is the anti-hallucination receipt: it is assembled by the
    agent from sourced data and is never taken from model output, so the copy
    can always be audited against what was actually known.
    """

    product_titles: list[str] = Field(default_factory=list)
    product_description: str = ""
    key_bullet_points: list[str] = Field(default_factory=list)
    marketing_angles: list[str] = Field(default_factory=list)
    facts_used: dict[str, Any] = Field(default_factory=dict)


class ProductOpportunityReport(BaseModel):
    """The validated output of Agent 1 (Product Research & Content)."""

    product_name: str = Field(min_length=1)
    product_category: str = "general"
    #: Where the information came from - adapter names, URLs, quote references.
    research_sources: list[str] = Field(default_factory=list)
    demand_summary: str = ""
    #: Demand is never claimed as verified unless a real source was consulted.
    demand_provenance: DataProvenance = DataProvenance.SIMULATED
    suppliers: list[SupplierOption] = Field(default_factory=list)
    competitors: list[CompetitorEntry] = Field(default_factory=list)
    margin_analysis: MarginAnalysis
    #: Plain-language risks, including every provenance caveat that applies.
    risks: list[str] = Field(default_factory=list)
    research_confidence: ConfidenceLevel
    recommendation: OpportunityRecommendation
    generated_content: GeneratedProductContent = Field(
        default_factory=GeneratedProductContent
    )
    #: Live web research receipt: which pages were read, which declared a
    #: structured price, and which were refused. Empty on the offline path, so
    #: an offline report is byte-identical to its pre-Phase-3.5 form.
    live_research: dict[str, Any] = Field(default_factory=dict)

    @property
    def best_supplier(self) -> SupplierOption | None:
        """Cheapest landed cost; ``None`` when no option was sourced."""
        if not self.suppliers:
            return None
        return min(self.suppliers, key=lambda option: option.total_landed_cost_usd)


def assess_opportunity(
    *,
    gross_margin_pct: float,
    cost_provenance: DataProvenance,
) -> tuple[ConfidenceLevel, OpportunityRecommendation]:
    """Derive confidence and recommendation from margin and provenance.

    Deterministic and deliberately conservative - the same inputs always give
    the same verdict, and no path returns ``HIGH`` on unverified supplier costs:

    * margin below :data:`MIN_VIABLE_MARGIN_PCT` -> ``REJECT``. Confidence is
      ``HIGH`` only when the costs that justify rejecting are verified;
      otherwise ``LOW``, because an unverified thin margin is not even a
      trustworthy rejection.
    * margin at or above :data:`STRONG_MARGIN_PCT` **and** verified costs ->
      ``HIGH`` + ``PROCEED``.
    * everything else -> ``TEST_CAUTIOUSLY``, with confidence capped below
      ``HIGH``: ``MEDIUM`` for verified or estimated costs, ``LOW`` when the
      costs are simulated (nothing real was consulted at all).
    """
    if gross_margin_pct < MIN_VIABLE_MARGIN_PCT:
        confidence = (
            ConfidenceLevel.HIGH
            if cost_provenance is DataProvenance.VERIFIED
            else ConfidenceLevel.LOW
        )
        return confidence, OpportunityRecommendation.REJECT

    if cost_provenance is DataProvenance.VERIFIED:
        if gross_margin_pct >= STRONG_MARGIN_PCT:
            return ConfidenceLevel.HIGH, OpportunityRecommendation.PROCEED
        # Verified costs, merely adequate margin: trust the numbers, not the upside.
        return ConfidenceLevel.MEDIUM, OpportunityRecommendation.TEST_CAUTIOUSLY

    if cost_provenance is DataProvenance.SIMULATED:
        return ConfidenceLevel.LOW, OpportunityRecommendation.TEST_CAUTIOUSLY
    return ConfidenceLevel.MEDIUM, OpportunityRecommendation.TEST_CAUTIOUSLY


def describe_provenance(provenance: DataProvenance) -> str:
    """Human-readable caveat used in copy and risk lists."""
    return {
        DataProvenance.VERIFIED: "verified",
        DataProvenance.ESTIMATED: "estimated - not a live quote",
        DataProvenance.SIMULATED: "simulated offline - not a live quote",
    }[provenance]


# ── API schemas ───────────────────────────────────────────────────────────


class SupplierQuoteInput(BaseModel):
    """A supplier quote supplied by the caller of the research endpoint.

    ``provenance`` defaults to ``ESTIMATED``: ``VERIFIED`` has to be declared
    explicitly, so a request can never launder a guess into a verified cost by
    omission.
    """

    supplier_name: str = Field(min_length=1)
    unit_cost_usd: float = Field(ge=0)
    shipping_cost_usd: float = Field(default=0.0, ge=0)
    shipping_days_min: int = Field(default=0, ge=0)
    shipping_days_max: int = Field(default=0, ge=0)
    in_stock: bool | None = None
    provenance: DataProvenance = DataProvenance.ESTIMATED
    source_reference: str = ""


class CompetitorObservationInput(BaseModel):
    """A competitor price the caller actually observed."""

    competitor_name: str = Field(min_length=1)
    observed_price_usd: float = Field(gt=0)
    market_positioning: str = ""
    provenance: DataProvenance = DataProvenance.ESTIMATED
    source_reference: str = ""


class OpportunityReportRequest(BaseModel):
    """Input for ``POST /api/v1/research/opportunity-report``.

    Everything except ``objective`` is optional: with no supplier quotes and no
    competitor observations the agent runs entirely on clearly-labelled
    simulated data and says so in the report.
    """

    objective: str = Field(
        default="Assess the product opportunity and draft listing content",
        min_length=1,
    )
    product_name: str | None = Field(default=None, min_length=1)
    product_category: str | None = None
    niche: str | None = None
    target_selling_price_usd: float | None = Field(default=None, gt=0)
    price_provenance: DataProvenance | None = None
    suppliers: list[SupplierQuoteInput] | None = None
    competitors: list[CompetitorObservationInput] | None = None
    demand_summary: str | None = None
    demand_provenance: DataProvenance | None = None
    research_sources: list[str] | None = None
    # ── Live web research (Phase 3.5) ─────────────────────────────────────
    #: Real supplier product pages to read. Each page's structured price
    #: (JSON-LD ``Product``/``Offer`` or ``product:price:amount``) becomes a
    #: ``VERIFIED`` landed-cost quote attributed to that URL; a page with no
    #: structured price contributes nothing and is listed in the report's risks.
    supplier_urls: list[str] = Field(default_factory=list)
    #: Real competitor product pages to read, for observed price points.
    competitor_urls: list[str] = Field(default_factory=list)
    #: Ask the live search backend for competitor URLs as well (a no-op when
    #: ``WEB_SEARCH_PROVIDER=none``).
    search_competitors: bool = False
    #: Search terms; defaults to the product name.
    search_query: str | None = None


class OpportunityReportResponse(BaseModel):
    """A validated report plus the tracked task that produced it."""

    parent_task_id: str
    task_id: str
    task_status: TaskStatus
    summary: str | None = None
    report: ProductOpportunityReport


class StoredOpportunityReport(BaseModel):
    """One archived report as returned by ``GET /api/v1/research/reports``.

    The summary fields are derived from the stored report itself, so the list
    view can never disagree with the report it links to.
    """

    report_id: str
    product_name: str
    product_category: str
    recommendation: OpportunityRecommendation
    research_confidence: ConfidenceLevel
    estimated_gross_margin_pct: float
    cost_provenance: DataProvenance
    source_task_id: str
    created_at: datetime
    updated_at: datetime
    report: ProductOpportunityReport


__all__ = [
    "ARITHMETIC_TOLERANCE",
    "MIN_VIABLE_MARGIN_PCT",
    "STRONG_MARGIN_PCT",
    "CompetitorEntry",
    "CompetitorObservationInput",
    "ConfidenceLevel",
    "DataProvenance",
    "GeneratedProductContent",
    "MarginAnalysis",
    "OpportunityRecommendation",
    "OpportunityReportRequest",
    "OpportunityReportResponse",
    "ProductOpportunityReport",
    "StoredOpportunityReport",
    "SupplierOption",
    "SupplierQuoteInput",
    "assess_opportunity",
    "describe_provenance",
]
