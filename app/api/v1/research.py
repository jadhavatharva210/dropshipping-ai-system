"""Product research API (Milestone 2B - Agent 1).

Two endpoints over the same agent the orchestrator already runs:

``POST /research/opportunity-report``
    Executes :class:`ProductResearchAgent` as a tracked task (through
    :class:`MainOrchestrator`, so it gets the retry, rate-limit and audit
    behaviour every other workflow has) and returns the validated
    :class:`ProductOpportunityReport` plus task metadata.
``GET /research/reports``
    Lists the reports archived in shared memory, oldest first.

Neither endpoint fabricates a result: when the agent refuses to invent a price
or a supplier cost, or when the AI provider fails, the task's real outcome is
surfaced as an error instead of an empty report.

Live research (Phase 3.5)
-------------------------
``supplier_urls`` / ``competitor_urls`` let a caller point the agent at real
product pages. Their structured prices (JSON-LD ``Product``/``Offer`` or
``price`` meta tags) become ``VERIFIED`` figures attributed to the source URL.
Nothing is fetched unless the caller supplies a URL, so the offline behaviour of
this endpoint is unchanged.

Provider ownership
------------------
The orchestrator is created per request and used through ``with``, so a provider
it created is always closed and one injected by a caller/test never is.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.specialists import REPORT_NAMESPACE
from app.core.enums import AgentType, TaskStatus
from app.db.session import get_db as get_db_session
from app.models.core import Task
from app.schemas.product_research import (
    OpportunityReportRequest,
    OpportunityReportResponse,
    ProductOpportunityReport,
    StoredOpportunityReport,
)
from app.services.memory import SharedMemoryService
from app.services.orchestrator import MainOrchestrator

router = APIRouter(tags=["research"])


def _orchestrator() -> MainOrchestrator:
    return MainOrchestrator()


@router.post(
    "/research/opportunity-report",
    response_model=OpportunityReportResponse,
)
def create_opportunity_report(
    payload: OpportunityReportRequest,
    db: Session = Depends(get_db_session),
) -> OpportunityReportResponse:
    """Run the research agent as a tracked task and return its report."""
    parameters: dict[str, Any] = {
        # `custom` decomposition: exactly one child task, assigned to Agent 1.
        "tasks": [{"agent_type": AgentType.PRODUCT_RESEARCH.value}],
        # mode="json" so enums reach the JSON column as plain strings.
        **payload.model_dump(mode="json", exclude_none=True, exclude={"objective"}),
    }

    orchestrator = _orchestrator()
    # `with` -> MainOrchestrator.close() runs on success *and* on failure, and
    # it closes only a provider the orchestrator itself created (releasing a
    # real provider's httpx pool). Closing an injected provider here would break
    # the caller that owns it - which is exactly what the previous explicit
    # `orchestrator.ai_provider.close()` did.
    with orchestrator:
        workflow = orchestrator.run_workflow(
            db,
            objective=payload.objective,
            workflow_type="custom",
            parameters=parameters,
        )

    children = workflow.get("children") or []
    if not children:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No research task was created",
        )

    child = children[0]
    task = db.get(Task, child["task_id"])
    output: dict[str, Any] = dict((task.output_payload if task else None) or {})
    data: dict[str, Any] = dict(output.get("data") or {})
    task_status = TaskStatus(child["status"])

    if task_status is not TaskStatus.COMPLETED:
        if task_status is TaskStatus.FAILED:
            if task is not None and task.error_message:
                # Provider or runtime failure - includes the AI rate-limit path
                # added in Milestone 2A, which fails fast without retrying.
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=task.error_message
                )
            # The agent refused to invent the missing input; that is a data
            # problem, not a server problem.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=data.get("reason") or output.get("summary") or "Research task failed",
            )
        # Every research tool is Level 1, so a gate here means the permission
        # policy was reconfigured. Report it rather than fake a report.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Research task is {task_status.value}: "
                f"{output.get('summary') or workflow.get('summary')}"
            ),
        )

    raw_report = data.get("opportunity_report")
    if not raw_report:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Research task completed without producing an opportunity report",
        )

    return OpportunityReportResponse(
        parent_task_id=str(workflow["parent_task_id"]),
        task_id=str(child["task_id"]),
        task_status=task_status,
        summary=output.get("summary"),
        report=ProductOpportunityReport.model_validate(raw_report),
    )


@router.get("/research/reports", response_model=list[StoredOpportunityReport])
def list_opportunity_reports(
    db: Session = Depends(get_db_session),
) -> list[StoredOpportunityReport]:
    """List archived opportunity reports (chronological, deterministic)."""
    entries = SharedMemoryService.list_namespace(db, namespace=REPORT_NAMESPACE)
    reports: list[StoredOpportunityReport] = []
    for entry in entries:
        value: dict[str, Any] = dict(entry.value or {})
        raw_report = value.get("report")
        if not isinstance(raw_report, dict):
            # Not a report record - skip it instead of failing the whole list.
            continue
        report = ProductOpportunityReport.model_validate(raw_report)
        reports.append(
            StoredOpportunityReport(
                report_id=entry.key,
                product_name=report.product_name,
                product_category=report.product_category,
                recommendation=report.recommendation,
                research_confidence=report.research_confidence,
                estimated_gross_margin_pct=report.margin_analysis.estimated_gross_margin_pct,
                cost_provenance=report.margin_analysis.cost_provenance,
                source_task_id=str(value.get("source_task_id") or entry.key),
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                report=report,
            )
        )
    return reports
