"""Advertising & analytics API (Phase 3 - Agent 2).

Two endpoints over the same agent the orchestrator runs:

``POST /advertising/strategy-report``
    Executes :class:`AdvertisingAnalyticsAgent` as a tracked task (through
    :class:`MainOrchestrator`, so it inherits the retry, rate-limit and audit
    behaviour every other workflow has) and returns the validated
    :class:`MarketingStrategyReport` plus task/approval metadata.

    One endpoint, two modes:

    * **pre-launch planning** - no ``impressions``/``clicks``/``spend_usd``/
      ``conversions``/``revenue_usd`` in the body. The report carries a plan, a
      creative brief set and a budget *recommendation*
      (``health_status = NO_DATA``).
    * **post-launch analysis** - metrics in the body. They are derived,
      validated and diagnosed deterministically.

    ``publish_live`` and ``commit_budget`` are the Level 2 flags. When set, the
    response comes back ``WAITING_APPROVAL`` with an ``approval_request_id`` -
    the report is returned, but no spend has been committed. Resuming happens
    through ``POST /api/v1/orchestrator/approvals/{id}/resume``.

``GET /advertising/reports``
    Lists the reports archived in shared memory (namespace
    ``marketing_reports``), oldest first.

Neither endpoint fabricates a result: a failed or rate-limited run is surfaced
as an error, never as an empty or invented report.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.specialists import AD_REPORT_NAMESPACE
from app.core.enums import AgentType, TaskStatus
from app.db.session import get_db as get_db_session
from app.models.core import Task
from app.schemas.advertising import (
    MarketingStrategyReport,
    StoredMarketingReport,
    StrategyReportRequest,
    StrategyReportResponse,
)
from app.services.memory import SharedMemoryService
from app.services.orchestrator import MainOrchestrator

router = APIRouter(tags=["advertising"])


def _orchestrator() -> MainOrchestrator:
    return MainOrchestrator()


@router.post(
    "/advertising/strategy-report",
    response_model=StrategyReportResponse,
)
def create_strategy_report(
    payload: StrategyReportRequest,
    db: Session = Depends(get_db_session),
) -> StrategyReportResponse:
    """Plan (and optionally analyse) a campaign as a tracked task."""
    # `mode="json"` so enums reach the task payload as plain strings; the
    # optional fields are dropped when None so the agent sees "not supplied"
    # rather than a zero it would have to interpret.
    request_fields = payload.model_dump(
        mode="json", exclude_none=True, exclude={"objective", "use_research_handoff"}
    )
    parameters: dict[str, Any] = {
        # `custom` decomposition: exactly one child task, assigned to Agent 2.
        "tasks": [{"agent_type": AgentType.ADVERTISING_ANALYTICS.value}],
        **request_fields,
    }

    orchestrator = _orchestrator()
    try:
        # `with` so a provider this request created is closed even on failure:
        # a real provider owns an httpx.Client pool, and leaking one per request
        # is exactly the 2A bug this closes over.
        with orchestrator:
            workflow = orchestrator.run_workflow(
                db,
                objective=payload.objective,
                workflow_type="custom",
                parameters=parameters,
            )
    except Exception as exc:  # provider/runtime failure -> a 502, not a trace
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Advertising task failed: {exc}",
        ) from exc

    children = workflow.get("children") or []
    if not children:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No advertising task was created",
        )

    child = children[0]
    task = db.get(Task, child["task_id"])
    output: dict[str, Any] = dict((task.output_payload if task else None) or {})
    data: dict[str, Any] = dict(output.get("data") or {})
    task_status = TaskStatus(child["status"])
    approval_request_id = workflow.get("approval_request_id")

    # A gated Level 2 request is a *success* for the analytical half: the
    # report exists and is being returned, while the spend waits for a human.
    if task_status is TaskStatus.WAITING_APPROVAL:
        if not approval_request_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Advertising task is awaiting approval without an approval request",
            )
    elif task_status is not TaskStatus.COMPLETED:
        if task_status is TaskStatus.FAILED:
            if task is not None and task.error_message:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY, detail=task.error_message
                )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=data.get("reason") or output.get("summary") or "Advertising task failed",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Advertising task is {task_status.value}: "
                f"{output.get('summary') or workflow.get('summary')}"
            ),
        )

    raw_report = data.get("marketing_report")
    if not raw_report:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Advertising task completed without producing a marketing strategy report",
        )

    return StrategyReportResponse(
        parent_task_id=str(workflow["parent_task_id"]),
        task_id=str(child["task_id"]),
        task_status=task_status,
        approval_request_id=approval_request_id,
        summary=output.get("summary"),
        report=MarketingStrategyReport.model_validate(raw_report),
    )


@router.get("/advertising/reports", response_model=list[StoredMarketingReport])
def list_strategy_reports(
    db: Session = Depends(get_db_session),
) -> list[StoredMarketingReport]:
    """List archived marketing strategy reports (chronological, deterministic)."""
    entries = SharedMemoryService.list_namespace(db, namespace=AD_REPORT_NAMESPACE)
    reports: list[StoredMarketingReport] = []
    for entry in entries:
        value: dict[str, Any] = dict(entry.value or {})
        raw_report = value.get("report")
        if not isinstance(raw_report, dict):
            # Not a report record - skip it instead of failing the whole list.
            continue
        report = MarketingStrategyReport.model_validate(raw_report)
        metrics = report.performance_metrics
        reports.append(
            StoredMarketingReport(
                report_id=entry.key,
                product_name=report.product_name,
                campaign_objective=report.campaign_objective,
                health_status=report.conversion_analysis.health_status,
                roas=metrics.roas if metrics is not None else None,
                source_task_id=str(value.get("source_task_id") or entry.key),
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                report=report,
            )
        )
    return reports
