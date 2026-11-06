"""Governance API: permission evaluation, approvals, audit logs.

Mounted under /api/v1 in app/main.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import AgentType, ApprovalStatus
from app.core.permissions import PermissionManager, TaskNotFoundError
from app.db.session import get_db
from app.models.core import ApprovalRequest
from app.schemas.governance import (
    ActionEvaluationRequest,
    ActionEvaluationResult,
    ApprovalApproveRequest,
    ApprovalDecisionRequest,
    ApprovalResponse,
    AuditLogResponse,
)
from app.services.approval import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalService,
)
from app.services.audit import AuditService

router = APIRouter(tags=["governance"])


@router.post(
    "/permissions/evaluate",
    response_model=ActionEvaluationResult,
    status_code=status.HTTP_200_OK,
)
def evaluate_action(
    payload: ActionEvaluationRequest, db: Session = Depends(get_db)
) -> ActionEvaluationResult:
    try:
        return PermissionManager.evaluate_action(db, payload)
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/approvals", response_model=list[ApprovalResponse])
def list_approvals(
    status_filter: ApprovalStatus | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
) -> list[ApprovalRequest]:
    stmt = select(ApprovalRequest).order_by(ApprovalRequest.created_at, ApprovalRequest.id)
    if status_filter is not None:
        stmt = stmt.where(ApprovalRequest.status == status_filter)
    return list(db.scalars(stmt).all())


@router.get("/approvals/{approval_id}", response_model=ApprovalResponse)
def get_approval(approval_id: str, db: Session = Depends(get_db)) -> ApprovalRequest:
    approval = db.get(ApprovalRequest, approval_id)
    if approval is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ApprovalRequest {approval_id!r} not found",
        )
    return approval


@router.post("/approvals/{approval_id}/approve", response_model=ApprovalResponse)
def approve_request(
    approval_id: str,
    payload: ApprovalApproveRequest,
    db: Session = Depends(get_db),
) -> ApprovalRequest:
    try:
        return ApprovalService.approve_request(
            db,
            approval_id,
            payload.approver_identity,
            execution_result=payload.execution_result,
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ApprovalAlreadyDecidedError, ApprovalExpiredError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/approvals/{approval_id}/reject", response_model=ApprovalResponse)
def reject_request(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    db: Session = Depends(get_db),
) -> ApprovalRequest:
    try:
        return ApprovalService.reject_request(
            db, approval_id, payload.approver_identity, payload.reason or ""
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ApprovalAlreadyDecidedError, ApprovalExpiredError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/approvals/expire-stale")
def expire_stale_requests(db: Session = Depends(get_db)) -> dict[str, int]:
    expired = ApprovalService.expire_stale_requests(db)
    return {"expired_count": expired}


@router.get("/audit-logs", response_model=list[AuditLogResponse])
def list_audit_logs(
    task_id: str | None = Query(default=None),
    agent_type: AgentType | None = Query(default=None),
    db: Session = Depends(get_db),
):
    return AuditService.list_logs(db, task_id=task_id, agent_type=agent_type)
