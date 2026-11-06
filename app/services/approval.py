"""Human approval state machine for Level 2 actions.

Lifecycle: PENDING -> APPROVED | REJECTED | EXPIRED (terminal).
Only PENDING requests can be decided; expired requests refuse execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session

from app.core.enums import ApprovalStatus, TaskStatus
from app.models.core import ApprovalRequest, Task
from app.services.audit import AuditService


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    """SQLite drops tzinfo on read; treat naive datetimes as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ApprovalNotFoundError(ValueError):
    """Raised when an approval request id does not exist."""


class ApprovalAlreadyDecidedError(ValueError):
    """Raised when a PENDING-only decision hits a terminal request."""


class ApprovalExpiredError(ValueError):
    """Raised when deciding a request past its expiry."""


class ApprovalService:
    """Decides, rejects and sweeps approval requests."""

    @staticmethod
    def _get_or_raise(db: Session, approval_id: str) -> ApprovalRequest:
        approval = db.get(ApprovalRequest, approval_id)
        if approval is None:
            raise ApprovalNotFoundError(f"ApprovalRequest {approval_id!r} not found")
        return approval

    @staticmethod
    def _ensure_pending(approval: ApprovalRequest) -> None:
        if approval.status != ApprovalStatus.PENDING:
            raise ApprovalAlreadyDecidedError(
                f"Approval request already decided (status={approval.status.value})"
            )

    @staticmethod
    def ensure_decidable(db: Session, approval_id: str) -> ApprovalRequest:
        """Fetch an approval and prove it can still be decided.

        Raises ApprovalNotFoundError / ApprovalAlreadyDecidedError, and flips
        stale requests to EXPIRED (+ audit) before raising ApprovalExpiredError.
        Callers must run this BEFORE executing any approved action.
        """
        approval = ApprovalService._get_or_raise(db, approval_id)
        ApprovalService._ensure_pending(approval)
        ApprovalService._expire_if_stale(db, approval)
        return approval

    @staticmethod
    def claim_approval(
        db: Session, approval_id: str, approver_identity: str
    ) -> ApprovalRequest:
        """Atomically claim a PENDING approval as APPROVED, before any execution.

        The conditional UPDATE ... WHERE status = PENDING is the concurrency
        guard: exactly one caller wins the claim, any other resume gets
        ApprovalAlreadyDecidedError. Task moves WAITING_APPROVAL -> IN_PROGRESS.
        """
        if not approver_identity or not approver_identity.strip():
            raise ValueError("A non-empty approver_identity is required")

        # Atomic claim: PENDING and still within expiry, checked at claim time.
        now = utc_now()
        # SQLite stores DATETIME(timezone=True) as a UTC-naive string, so the
        # comparison parameter must be naive UTC to compare correctly.
        now_db = now.replace(tzinfo=None)
        result = db.execute(
            update(ApprovalRequest)
            .where(
                ApprovalRequest.id == approval_id,
                ApprovalRequest.status == ApprovalStatus.PENDING,
                or_(
                    ApprovalRequest.expires_at.is_(None),
                    ApprovalRequest.expires_at > now_db,
                ),
            )
            .values(
                status=ApprovalStatus.APPROVED,
                approver_identity=approver_identity,
                decided_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            db.rollback()
            current = db.get(ApprovalRequest, approval_id)
            if current is None:
                raise ApprovalNotFoundError(f"ApprovalRequest {approval_id!r} not found")
            if current.status == ApprovalStatus.EXPIRED:
                raise ApprovalExpiredError("Approval request has expired")
            if current.status == ApprovalStatus.PENDING:
                # PENDING but past expiry at claim time: flip + audit, then refuse.
                ApprovalService._expire_if_stale(db, current)
            raise ApprovalAlreadyDecidedError(
                f"Approval request already decided (status={current.status.value})"
            )
        db.expire_all()
        approval = db.get(ApprovalRequest, approval_id)

        task = db.get(Task, approval.task_id)
        if task is not None and task.status == TaskStatus.WAITING_APPROVAL:
            task.status = TaskStatus.IN_PROGRESS

        AuditService.log_event(
            db,
            agent_type=approval.requesting_agent,
            action_name="approval.approved",
            permission_level=approval.task.permission_level,
            status="APPROVED_BY_HUMAN",
            task_id=approval.task_id,
            details={
                "approval_request_id": approval.id,
                "approver_identity": approver_identity,
            },
        )
        db.commit()
        db.refresh(approval)
        return approval

    @staticmethod
    def _expire_if_stale(db: Session, approval: ApprovalRequest) -> None:
        """Mark a stale request EXPIRED + audit it; raises ApprovalExpiredError."""
        if approval.expires_at is not None and _as_utc(approval.expires_at) <= utc_now():
            approval.status = ApprovalStatus.EXPIRED
            approval.decided_at = utc_now()
            AuditService.log_event(
                db,
                agent_type=approval.requesting_agent,
                action_name="approval.expired",
                permission_level=approval.task.permission_level,
                status="APPROVAL_EXPIRED",
                task_id=approval.task_id,
                details={"approval_request_id": approval.id},
            )
            db.commit()
            raise ApprovalExpiredError(
                "Approval request has expired; execution refused"
            )

    @staticmethod
    def approve_request(
        db: Session,
        approval_id: str,
        approver_identity: str,
        execution_result: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        approval = ApprovalService.ensure_decidable(db, approval_id)

        approval.status = ApprovalStatus.APPROVED
        approval.approver_identity = approver_identity
        approval.decided_at = utc_now()
        if execution_result is not None:
            approval.execution_result = execution_result

        task = db.get(Task, approval.task_id)
        if task is not None and task.status == TaskStatus.WAITING_APPROVAL:
            task.status = (
                TaskStatus.COMPLETED if execution_result is not None else TaskStatus.IN_PROGRESS
            )

        AuditService.log_event(
            db,
            agent_type=approval.requesting_agent,
            action_name="approval.approved",
            permission_level=approval.task.permission_level,
            status="APPROVED_BY_HUMAN",
            task_id=approval.task_id,
            details={
                "approval_request_id": approval.id,
                "approver_identity": approver_identity,
            },
        )
        db.commit()
        db.refresh(approval)
        return approval

    @staticmethod
    def reject_request(
        db: Session,
        approval_id: str,
        approver_identity: str,
        reason: str,
    ) -> ApprovalRequest:
        if not reason or not reason.strip():
            raise ValueError("A non-empty rejection reason is required")

        approval = ApprovalService.ensure_decidable(db, approval_id)

        approval.status = ApprovalStatus.REJECTED
        approval.approver_identity = approver_identity
        approval.rejection_reason = reason
        approval.decided_at = utc_now()

        task = db.get(Task, approval.task_id)
        if task is not None:
            task.status = TaskStatus.BLOCKED

        AuditService.log_event(
            db,
            agent_type=approval.requesting_agent,
            action_name="approval.rejected",
            permission_level=approval.task.permission_level,
            status="REJECTED_BY_HUMAN",
            task_id=approval.task_id,
            details={
                "approval_request_id": approval.id,
                "approver_identity": approver_identity,
                "reason": reason,
            },
        )
        db.commit()
        db.refresh(approval)
        return approval

    @staticmethod
    def expire_stale_requests(db: Session) -> int:
        """Sweep PENDING approvals past expiry -> EXPIRED, block their tasks."""
        stmt = select(ApprovalRequest).where(
            ApprovalRequest.status == ApprovalStatus.PENDING,
            ApprovalRequest.expires_at.is_not(None),
        )
        now = utc_now()
        stale = [
            approval
            for approval in db.scalars(stmt).all()
            if _as_utc(approval.expires_at) <= now
        ]
        for approval in stale:
            approval.status = ApprovalStatus.EXPIRED
            approval.decided_at = now
            task = db.get(Task, approval.task_id)
            if task is not None and task.status == TaskStatus.WAITING_APPROVAL:
                task.status = TaskStatus.BLOCKED
                task.error_message = "Approval request expired"
            AuditService.log_event(
                db,
                agent_type=approval.requesting_agent,
                action_name="approval.expired",
                permission_level=approval.task.permission_level,
                status="APPROVAL_EXPIRED",
                task_id=approval.task_id,
                details={"approval_request_id": approval.id},
            )
        db.commit()
        return len(stale)
