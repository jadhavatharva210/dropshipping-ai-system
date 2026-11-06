"""Operations API: workflows, task inspection, memory, agent listing."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db as get_db_session
from app.models.core import Task
from app.schemas.orchestrator import (
    AgentInfoResponse,
    MemoryEntryResponse,
    TaskResponse,
    WorkflowRunRequest,
    WorkflowReport,
)
from app.schemas.governance import ApprovalDecisionRequest
from app.services.approval import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
)
from app.services.memory import SharedMemoryService
from app.services.orchestrator import MainOrchestrator

router = APIRouter(tags=["orchestrator"])


def _orchestrator() -> MainOrchestrator:
    return MainOrchestrator()


@router.post("/orchestrator/workflows", response_model=WorkflowReport)
def run_workflow(
    payload: WorkflowRunRequest, db: Session = Depends(get_db_session)
) -> dict:
    try:
        return _orchestrator().run_workflow(
            db,
            objective=payload.objective,
            workflow_type=payload.workflow_type,
            parameters=payload.parameters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/orchestrator/approvals/{approval_id}/resume", response_model=WorkflowReport)
def resume_workflow(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    db: Session = Depends(get_db_session),
) -> dict:
    try:
        return _orchestrator().resume_after_approval(
            db, approval_id=approval_id, approver_identity=payload.approver_identity
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ApprovalAlreadyDecidedError, ApprovalExpiredError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/tasks", response_model=list[TaskResponse])
def list_tasks(db: Session = Depends(get_db_session)) -> list[Task]:
    stmt = select(Task).order_by(Task.created_at, Task.id)
    return list(db.scalars(stmt).all())


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db_session)) -> Task:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id!r} not found"
        )
    return task


@router.get("/memory/{namespace}", response_model=list[MemoryEntryResponse])
def list_memory(namespace: str, db: Session = Depends(get_db_session)):
    return SharedMemoryService.list_namespace(db, namespace=namespace)


@router.get("/agents", response_model=list[AgentInfoResponse])
def list_agents() -> list[AgentInfoResponse]:
    orchestrator = _orchestrator()
    return [
        AgentInfoResponse(
            agent_type=agent.agent_type,
            name=type(agent).__name__,
            description=agent.description,
        )
        for agent in orchestrator.agent_registry.list_agents()
    ]
