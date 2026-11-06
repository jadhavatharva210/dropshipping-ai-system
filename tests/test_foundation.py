"""Milestone 1A foundation suite.

What is covered
---------------
1. **Configuration** - Pydantic v2 settings, defaults, normalisation, and the
   validation that makes a bad ``.env`` fail at startup instead of mid-request.
2. **Enums** - the exact member sets the architecture specifies.
3. **HTTP** - ``GET /api/v1/health`` over FastAPI's httpx-based ``TestClient``,
   plus OpenAPI registration.
4. **Startup** - the lifespan creating the schema in development and refusing
   to do so in production (where Alembic owns migrations).
5. **Models** - create/read/update on ``Task``, ``ApprovalRequest``,
   ``AuditLog`` and ``SharedMemory``, JSON payload fidelity, UTC timestamps.
6. **Relationships** - task tree, task -> approvals, task -> audit trail, and
   the delete semantics of each foreign key.
7. **Constraints** - enum rejection at *both* the ORM layer and the database
   layer, foreign-key enforcement on SQLite, and unique keys.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from sqlalchemy import Engine, func, inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateTable

from app.core.config import (
    API_V1_PREFIX,
    APP_NAME,
    APP_VERSION,
    ARCHITECTURE_PHASE,
    Settings,
    settings,
)
from app.core.enums import AgentType, ApprovalStatus, PermissionLevel, TaskStatus
from app.db import session as session_module
from app.db.base import Base, utcnow
from app.db.session import build_engine, get_db
from app import main as app_main
from app.main import HealthResponse, create_app
from app.models.core import (
    ApprovalRequest,
    AuditLog,
    SharedMemory,
    Task,
)

#: Timestamp literal used by the raw-SQL tests (SQLite DATETIME format).
_NOW = "2026-01-01 00:00:00.000000"

HEALTH_URL = f"{API_V1_PREFIX}/health"


# ── Raw-SQL helpers: these bypass the ORM on purpose, so anything they prove
# ── is enforced by the database itself and not by Python.
def _raw_task_insert(
    task_id: str,
    *,
    agent: str = "PRODUCT_RESEARCH",
    status: str = "PENDING",
    level: str = "LEVEL_1_AUTO",
    parent_task_id: str | None = None,
):
    return text(
        "INSERT INTO tasks (id, parent_task_id, assigned_agent, title, objective, "
        "status, permission_level, retry_count, created_at, updated_at) "
        "VALUES (:id, :parent, :agent, 'title', 'objective', :status, :level, "
        "0, :now, :now)"
    ).bindparams(
        id=task_id,
        parent=parent_task_id,
        agent=agent,
        status=status,
        level=level,
        now=_NOW,
    )


def _raw_audit_insert(
    log_id: str, *, agent: str = "MAIN_ORCHESTRATOR", level: str = "LEVEL_1_AUTO"
):
    return text(
        "INSERT INTO audit_logs (id, task_id, agent_type, action_name, "
        "permission_level, status, created_at) "
        "VALUES (:id, NULL, :agent, 'action', :level, 'success', :now)"
    ).bindparams(id=log_id, agent=agent, level=level, now=_NOW)


def _raw_approval_insert(
    approval_id: str, task_id: str, *, status: str = "PENDING", agent: str = "PRODUCT_RESEARCH"
):
    return text(
        "INSERT INTO approval_requests (id, task_id, requesting_agent, "
        "action_description, status, created_at) "
        "VALUES (:id, :task_id, :agent, 'action', :status, :now)"
    ).bindparams(
        id=approval_id, task_id=task_id, agent=agent, status=status, now=_NOW
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("field", "default"),
    [
        ("app_env", "development"),
        ("database_url", "sqlite:///./dropshipping_dev.db"),
        ("ai_provider", "mock"),
        ("default_approval_expiry_minutes", 60),
    ],
)
def test_settings_defaults_match_env_example(field: str, default: Any) -> None:
    """The .env.example contract: these four keys, these four defaults."""
    assert Settings.model_fields[field].default == default


def test_env_example_stays_in_sync_with_settings() -> None:
    """Every documented key must exist and agree with the code default."""
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    documented: dict[str, str] = {}
    for raw_line in env_example.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        documented[key.strip().lower()] = value.strip()

    assert documented, ".env.example parsed to nothing - check the file format"
    for key, value in documented.items():
        assert key in Settings.model_fields, f"{key.upper()} is not a Settings field"
        assert str(Settings.model_fields[key].default) == value, (
            f"{key.upper()}: .env.example documents {value!r} but the code "
            f"defaults to {Settings.model_fields[key].default!r}"
        )


def test_settings_loaded_from_process_environment() -> None:
    """conftest forces these; proves env vars win over defaults/.env."""
    assert settings.app_env == "development"
    assert settings.database_url == "sqlite://"
    assert settings.ai_provider == "mock"
    assert settings.is_sqlite is True
    assert settings.is_postgres is False
    assert settings.is_development is True
    assert settings.is_production is False


def test_settings_exposes_approval_expiry_as_timedelta() -> None:
    assert settings.approval_expiry == timedelta(minutes=60)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"app_env": " Production "}, {"app_env": "production"}),
        ({"ai_provider": "MOCK"}, {"ai_provider": "mock"}),
        ({"database_url": " postgresql+psycopg://u:p@localhost/db "},
         {"database_url": "postgresql+psycopg://u:p@localhost/db",
          "is_postgres": True}),
    ],
)
def test_settings_normalise_input(overrides: dict[str, Any], expected: dict[str, Any]) -> None:
    config = Settings(_env_file=None, **overrides)
    for key, value in expected.items():
        assert getattr(config, key) == value


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"app_env": "dev"}, "APP_ENV"),
        ({"app_env": ""}, "APP_ENV"),
        ({"database_url": "mysql://user:pw@localhost/db"}, "Unsupported DATABASE_URL"),
        ({"database_url": ""}, "must not be empty"),
        ({"default_approval_expiry_minutes": 0}, "greater than 0"),
        ({"default_approval_expiry_minutes": -5}, "greater than 0"),
        ({"ai_provider": "skynet"}, "ai_provider"),
    ],
)
def test_settings_reject_invalid_configuration(
    overrides: dict[str, Any], message: str
) -> None:
    """A typo in .env must fail fast, at import time, with a readable message."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, **overrides)
    assert message in str(excinfo.value)


def test_application_identity_constants() -> None:
    assert APP_NAME == "Dropshipping AI System"
    assert APP_VERSION == "0.1.0"
    assert API_V1_PREFIX == "/api/v1"
    assert ARCHITECTURE_PHASE == "1A-foundation"


# ══════════════════════════════════════════════════════════════════════════
# 2. Enums
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("enum_cls", "expected_members"),
    [
        (
            PermissionLevel,
            ("LEVEL_1_AUTO", "LEVEL_2_APPROVAL", "LEVEL_3_BLOCKED"),
        ),
        (
            AgentType,
            (
                "MAIN_ORCHESTRATOR",
                "PRODUCT_RESEARCH",
                "ADVERTISING_ANALYTICS",
                "DELIVERY_OPERATIONS",
            ),
        ),
        (
            TaskStatus,
            (
                "PENDING",
                "IN_PROGRESS",
                "WAITING_APPROVAL",
                "COMPLETED",
                "FAILED",
                "BLOCKED",
            ),
        ),
        (ApprovalStatus, ("PENDING", "APPROVED", "REJECTED", "EXPIRED")),
    ],
)
def test_enum_members_match_the_specification(enum_cls, expected_members) -> None:
    """Guards the shared vocabulary: names *and* values, in order."""
    assert tuple(member.name for member in enum_cls) == expected_members
    assert tuple(member.value for member in enum_cls) == expected_members
    assert issubclass(enum_cls, str), "enums must serialise as plain strings"


@pytest.mark.parametrize(
    ("enum_cls", "value"),
    [
        (AgentType, "NOT_AN_AGENT"),
        (TaskStatus, "DOING_STUFF"),
        (PermissionLevel, "LEVEL_99"),
        (ApprovalStatus, "MAYBE"),
    ],
)
def test_invalid_enum_values_rejected_by_python(enum_cls, value: str) -> None:
    with pytest.raises(ValueError):
        enum_cls(value)


def test_permission_level_semantics() -> None:
    assert PermissionLevel.LEVEL_1_AUTO.is_autonomous is True
    assert PermissionLevel.LEVEL_1_AUTO.requires_approval is False
    assert PermissionLevel.LEVEL_2_APPROVAL.requires_approval is True
    assert PermissionLevel.LEVEL_2_APPROVAL.is_autonomous is False
    assert PermissionLevel.LEVEL_3_BLOCKED.is_blocked is True
    assert PermissionLevel.LEVEL_3_BLOCKED.requires_approval is False


def test_task_status_semantics() -> None:
    assert TaskStatus.PENDING.is_terminal is False
    assert TaskStatus.IN_PROGRESS.is_active is True
    assert TaskStatus.WAITING_APPROVAL.is_terminal is False
    for terminal in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED):
        assert terminal.is_terminal is True
        assert terminal.is_active is False


def test_approval_status_semantics() -> None:
    assert ApprovalStatus.PENDING.is_decided is False
    assert ApprovalStatus.PENDING.allows_execution is False
    assert ApprovalStatus.APPROVED.allows_execution is True
    for decided in (ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED, ApprovalStatus.APPROVED):
        assert decided.is_decided is True
    assert ApprovalStatus.REJECTED.allows_execution is False


def test_agent_type_orchestrator_flag() -> None:
    assert AgentType.MAIN_ORCHESTRATOR.is_orchestrator is True
    for worker in (
        AgentType.PRODUCT_RESEARCH,
        AgentType.ADVERTISING_ANALYTICS,
        AgentType.DELIVERY_OPERATIONS,
    ):
        assert worker.is_orchestrator is False


# ══════════════════════════════════════════════════════════════════════════
# 3. HTTP surface
# ══════════════════════════════════════════════════════════════════════════


def test_health_endpoint(client: TestClient) -> None:
    response = client.get(HEALTH_URL)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "architecture_phase": ARCHITECTURE_PHASE,
    }


def test_health_client_is_httpx_based(client: TestClient) -> None:
    """FastAPI's TestClient *is* an httpx client - same API as production HTTP."""
    assert isinstance(client, httpx.Client)
    response = client.get(HEALTH_URL)
    assert isinstance(response, httpx.Response)


def test_health_endpoint_is_registered_in_openapi(app: FastAPI) -> None:
    schema = app.openapi()
    assert HEALTH_URL in schema["paths"]
    assert schema["info"]["title"] == APP_NAME
    assert schema["info"]["version"] == APP_VERSION
    # The response is described by a real Pydantic v2 model, not a bare dict.
    assert issubclass(HealthResponse, BaseModel)
    assert set(HealthResponse.model_fields) == {
        "status",
        "environment",
        "architecture_phase",
    }


def test_health_endpoint_reports_the_active_environment(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "app_env", "staging")
    with TestClient(app) as client:
        assert client.get(HEALTH_URL).json()["environment"] == "staging"


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get(f"{API_V1_PREFIX}/does-not-exist").status_code == 404


# ══════════════════════════════════════════════════════════════════════════
# 4. Startup / lifespan
# ══════════════════════════════════════════════════════════════════════════


def test_lifespan_creates_the_schema_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh clone must boot with zero manual database setup."""
    fresh = build_engine("sqlite://")
    assert inspect(fresh).has_table("tasks") is False
    monkeypatch.setattr(session_module, "engine", fresh)
    monkeypatch.setattr(settings, "app_env", "development")
    try:
        with TestClient(create_app()) as client:
            assert client.get(HEALTH_URL).status_code == 200
            inspector = inspect(fresh)
            for table in ("tasks", "approval_requests", "audit_logs", "shared_memory"):
                assert inspector.has_table(table), f"{table} was not created at startup"
    finally:
        fresh.dispose()


def test_lifespan_releases_connections_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shutdown hook must release the pool - spied, not engine-dependent.

    ``tests/conftest.py`` no-ops this hook for fixture-driven apps (the harness
    owns that engine), so it is asserted here against a bare app instead.
    """
    calls: list[bool] = []
    monkeypatch.setattr(app_main, "dispose_engine", lambda: calls.append(True))
    monkeypatch.setattr(settings, "app_env", "production")  # skip create_all

    with TestClient(create_app()):
        assert calls == [], "connections must not be released before shutdown"
    assert calls == [True], "shutdown must release the connection pool exactly once"


def test_lifespan_does_not_create_the_schema_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside development, Alembic owns the schema - the app must not mutate it."""
    fresh = build_engine("sqlite://")
    monkeypatch.setattr(session_module, "engine", fresh)
    monkeypatch.setattr(settings, "app_env", "production")
    try:
        with TestClient(create_app()) as client:
            assert client.get(HEALTH_URL).json()["environment"] == "production"
        assert inspect(fresh).has_table("tasks") is False
    finally:
        fresh.dispose()


def test_all_core_tables_are_registered_on_metadata(engine: Engine) -> None:
    expected = {"tasks", "approval_requests", "audit_logs", "shared_memory"}
    assert expected <= set(Base.metadata.tables)
    assert expected <= set(inspect(engine).get_table_names())


def test_get_db_dependency_yields_a_working_session(
    monkeypatch: pytest.MonkeyPatch, engine: Engine, session_factory
) -> None:
    monkeypatch.setattr(session_module, "SessionLocal", session_factory)
    generator = get_db()
    session = next(generator)
    try:
        assert isinstance(session, Session)
        assert session.get_bind() is engine
        assert session.scalar(select(func.count()).select_from(Task)) == 0
    finally:
        with pytest.raises(StopIteration):
            next(generator)  # runs the `finally: db.close()` branch


# ══════════════════════════════════════════════════════════════════════════
# 5. Schema contract
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("model", "expected_columns"),
    [
        (
            Task,
            {
                "id",
                "parent_task_id",
                "assigned_agent",
                "title",
                "objective",
                "status",
                "permission_level",
                "input_payload",
                "output_payload",
                "error_message",
                "retry_count",
                "created_at",
                "updated_at",
            },
        ),
        (
            ApprovalRequest,
            {
                "id",
                "task_id",
                "requesting_agent",
                "action_description",
                "context_summary",
                "requested_parameters",
                "status",
                "approver_identity",
                "rejection_reason",
                "created_at",
                "expires_at",
                "decided_at",
                "execution_result",
            },
        ),
        (
            AuditLog,
            {
                "id",
                "task_id",
                "agent_type",
                "action_name",
                "permission_level",
                "status",
                "details",
                "created_at",
            },
        ),
        (
            SharedMemory,
            {
                "id",
                "namespace",
                "key",
                "value",
                "updated_by_agent",
                "created_at",
                "updated_at",
            },
        ),
    ],
)
def test_model_columns_match_the_specification(model, expected_columns) -> None:
    assert set(model.__table__.columns.keys()) == expected_columns


@pytest.mark.parametrize("model", [Task, ApprovalRequest, AuditLog, SharedMemory])
def test_primary_keys_are_uuid_strings(model) -> None:
    """UUID4 stored as VARCHAR(36): identical DDL on SQLite and PostgreSQL."""
    pk_columns = list(model.__table__.primary_key.columns)
    assert len(pk_columns) == 1
    pk = pk_columns[0]
    assert pk.name == "id"
    assert pk.type.__class__.__name__ == "String"
    assert pk.type.length == 36


def test_uuid_default_generates_unique_ids(db: Session) -> None:
    ids = set()
    for index in range(25):
        record = SharedMemory(
            namespace="test", key=f"key-{index}", value={"index": index},
            updated_by_agent=AgentType.MAIN_ORCHESTRATOR,
        )
        db.add(record)
        db.flush()
        ids.add(record.id)
    db.commit()
    assert len(ids) == 25
    for generated in ids:
        assert uuid.UUID(generated).version == 4


def test_enum_columns_carry_database_check_constraints(engine: Engine) -> None:
    """CHECK constraints must exist in the DDL, not only in Python."""
    expectations = {
        Task: ["assigned_agent", "status", "permission_level"],
        ApprovalRequest: ["requesting_agent", "status"],
        AuditLog: ["agent_type", "permission_level"],
        SharedMemory: ["updated_by_agent"],
    }
    for model, columns in expectations.items():
        ddl = str(CreateTable(model.__table__).compile(bind=engine))
        for column in columns:
            assert f"CHECK ({column} IN (" in ddl, f"{model.__tablename__}.{column}"


def test_constraint_names_are_deterministic_and_unique(engine: Engine) -> None:
    """Stable names are what make Alembic autogenerate reproducible."""
    for table in Base.metadata.sorted_tables:
        names = [constraint.name for constraint in table.constraints if constraint.name]
        duplicates = {name for name in names if names.count(name) > 1}
        assert not duplicates, f"{table.name} has duplicate constraint names: {duplicates}"
    fk = Task.__table__.c.parent_task_id.foreign_keys.copy().pop()
    assert fk.constraint.name == "fk_tasks_parent_task_id_tasks"


# ══════════════════════════════════════════════════════════════════════════
# 6. Task
# ══════════════════════════════════════════════════════════════════════════


def test_create_task_with_enums(db: Session) -> None:
    task = Task(
        assigned_agent=AgentType.MAIN_ORCHESTRATOR,
        title="Coordinate product launch",
        objective="Delegate research, advertising and delivery sub-tasks",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        input_payload={"launch_date": "2026-12-01"},
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    assert task.id is not None
    assert task.assigned_agent is AgentType.MAIN_ORCHESTRATOR
    assert task.status is TaskStatus.PENDING
    assert task.permission_level is PermissionLevel.LEVEL_1_AUTO
    assert task.input_payload == {"launch_date": "2026-12-01"}


def test_task_defaults_are_applied(db: Session) -> None:
    """status/retry_count/timestamps come from the model, not the caller."""
    task = Task(
        assigned_agent=AgentType.DELIVERY_OPERATIONS,
        title="Track shipment",
        objective="Poll carrier API until delivered",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    assert task.status is TaskStatus.PENDING
    assert task.retry_count == 0
    assert task.parent_task_id is None
    assert task.output_payload is None
    assert task.error_message is None


def test_task_timestamps_are_utc_aware(db: Session, sample_task: Task) -> None:
    """SQLite drops tzinfo; UTCDateTime puts it back so comparisons are safe."""
    before = utcnow()
    for stamp in (sample_task.created_at, sample_task.updated_at):
        assert isinstance(stamp, datetime)
        assert stamp.tzinfo is not None
        assert stamp.utcoffset() == timedelta(0)
        assert stamp <= before + timedelta(seconds=5)


def test_task_updated_at_is_refreshed_on_write(db: Session, sample_task: Task) -> None:
    created = sample_task.created_at
    updated = sample_task.updated_at
    sample_task.title = "Find winning products (v2)"
    db.commit()
    db.refresh(sample_task)
    assert sample_task.updated_at >= updated
    assert sample_task.created_at == created, "created_at must never move"


def test_task_parent_child_hierarchy(db: Session, sample_task: Task) -> None:
    child = Task(
        parent_task_id=sample_task.id,
        assigned_agent=AgentType.ADVERTISING_ANALYTICS,
        title="Analyse ad creatives",
        objective="Score the top 5 creatives for the niche",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
    )
    db.add(child)
    db.commit()
    db.refresh(sample_task)

    assert [task.id for task in sample_task.children] == [child.id]
    assert child.parent is not None
    assert child.parent.id == sample_task.id
    assert sample_task.parent is None


def test_task_hierarchy_supports_multiple_levels(db: Session, sample_task: Task) -> None:
    child = Task(
        parent_task_id=sample_task.id,
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Shortlist suppliers",
        objective="Find 3 suppliers with <7 day delivery",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
    )
    db.add(child)
    db.commit()
    grandchild = Task(
        parent_task_id=child.id,
        assigned_agent=AgentType.DELIVERY_OPERATIONS,
        title="Verify shipping times",
        objective="Order samples and measure real delivery time",
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
    )
    db.add(grandchild)
    db.commit()
    db.refresh(sample_task)

    assert len(sample_task.children) == 1
    assert len(sample_task.children[0].children) == 1
    assert grandchild.parent.parent.id == sample_task.id


def test_task_status_transitions(db: Session, sample_task: Task) -> None:
    for status in (
        TaskStatus.IN_PROGRESS,
        TaskStatus.WAITING_APPROVAL,
        TaskStatus.IN_PROGRESS,
        TaskStatus.COMPLETED,
    ):
        sample_task.status = status
        db.commit()
        db.refresh(sample_task)
        assert sample_task.status is status
    assert sample_task.is_terminal is True


def test_task_records_output_error_and_retry(db: Session, sample_task: Task) -> None:
    sample_task.output_payload = {"products_found": 10, "top_pick": "LED dog collar"}
    sample_task.status = TaskStatus.COMPLETED
    db.commit()
    db.refresh(sample_task)
    assert sample_task.output_payload["top_pick"] == "LED dog collar"

    sample_task.status = TaskStatus.FAILED
    sample_task.error_message = "Supplier API timeout after 30s"
    sample_task.retry_count = 2
    db.commit()
    db.refresh(sample_task)
    assert sample_task.error_message == "Supplier API timeout after 30s"
    assert sample_task.retry_count == 2
    assert sample_task.is_terminal is True


def test_task_blocked_by_permission_level(db: Session, sample_task: Task) -> None:
    """LEVEL_3_BLOCKED work is recorded, never executed."""
    sample_task.permission_level = PermissionLevel.LEVEL_3_BLOCKED
    sample_task.status = TaskStatus.BLOCKED
    db.commit()
    db.refresh(sample_task)
    assert sample_task.permission_level.is_blocked is True
    assert sample_task.status is TaskStatus.BLOCKED


def test_json_payloads_round_trip_exactly(db: Session, sample_task: Task) -> None:
    payload = {
        "nested": {"list": [1, 2, 3], "flag": True, "nothing": None},
        "unicode": "पेट कैमरा 🐶",
        "float": 19.99,
        "empty": {},
    }
    sample_task.input_payload = payload
    db.commit()
    db.expunge_all()

    reloaded = db.get(Task, sample_task.id)
    assert reloaded is not None
    assert reloaded.input_payload == payload


def test_in_place_json_edits_are_persisted(db: Session, sample_task: Task) -> None:
    """MutableDict: a top-level in-place edit must not be silently lost."""
    sample_task.output_payload = {"products_found": 0}
    db.commit()

    sample_task.output_payload["products_found"] = 10
    sample_task.output_payload["top_pick"] = "LED dog collar"
    db.commit()
    db.expunge_all()

    reloaded = db.get(Task, sample_task.id)
    assert reloaded.output_payload == {"products_found": 10, "top_pick": "LED dog collar"}


def test_nested_json_edits_need_reassignment(db: Session, sample_task: Task) -> None:
    """MutableDict tracks the *outer* object only.

    Nested containers must be reassigned, and the owning instance has to stay
    reachable - a payload dict taken from an instance that is then collected has
    no parent to notify, so the edit would be dropped. Pinned here so the rule
    is documented by a test rather than by a bug report.
    """
    sample_task.output_payload = {"products": []}
    db.commit()

    sample_task.output_payload["products"].append("invisible")  # nested: not tracked
    db.commit()
    db.expunge_all()

    reloaded = db.get(Task, sample_task.id)
    assert reloaded.output_payload == {"products": []}

    reloaded.output_payload["products"] = ["LED dog collar"]  # outer key: tracked
    db.commit()
    db.expunge_all()

    final = db.get(Task, sample_task.id)
    assert final.output_payload == {"products": ["LED dog collar"]}


def test_payload_columns_reject_non_object_values(db: Session, sample_task: Task) -> None:
    """Object-shaped payloads enforce their contract at assignment time."""
    with pytest.raises(ValueError, match="does not accept objects of type"):
        sample_task.input_payload = ["not", "a", "dict"]  # type: ignore[assignment]


def test_tasks_can_be_queried_by_enum(db: Session, sample_task: Task) -> None:
    other = Task(
        assigned_agent=AgentType.ADVERTISING_ANALYTICS,
        title="Draft ad copy",
        objective="Write 3 hooks for the winning product",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        status=TaskStatus.IN_PROGRESS,
    )
    db.add(other)
    db.commit()

    pending = db.scalars(select(Task).where(Task.status == TaskStatus.PENDING)).all()
    assert [task.id for task in pending] == [sample_task.id]
    # A valid raw string works too (validate_strings allows known values).
    assert db.scalar(
        select(func.count()).select_from(Task).where(Task.status == "IN_PROGRESS")
    ) == 1
    assert db.scalar(
        select(func.count())
        .select_from(Task)
        .where(Task.assigned_agent == AgentType.PRODUCT_RESEARCH)
    ) == 1


# ══════════════════════════════════════════════════════════════════════════
# 7. ApprovalRequest
# ══════════════════════════════════════════════════════════════════════════


def test_create_approval_request_linked_to_task(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.PRODUCT_RESEARCH,
        action_description="Place a test order with the supplier",
        context_summary="Verify product quality before spending on ads",
        requested_parameters={"supplier": "AliExpress", "budget_usd": 50},
        expires_at=utcnow() + timedelta(minutes=60),
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)

    assert approval.id is not None
    assert approval.status is ApprovalStatus.PENDING
    assert approval.task.id == sample_task.id
    assert [item.id for item in sample_task.approvals] == [approval.id]
    assert approval.requested_parameters == {"supplier": "AliExpress", "budget_usd": 50}
    assert approval.decided_at is None


def test_approval_expiry_defaults_from_settings(db: Session, sample_task: Task) -> None:
    """DEFAULT_APPROVAL_EXPIRY_MINUTES must actually drive the model."""
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.ADVERTISING_ANALYTICS,
        action_description="Launch campaign with $20/day budget",
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)

    assert approval.expires_at is not None
    window = approval.expires_at - approval.created_at
    assert settings.approval_expiry <= window <= settings.approval_expiry + timedelta(seconds=5)
    assert approval.is_expired is False


def test_approval_request_is_expired_flag(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.DELIVERY_OPERATIONS,
        action_description="Book courier pickup",
        expires_at=utcnow() - timedelta(minutes=1),
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)

    assert approval.status is ApprovalStatus.PENDING
    assert approval.is_expired is True


def test_approval_decision_flow(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.DELIVERY_OPERATIONS,
        action_description="Book courier pickup",
        requested_parameters={"courier": "Delhivery"},
    )
    db.add(approval)
    db.commit()

    approval.status = ApprovalStatus.APPROVED
    approval.approver_identity = "atharva"
    approval.decided_at = utcnow()
    approval.execution_result = {"awb": "1234567890", "pickup_scheduled": True}
    db.commit()
    db.refresh(approval)

    assert approval.status is ApprovalStatus.APPROVED
    assert approval.status.allows_execution is True
    assert approval.approver_identity == "atharva"
    assert approval.execution_result["awb"] == "1234567890"
    assert approval.decided_at is not None
    assert approval.decided_at.tzinfo is not None


def test_approval_rejection_flow(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.ADVERTISING_ANALYTICS,
        action_description="Increase daily ad budget to $500",
        requested_parameters={"daily_budget_usd": 500},
    )
    db.add(approval)
    db.commit()

    approval.status = ApprovalStatus.REJECTED
    approval.rejection_reason = "Weekly budget cap already reached"
    approval.decided_at = utcnow()
    db.commit()
    db.refresh(approval)

    assert approval.status is ApprovalStatus.REJECTED
    assert approval.status.allows_execution is False
    assert approval.rejection_reason == "Weekly budget cap already reached"
    assert approval.execution_result is None


def test_task_waits_for_approval_then_resumes(db: Session, sample_task: Task) -> None:
    """The two-way approval model, end to end at the persistence layer."""
    sample_task.status = TaskStatus.WAITING_APPROVAL
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=sample_task.assigned_agent,
        action_description="Order 5 units of the shortlisted product",
        requested_parameters={"units": 5},
    )
    db.add(approval)
    db.commit()
    assert sample_task.status is TaskStatus.WAITING_APPROVAL

    approval.status = ApprovalStatus.APPROVED
    approval.decided_at = utcnow()
    sample_task.status = TaskStatus.IN_PROGRESS
    db.commit()
    db.refresh(sample_task)

    assert sample_task.status is TaskStatus.IN_PROGRESS
    assert db.scalar(
        select(func.count())
        .select_from(ApprovalRequest)
        .where(ApprovalRequest.status == ApprovalStatus.APPROVED)
    ) == 1


# ══════════════════════════════════════════════════════════════════════════
# 8. AuditLog
# ══════════════════════════════════════════════════════════════════════════


def test_create_audit_log_records(db: Session, sample_task: Task) -> None:
    entry = AuditLog(
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name="task.created",
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
        status="success",
        details={"title": sample_task.title},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    assert entry.id is not None
    assert entry.task.id == sample_task.id
    assert [row.id for row in sample_task.audit_entries] == [entry.id]
    assert entry.created_at.tzinfo is not None
    assert entry.details == {"title": sample_task.title}


def test_audit_log_without_task(db: Session) -> None:
    entry = AuditLog(
        task_id=None,
        agent_type=AgentType.MAIN_ORCHESTRATOR,
        action_name="system.startup",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        status="success",
        details={"version": APP_VERSION},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    assert entry.task is None
    assert entry.task_id is None


def test_audit_log_records_blocked_action(db: Session, sample_task: Task) -> None:
    entry = AuditLog(
        task_id=sample_task.id,
        agent_type=AgentType.ADVERTISING_ANALYTICS,
        action_name="ads.pause_all_campaigns",
        permission_level=PermissionLevel.LEVEL_3_BLOCKED,
        status="blocked",
        details={"reason": "LEVEL_3 actions are never executed by an agent"},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    assert entry.status == "blocked"
    assert entry.permission_level.is_blocked is True


def test_audit_trail_is_ordered(db: Session, sample_task: Task) -> None:
    for index, action in enumerate(["task.created", "task.started", "task.completed"]):
        db.add(
            AuditLog(
                task_id=sample_task.id,
                agent_type=AgentType.MAIN_ORCHESTRATOR,
                action_name=action,
                permission_level=PermissionLevel.LEVEL_1_AUTO,
                status="success",
                details={"step": index},
                created_at=utcnow() + timedelta(seconds=index),
            )
        )
    db.commit()
    db.refresh(sample_task)
    assert [entry.action_name for entry in sample_task.audit_entries] == [
        "task.created",
        "task.started",
        "task.completed",
    ]


# ══════════════════════════════════════════════════════════════════════════
# 9. SharedMemory
# ══════════════════════════════════════════════════════════════════════════


def test_create_and_update_shared_memory(db: Session) -> None:
    record = SharedMemory(
        namespace="product-research",
        key="winning_products",
        value={"products": ["LED dog collar", "pet camera"]},
        updated_by_agent=AgentType.PRODUCT_RESEARCH,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    assert record.id is not None
    assert record.created_at.tzinfo is not None
    assert record.updated_by_agent is AgentType.PRODUCT_RESEARCH

    record.value = {"products": ["LED dog collar", "pet camera", "slow feeder"]}
    record.updated_by_agent = AgentType.MAIN_ORCHESTRATOR
    db.commit()
    db.refresh(record)
    assert len(record.value["products"]) == 3
    assert record.updated_by_agent is AgentType.MAIN_ORCHESTRATOR


def test_shared_memory_accepts_any_json_shape(db: Session) -> None:
    """value is plain JSON (not MutableDict), so any shape is allowed."""
    values: list[Any] = [{"a": 1}, ["x", "y"], "scalar", 42, 4.2, True, None]
    for index, value in enumerate(values):
        db.add(
            SharedMemory(
                namespace="shapes",
                key=f"key-{index}",
                value=value,
                updated_by_agent=AgentType.MAIN_ORCHESTRATOR,
            )
        )
    db.commit()
    stored = db.scalars(
        select(SharedMemory).where(SharedMemory.namespace == "shapes")
    ).all()
    assert len(stored) == len(values)
    by_key = {record.key: record.value for record in stored}
    for index, value in enumerate(values):
        assert by_key[f"key-{index}"] == value


def test_shared_memory_is_shared_between_agents(db: Session) -> None:
    """One writer, one reader, no direct coupling between the agents."""
    writer = SharedMemory(
        namespace="advertising",
        key="best_creative_id",
        value={"creative_id": "cr-77", "ctr": 0.043},
        updated_by_agent=AgentType.ADVERTISING_ANALYTICS,
    )
    db.add(writer)
    db.commit()

    reader = db.scalar(
        select(SharedMemory).where(
            SharedMemory.namespace == "advertising",
            SharedMemory.key == "best_creative_id",
        )
    )
    assert reader is not None
    assert reader.value["creative_id"] == "cr-77"
    assert reader.updated_by_agent is AgentType.ADVERTISING_ANALYTICS


def test_shared_memory_rejects_duplicate_namespace_key(db: Session) -> None:
    db.add_all(
        [
            SharedMemory(
                namespace="dup", key="same", value={"n": 1},
                updated_by_agent=AgentType.PRODUCT_RESEARCH,
            ),
            SharedMemory(
                namespace="dup", key="same", value={"n": 2},
                updated_by_agent=AgentType.DELIVERY_OPERATIONS,
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    assert db.scalar(select(func.count()).select_from(SharedMemory)) == 0


# ══════════════════════════════════════════════════════════════════════════
# 10. Relationships and referential integrity
# ══════════════════════════════════════════════════════════════════════════


def test_task_approval_audit_chain(db: Session, sample_task: Task) -> None:
    approval = ApprovalRequest(
        task_id=sample_task.id,
        requesting_agent=AgentType.PRODUCT_RESEARCH,
        action_description="Approve the supplier shortlist",
    )
    audit = AuditLog(
        task_id=sample_task.id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name="approval.requested",
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
        status="success",
    )
    db.add_all([approval, audit])
    db.commit()
    db.refresh(sample_task)

    assert len(sample_task.approvals) == 1
    assert len(sample_task.audit_entries) == 1
    assert sample_task.approvals[0].requesting_agent is AgentType.PRODUCT_RESEARCH
    assert sample_task.audit_entries[0].action_name == "approval.requested"
    assert sample_task.approvals[0].task is sample_task


def test_sqlite_foreign_keys_are_enforced(engine: Engine) -> None:
    """SQLite ignores FKs unless the pragma is on - build_engine switches it on."""
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_orphan_approval_request_is_rejected(db: Session) -> None:
    approval = ApprovalRequest(
        task_id=str(uuid.uuid4()),  # no such task
        requesting_agent=AgentType.PRODUCT_RESEARCH,
        action_description="Spend money on a task that does not exist",
    )
    db.add(approval)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    assert db.scalar(select(func.count()).select_from(ApprovalRequest)) == 0


def test_orphan_child_task_is_rejected(db: Session) -> None:
    child = Task(
        parent_task_id=str(uuid.uuid4()),
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Orphan",
        objective="Should never be persisted",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
    )
    db.add(child)
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_deleting_a_task_cascades_and_preserves_the_audit_trail(
    db: Session, sample_task: Task
) -> None:
    task_id = sample_task.id
    child = Task(
        parent_task_id=task_id,
        assigned_agent=AgentType.ADVERTISING_ANALYTICS,
        title="Child",
        objective="Deleted with its parent",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
    )
    approval = ApprovalRequest(
        task_id=task_id,
        requesting_agent=AgentType.PRODUCT_RESEARCH,
        action_description="Deleted with its task",
    )
    audit = AuditLog(
        task_id=task_id,
        agent_type=AgentType.PRODUCT_RESEARCH,
        action_name="task.created",
        permission_level=PermissionLevel.LEVEL_1_AUTO,
        status="success",
    )
    db.add_all([child, approval, audit])
    db.commit()
    audit_id = audit.id

    db.delete(sample_task)
    db.commit()

    assert db.get(Task, task_id) is None
    assert db.scalar(select(func.count()).select_from(Task)) == 0, "children must cascade"
    assert db.scalar(select(func.count()).select_from(ApprovalRequest)) == 0

    surviving = db.get(AuditLog, audit_id)
    assert surviving is not None, "audit rows must outlive their task"
    # expire_on_commit=False keeps the pre-delete value in memory, so re-read it:
    # the database applied ON DELETE SET NULL, the identity map has not noticed.
    db.refresh(surviving)
    assert surviving.task_id is None
    assert surviving.task is None
    assert surviving.action_name == "task.created"


def test_deleting_a_task_at_the_database_level_cascades(db: Session, sample_task: Task) -> None:
    """Proves the ON DELETE rules live in the DDL, not only in the ORM."""
    db.execute(_raw_approval_insert("a-raw", sample_task.id))
    db.execute(_raw_audit_insert("l-raw"))
    db.execute(
        text("UPDATE audit_logs SET task_id = :tid WHERE id = 'l-raw'").bindparams(
            tid=sample_task.id
        )
    )
    db.commit()

    db.execute(text("DELETE FROM tasks WHERE id = :tid").bindparams(tid=sample_task.id))
    db.commit()

    assert db.scalar(select(func.count()).select_from(ApprovalRequest)) == 0
    assert db.scalar(select(func.count()).select_from(AuditLog)) == 1
    assert db.scalar(select(AuditLog.task_id)) is None


# ══════════════════════════════════════════════════════════════════════════
# 11. Enum constraints, enforced twice
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("description", "statement"),
    [
        ("tasks.assigned_agent", _raw_task_insert("bad-agent", agent="NOT_AN_AGENT")),
        ("tasks.status", _raw_task_insert("bad-status", status="DOING_STUFF")),
        (
            "tasks.permission_level",
            _raw_task_insert("bad-level", level="LEVEL_99"),
        ),
        ("audit_logs.agent_type", _raw_audit_insert("bad-audit-agent", agent="ROBOT")),
        (
            "audit_logs.permission_level",
            _raw_audit_insert("bad-audit-level", level="LEVEL_0"),
        ),
    ],
)
def test_database_check_constraints_reject_invalid_enum_values(
    db: Session, description: str, statement
) -> None:
    """Raw SQL cannot smuggle an unknown enum value past the CHECK constraint."""
    with pytest.raises(IntegrityError):
        db.execute(statement)
    db.rollback()
    assert db.execute(text(f"SELECT COUNT(*) FROM {description.split('.')[0]}")).scalar() == 0


def test_database_rejects_invalid_approval_status(db: Session, sample_task: Task) -> None:
    with pytest.raises(IntegrityError):
        db.execute(_raw_approval_insert("bad-approval", sample_task.id, status="MAYBE"))
    db.rollback()
    assert db.scalar(select(func.count()).select_from(ApprovalRequest)) == 0


def test_database_rejects_invalid_shared_memory_agent(db: Session) -> None:
    with pytest.raises(IntegrityError):
        db.execute(
            text(
                "INSERT INTO shared_memory (id, namespace, key, value, "
                "updated_by_agent, created_at, updated_at) "
                "VALUES ('sm-bad', 'ns', 'k', '{}', 'GHOST_AGENT', :now, :now)"
            ).bindparams(now=_NOW)
        )
    db.rollback()


def test_orm_rejects_unknown_enum_string(db: Session, sample_task: Task) -> None:
    """validate_strings=True: a typo fails in Python, before it reaches SQL."""
    sample_task.status = "DOING_STUFF"  # type: ignore[assignment]
    with pytest.raises(StatementError) as excinfo:
        db.flush()
    assert isinstance(excinfo.value.orig, LookupError)
    db.rollback()


def test_orm_rejects_value_from_the_wrong_enum(db: Session, sample_task: Task) -> None:
    sample_task.assigned_agent = ApprovalStatus.PENDING  # type: ignore[assignment]
    with pytest.raises(StatementError):
        db.flush()
    db.rollback()


def test_orm_rejects_non_enum_value(db: Session, sample_task: Task) -> None:
    sample_task.permission_level = 3  # type: ignore[assignment]
    with pytest.raises(StatementError):
        db.flush()
    db.rollback()


def test_required_fields_cannot_be_null(db: Session) -> None:
    """NOT NULL is part of the contract for title/objective/permission level."""
    db.add(
        Task(
            assigned_agent=AgentType.PRODUCT_RESEARCH,
            title=None,  # type: ignore[arg-type]
            objective="Missing a title",
            permission_level=PermissionLevel.LEVEL_1_AUTO,
        )
    )
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ══════════════════════════════════════════════════════════════════════════
# 12. Backend portability (SQLite dev -> PostgreSQL prod)
# ══════════════════════════════════════════════════════════════════════════


def test_schema_compiles_for_postgresql_unchanged() -> None:
    """Swapping DATABASE_URL must be the *only* change needed for Postgres.

    Compiling against the PostgreSQL dialect needs no driver and no server, so
    this runs in CI while still proving the schema is not SQLite-specific.
    """
    dialect = postgresql.dialect()
    ddl = {
        table.name: str(CreateTable(table).compile(dialect=dialect))
        for table in Base.metadata.sorted_tables
    }

    assert set(ddl) == {"tasks", "approval_requests", "audit_logs", "shared_memory"}
    # UTCDateTime -> real timestamptz, not a naive TIMESTAMP.
    assert ddl["tasks"].count("TIMESTAMP WITH TIME ZONE") == 2
    assert "TIMESTAMP WITH TIME ZONE" in ddl["approval_requests"]
    assert "TIMESTAMP WITH TIME ZONE" in ddl["audit_logs"]
    # Enums stay portable VARCHAR + CHECK instead of CREATE TYPE.
    assert "VARCHAR(32)" in ddl["tasks"]
    assert "CREATE TYPE" not in ddl["tasks"]
    assert ddl["tasks"].count("CHECK (") == 3
    # Payloads are native JSON.
    assert "input_payload JSON" in ddl["tasks"]
    assert "value JSON" in ddl["shared_memory"]
    # Delete semantics travel with the schema.
    assert "ON DELETE CASCADE" in ddl["approval_requests"]
    assert "ON DELETE CASCADE" in ddl["tasks"]
    assert "ON DELETE SET NULL" in ddl["audit_logs"]
    assert "UNIQUE (namespace, key)" in ddl["shared_memory"]


def test_build_engine_pools_per_backend() -> None:
    """In-memory SQLite shares one connection; everything else gets a pool."""
    memory_engine = build_engine("sqlite://")
    file_engine = build_engine("sqlite:///./_probe_never_used.db")
    try:
        assert isinstance(memory_engine.pool, StaticPool)
        assert not isinstance(file_engine.pool, StaticPool)
        # create_engine is lazy: no connection, therefore no file on disk yet.
        assert file_engine.url.database is not None
        assert file_engine.url.database.endswith("_probe_never_used.db")
    finally:
        memory_engine.dispose()
        file_engine.dispose()


def test_file_backed_sqlite_default_works_end_to_end(tmp_path) -> None:
    """The .env.example default (a file-backed SQLite URL) must just work."""
    db_file = tmp_path / "dropshipping_dev.db"
    file_engine = build_engine(f"sqlite:///{db_file}")
    factory = sessionmaker(bind=file_engine, expire_on_commit=False)
    try:
        Base.metadata.create_all(bind=file_engine)
        with file_engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1

        with factory() as session:
            session.add(
                SharedMemory(
                    namespace="smoke",
                    key="boot",
                    value={"ok": True},
                    updated_by_agent=AgentType.MAIN_ORCHESTRATOR,
                )
            )
            session.commit()

        with factory() as session:  # brand-new connection, data must survive
            assert session.scalar(select(func.count()).select_from(SharedMemory)) == 1
        assert db_file.exists()
    finally:
        Base.metadata.drop_all(bind=file_engine)
        file_engine.dispose()
