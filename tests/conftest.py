"""Shared fixtures: in-memory SQLite DB, clean session per test, TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.enums import AgentType, PermissionLevel, TaskStatus
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from app.models.core import Task

TEST_DATABASE_URL = "sqlite://"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@pytest.fixture(scope="function")
def db() -> Session:
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db: Session) -> TestClient:
    app = create_app()

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture(scope="function")
def sample_task(db: Session) -> Task:
    task = Task(
        assigned_agent=AgentType.PRODUCT_RESEARCH,
        title="Find winning products",
        objective="Research 10 trending products under $20",
        status=TaskStatus.PENDING,
        permission_level=PermissionLevel.LEVEL_2_APPROVAL,
        input_payload={"niche": "pet accessories", "max_price": 20},
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task
