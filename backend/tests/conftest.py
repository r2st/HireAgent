"""Shared pytest fixtures.

Tests run against an in-memory SQLite database so no server is required. The
environment is configured before ``app`` is imported so settings pick it up.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("ENCRYPTION_KEY", "test-encryption-key-material")
os.environ.setdefault("LLM_ENABLED", "false")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.api.deps import get_session  # noqa: E402
from app.core.ratelimit import reset_local_windows  # noqa: E402
from app.integrations import calendar as calendar_module  # noqa: E402
from app.integrations import openrouter as openrouter_module  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402
from app.models.enums import UserRole  # noqa: E402
from app.services import storage as storage_module  # noqa: E402
from app.workers import reminders as reminders_module  # noqa: E402


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path):
    """Point resume storage at a per-test directory.

    Without this, uploads in tests would write into the repository's ``var/``
    directory and leak between runs.
    """
    storage_module.set_storage(storage_module.LocalStorage(tmp_path / "storage"))
    yield tmp_path / "storage"
    storage_module.set_storage(None)


@pytest.fixture(autouse=True)
def no_llm():
    """Fail loudly rather than call OpenRouter if a test forgets to stub it.

    ``LLM_ENABLED=false`` already short-circuits the real client, but resetting
    the module-level singleton stops one test's fake client leaking into the
    next.
    """
    openrouter_module.set_llm_client(None)
    yield
    openrouter_module.set_llm_client(None)


@pytest.fixture(autouse=True)
def no_calendar():
    """Reset the calendar provider registry around every test.

    Left alone, one test's fake provider would answer another test's
    availability lookup. Reset to ``None`` means the registry rebuilds from
    settings, which in tests configures no OAuth client and so yields the
    unavailable providers.
    """
    calendar_module.set_providers(None)
    reminders_module.set_notifier(None)
    yield
    calendar_module.set_providers(None)
    reminders_module.set_notifier(None)


@pytest_asyncio.fixture
async def engine():
    """A fresh in-memory database per test.

    StaticPool keeps every connection pointed at the same in-memory database;
    without it each connection would get its own empty one.
    """
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest_asyncio.fixture
async def client(engine) -> AsyncGenerator[AsyncClient, None]:
    """An HTTP client bound to the app with the test database injected."""
    reset_local_windows()
    app = create_app()
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _override_session() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as s:
            yield s

    app.dependency_overrides[get_session] = _override_session

    # Skip the real lifespan: it would open the production engine and Redis.
    app.state.redis = None
    app.state.rate_limiter = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"


@pytest_asyncio.fixture
async def registered(client: AsyncClient) -> dict:
    """Register an organization and return its payload plus auth headers."""
    email = unique_email("admin")
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "organization_name": "Acme Talent",
            "organization_type": "company",
            "full_name": "Ada Admin",
            "email": email,
            "password": "Str0ngPassword1",
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return {
        **data,
        "email": email,
        "password": "Str0ngPassword1",
        "headers": {"Authorization": f"Bearer {data['tokens']['access_token']}"},
    }


@pytest_asyncio.fixture
async def auth_headers(registered: dict) -> dict[str, str]:
    return registered["headers"]


@pytest_asyncio.fixture
async def second_org(client: AsyncClient) -> dict:
    """A second, unrelated tenant — used to prove cross-tenant isolation."""
    email = unique_email("other")
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "organization_name": "Rival Recruiters",
            "organization_type": "agency",
            "full_name": "Bob Boss",
            "email": email,
            "password": "Str0ngPassword2",
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return {
        **data,
        "email": email,
        "headers": {"Authorization": f"Bearer {data['tokens']['access_token']}"},
    }


async def make_user(
    client: AsyncClient,
    admin_headers: dict[str, str],
    role: UserRole,
    password: str = "Str0ngPassword9",
) -> dict:
    """Create an extra user in the admin's org and log them in."""
    email = unique_email(role.value)
    resp = await client.post(
        "/api/v1/auth/users",
        headers=admin_headers,
        json={
            "email": email,
            "full_name": f"{role.value.title()} User",
            "role": role.value,
            "password": password,
        },
    )
    assert resp.status_code == 201, resp.text
    user = resp.json()

    login = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": password}
    )
    assert login.status_code == 200, login.text
    tokens = login.json()["tokens"]
    return {
        "user": user,
        "headers": {"Authorization": f"Bearer {tokens['access_token']}"},
    }
