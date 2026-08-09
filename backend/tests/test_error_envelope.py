"""Every error response uses one body shape.

Clients parse ``detail`` to show a message, so ``detail`` must always be an
object with ``code`` and ``message`` — never a bare string. Starlette raises
some errors (unmatched route, wrong method) before any of our code runs, and
FastAPI raises others (request validation), so the shape is easy to break
without noticing.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_session
from app.core.errors import AppError, ConflictError
from app.core.ratelimit import RateLimiter, reset_local_windows
from app.main import create_app


def assert_envelope(resp, status: int, code: str) -> None:
    assert resp.status_code == status, resp.text
    detail = resp.json()["detail"]
    assert isinstance(detail, dict), f"detail must be an object, got {detail!r}"
    assert detail["code"] == code
    assert isinstance(detail["message"], str) and detail["message"]


class TestFrameworkErrors:
    """Errors raised before or around our handlers still get the envelope."""

    async def test_unmatched_route(self, client: AsyncClient) -> None:
        assert_envelope(await client.get("/api/v1/no-such-thing"), 404, "not_found")

    async def test_wrong_method(self, client: AsyncClient) -> None:
        assert_envelope(
            await client.delete("/api/v1/auth/login"), 405, "method_not_allowed"
        )

    async def test_missing_bearer_token(self, client: AsyncClient) -> None:
        assert_envelope(
            await client.get("/api/v1/jobs"), 401, "authentication_error"
        )

    async def test_request_validation_error(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        """FastAPI's 422 keeps its own body, but still nests under ``detail``."""
        resp = await client.post("/api/v1/jobs", headers=auth_headers, json={})
        assert resp.status_code == 422
        assert "detail" in resp.json()

    async def test_a_served_route_is_untouched(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAppErrorFallback:
    """An AppError that escapes a route matches an explicitly converted one."""

    @pytest_asyncio.fixture
    async def raising_client(self, engine) -> AsyncGenerator[AsyncClient, None]:
        app = create_app()

        @app.get("/boom")
        async def boom() -> None:
            raise ConflictError("it clashed", details={"field": "slug"})

        @app.get("/boom-converted")
        async def boom_converted() -> None:
            raise ConflictError("it clashed", details={"field": "slug"}).to_http()

        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async def _override():
            async with factory() as s:
                yield s

        app.dependency_overrides[get_session] = _override
        app.state.redis = None
        app.state.rate_limiter = None
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    async def test_raw_and_converted_bodies_are_identical(
        self, raising_client: AsyncClient
    ) -> None:
        raw = await raising_client.get("/boom")
        converted = await raising_client.get("/boom-converted")
        assert raw.status_code == converted.status_code == 409
        assert raw.json() == converted.json()
        assert_envelope(raw, 409, "conflict")
        assert raw.json()["detail"]["details"] == {"field": "slug"}

    @pytest.mark.parametrize(
        "exc_class,status",
        [(AppError, 400), (ConflictError, 409)],
    )
    async def test_details_are_omitted_when_absent(
        self, exc_class: type[AppError], status: int
    ) -> None:
        app = create_app()

        @app.get("/plain")
        async def plain() -> None:
            raise exc_class("no details here")

        app.state.redis = None
        app.state.rate_limiter = None
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/plain")
        assert resp.status_code == status
        assert "details" not in resp.json()["detail"]


class TestRateLimitError:
    """The 429 is written by middleware, which bypasses the handlers entirely."""

    @pytest_asyncio.fixture
    async def throttled_client(self, engine) -> AsyncGenerator[AsyncClient, None]:
        reset_local_windows()
        app = create_app()
        factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async def _override():
            async with factory() as s:
                yield s

        # Bound to the test database so a request that gets past the quota
        # cannot reach for the real engine.
        app.dependency_overrides[get_session] = _override
        app.state.redis = None
        # A limit of 1 means the second request is refused.
        app.state.rate_limiter = RateLimiter(None, limit=1)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
        reset_local_windows()

    async def test_over_quota_uses_the_envelope(
        self, throttled_client: AsyncClient
    ) -> None:
        first = await throttled_client.get("/api/v1/jobs")
        assert first.status_code != 429

        second = await throttled_client.get("/api/v1/jobs")
        assert_envelope(second, 429, "rate_limited")
        assert second.headers["Retry-After"] == "60"

    async def test_health_is_exempt_from_the_quota(
        self, throttled_client: AsyncClient
    ) -> None:
        for _ in range(5):
            assert (await throttled_client.get("/health")).status_code == 200
