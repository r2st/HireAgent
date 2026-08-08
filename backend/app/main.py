"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import AppError
from app.core.ratelimit import RateLimiter
from app.core.redis import close_redis, get_redis
from app.core.security import TokenError, decode_token
from app.db.session import dispose_engine

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("hireagent")

# Endpoints that must stay reachable even when a caller is over quota.
RATE_LIMIT_EXEMPT = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.redis = await get_redis()
    app.state.rate_limiter = RateLimiter(app.state.redis)
    if app.state.redis is None:
        logger.warning("Redis unavailable — using in-process rate limiting fallback")
    yield
    await close_redis()
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="AI Recruitment Automation Platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id, time the request, and enforce the org quota."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()

        limiter: RateLimiter | None = getattr(app.state, "rate_limiter", None)
        remaining: int | None = None
        if limiter is not None and request.url.path not in RATE_LIMIT_EXEMPT:
            # Rate limit by organization where we can read one from the token,
            # else by client IP so unauthenticated traffic is still bounded.
            bucket = _rate_limit_bucket(request)
            allowed, remaining = await limiter.check(bucket)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": "rate_limited",
                        "message": (
                            f"Rate limit of {limiter.limit} requests/minute exceeded"
                        ),
                    },
                    headers={"Retry-After": "60", "X-Request-ID": request_id},
                )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = (
            f"{(time.perf_counter() - started) * 1000:.1f}"
        )
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        payload = {"code": exc.code, "message": exc.message}
        if exc.details is not None:
            payload["details"] = exc.details
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.get("/health", tags=["system"])
    async def health() -> dict:
        return {"status": "ok", "service": settings.app_name}

    @app.get("/ready", tags=["system"])
    async def ready() -> dict:
        """Readiness probe: reports database and Redis reachability."""
        from sqlalchemy import text

        from app.db.session import SessionFactory

        db_ok = False
        try:
            async with SessionFactory() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception as exc:
            logger.warning("Readiness DB check failed: %s", exc)

        redis_ok = getattr(app.state, "redis", None) is not None
        return {
            "status": "ok" if db_ok else "degraded",
            "database": db_ok,
            "redis": redis_ok,
        }

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


def _rate_limit_bucket(request: Request) -> str:
    """Prefer the organization claim; fall back to client IP."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        try:
            payload = decode_token(auth[7:].strip(), expected_type="access")
            return f"org:{payload['org']}"
        except (TokenError, KeyError):
            pass
    forwarded = request.headers.get("x-forwarded-for")
    ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    return f"ip:{ip}"


app = create_app()
