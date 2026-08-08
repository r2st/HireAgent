"""Shared Redis client."""

from __future__ import annotations

from redis.asyncio import Redis, from_url

from app.core.config import settings

_client: Redis | None = None


async def get_redis() -> Redis | None:
    """Return a connected client, or ``None`` when Redis is unreachable.

    Callers must treat ``None`` as "degrade gracefully" rather than an error:
    rate limiting and queueing both have local fallbacks.
    """
    global _client
    if _client is not None:
        return _client
    try:
        client: Redis = from_url(
            settings.redis_url, encoding="utf-8", decode_responses=True
        )
        await client.ping()
        _client = client
        return _client
    except Exception:
        return None


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
