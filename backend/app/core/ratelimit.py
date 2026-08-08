"""Per-organization rate limiting (design §6.1: 200 requests/minute).

Uses a Redis fixed-window counter when Redis is reachable and falls back to an
in-process window otherwise, so local development and tests work without Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict

from redis.asyncio import Redis

from app.core.config import settings

_local_windows: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))


class RateLimiter:
    def __init__(self, redis: Redis | None = None, limit: int | None = None) -> None:
        self._redis = redis
        self.limit = limit or settings.rate_limit_per_minute

    async def check(self, key: str) -> tuple[bool, int]:
        """Consume one unit for ``key``.

        Returns ``(allowed, remaining)``.
        """
        window = int(time.time() // 60)
        bucket = f"ratelimit:{key}:{window}"

        if self._redis is not None:
            try:
                count = await self._redis.incr(bucket)
                if count == 1:
                    # Expire slightly after the window so a clock skew cannot
                    # leave the key resident forever.
                    await self._redis.expire(bucket, 90)
                return count <= self.limit, max(0, self.limit - int(count))
            except Exception:
                # Redis being down must not take the API down with it.
                pass

        stored_window, count = _local_windows[key]
        count = count + 1 if stored_window == window else 1
        _local_windows[key] = (window, count)
        return count <= self.limit, max(0, self.limit - count)


def reset_local_windows() -> None:
    """Clear the in-process fallback state (used by tests)."""
    _local_windows.clear()
