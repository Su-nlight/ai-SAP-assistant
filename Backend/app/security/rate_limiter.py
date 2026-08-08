"""
Rate limiting.

In-memory fixed-window counter per client key (user id or IP). Correct for
a single-process dev/demo deployment; production with multiple uvicorn
workers needs the counters centralized. The `RateLimiter` interface is
kept minimal specifically so a `RedisRateLimiter` (same interface, `INCR`
+ `EXPIRE` in Redis) is a drop-in swap — see docs/ARCHITECTURE.md roadmap.
"""
from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request, status

from app.core.config import get_settings


class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - self._window_seconds
        with self._lock:
            recent = [t for t in self._hits[key] if t > window_start]
            if len(recent) >= self._max_requests:
                self._hits[key] = recent
                return False
            recent.append(now)
            self._hits[key] = recent
            return True


_settings = get_settings()
_limiter = InMemoryRateLimiter(
    max_requests=_settings.RATE_LIMIT_REQUESTS, window_seconds=_settings.RATE_LIMIT_WINDOW_SECONDS
)


async def enforce_rate_limit(request: Request) -> None:
    client_key = request.client.host if request.client else "unknown"
    if not _limiter.allow(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down.",
        )
