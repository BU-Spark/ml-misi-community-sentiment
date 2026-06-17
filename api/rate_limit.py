"""Lightweight request rate limiting with Redis or in-process fallback."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple


class RateLimiter:
    """Fixed-window counter. Redis when available; thread-safe memory otherwise."""

    def __init__(self, redis_client=None, *, key_prefix: str = "rethinkai:ratelimit:") -> None:
        self._redis = redis_client
        self._prefix = key_prefix
        self._memory: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def _key(self, bucket: str, identifier: str) -> str:
        return f"{self._prefix}{bucket}:{identifier}"

    def check(self, bucket: str, identifier: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if limit <= 0 or window_seconds <= 0:
            return True, 0

        key = self._key(bucket, identifier)
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.set(key, 0, ex=window_seconds, nx=True)
                pipe.incr(key)
                results = pipe.execute()
                count = int(results[1])
                ttl = int(self._redis.ttl(key))
                if ttl < 0:
                    ttl = window_seconds
                if count > limit:
                    return False, max(1, ttl)
                return True, 0
            except Exception:
                pass

        now = time.time()
        with self._lock:
            count, window_start = self._memory.get(key, (0, now))
            if now - window_start >= window_seconds:
                count = 0
                window_start = now
            count += 1
            self._memory[key] = (count, window_start)
            retry_after = max(1, int(window_seconds - (now - window_start)))
            if count > limit:
                return False, retry_after
        return True, 0
