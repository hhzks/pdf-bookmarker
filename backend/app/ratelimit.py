"""Per-key sliding-window rate limiter."""
import time
from collections import defaultdict, deque
from threading import Lock


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 3600):
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit for key and return True if it is within the limit."""
        now = time.time() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True

    def cleanup_expired(self, now: float | None = None) -> None:
        """Drop keys whose hits have all aged out of the window."""
        now = time.time() if now is None else now
        with self._lock:
            stale = [
                key for key, hits in self._hits.items()
                if not hits or now - hits[-1] > self._window
            ]
            for key in stale:
                del self._hits[key]
