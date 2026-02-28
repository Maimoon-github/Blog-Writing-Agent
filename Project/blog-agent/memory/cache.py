import hashlib
import json
import time
from pathlib import Path

import structlog

from app.config import CACHE_DIR, CACHE_TTL_SECONDS


class CacheStore:
    """Disk-based JSON cache with a configurable TTL."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = CACHE_TTL_SECONDS
        self.logger = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        """Map an arbitrary string key to a deterministic cache file path."""
        digest = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict | None:
        """Return the cached value for *key*, or ``None`` on a miss / expiry."""
        path = self._key_to_path(key)

        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.logger.warning("cache.read_error", key=key, path=str(path))
            return None

        if time.time() - payload["timestamp"] > self.ttl:
            self.logger.debug("cache.expired", key=key, path=str(path))
            path.unlink(missing_ok=True)
            return None

        self.logger.debug("cache.hit", key=key)
        return payload["data"]

    def set(self, key: str, value: dict) -> None:
        """Persist *value* to disk under *key*."""
        path = self._key_to_path(key)
        payload = {"timestamp": time.time(), "data": value}

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("cache.write", key=key, path=str(path))

    def delete(self, key: str) -> None:
        """Remove the cache entry for *key* if it exists."""
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            self.logger.debug("cache.delete", key=key, path=str(path))

    def clear_expired(self) -> int:
        """Scan the cache directory and delete every expired (or unreadable) entry.

        Returns:
            The number of files that were removed.
        """
        removed = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                expired = time.time() - payload["timestamp"] > self.ttl
            except (json.JSONDecodeError, KeyError, OSError):
                expired = True  # treat unreadable files as expired

            if expired:
                cache_file.unlink(missing_ok=True)
                removed += 1
                self.logger.debug("cache.cleared_expired", path=str(cache_file))

        self.logger.info("cache.clear_expired_done", removed=removed)
        return removed


# ---------------------------------------------------------------------------
# Module-level singleton – import and use this throughout the application.
# ---------------------------------------------------------------------------
cache = CacheStore()