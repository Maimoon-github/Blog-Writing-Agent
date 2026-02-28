# import hashlib
# import json
# import time
# from pathlib import Path

# import structlog

# from app.config import CACHE_DIR, CACHE_TTL_SECONDS


# class CacheStore:
#     """Disk-based JSON cache with a configurable TTL."""

#     def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.ttl = CACHE_TTL_SECONDS
#         self.logger = structlog.get_logger(__name__)

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _key_to_path(self, key: str) -> Path:
#         """Map an arbitrary string key to a deterministic cache file path."""
#         digest = hashlib.md5(key.encode()).hexdigest()
#         return self.cache_dir / f"{digest}.json"

#     # ------------------------------------------------------------------
#     # Public API
#     # ------------------------------------------------------------------

#     def get(self, key: str) -> dict | None:
#         """Return the cached value for *key*, or ``None`` on a miss / expiry."""
#         path = self._key_to_path(key)

#         if not path.exists():
#             return None

#         try:
#             payload = json.loads(path.read_text(encoding="utf-8"))
#         except (json.JSONDecodeError, OSError):
#             self.logger.warning("cache.read_error", key=key, path=str(path))
#             return None

#         if time.time() - payload["timestamp"] > self.ttl:
#             self.logger.debug("cache.expired", key=key, path=str(path))
#             path.unlink(missing_ok=True)
#             return None

#         self.logger.debug("cache.hit", key=key)
#         return payload["data"]

#     def set(self, key: str, value: dict) -> None:
#         """Persist *value* to disk under *key*."""
#         path = self._key_to_path(key)
#         payload = {"timestamp": time.time(), "data": value}

#         path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
#         self.logger.info("cache.write", key=key, path=str(path))

#     def delete(self, key: str) -> None:
#         """Remove the cache entry for *key* if it exists."""
#         path = self._key_to_path(key)
#         if path.exists():
#             path.unlink()
#             self.logger.debug("cache.delete", key=key, path=str(path))

#     def clear_expired(self) -> int:
#         """Scan the cache directory and delete every expired (or unreadable) entry.

#         Returns:
#             The number of files that were removed.
#         """
#         removed = 0

#         for cache_file in self.cache_dir.glob("*.json"):
#             try:
#                 payload = json.loads(cache_file.read_text(encoding="utf-8"))
#                 expired = time.time() - payload["timestamp"] > self.ttl
#             except (json.JSONDecodeError, KeyError, OSError):
#                 expired = True  # treat unreadable files as expired

#             if expired:
#                 cache_file.unlink(missing_ok=True)
#                 removed += 1
#                 self.logger.debug("cache.cleared_expired", path=str(cache_file))

#         self.logger.info("cache.clear_expired_done", removed=removed)
#         return removed


# # ---------------------------------------------------------------------------
# # Module-level singleton – import and use this throughout the application.
# # ---------------------------------------------------------------------------
# cache = CacheStore()

































"""Disk-based JSON cache with TTL, atomic writes, and thread safety.

This module provides a persistent cache for serializable Python dictionaries.
Each cache entry is stored as a separate JSON file with an embedded timestamp.
The cache supports time-to-live (TTL) expiration, atomic writes, and thread-safe
operations through an instance-level lock.

Typical usage:
    from memory.cache import cache
    
    # Store data
    cache.set("user:123", {"name": "Alice", "role": "admin"})
    
    # Retrieve data (returns None if expired or missing)
    user = cache.get("user:123")
    
    # Clean up expired entries
    removed = cache.clear_expired()
"""

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from app.config import CACHE_DIR, CACHE_TTL_SECONDS

# Get a module-level logger
logger = structlog.get_logger(__name__)


class CacheStore:
    """A thread-safe disk cache with time-to-live expiration.

    This class implements a persistent cache where each key is mapped to a
    JSON file via MD5 hashing. All operations are protected by an instance
    lock to prevent race conditions in multi-threaded environments.

    Attributes:
        cache_dir: Directory where cache files are stored.
        ttl: Time-to-live in seconds for cache entries.
        logger: Structured logger bound with cache directory context.
        _lock: Reentrant lock for thread-safe operations.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        ttl: int = CACHE_TTL_SECONDS
    ) -> None:
        """Initialize the cache store.

        Args:
            cache_dir: Directory path for cache files. Created if it doesn't exist.
            ttl: Time-to-live in seconds. Defaults to app.config.CACHE_TTL_SECONDS.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.logger = logger.bind(cache_dir=str(self.cache_dir))
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _key_to_path(self, key: str) -> Path:
        """Convert a cache key to a deterministic filesystem path.

        Uses MD5 hash to ensure:
        - Consistent filename for the same key
        - Safe filesystem characters regardless of key content
        - Even distribution across the filesystem

        Args:
            key: Arbitrary string key.

        Returns:
            Path object pointing to the cache file.
        """
        # MD5 is sufficient for cache key hashing (not security-critical)
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_payload(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read and validate a cache file payload.

        Args:
            path: Path to the cache file.

        Returns:
            The parsed payload dictionary if valid, None otherwise.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            self.logger.warning(
                "cache.corrupted_file",
                path=str(path),
                error=str(e),
            )
            # Delete corrupted file to avoid future issues
            try:
                path.unlink(missing_ok=True)
            except OSError as unlink_err:
                self.logger.error(
                    "cache.delete_failed",
                    path=str(path),
                    error=str(unlink_err),
                )
            return None
        except OSError as e:
            self.logger.error(
                "cache.read_failed",
                path=str(path),
                error=str(e),
            )
            return None

        # Validate required fields
        if not isinstance(payload, dict) or "timestamp" not in payload or "data" not in payload:
            self.logger.warning(
                "cache.invalid_format",
                path=str(path),
                fields=list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
            )
            # Delete malformed file
            try:
                path.unlink(missing_ok=True)
            except OSError as unlink_err:
                self.logger.error(
                    "cache.delete_failed",
                    path=str(path),
                    error=str(unlink_err),
                )
            return None

        return payload

    def _is_expired(self, payload: Dict[str, Any]) -> bool:
        """Check if a cache payload is expired.

        Args:
            payload: Cache payload containing 'timestamp' field.

        Returns:
            True if the entry is expired, False otherwise.
        """
        age = time.time() - payload["timestamp"]
        return age > self.ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a value from cache.

        Args:
            key: Cache key to look up.

        Returns:
            The cached value if present and not expired, otherwise None.
        """
        with self._lock:
            path = self._key_to_path(key)

            if not path.exists():
                self.logger.debug("cache.miss", key=key)
                return None

            payload = self._read_payload(path)
            if payload is None:
                return None

            if self._is_expired(payload):
                self.logger.debug(
                    "cache.expired",
                    key=key,
                    age=time.time() - payload["timestamp"],
                    ttl=self.ttl,
                )
                # Delete expired file
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    self.logger.error(
                        "cache.delete_failed",
                        key=key,
                        path=str(path),
                        error=str(e),
                    )
                return None

            self.logger.debug(
                "cache.hit",
                key=key,
                age=time.time() - payload["timestamp"],
            )
            return payload["data"]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store a value in cache with current timestamp.

        Uses atomic write pattern:
        1. Write to a temporary file in the same directory
        2. Rename the temporary file to the target path
        This ensures the cache file is never partially written.

        Args:
            key: Cache key.
            value: Dictionary data to cache. Must be JSON-serializable.
        """
        with self._lock:
            path = self._key_to_path(key)
            payload = {
                "timestamp": time.time(),
                "data": value,
            }

            # Atomic write using temporary file
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    suffix=".tmp",
                    delete=False,
                ) as tf:
                    json.dump(payload, tf, ensure_ascii=False, indent=2)
                    temp_path = tf.name

                # Atomic rename (os.replace is atomic on most platforms)
                os.replace(temp_path, path)

                self.logger.info(
                    "cache.write",
                    key=key,
                    path=str(path),
                    size_bytes=path.stat().st_size if path.exists() else None,
                )

            except (OSError, json.JSONEncodeError) as e:
                self.logger.error(
                    "cache.write_failed",
                    key=key,
                    error=str(e),
                )
                # Clean up temporary file if it exists
                if "temp_path" in locals():
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass

    def delete(self, key: str) -> None:
        """Remove a specific key from cache.

        Args:
            key: Cache key to delete.
        """
        with self._lock:
            path = self._key_to_path(key)

            if path.exists():
                try:
                    path.unlink()
                    self.logger.debug("cache.delete", key=key, path=str(path))
                except OSError as e:
                    self.logger.error(
                        "cache.delete_failed",
                        key=key,
                        path=str(path),
                        error=str(e),
                    )
            else:
                self.logger.debug("cache.delete_miss", key=key)

    def clear_expired(self) -> int:
        """Remove all expired cache entries.

        Scans all .json files in the cache directory, validates each,
        and deletes those that are expired or corrupted.

        Returns:
            Number of files successfully removed.
        """
        removed = 0
        # No lock during scan to avoid blocking other operations,
        # but we lock per file when deleting
        for cache_file in self.cache_dir.glob("*.json"):
            # Extract key from filename for logging (optional)
            # Since we don't have the original key, we use the filename
            file_stem = cache_file.stem

            try:
                payload = self._read_payload(cache_file)
                if payload is None:
                    # Corrupted or unreadable - treat as expired and delete
                    with self._lock:
                        try:
                            cache_file.unlink(missing_ok=True)
                            removed += 1
                            self.logger.debug(
                                "cache.cleared_corrupted",
                                path=str(cache_file),
                            )
                        except OSError as e:
                            self.logger.error(
                                "cache.clear_failed",
                                path=str(cache_file),
                                error=str(e),
                            )
                    continue

                if self._is_expired(payload):
                    with self._lock:
                        try:
                            cache_file.unlink(missing_ok=True)
                            removed += 1
                            self.logger.debug(
                                "cache.cleared_expired",
                                path=str(cache_file),
                                age=time.time() - payload["timestamp"],
                                ttl=self.ttl,
                            )
                        except OSError as e:
                            self.logger.error(
                                "cache.clear_failed",
                                path=str(cache_file),
                                error=str(e),
                            )
            except Exception as e:
                # Unexpected error - log and skip
                self.logger.error(
                    "cache.clear_error",
                    path=str(cache_file),
                    error=str(e),
                )

        self.logger.info("cache.clear_expired_done", removed=removed)
        return removed

    # ------------------------------------------------------------------
    # Async wrappers (for asyncio-based applications)
    # ------------------------------------------------------------------
    # These methods allow using the synchronous cache in asyncio code
    # without blocking the event loop, by offloading to a thread pool.

    async def get_async(self, key: str) -> Optional[Dict[str, Any]]:
        """Asynchronous version of get().

        Runs the synchronous get() in a thread pool to avoid blocking
        the asyncio event loop.

        Args:
            key: Cache key to look up.

        Returns:
            The cached value if present and not expired, otherwise None.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get, key)

    async def set_async(self, key: str, value: Dict[str, Any]) -> None:
        """Asynchronous version of set().

        Runs the synchronous set() in a thread pool to avoid blocking
        the asyncio event loop.

        Args:
            key: Cache key.
            value: Dictionary data to cache.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.set, key, value)

    async def delete_async(self, key: str) -> None:
        """Asynchronous version of delete().

        Runs the synchronous delete() in a thread pool to avoid blocking
        the asyncio event loop.

        Args:
            key: Cache key to delete.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.delete, key)

    async def clear_expired_async(self) -> int:
        """Asynchronous version of clear_expired().

        Runs the synchronous clear_expired() in a thread pool to avoid
        blocking the asyncio event loop.

        Returns:
            Number of files successfully removed.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.clear_expired)


# ----------------------------------------------------------------------
# Module-level singleton – import and use this throughout the application.
# ----------------------------------------------------------------------
cache = CacheStore()