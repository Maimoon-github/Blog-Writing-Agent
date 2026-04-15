"""Thread-safe disk cache for BWAgent."""

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from config.settings import CACHE_DIR

logger = structlog.get_logger(__name__)


class CacheStore:
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = 86400) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._lock = threading.RLock()
        self.logger = logger.bind(cache_dir=str(self.cache_dir))

    def _key_to_path(self, key: str) -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_payload(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("cache.read_failed", path=str(path), error=str(exc))
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        if not isinstance(payload, dict) or "timestamp" not in payload or "data" not in payload:
            self.logger.warning("cache.invalid_payload", path=str(path))
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return payload

    def _is_expired(self, payload: Dict[str, Any]) -> bool:
        return time.time() - payload["timestamp"] > self.ttl

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            path = self._key_to_path(key)
            if not path.exists():
                self.logger.debug("cache.miss", key=key)
                return None
            payload = self._read_payload(path)
            if not payload:
                return None
            if self._is_expired(payload):
                self.logger.debug("cache.expired", key=key)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                return None
            self.logger.debug("cache.hit", key=key)
            return payload["data"]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            path = self._key_to_path(key)
            payload = {"timestamp": time.time(), "data": value}
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir, delete=False) as tf:
                    json.dump(payload, tf, ensure_ascii=False, indent=2)
                    temp_path = tf.name
                os.replace(temp_path, path)
                self.logger.debug("cache.write", key=key, path=str(path))
            except Exception as exc:
                self.logger.error("cache.write_failed", key=key, error=str(exc))
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def delete(self, key: str) -> None:
        with self._lock:
            path = self._key_to_path(key)
            try:
                path.unlink(missing_ok=True)
                self.logger.debug("cache.delete", key=key)
            except Exception as exc:
                self.logger.error("cache.delete_failed", key=key, error=str(exc))

    async def get_async(self, key: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get, key)

    async def set_async(self, key: str, value: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.set, key, value)


cache = CacheStore()
