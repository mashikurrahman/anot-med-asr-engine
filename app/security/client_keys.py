#!/usr/bin/env python3
"""
Per-client API keys and daily usage limits.

ANOT_ASR_CLIENT_KEYS = "name:key[:daily_limit], name2:key2:50"   (limit omitted or 0 = unlimited)

Each outside client gets its own key, so it can be revoked on its own and its usage counted on its own.
The main ANOT_ASR_API_KEY stays unlimited. Counts are per UTC day.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ClientKeys = Dict[str, Tuple[str, Optional[int]]]  # key -> (client name, daily limit or None)


def parse_client_keys(raw: str) -> ClientKeys:
    """Parse the setting. Malformed entries are skipped (never raise: a typo must not take the engine down)."""
    out: ClientKeys = {}
    for entry in (raw or "").split(","):
        parts = [p.strip() for p in entry.strip().split(":")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        limit: Optional[int] = None
        if len(parts) >= 3 and parts[2]:
            try:
                limit = int(parts[2]) or None
            except ValueError:
                limit = None
        out[parts[1]] = (parts[0], limit)
    return out


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class DailyUsage:
    """Per-client request counter, reset each UTC day. In memory (one container)."""

    def __init__(self, today: Callable[[], str] = _utc_day) -> None:
        self._today = today
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}

    def allow(self, client: str, limit: Optional[int]) -> bool:
        """Count one request; False when the client is already at its daily limit."""
        if limit is None:
            return True
        key = f"{client}:{self._today()}"
        with self._lock:
            used = self._counts.get(key, 0)
            if used >= limit:
                return False
            self._counts[key] = used + 1
            return True


class SharedDailyUsage(DailyUsage):
    """Same counter, shared by every container through a Modal Dict, so the limit holds when Modal runs several copies.
    Approximate (read then write), which is fine for a courtesy limit. Falls back to this container's count if Modal is unreachable."""

    def __init__(self, name: str = "anot-asr-usage", today: Callable[[], str] = _utc_day) -> None:
        super().__init__(today)
        self._name = name
        self._dict = None

    def allow(self, client: str, limit: Optional[int]) -> bool:
        if limit is None:
            return True
        try:
            if self._dict is None:
                import modal
                self._dict = modal.Dict.from_name(self._name, create_if_missing=True)
            key = f"{client}:{self._today()}"
            used = int(self._dict.get(key, 0))
            if used >= limit:
                return False
            self._dict[key] = used + 1
            return True
        except Exception as exc:  # never block a client because the counter store is down
            logger.warning("Shared usage counter unavailable (%s); using this container's count", exc)
            return super().allow(client, limit)
