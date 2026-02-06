"""Models domain — shared data models and utilities."""

from __future__ import annotations

import hashlib


def agent_fingerprint(agent_id: str | None) -> str | None:
    """Return a deterministic fingerprint for the given agent ID.

    Uses SHA-256 truncated to 16 hex characters. Returns ``None`` if
    *agent_id* is ``None``.
    """
    if agent_id is None:
        return None
    return hashlib.sha256(agent_id.encode()).hexdigest()[:16]
