"""Scenario-eval failure reporting helpers."""

from __future__ import annotations

import json
from typing import Any


def build_failure_context(
    *,
    scenario: str,
    query: str,
    response: dict[str, Any],
    fragments: list[str] | None = None,
    notes: str | None = None,
) -> str:
    """Build a compact, readable context payload for assertion failures."""
    payload: dict[str, Any] = {
        "scenario": scenario,
        "query": query,
        "meta": response.get("meta"),
        "memories": response.get("memories", [])[:5],
        "contradictions": response.get("contradictions", [])[:5],
    }
    if fragments:
        payload["fragments"] = fragments
    if notes:
        payload["notes"] = notes
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
