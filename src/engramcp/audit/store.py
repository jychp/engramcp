"""Async JSONL audit logger."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

from pydantic import ValidationError

from engramcp.audit.schemas import AuditEvent
from engramcp.audit.schemas import AuditEventType
from engramcp.config import AuditConfig

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit log with async I/O.

    Uses ``asyncio.to_thread`` for file operations to avoid blocking
    the event loop, guarded by an ``asyncio.Lock`` for serialization.
    """

    def __init__(self, config: AuditConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def log(self, event: AuditEvent) -> None:
        """Append *event* as a single JSON line to the audit file."""
        if not self.config.enabled:
            return
        line = event.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(
                partial(self._append, self.config.file_path, line),
            )

    @staticmethod
    def _append(path: str, line: str) -> None:
        with open(path, "a") as fh:
            fh.write(line)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def read_events(
        self,
        *,
        event_type: AuditEventType | None = None,
        since: float | None = None,
    ) -> list[AuditEvent]:
        """Read events back from the audit file, optionally filtered."""
        path = Path(self.config.file_path)
        if not path.exists():
            return []

        async with self._lock:
            raw = await asyncio.to_thread(path.read_text)
        events: list[AuditEvent] = []
        for line_no, line in enumerate(raw.strip().splitlines(), start=1):
            try:
                evt = AuditEvent.model_validate_json(line)
            except ValidationError:
                logger.warning(
                    "Skipping malformed audit event line %d in %s",
                    line_no,
                    path,
                )
                continue
            if event_type is not None and evt.event_type != event_type:
                continue
            if since is not None and evt.timestamp < since:
                continue
            events.append(evt)
        return events
