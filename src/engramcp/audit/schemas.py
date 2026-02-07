"""Audit event types and data models."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class AuditEventType(str, Enum):
    """Categories of auditable events."""

    SEND_MEMORY = "SEND_MEMORY"
    GET_MEMORY = "GET_MEMORY"
    CORRECT_MEMORY = "CORRECT_MEMORY"
    CONFIDENCE_CHANGE = "CONFIDENCE_CHANGE"
    CONSOLIDATION_RUN = "CONSOLIDATION_RUN"
    NODE_CREATED = "NODE_CREATED"
    RELATION_CREATED = "RELATION_CREATED"


class AuditEvent(BaseModel):
    """A single immutable audit log entry."""

    model_config = {"frozen": True}

    timestamp: float = Field(
        default_factory=time.time,
        description="Unix epoch when the event occurred.",
    )
    event_type: AuditEventType = Field(
        description="Category of the audited action.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary event-specific data.",
    )
