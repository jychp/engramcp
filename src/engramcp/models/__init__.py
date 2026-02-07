"""Models domain — shared data models and utilities."""

from __future__ import annotations

import hashlib

from engramcp.models.confidence import Credibility
from engramcp.models.confidence import credibility_from_int
from engramcp.models.confidence import degrade_credibility
from engramcp.models.confidence import NATORating
from engramcp.models.confidence import Reliability
from engramcp.models.confidence import worst_reliability
from engramcp.models.nodes import Agent
from engramcp.models.nodes import AgentType
from engramcp.models.nodes import Artifact
from engramcp.models.nodes import ArtifactType
from engramcp.models.nodes import Concept
from engramcp.models.nodes import Decision
from engramcp.models.nodes import DerivedStatus
from engramcp.models.nodes import Event
from engramcp.models.nodes import Fact
from engramcp.models.nodes import FactStatus
from engramcp.models.nodes import LABEL_TO_MODEL
from engramcp.models.nodes import MemoryBase
from engramcp.models.nodes import MemoryNode
from engramcp.models.nodes import Observation
from engramcp.models.nodes import Outcome
from engramcp.models.nodes import Pattern
from engramcp.models.nodes import Rule
from engramcp.models.nodes import Source
from engramcp.models.nodes import TemporalPrecision
from engramcp.models.relations import CausedBy
from engramcp.models.relations import Cites
from engramcp.models.relations import Concerns
from engramcp.models.relations import Contradicts
from engramcp.models.relations import DecidedBy
from engramcp.models.relations import DerivedFrom
from engramcp.models.relations import ExtractionMethod
from engramcp.models.relations import Followed
from engramcp.models.relations import Generalizes
from engramcp.models.relations import InstanceOf
from engramcp.models.relations import LeadsTo
from engramcp.models.relations import Mentions
from engramcp.models.relations import ObservedBy
from engramcp.models.relations import ParticipatedIn
from engramcp.models.relations import PossiblySameAs
from engramcp.models.relations import Preceded
from engramcp.models.relations import Relationship
from engramcp.models.relations import RelationshipBase
from engramcp.models.relations import ResolutionStatus
from engramcp.models.relations import SourcedFrom
from engramcp.models.relations import Supports

__all__ = [
    # Confidence
    "Credibility",
    "NATORating",
    "Reliability",
    "credibility_from_int",
    "degrade_credibility",
    "worst_reliability",
    # Nodes
    "Agent",
    "AgentType",
    "Artifact",
    "ArtifactType",
    "Concept",
    "Decision",
    "DerivedStatus",
    "Event",
    "Fact",
    "FactStatus",
    "LABEL_TO_MODEL",
    "MemoryBase",
    "MemoryNode",
    "Observation",
    "Outcome",
    "Pattern",
    "Rule",
    "Source",
    "TemporalPrecision",
    # Relations
    "CausedBy",
    "Cites",
    "Concerns",
    "Contradicts",
    "DecidedBy",
    "DerivedFrom",
    "ExtractionMethod",
    "Followed",
    "Generalizes",
    "InstanceOf",
    "LeadsTo",
    "Mentions",
    "ObservedBy",
    "ParticipatedIn",
    "PossiblySameAs",
    "Preceded",
    "Relationship",
    "RelationshipBase",
    "ResolutionStatus",
    "SourcedFrom",
    "Supports",
    # Utilities
    "agent_fingerprint",
]


def agent_fingerprint(agent_id: str | None) -> str | None:
    """Return a deterministic fingerprint for the given agent ID.

    Uses SHA-256 truncated to 16 hex characters. Returns ``None`` if
    *agent_id* is ``None``.
    """
    if agent_id is None:
        return None
    return hashlib.sha256(agent_id.encode()).hexdigest()[:16]
