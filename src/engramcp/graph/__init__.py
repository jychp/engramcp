"""Graph domain — Neo4j knowledge graph storage and schema management."""

from engramcp.graph.entity_resolution import EntityResolver
from engramcp.graph.entity_resolution import ExistingEntity
from engramcp.graph.entity_resolution import MergeExecutor
from engramcp.graph.entity_resolution import MergeResult
from engramcp.graph.entity_resolution import normalize_name
from engramcp.graph.entity_resolution import ResolutionAction
from engramcp.graph.entity_resolution import ResolutionCandidate
from engramcp.graph.schema import init_schema
from engramcp.graph.store import GraphStore
from engramcp.graph.traceability import IndependenceResult
from engramcp.graph.traceability import SourceChain
from engramcp.graph.traceability import SourceTraceability

__all__ = [
    "EntityResolver",
    "ExistingEntity",
    "GraphStore",
    "IndependenceResult",
    "MergeExecutor",
    "MergeResult",
    "ResolutionAction",
    "ResolutionCandidate",
    "SourceChain",
    "SourceTraceability",
    "init_schema",
    "normalize_name",
]
