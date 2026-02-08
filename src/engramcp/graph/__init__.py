"""Graph domain — Neo4j knowledge graph storage and schema management.

Exports are loaded lazily to avoid import cycles between graph and engine
packages during test/bootstrap imports.
"""

from __future__ import annotations

from importlib import import_module

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


_EXPORT_TO_MODULE = {
    "EntityResolver": "engramcp.graph.entity_resolution",
    "ExistingEntity": "engramcp.graph.entity_resolution",
    "MergeExecutor": "engramcp.graph.entity_resolution",
    "MergeResult": "engramcp.graph.entity_resolution",
    "ResolutionAction": "engramcp.graph.entity_resolution",
    "ResolutionCandidate": "engramcp.graph.entity_resolution",
    "normalize_name": "engramcp.graph.entity_resolution",
    "init_schema": "engramcp.graph.schema",
    "GraphStore": "engramcp.graph.store",
    "IndependenceResult": "engramcp.graph.traceability",
    "SourceChain": "engramcp.graph.traceability",
    "SourceTraceability": "engramcp.graph.traceability",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
