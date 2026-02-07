"""Graph domain — Neo4j knowledge graph storage and schema management."""

from engramcp.graph.schema import init_schema
from engramcp.graph.store import GraphStore

__all__ = ["GraphStore", "init_schema"]
