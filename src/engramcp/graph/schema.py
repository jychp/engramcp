"""Neo4j schema initialization — indexes and constraints.

All statements use ``IF NOT EXISTS`` so they are safe to run repeatedly
(idempotent). Only the uniqueness constraint is enforced at the DB level;
NOT NULL constraints are handled by Pydantic validation (Neo4j Community
Edition does not support existence constraints).
"""

from __future__ import annotations

from neo4j import AsyncDriver

# ---------------------------------------------------------------------------
# Constraint statements (Community Edition: uniqueness only)
# ---------------------------------------------------------------------------

_CONSTRAINTS = [
    "CREATE CONSTRAINT mem_unique_id IF NOT EXISTS FOR (n:Memory) REQUIRE n.id IS UNIQUE",
]

# ---------------------------------------------------------------------------
# Index statements
# ---------------------------------------------------------------------------

_NODE_INDEXES = [
    # Primary lookup
    "CREATE INDEX mem_id IF NOT EXISTS FOR (n:Memory) ON (n.id)",
    # Temporal queries
    "CREATE INDEX event_occurred IF NOT EXISTS FOR (n:Event) ON (n.occurred_at)",
    "CREATE INDEX decision_occurred IF NOT EXISTS FOR (n:Decision) ON (n.occurred_at)",
    "CREATE INDEX outcome_occurred IF NOT EXISTS FOR (n:Outcome) ON (n.occurred_at)",
    # Status filtering
    "CREATE INDEX fact_status IF NOT EXISTS FOR (n:Fact) ON (n.status)",
    "CREATE INDEX pattern_status IF NOT EXISTS FOR (n:Pattern) ON (n.status)",
    # Agent lookup
    "CREATE INDEX agent_name IF NOT EXISTS FOR (n:Agent) ON (n.name)",
    # Source lookup
    "CREATE INDEX source_type IF NOT EXISTS FOR (n:Source) ON (n.type)",
    "CREATE INDEX source_reliability IF NOT EXISTS FOR (n:Source) ON (n.reliability)",
    # Ingestion tracking
    "CREATE INDEX mem_ingested IF NOT EXISTS FOR (n:Memory) ON (n.ingested_at)",
    # Derivation tracking
    "CREATE INDEX derived_run IF NOT EXISTS FOR (n:Derived) ON (n.derivation_run_id)",
    "CREATE INDEX derived_depth IF NOT EXISTS FOR (n:Derived) ON (n.derivation_depth)",
]

_REL_INDEXES = [
    # Credibility filtering on traceability
    "CREATE INDEX sourced_credibility IF NOT EXISTS FOR ()-[r:SOURCED_FROM]-() ON (r.credibility)",
    # Contradiction management
    "CREATE INDEX contradiction_status IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.resolution_status)",
    "CREATE INDEX contradiction_detected IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.detected_at)",
    # Temporal relation ordering
    "CREATE INDEX preceded_gap IF NOT EXISTS FOR ()-[r:PRECEDED]-() ON (r.gap_seconds)",
    # Participation role filtering
    "CREATE INDEX participation_role IF NOT EXISTS FOR ()-[r:PARTICIPATED_IN]-() ON (r.role)",
    # Derivation tracking
    "CREATE INDEX derivation_run IF NOT EXISTS FOR ()-[r:DERIVED_FROM]-() ON (r.derivation_run_id)",
]

# Full-text index (separate because it uses a different syntax)
_FULLTEXT_INDEXES = [
    (
        "CREATE FULLTEXT INDEX mem_content IF NOT EXISTS "
        "FOR (n:Fact|Event|Observation|Decision|Outcome|Pattern|Concept|Rule) "
        "ON EACH [n.content]"
    ),
]


async def init_schema(driver: AsyncDriver) -> None:
    """Create all indexes and constraints (idempotent).

    Runs each statement in its own transaction to avoid batching issues
    with schema commands in Neo4j.
    """
    all_statements = _CONSTRAINTS + _NODE_INDEXES + _REL_INDEXES + _FULLTEXT_INDEXES
    async with driver.session() as session:
        for stmt in all_statements:
            await session.run(stmt)
