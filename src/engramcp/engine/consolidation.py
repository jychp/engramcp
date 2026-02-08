"""Consolidation pipeline orchestrator — Layer 4.

Wires together extraction, entity resolution, graph integration, and
audit logging into a single ``ConsolidationPipeline.run(fragments)``
method.  The pipeline coordinates existing components — it does not
contain complex logic itself.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from engramcp.audit.schemas import AuditEvent
from engramcp.audit.schemas import AuditEventType
from engramcp.audit.store import AuditLogger
from engramcp.config import ConsolidationConfig
from engramcp.engine.extraction import ExtractionEngine
from engramcp.engine.schemas import ExtractedClaim
from engramcp.engine.schemas import ExtractedEntity
from engramcp.engine.schemas import ExtractionResult
from engramcp.graph.entity_resolution import EntityResolver
from engramcp.graph.entity_resolution import ExistingEntity
from engramcp.graph.entity_resolution import MergeExecutor
from engramcp.graph.entity_resolution import ResolutionAction
from engramcp.graph.store import GraphStore
from engramcp.memory.schemas import MemoryFragment
from engramcp.models.confidence import Credibility
from engramcp.models.confidence import Reliability
from engramcp.models.nodes import Agent
from engramcp.models.nodes import AgentType
from engramcp.models.nodes import Artifact
from engramcp.models.nodes import ArtifactType
from engramcp.models.nodes import Concept
from engramcp.models.nodes import Decision
from engramcp.models.nodes import Event
from engramcp.models.nodes import Fact
from engramcp.models.nodes import MemoryBase
from engramcp.models.nodes import MemoryNode
from engramcp.models.nodes import Observation
from engramcp.models.nodes import Outcome
from engramcp.models.nodes import Pattern
from engramcp.models.nodes import Rule
from engramcp.models.nodes import Source
from engramcp.models.relations import CausedBy
from engramcp.models.relations import Concerns
from engramcp.models.relations import Contradicts
from engramcp.models.relations import DecidedBy
from engramcp.models.relations import DerivedFrom
from engramcp.models.relations import Followed
from engramcp.models.relations import Generalizes
from engramcp.models.relations import InstanceOf
from engramcp.models.relations import LeadsTo
from engramcp.models.relations import Mentions
from engramcp.models.relations import ObservedBy
from engramcp.models.relations import ParticipatedIn
from engramcp.models.relations import PossiblySameAs
from engramcp.models.relations import Preceded
from engramcp.models.relations import RelationshipBase
from engramcp.models.relations import ResolutionStatus
from engramcp.models.relations import SourcedFrom
from engramcp.models.relations import Supports

# ---------------------------------------------------------------------------
# Helper maps
# ---------------------------------------------------------------------------

_ENTITY_TYPE_TO_MODEL: dict[str, type] = {
    "Agent": Agent,
    "Artifact": Artifact,
}

_CLAIM_TYPE_TO_MODEL: dict[str, type] = {
    "Fact": Fact,
    "Event": Event,
    "Observation": Observation,
    "Decision": Decision,
    "Outcome": Outcome,
}

_REL_TYPE_TO_MODEL: dict[str, type[RelationshipBase]] = {
    "CAUSED_BY": CausedBy,
    "LEADS_TO": LeadsTo,
    "PRECEDED": Preceded,
    "FOLLOWED": Followed,
    "SUPPORTS": Supports,
    "PARTICIPATED_IN": ParticipatedIn,
    "DECIDED_BY": DecidedBy,
    "OBSERVED_BY": ObservedBy,
    "MENTIONS": Mentions,
    "CONCERNS": Concerns,
}

# Default entity constructor kwargs
_ENTITY_DEFAULTS: dict[str, dict] = {
    "Agent": {"type": AgentType.person},
    "Artifact": {"type": ArtifactType.document},
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationRunResult:
    """Summary of a single pipeline execution."""

    run_id: str
    fragments_processed: int = 0
    entities_created: int = 0
    entities_merged: int = 0
    entities_linked: int = 0
    claims_created: int = 0
    relations_created: int = 0
    contradictions_detected: int = 0
    patterns_created: int = 0
    concepts_created: int = 0
    rules_created: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node_to_existing_entity(node: MemoryNode) -> ExistingEntity:
    """Convert a graph node to an ``ExistingEntity`` for entity resolution."""
    # Derive type from the concrete class name
    entity_type = type(node).__name__
    return ExistingEntity(
        node_id=node.id,
        name=getattr(node, "name", ""),
        type=entity_type,
        aliases=getattr(node, "aliases", []),
        properties={},
        fragment_ids=getattr(node, "source_fragment_ids", []),
    )


def _parse_datetime(iso_str: str | None) -> datetime:
    """Parse an ISO-8601 string to datetime, fallback to UTC now."""
    if not iso_str:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


_NEGATION_TOKENS = {"not", "never", "no", "none", "cannot", "can't", "without"}


def _tokenize_for_claim(text: str) -> list[str]:
    cleaned = "".join(
        ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text
    )
    return [tok for tok in cleaned.split() if tok]


def _claim_base_and_polarity(text: str) -> tuple[str, bool]:
    tokens = _tokenize_for_claim(text)
    has_negation = any(tok in _NEGATION_TOKENS for tok in tokens)
    base_tokens = [tok for tok in tokens if tok not in _NEGATION_TOKENS]
    return (" ".join(base_tokens), has_negation)


def _claims_contradict(content_a: str, content_b: str) -> bool:
    base_a, neg_a = _claim_base_and_polarity(content_a)
    base_b, neg_b = _claim_base_and_polarity(content_b)
    return bool(base_a) and base_a == base_b and neg_a != neg_b


def _stable_claim_id(claim: ExtractedClaim) -> str:
    """Build a deterministic claim-node ID for idempotent consolidation."""
    signature = {
        "claim_type": claim.claim_type,
        "content": claim.content,
        "properties": claim.properties,
        "temporal_info": (
            claim.temporal_info.model_dump(mode="json")
            if claim.temporal_info is not None
            else None
        ),
        "involved_entities": sorted(claim.involved_entities),
        "source_fragment_ids": sorted(claim.source_fragment_ids),
    }
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"claim_{digest}"


def _stable_source_id(fragment_id: str) -> str:
    """Build a deterministic source-node ID for one fragment."""
    return f"source_{fragment_id}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ConsolidationPipeline:
    """Orchestrates the full consolidation flow: extract, resolve, integrate, audit."""

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        entity_resolver: EntityResolver,
        merge_executor: MergeExecutor,
        graph_store: GraphStore,
        audit_logger: AuditLogger,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self._extraction = extraction_engine
        self._resolver = entity_resolver
        self._merger = merge_executor
        self._graph = graph_store
        self._audit = audit_logger
        self._config = config or ConsolidationConfig()

    async def run(self, fragments: list[MemoryFragment]) -> ConsolidationRunResult:
        """Full pipeline: extract, resolve entities, integrate, audit."""
        run_id = uuid.uuid4().hex
        result = ConsolidationRunResult(run_id=run_id)

        if not fragments:
            return result

        result.fragments_processed = len(fragments)

        # --- 1. Extract ---
        extraction = await self._extraction.extract(fragments)
        result.errors.extend(extraction.errors)
        existing_claims = await self._graph.find_claim_nodes()

        # --- 2. Resolve entities ---
        name_to_node_id: dict[str, str] = {}
        await self._resolve_entities(extraction, name_to_node_id, result)

        # --- 3. Create claims ---
        claim_node_ids: list[tuple[str, ExtractedClaim]] = []
        await self._create_claims(extraction, name_to_node_id, claim_node_ids, result)

        # --- 4. Create sources + SOURCED_FROM ---
        await self._create_sources(fragments, extraction, claim_node_ids, result)

        # --- 5. Create extracted relations ---
        await self._create_relations(extraction, name_to_node_id, result)

        # --- 6. Detect contradictions with existing claims ---
        await self._detect_contradictions(existing_claims, claim_node_ids, result)

        # --- 7. Abstraction ---
        await self._run_abstraction(claim_node_ids, extraction, result)

        # --- 8. Audit ---
        await self._audit.log(
            AuditEvent(
                event_type=AuditEventType.CONSOLIDATION_RUN,
                payload={
                    "run_id": run_id,
                    "fragments_processed": result.fragments_processed,
                    "entities_created": result.entities_created,
                    "entities_merged": result.entities_merged,
                    "entities_linked": result.entities_linked,
                    "claims_created": result.claims_created,
                    "relations_created": result.relations_created,
                    "contradictions_detected": result.contradictions_detected,
                    "patterns_created": result.patterns_created,
                    "concepts_created": result.concepts_created,
                    "rules_created": result.rules_created,
                    "error_count": len(result.errors),
                },
            )
        )

        return result

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    async def _resolve_entities(
        self,
        extraction: ExtractionResult,
        name_to_node_id: dict[str, str],
        result: ConsolidationRunResult,
    ) -> None:
        for entity in extraction.entities:
            if entity.type not in _ENTITY_TYPE_TO_MODEL:
                result.errors.append(
                    f"Unknown entity type '{entity.type}', skipping entity '{entity.name}'"
                )
                continue

            # Query existing entities of same type
            existing_nodes = await self._graph.find_by_label(entity.type)
            existing = [_node_to_existing_entity(n) for n in existing_nodes]

            candidate = await self._resolver.resolve(entity, existing)

            if candidate.action == ResolutionAction.create_new:
                node_id = await self._create_entity_node(entity, result)
                name_to_node_id[entity.name] = node_id

            elif candidate.action == ResolutionAction.merge:
                if candidate.existing_node_id is None:
                    result.errors.append(
                        f"Resolver returned merge without target for '{entity.name}'"
                    )
                    continue
                # Create new node, then merge into existing
                new_node_id = await self._create_entity_node(entity, result)
                merge_result = await self._merger.execute_merge(
                    survivor_id=candidate.existing_node_id,
                    absorbed_id=new_node_id,
                    merge_run_id=result.run_id,
                )
                name_to_node_id[entity.name] = merge_result.survivor_id
                # Undo the created count — node was absorbed
                result.entities_created -= 1
                result.entities_merged += 1

            elif candidate.action in (ResolutionAction.link, ResolutionAction.review):
                if candidate.existing_node_id is None:
                    result.errors.append(
                        f"Resolver returned link/review without target for '{entity.name}'"
                    )
                    continue
                new_node_id = await self._create_entity_node(entity, result)
                name_to_node_id[entity.name] = new_node_id
                # Create POSSIBLY_SAME_AS relationship
                rel = PossiblySameAs(
                    similarity_score=candidate.score,
                    detection_method=candidate.method,
                )
                await self._graph.create_relationship(
                    new_node_id, candidate.existing_node_id, rel
                )
                result.entities_linked += 1

    async def _create_entity_node(
        self,
        entity: ExtractedEntity,
        result: ConsolidationRunResult,
    ) -> str:
        model_cls = _ENTITY_TYPE_TO_MODEL[entity.type]
        defaults = _ENTITY_DEFAULTS.get(entity.type, {})
        kwargs = {
            "name": entity.name,
            **defaults,
            **entity.properties,
        }
        if entity.aliases:
            kwargs["aliases"] = entity.aliases
        node = model_cls(**kwargs)
        node_id = await self._graph.create_node(node)
        result.entities_created += 1

        await self._audit.log(
            AuditEvent(
                event_type=AuditEventType.NODE_CREATED,
                payload={
                    "run_id": result.run_id,
                    "node_id": node_id,
                    "node_type": entity.type,
                    "name": entity.name,
                },
            )
        )
        return node_id

    # ------------------------------------------------------------------
    # Claim creation
    # ------------------------------------------------------------------

    async def _create_claims(
        self,
        extraction: ExtractionResult,
        name_to_node_id: dict[str, str],
        claim_node_ids: list[tuple[str, ExtractedClaim]],
        result: ConsolidationRunResult,
    ) -> None:
        for claim in extraction.claims:
            if claim.claim_type not in _CLAIM_TYPE_TO_MODEL:
                result.errors.append(
                    f"Unknown claim type '{claim.claim_type}', skipping claim"
                )
                continue

            model_cls = _CLAIM_TYPE_TO_MODEL[claim.claim_type]
            kwargs: dict = {"content": claim.content, **claim.properties}

            # Temporal claims need occurred_at
            if claim.claim_type in ("Event", "Decision", "Outcome"):
                occurred_at = _parse_datetime(
                    claim.temporal_info.occurred_at if claim.temporal_info else None
                )
                kwargs["occurred_at"] = occurred_at

            stable_claim_id = _stable_claim_id(claim)
            existing_claim = await self._graph.get_node(stable_claim_id)
            if isinstance(existing_claim, MemoryBase):
                claim_node_ids.append((stable_claim_id, claim))
                continue

            try:
                node = model_cls(id=stable_claim_id, **kwargs)
            except Exception as exc:
                result.errors.append(f"Failed to create {claim.claim_type} node: {exc}")
                continue

            node_id = await self._graph.create_node(node)
            claim_node_ids.append((node_id, claim))
            result.claims_created += 1

            await self._audit.log(
                AuditEvent(
                    event_type=AuditEventType.NODE_CREATED,
                    payload={
                        "run_id": result.run_id,
                        "node_id": node_id,
                        "node_type": claim.claim_type,
                        "content": claim.content[:100],
                    },
                )
            )

            # Link claim to involved entities via CONCERNS
            for entity_name in claim.involved_entities:
                entity_node_id = name_to_node_id.get(entity_name)
                if entity_node_id:
                    concerns = Concerns()
                    await self._create_relationship_once(
                        node_id, entity_node_id, "CONCERNS", concerns
                    )

    # ------------------------------------------------------------------
    # Source traceability
    # ------------------------------------------------------------------

    async def _create_sources(
        self,
        fragments: list[MemoryFragment],
        extraction: ExtractionResult,
        claim_node_ids: list[tuple[str, ExtractedClaim]],
        result: ConsolidationRunResult,
    ) -> None:
        # Build fragment_id -> fragment lookup
        frag_map = {f.id: f for f in fragments}

        # Build fragment_id -> claim_node_ids mapping
        frag_to_claim_ids: dict[str, list[str]] = {}
        for claim_id, claim in claim_node_ids:
            for fid in claim.source_fragment_ids:
                frag_to_claim_ids.setdefault(fid, []).append(claim_id)

        # Create Source nodes for fragments that produced claims
        for frag_id, claim_ids in frag_to_claim_ids.items():
            frag = frag_map.get(frag_id)
            if not frag:
                continue

            source_node_id = _stable_source_id(frag_id)
            source = Source(
                id=source_node_id,
                type=frag.type or "unknown",
                agent_id=frag.agent_id,
                reliability=Reliability.C,  # default; could be refined
            )
            existing_source = await self._graph.get_node(source_node_id)
            if not isinstance(existing_source, MemoryBase):
                source_node_id = await self._graph.create_node(source)
                await self._audit.log(
                    AuditEvent(
                        event_type=AuditEventType.NODE_CREATED,
                        payload={
                            "run_id": result.run_id,
                            "node_id": source_node_id,
                            "node_type": "Source",
                            "fragment_id": frag_id,
                        },
                    )
                )

            # SOURCED_FROM for each claim from this fragment
            for claim_id in claim_ids:
                sourced_from = SourcedFrom(credibility=Credibility.SIX)
                created = await self._create_relationship_once(
                    claim_id, source_node_id, "SOURCED_FROM", sourced_from
                )
                if created:
                    await self._audit.log(
                        AuditEvent(
                            event_type=AuditEventType.RELATION_CREATED,
                            payload={
                                "run_id": result.run_id,
                                "rel_type": "SOURCED_FROM",
                                "from_id": claim_id,
                                "to_id": source_node_id,
                            },
                        )
                    )

    # ------------------------------------------------------------------
    # Relation creation
    # ------------------------------------------------------------------

    async def _create_relations(
        self,
        extraction: ExtractionResult,
        name_to_node_id: dict[str, str],
        result: ConsolidationRunResult,
    ) -> None:
        for rel in extraction.relations:
            from_id = name_to_node_id.get(rel.from_entity)
            to_id = name_to_node_id.get(rel.to_entity)

            if not from_id or not to_id:
                missing = []
                if not from_id:
                    missing.append(rel.from_entity)
                if not to_id:
                    missing.append(rel.to_entity)
                result.errors.append(
                    f"Cannot create relation '{rel.relation_type}': "
                    f"unresolved entities {missing}"
                )
                continue

            model_cls = _REL_TYPE_TO_MODEL.get(rel.relation_type)
            if model_cls is None:
                result.errors.append(
                    f"Unknown relation type '{rel.relation_type}', skipping"
                )
                continue

            try:
                rel_model = model_cls(**rel.properties)
            except Exception as exc:
                result.errors.append(f"Failed to construct {rel.relation_type}: {exc}")
                continue

            await self._graph.create_relationship(from_id, to_id, rel_model)
            result.relations_created += 1

            await self._audit.log(
                AuditEvent(
                    event_type=AuditEventType.RELATION_CREATED,
                    payload={
                        "run_id": result.run_id,
                        "rel_type": rel.relation_type,
                        "from_id": from_id,
                        "to_id": to_id,
                    },
                )
            )

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    async def _detect_contradictions(
        self,
        existing_claims: list[MemoryNode],
        claim_node_ids: list[tuple[str, ExtractedClaim]],
        result: ConsolidationRunResult,
    ) -> None:
        existing_pairs = [
            (node.id, getattr(node, "content", ""))
            for node in existing_claims
            if hasattr(node, "content")
        ]

        for new_claim_id, new_claim in claim_node_ids:
            for existing_id, existing_content in existing_pairs:
                if _claims_contradict(new_claim.content, existing_content):
                    contradiction = Contradicts(
                        detection_run_id=result.run_id,
                        resolution_status=ResolutionStatus.unresolved,
                    )
                    created = await self._create_relationship_once(
                        new_claim_id,
                        existing_id,
                        "CONTRADICTS",
                        contradiction,
                    )
                    if created:
                        result.contradictions_detected += 1
                        await self._audit.log(
                            AuditEvent(
                                event_type=AuditEventType.RELATION_CREATED,
                                payload={
                                    "run_id": result.run_id,
                                    "rel_type": "CONTRADICTS",
                                    "from_id": new_claim_id,
                                    "to_id": existing_id,
                                },
                            )
                        )

    # ------------------------------------------------------------------
    # Abstraction
    # ------------------------------------------------------------------

    async def _run_abstraction(
        self,
        claim_node_ids: list[tuple[str, ExtractedClaim]],
        extraction: ExtractionResult,
        result: ConsolidationRunResult,
    ) -> None:
        min_occurrences = max(2, self._config.pattern_min_occurrences)
        buckets: dict[str, list[str]] = {}
        for claim_id, claim in claim_node_ids:
            key, _ = _claim_base_and_polarity(claim.content)
            if not key:
                continue
            buckets.setdefault(key, []).append(claim_id)

        pattern_ids: list[str] = []
        for key, claim_ids in buckets.items():
            if len(claim_ids) < min_occurrences:
                continue

            pattern = Pattern(
                content=f"Recurring pattern: {key}",
                derivation_run_id=result.run_id,
            )
            pattern_id = await self._graph.create_node(pattern)
            pattern_ids.append(pattern_id)
            result.patterns_created += 1

            await self._audit.log(
                AuditEvent(
                    event_type=AuditEventType.NODE_CREATED,
                    payload={
                        "run_id": result.run_id,
                        "node_id": pattern_id,
                        "node_type": "Pattern",
                    },
                )
            )

            for claim_id in claim_ids:
                await self._graph.create_relationship(
                    pattern_id,
                    claim_id,
                    DerivedFrom(
                        derivation_run_id=result.run_id,
                        derivation_method="frequency_detection",
                    ),
                )
                await self._graph.create_relationship(
                    pattern_id, claim_id, Generalizes()
                )
                await self._graph.create_relationship(
                    claim_id, pattern_id, InstanceOf()
                )

        concept_id: str | None = None
        if len(pattern_ids) >= 2:
            concept = Concept(
                content=f"Concept derived from {len(pattern_ids)} patterns",
                derivation_run_id=result.run_id,
            )
            concept_id = await self._graph.create_node(concept)
            result.concepts_created += 1

            await self._audit.log(
                AuditEvent(
                    event_type=AuditEventType.NODE_CREATED,
                    payload={
                        "run_id": result.run_id,
                        "node_id": concept_id,
                        "node_type": "Concept",
                    },
                )
            )

            for pattern_id in pattern_ids:
                await self._graph.create_relationship(
                    concept_id,
                    pattern_id,
                    DerivedFrom(
                        derivation_run_id=result.run_id,
                        derivation_method="pattern_clustering",
                    ),
                )
                await self._graph.create_relationship(
                    concept_id, pattern_id, Generalizes()
                )
                await self._graph.create_relationship(
                    pattern_id, concept_id, InstanceOf()
                )

        has_causal_signal = any(
            rel.relation_type in {"CAUSED_BY", "LEADS_TO"}
            for rel in extraction.relations
        )
        if concept_id and has_causal_signal:
            rule = Rule(
                content="Causal rule derived from recurring concept patterns",
                derivation_run_id=result.run_id,
            )
            rule_id = await self._graph.create_node(rule)
            result.rules_created += 1

            await self._audit.log(
                AuditEvent(
                    event_type=AuditEventType.NODE_CREATED,
                    payload={
                        "run_id": result.run_id,
                        "node_id": rule_id,
                        "node_type": "Rule",
                    },
                )
            )

            await self._graph.create_relationship(
                rule_id,
                concept_id,
                DerivedFrom(
                    derivation_run_id=result.run_id,
                    derivation_method="causal_abstraction",
                ),
            )
            await self._graph.create_relationship(rule_id, concept_id, Generalizes())
            await self._graph.create_relationship(concept_id, rule_id, InstanceOf())

    async def _create_relationship_once(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        rel_model: RelationshipBase,
    ) -> bool:
        """Create relationship only when an identical edge does not already exist."""
        existing = await self._graph.get_relationships(
            from_id, rel_type=rel_type, direction="outgoing"
        )
        if any(rel.get("to_id") == to_id for rel in existing):
            return False
        await self._graph.create_relationship(from_id, to_id, rel_model)
        return True
