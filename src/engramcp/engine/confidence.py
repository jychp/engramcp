"""Confidence engine — Layer 3.

Computes NATO two-dimensional confidence ratings for knowledge nodes:
  - Dimension 1 (Reliability): from the source itself
  - Dimension 2 (Credibility): from corroboration, contradiction, and propagation

Also handles upward propagation (Fact → Pattern → Concept → Rule) and
downward cascade (contest/retraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from engramcp.graph.store import GraphStore
from engramcp.graph.traceability import SourceTraceability
from engramcp.models.confidence import Credibility
from engramcp.models.confidence import degrade_credibility
from engramcp.models.confidence import Reliability
from engramcp.models.confidence import worst_reliability
from engramcp.models.nodes import DerivedStatus
from engramcp.models.nodes import FactStatus

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceConfig:
    """Tuneable parameters for the confidence engine."""

    default_reliability: Reliability = Reliability.F
    initial_credibility: Credibility = Credibility.THREE
    min_independent_sources: int = 2
    depth_decay_steps: int = 1


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredibilityAssessment:
    """Result of assessing a fact's credibility."""

    credibility: Credibility
    supporting_count: int
    contradicting_count: int
    reason: str


@dataclass(frozen=True)
class PropagatedRating:
    """Computed confidence for a derived node."""

    reliability: Reliability
    credibility: Credibility
    source_count: int
    reason: str


@dataclass(frozen=True)
class CascadeResult:
    """Result of cascading a contest through the derivation hierarchy."""

    affected_nodes: list[str] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# ConfidenceEngine
# ---------------------------------------------------------------------------


class ConfidenceEngine:
    """Compute and propagate NATO confidence ratings."""

    def __init__(
        self,
        graph_store: GraphStore,
        traceability: SourceTraceability,
        config: ConfidenceConfig | None = None,
    ) -> None:
        self._store = graph_store
        self._trace = traceability
        self._config = config or ConfidenceConfig()

    # ---- Dimension 1: Reliability ----

    def reliability_from_hint(self, confidence_hint: str | None) -> Reliability:
        """Parse the first character of *confidence_hint* as a ``Reliability``.

        Returns ``config.default_reliability`` when the hint is absent
        or unparseable.
        """
        if not confidence_hint:
            return self._config.default_reliability
        try:
            return Reliability(confidence_hint[0].upper())
        except (ValueError, IndexError):
            return self._config.default_reliability

    # ---- Dimension 2: Credibility ----

    async def assess_credibility(self, fact_id: str) -> CredibilityAssessment:
        """Assess a fact's credibility based on its sources and contradictions."""
        sources = await self._trace.trace_fact_to_sources(fact_id)
        if not sources:
            return CredibilityAssessment(
                credibility=Credibility.SIX,
                supporting_count=0,
                contradicting_count=0,
                reason="No sources — truth cannot be judged",
            )

        # Count contradictions
        contradictions = await self._store.get_relationships(
            fact_id, rel_type="CONTRADICTS", direction="outgoing"
        )
        contradicting_count = len(contradictions)

        # Check for stronger contradicting sources
        if contradicting_count > 0:
            fact_reliabilities = [s.reliability for s in sources]
            best_fact_rel = min(
                fact_reliabilities, key=lambda r: list(Reliability).index(r)
            )

            has_stronger_contradiction = False
            for contra in contradictions:
                contra_sources = await self._trace.trace_fact_to_sources(
                    contra["to_id"]
                )
                for cs in contra_sources:
                    contra_idx = list(Reliability).index(cs.reliability)
                    fact_idx = list(Reliability).index(best_fact_rel)
                    if contra_idx < fact_idx:
                        has_stronger_contradiction = True
                        break
                if has_stronger_contradiction:
                    break

            if has_stronger_contradiction:
                return CredibilityAssessment(
                    credibility=Credibility.FIVE,
                    supporting_count=len(sources),
                    contradicting_count=contradicting_count,
                    reason="Contradicted by stronger source — improbable",
                )

            return CredibilityAssessment(
                credibility=Credibility.FOUR,
                supporting_count=len(sources),
                contradicting_count=contradicting_count,
                reason="Contradicted — doubtful",
            )

        # Check corroboration
        independent_count, _ = await self.check_corroboration(fact_id)
        if independent_count >= self._config.min_independent_sources:
            # Upgrade: more independent sources → better credibility
            upgrade_steps = independent_count - 1  # At least 1 step for 2 sources
            initial_idx = list(Credibility).index(self._config.initial_credibility)
            new_idx = max(initial_idx - upgrade_steps, 0)
            credibility = list(Credibility)[new_idx]
            return CredibilityAssessment(
                credibility=credibility,
                supporting_count=independent_count,
                contradicting_count=0,
                reason=f"Corroborated by {independent_count} independent sources",
            )

        return CredibilityAssessment(
            credibility=self._config.initial_credibility,
            supporting_count=len(sources),
            contradicting_count=0,
            reason="Uncorroborated — possibly true",
        )

    # ---- Corroboration ----

    async def check_corroboration(self, fact_id: str) -> tuple[int, list[str]]:
        """Count independent sources for a fact.

        Returns ``(independent_count, independent_source_ids)``.
        Uses union-find on independence to group dependent sources.
        """
        sources = await self._trace.trace_fact_to_sources(fact_id)
        if not sources:
            return 0, []

        source_ids = [s.id for s in sources]
        if len(source_ids) == 1:
            return 1, source_ids

        # Group sources by independence using union-find
        parent: dict[str, str] = {sid: sid for sid in source_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                result = await self._trace.check_independence(
                    source_ids[i], source_ids[j]
                )
                if not result.independent:
                    union(source_ids[i], source_ids[j])

        # Count distinct groups
        groups: set[str] = {find(sid) for sid in source_ids}
        # Return one representative per group
        representatives = list(groups)
        return len(representatives), representatives

    # ---- Upward propagation ----

    async def propagate_upward(self, derived_node_id: str) -> PropagatedRating:
        """Compute the propagated confidence for a derived node.

        Traces down through ``DERIVED_FROM`` to find base facts, then
        collects their source reliabilities and credibilities.
        """
        node = await self._store.get_node(derived_node_id)
        if node is None:
            return PropagatedRating(
                reliability=self._config.default_reliability,
                credibility=Credibility.SIX,
                source_count=0,
                reason="Node not found",
            )

        # Get derivation depth (Pattern=1, Concept=2, Rule=3)
        depth = getattr(node, "derivation_depth", 1)

        # Collect all base facts by walking DERIVED_FROM recursively
        base_facts = await self._collect_base_facts(derived_node_id, set())
        if not base_facts:
            return PropagatedRating(
                reliability=self._config.default_reliability,
                credibility=Credibility.SIX,
                source_count=0,
                reason="No base facts found in derivation chain",
            )

        # Collect all source reliabilities and best credibility
        reliabilities: list[Reliability] = []
        best_credibility_idx = len(list(Credibility)) - 1  # Start at worst

        for fact_id in base_facts:
            sources = await self._trace.trace_fact_to_sources(fact_id)
            for src in sources:
                reliabilities.append(src.reliability)
            # Get SOURCED_FROM credibilities
            rels = await self._store.get_relationships(
                fact_id, rel_type="SOURCED_FROM", direction="outgoing"
            )
            for rel in rels:
                cred_val = rel["props"].get("credibility")
                if cred_val:
                    try:
                        cred = Credibility(str(cred_val))
                    except ValueError:
                        continue
                    idx = list(Credibility).index(cred)
                    best_credibility_idx = min(best_credibility_idx, idx)

        if not reliabilities:
            return PropagatedRating(
                reliability=self._config.default_reliability,
                credibility=Credibility.SIX,
                source_count=0,
                reason="No sources found for base facts",
            )

        # Reliability: worst among all sources
        worst_rel = worst_reliability(*reliabilities)

        # Credibility: best source credibility, degraded by depth
        base_cred = list(Credibility)[best_credibility_idx]
        degraded = degrade_credibility(
            base_cred, steps=depth * self._config.depth_decay_steps
        )

        return PropagatedRating(
            reliability=worst_rel,
            credibility=degraded,
            source_count=len(reliabilities),
            reason=f"Propagated from {len(base_facts)} base facts at depth {depth}",
        )

    async def _collect_base_facts(self, node_id: str, visited: set[str]) -> list[str]:
        """Recursively collect base fact IDs through DERIVED_FROM chains."""
        if node_id in visited:
            return []
        visited.add(node_id)

        rels = await self._store.get_relationships(
            node_id, rel_type="DERIVED_FROM", direction="outgoing"
        )
        if not rels:
            # This might be a base fact itself
            node = await self._store.get_node(node_id)
            if (
                node is not None
                and hasattr(node, "content")
                and not hasattr(node, "derivation_depth")
            ):
                return [node_id]
            return []

        base_facts: list[str] = []
        for rel in rels:
            target_id = rel["to_id"]
            target = await self._store.get_node(target_id)
            if target is None:
                continue
            if hasattr(target, "derivation_depth"):
                # Derived node — recurse deeper
                base_facts.extend(await self._collect_base_facts(target_id, visited))
            else:
                # Base knowledge node (Fact, Event, etc.)
                base_facts.append(target_id)

        return base_facts

    # ---- Downward cascade ----

    async def cascade_contest(self, contested_node_id: str) -> CascadeResult:
        """Cascade a contest through the derivation hierarchy.

        Finds all derived nodes (Pattern, Concept, Rule) that depend on
        *contested_node_id* and recalculates their status. Nodes with
        all contributing facts contested/retracted are dissolved.

        Never deletes nodes — only changes status.
        """
        affected: list[str] = []
        await self._cascade_to_dependents(contested_node_id, affected, set())
        return CascadeResult(
            affected_nodes=affected,
            reason=f"Cascaded contest from {contested_node_id} to {len(affected)} nodes",
        )

    async def _cascade_to_dependents(
        self, node_id: str, affected: list[str], visited: set[str]
    ) -> None:
        """Recursively cascade contest to nodes derived from *node_id*."""
        if node_id in visited:
            return
        visited.add(node_id)

        # Find nodes that have DERIVED_FROM pointing to this node
        # i.e. (derived)-[:DERIVED_FROM]->(node_id)
        rels = await self._store.get_relationships(
            node_id, rel_type="DERIVED_FROM", direction="incoming"
        )

        for rel in rels:
            derived_id = rel["from_id"]
            derived = await self._store.get_node(derived_id)
            if derived is None:
                continue
            if not hasattr(derived, "derivation_depth"):
                continue

            # Check if all contributing facts are contested/retracted
            should_dissolve = await self._all_sources_contested(derived_id)
            if should_dissolve:
                await self._store.update_node(
                    derived_id, status=DerivedStatus.dissolved.value
                )
                if derived_id not in affected:
                    affected.append(derived_id)
            else:
                # Mark as contested but not dissolved
                if derived.status == DerivedStatus.active:
                    await self._store.update_node(
                        derived_id, status=DerivedStatus.contested.value
                    )
                if derived_id not in affected:
                    affected.append(derived_id)

            # Recurse to higher-level derived nodes
            await self._cascade_to_dependents(derived_id, affected, visited)

    async def _all_sources_contested(self, derived_node_id: str) -> bool:
        """Check if all base facts of a derived node are contested or retracted."""
        base_facts = await self._collect_base_facts(derived_node_id, set())
        if not base_facts:
            return True  # No sources = dissolve

        for fact_id in base_facts:
            fact = await self._store.get_node(fact_id)
            if fact is None:
                continue
            status = getattr(fact, "status", None)
            if status not in (FactStatus.contested, FactStatus.retracted):
                return False  # At least one active fact
        return True
