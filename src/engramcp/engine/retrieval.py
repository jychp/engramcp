"""Retrieval engine (Layer 6) with WM-first strategy and graph fallback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from engramcp.engine.concepts import ConceptRegistry
from engramcp.engine.demand import QueryDemandTracker
from engramcp.engine.demand import QueryPattern
from engramcp.memory.schemas import MemoryFragment
from engramcp.memory.store import WorkingMemory
from engramcp.models.schemas import Contradiction
from engramcp.models.schemas import ContradictionNature
from engramcp.models.schemas import GetMemoryInput
from engramcp.models.schemas import GetMemoryResult
from engramcp.models.schemas import MemoryEntry
from engramcp.models.schemas import MetaInfo
from engramcp.models.schemas import SourceEntry
from engramcp.observability import record_latency

_RELIABILITY_ORDER = "ABCDEF"


class RetrievalScorer(Protocol):
    """Scores retrieval candidates for ranking."""

    def score_working_memory(self, fragment: MemoryFragment) -> float:
        """Return a sortable score for a working-memory fragment."""

    def score_graph_memory(self, memory: MemoryEntry) -> float:
        """Return a sortable score for a graph-backed memory entry."""


class GraphRetriever(Protocol):
    """Minimal graph retrieval protocol used by Layer 6."""

    async def find_claim_nodes(self) -> Sequence[object]:
        """Return claim-like nodes as a fallback retrieval source."""


@dataclass(frozen=True)
class RecencyConfidenceScorer:
    """Default scorer combining recency and NATO confidence quality."""

    def score_working_memory(self, fragment: MemoryFragment) -> float:
        confidence_bonus = _confidence_bonus(fragment.confidence)
        return fragment.timestamp + confidence_bonus

    def score_graph_memory(self, memory: MemoryEntry) -> float:
        # Graph entries currently expose no confidence relation in this projection.
        return _confidence_bonus(memory.confidence)


class RetrievalEngine:
    """Layer 6 retrieval service with WM-first and graph fallback behavior."""

    def __init__(
        self,
        working_memory: WorkingMemory,
        *,
        graph_retriever: GraphRetriever | None = None,
        scorer: RetrievalScorer | None = None,
        demand_tracker: QueryDemandTracker | None = None,
        concept_registry: ConceptRegistry | None = None,
    ) -> None:
        self._wm = working_memory
        self._graph = graph_retriever
        self._scorer = scorer or RecencyConfidenceScorer()
        self._demand_tracker = demand_tracker or QueryDemandTracker()
        self._concept_registry = concept_registry or ConceptRegistry()

    async def retrieve(self, request: GetMemoryInput) -> GetMemoryResult:
        """Retrieve memories using WM-first selection with graph fallback."""
        start = perf_counter()
        wm_matches = await self._wm.search(
            request.query, min_confidence=request.min_confidence
        )
        self._record_retrieval_shape(
            matches=wm_matches,
            compact=request.compact,
            include_sources=request.include_sources,
            include_contradictions=request.include_contradictions,
        )

        wm_matches.sort(key=self._scorer.score_working_memory, reverse=True)
        graph_matches: list[MemoryEntry] = []
        graph_contradictions: list[Contradiction] = []
        if not wm_matches:
            graph_matches, graph_contradictions = await self._search_graph(request)
            graph_matches.sort(key=self._scorer.score_graph_memory, reverse=True)

        total_found = len(wm_matches) + len(graph_matches)
        truncated = total_found > request.limit

        selected_wm = wm_matches[: request.limit]
        memories = [
            self._to_memory_entry(fragment, request) for fragment in selected_wm
        ]
        remaining = request.limit - len(memories)
        if remaining > 0 and graph_matches:
            memories.extend(graph_matches[:remaining])

        elapsed_ms = (perf_counter() - start) * 1000
        meta = MetaInfo(
            query=request.query,
            total_found=total_found,
            returned=len(memories),
            truncated=truncated,
            max_depth_used=request.max_depth,
            min_confidence_applied=request.min_confidence,
            retrieval_ms=int(elapsed_ms),
            working_memory_hits=len(selected_wm),
            graph_hits=min(len(graph_matches), max(remaining, 0)),
        )
        record_latency(
            operation="retrieval_engine.retrieve",
            duration_ms=elapsed_ms,
            ok=True,
        )
        return GetMemoryResult(
            memories=memories,
            contradictions=graph_contradictions,
            meta=meta,
        )

    def query_demand_count(
        self,
        *,
        node_types: Sequence[str] | None = None,
        properties: Sequence[str] | None = None,
    ) -> int:
        """Expose current demand count for a normalized retrieval shape."""
        pattern = QueryPattern.from_parts(node_types=node_types, properties=properties)
        return self._demand_tracker.count(pattern)

    def concept_candidate_count(self) -> int:
        """Expose current concept-candidate count (for tests/introspection)."""
        return self._concept_registry.candidate_count()

    def _record_retrieval_shape(
        self,
        *,
        matches: list[MemoryFragment],
        compact: bool,
        include_sources: bool,
        include_contradictions: bool,
    ) -> None:
        node_types = [m.dynamic_type or m.type for m in matches]
        properties = ["content", "confidence", "min_confidence"]

        if not compact:
            properties.extend(["participants", "causal_chain"])
            if include_sources:
                properties.append("sources")
        if include_contradictions:
            properties.append("contradictions")

        signal = self._demand_tracker.record_call(
            node_types=node_types, properties=properties
        )
        if signal is not None:
            self._concept_registry.observe_signal(signal)

    async def _search_graph(
        self, request: GetMemoryInput
    ) -> tuple[list[MemoryEntry], list[Contradiction]]:
        """Search graph claims by query content with legacy compatibility fallback."""
        if self._graph is None:
            return [], []

        find_context = getattr(self._graph, "find_claim_context_by_content", None)
        if callable(find_context):
            contexts = await find_context(
                request.query,
                limit=request.limit,
                max_depth=request.max_depth,
                include_sources=request.include_sources,
                include_contradictions=request.include_contradictions,
            )
            return self._context_to_graph_entries(contexts, request)

        find_by_content = getattr(self._graph, "find_claim_nodes_by_content", None)
        if callable(find_by_content):
            nodes = await find_by_content(request.query, limit=request.limit)
        else:
            nodes = await self._graph.find_claim_nodes()

        entries: list[MemoryEntry] = []
        query = request.query.casefold()
        for node in nodes:
            node_id = getattr(node, "id", None)
            content = getattr(node, "content", None)
            if not node_id or not content:
                continue
            if query and query not in content.casefold():
                continue

            node_labels = set(getattr(node, "node_labels", []) or [])
            base_type, dynamic_type = _labels_to_types(
                node_labels, fallback=type(node).__name__
            )
            confidence = getattr(node, "confidence", None)
            if not _confidence_passes_minimum(confidence, request.min_confidence):
                continue

            entries.append(
                MemoryEntry(
                    id=node_id,
                    type=base_type,
                    dynamic_type=dynamic_type,
                    content=content,
                    confidence=confidence,
                    properties={},
                    participants=[] if request.compact else [],
                    causal_chain=[] if request.compact else [],
                    sources=[],
                )
            )
        return entries, []

    def _context_to_graph_entries(
        self, contexts: Sequence[dict], request: GetMemoryInput
    ) -> tuple[list[MemoryEntry], list[Contradiction]]:
        entries: list[MemoryEntry] = []
        contradictions: list[Contradiction] = []
        for context in contexts:
            node = context.get("node", {})
            node_id = str(node.get("id", ""))
            content = str(node.get("content", ""))
            if not node_id or not content:
                continue

            base_type, dynamic_type = _labels_to_types(
                node.get("labels", []), fallback="Fact"
            )
            confidence = node.get("confidence")
            if not _confidence_passes_minimum(confidence, request.min_confidence):
                continue

            causal_chain = []
            if not request.compact:
                for link in context.get("causal_chain", []):
                    relation = str(link.get("relation", "")).strip()
                    target_id = str(link.get("target_id", "")).strip()
                    target_summary = str(link.get("target_summary", "")).strip()
                    if not relation or not target_id:
                        continue
                    causal_chain.append(
                        {
                            "relation": relation.casefold(),
                            "target_id": target_id,
                            "target_summary": target_summary,
                            "confidence": link.get("confidence"),
                        }
                    )

            sources: list[SourceEntry] = []
            if not request.compact and request.include_sources:
                sources = [SourceEntry(**src) for src in context.get("sources", [])]

            entries.append(
                MemoryEntry(
                    id=node_id,
                    type=base_type,
                    dynamic_type=dynamic_type,
                    content=content,
                    confidence=confidence,
                    properties=node.get("properties", {}) or {},
                    participants=[],
                    causal_chain=causal_chain,
                    sources=sources,
                )
            )

            if request.compact or not request.include_contradictions:
                continue

            for contradiction_raw in context.get("contradictions", []):
                contradictions.append(
                    self._to_contradiction(contradiction_raw, default_memory_id=node_id)
                )

        return entries, contradictions

    def _to_memory_entry(
        self, fragment: MemoryFragment, request: GetMemoryInput
    ) -> MemoryEntry:
        sources = []
        if not request.compact and request.include_sources and fragment.sources:
            sources = [SourceEntry(**source) for source in fragment.sources]

        return MemoryEntry(
            id=fragment.id,
            type=fragment.type,
            dynamic_type=fragment.dynamic_type,
            content=fragment.content,
            confidence=fragment.confidence,
            properties=fragment.properties,
            participants=[] if request.compact else fragment.participants,
            causal_chain=[] if request.compact else fragment.causal_chain,
            sources=sources,
        )

    def _to_contradiction(self, raw: dict, *, default_memory_id: str) -> Contradiction:
        contradictory = raw.get("memory") or {}
        base_type, dynamic_type = _labels_to_types(
            contradictory.get("labels", []), fallback="Fact"
        )
        contradicting_memory = MemoryEntry(
            id=str(contradictory.get("id", "")),
            type=base_type,
            dynamic_type=dynamic_type,
            content=str(contradictory.get("content", "")),
            confidence=contradictory.get("confidence"),
            properties=contradictory.get("properties", {}) or {},
            participants=[],
            causal_chain=[],
            sources=[],
        )
        nature = _parse_contradiction_nature(raw.get("nature"))
        return Contradiction(
            id=str(raw.get("id", "")),
            memory_id=str(raw.get("memory_id") or default_memory_id),
            contradicting_memory=contradicting_memory,
            nature=nature,
            resolution_status=str(raw.get("resolution_status", "unresolved")),
            detected_at=raw.get("detected_at"),
        )


def _confidence_bonus(confidence: str | None) -> float:
    """Convert NATO confidence text into a tiny bonus for stable sorting."""
    if not confidence or len(confidence) < 2:
        return 0.0

    letter = confidence[0].upper()
    number = confidence[1:]
    letters = "ABCDEF"

    try:
        letter_quality = (len(letters) - letters.index(letter)) / 10_000
        number_quality = (7 - int(number)) / 100_000
    except (ValueError, IndexError):
        return 0.0

    return letter_quality + number_quality


def _confidence_passes_minimum(confidence: object, minimum: str) -> bool:
    """Apply NATO minimum confidence filtering with AND semantics."""
    if minimum == "F6":
        return True
    if not isinstance(confidence, str) or len(confidence) < 2:
        return False

    mem_letter = confidence[0].upper()
    mem_number_raw = confidence[1:]
    flt_letter = minimum[0].upper()
    flt_number_raw = minimum[1:]
    try:
        letter_ok = _RELIABILITY_ORDER.index(mem_letter) <= _RELIABILITY_ORDER.index(
            flt_letter
        )
        number_ok = int(mem_number_raw) <= int(flt_number_raw)
    except (ValueError, IndexError):
        return False
    return letter_ok and number_ok


def _parse_contradiction_nature(value: object) -> ContradictionNature:
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return ContradictionNature(normalized)
        except ValueError:
            pass
    return ContradictionNature.factual_conflict


def _labels_to_types(
    labels: Sequence[str] | set[str], *, fallback: str
) -> tuple[str, str | None]:
    ordered_base = [
        "Fact",
        "Event",
        "Observation",
        "Decision",
        "Outcome",
        "Pattern",
        "Concept",
        "Rule",
        "Agent",
        "Artifact",
        "Source",
    ]
    raw_labels = [str(label) for label in labels]
    label_set = set(raw_labels)
    base_type = next((name for name in ordered_base if name in label_set), fallback)
    structural_labels = {"Memory", "Temporal", "Derived", base_type}
    dynamic_type = next(
        (label for label in raw_labels if label not in structural_labels),
        None,
    )
    return base_type, dynamic_type
