"""Three-level entity resolution for the consolidation pipeline.

Maps extracted entity names to existing graph nodes or decides to create new
ones. The resolver is **pure logic** — it takes an extracted entity plus a
list of existing entities and returns a resolution decision. Graph mutations
are handled by the separate ``MergeExecutor``.

Levels:
    1. Deterministic normalization + exact match (incl. aliases)
    2. Fuzzy scoring (name similarity, context overlap, property compatibility)
    3. LLM-assisted disambiguation (optional, expensive)
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING

from engramcp.config import EntityResolutionConfig
from engramcp.engine.schemas import ExtractedEntity

if TYPE_CHECKING:
    from engramcp.engine.extraction import LLMAdapter
    from engramcp.graph.store import GraphStore

# ---------------------------------------------------------------------------
# Titles to strip during normalization
# ---------------------------------------------------------------------------

TITLES = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "esq",
        "jr",
        "sr",
        "ii",
        "iii",
        "iv",
    }
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ExistingEntity:
    """Lightweight representation of an entity already in the graph."""

    node_id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    fragment_ids: list[str] = field(default_factory=list)


class ResolutionAction(str, Enum):
    merge = "merge"
    review = "review"
    link = "link"
    create_new = "create_new"


@dataclass
class ResolutionCandidate:
    entity_name: str
    existing_node_id: str | None
    existing_name: str | None
    score: float
    action: ResolutionAction
    method: str  # "level_1", "level_2", "level_3"


@dataclass
class MergeResult:
    survivor_id: str
    absorbed_id: str
    aliases_added: list[str]
    relations_transferred: int


# ---------------------------------------------------------------------------
# Level 1 — Deterministic normalization
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """Deterministic name normalization.

    1. Unicode NFC normalization
    2. Strip leading/trailing whitespace
    3. Handle "Last, First" -> "First Last" (single comma reorder)
    4. Remove title tokens and periods
    5. Collapse whitespace
    6. Lowercase
    """
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", name)
    # 2. Strip whitespace
    text = text.strip()
    # 3. Comma reorder: "Epstein, Jeffrey" -> "Jeffrey Epstein"
    if "," in text:
        parts = text.split(",", maxsplit=1)
        if len(parts) == 2:
            text = f"{parts[1].strip()} {parts[0].strip()}"
    # 4. Remove periods (before token split to handle "Mr.")
    text = text.replace(".", "")
    # 5. Lowercase
    text = text.lower()
    # 6. Remove title tokens
    tokens = text.split()
    tokens = [t for t in tokens if t not in TITLES]
    # 7. Collapse whitespace via join
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Level 2 — Scoring functions (pure)
# ---------------------------------------------------------------------------


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on name tokens (after normalization)."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance — O(n*m), no external deps."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            # Insertion, deletion, substitution
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(
                    prev_row[j + 1] + 1,  # deletion
                    curr_row[j] + 1,  # insertion
                    prev_row[j] + cost,  # substitution
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """0.0 = identical, 1.0 = completely different."""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    return _levenshtein(a, b) / max_len


def name_similarity(a: str, b: str) -> float:
    """Combined name similarity: max(jaccard, 1 - edit_distance)."""
    jaccard = token_jaccard(a, b)
    edit_sim = 1.0 - normalized_edit_distance(a, b)
    return max(jaccard, edit_sim)


def context_overlap(frag_ids_a: set[str], frag_ids_b: set[str]) -> float:
    """Jaccard similarity on fragment ID sets. 0 if both empty."""
    if not frag_ids_a and not frag_ids_b:
        return 0.0
    intersection = frag_ids_a & frag_ids_b
    union = frag_ids_a | frag_ids_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


# Keys that trigger a blocking conflict when both present and different
_BLOCKING_KEYS = frozenset({"date_of_birth", "dob", "ssn", "email", "national_id"})


def property_compatibility(props_a: dict, props_b: dict) -> float | None:
    """Check property compatibility.

    Returns ``None`` if a blocking conflict exists (e.g. different DOB).
    Otherwise returns 0.0-1.0 based on shared compatible keys.
    """
    shared_keys = set(props_a) & set(props_b)
    if not shared_keys:
        # No overlap — neutral, return 0.0 (no signal)
        return 0.0

    compatible = 0
    for key in shared_keys:
        if props_a[key] == props_b[key]:
            compatible += 1
        elif key in _BLOCKING_KEYS:
            return None  # blocking conflict

    return compatible / len(shared_keys)


def composite_score(
    name_sim: float,
    context_sim: float,
    prop_compat: float | None,
    config: EntityResolutionConfig,
) -> float:
    """Weighted composite score. Blocking properties -> 0.0."""
    if prop_compat is None:
        return 0.0
    return (
        name_sim * config.name_similarity_weight
        + context_sim * config.context_overlap_weight
        + prop_compat * config.property_compatibility_weight
    )


# ---------------------------------------------------------------------------
# Anti-pattern guards
# ---------------------------------------------------------------------------


def _is_single_token(name: str) -> bool:
    """Single-token names should never auto-merge, only link."""
    tokens = name.strip().split()
    return len(tokens) == 1


def _is_cross_type(type_a: str, type_b: str) -> bool:
    """Different entity types are never merge candidates."""
    return type_a != type_b


# ---------------------------------------------------------------------------
# Level 3 — LLM disambiguation prompt
# ---------------------------------------------------------------------------


def build_disambiguation_prompt(
    entity_a_name: str,
    entity_a_context: str,
    entity_b_name: str,
    entity_b_context: str,
) -> str:
    """Build a prompt asking the LLM to decide SAME, DIFFERENT, or UNCERTAIN."""
    return (
        "You are an entity resolution assistant. "
        "Given two entity references and their contexts, determine if they "
        "refer to the same real-world entity.\n\n"
        f'Entity A: "{entity_a_name}"\n'
        f"Context A: {entity_a_context}\n\n"
        f'Entity B: "{entity_b_name}"\n'
        f"Context B: {entity_b_context}\n\n"
        "Reply with exactly one word: SAME, DIFFERENT, or UNCERTAIN."
    )


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolve extracted entities against existing graph entities (3-level cascade)."""

    def __init__(
        self,
        config: EntityResolutionConfig | None = None,
        llm: LLMAdapter | None = None,
    ) -> None:
        self._config = config or EntityResolutionConfig()
        self._llm = llm

    async def resolve(
        self,
        entity: ExtractedEntity,
        existing: list[ExistingEntity],
    ) -> ResolutionCandidate:
        """Resolve *entity* against *existing* graph entities.

        Level 1: exact match on normalized name + aliases
        Level 2: fuzzy scoring -> threshold-based decision
        Level 3: LLM disambiguation for ambiguous cases (if enabled)
        """
        if not existing:
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_1",
            )

        normalized = normalize_name(entity.name)

        # --- Level 1: exact match on normalized name + aliases ---
        # Single-token guard applies even at level 1: downgrade to link.
        single_token = _is_single_token(normalized)

        for ex in existing:
            if _is_cross_type(entity.type, ex.type):
                continue
            ex_normalized = normalize_name(ex.name)
            matched = normalized == ex_normalized
            if not matched:
                # Check aliases
                matched = any(
                    normalized == normalize_name(alias) for alias in ex.aliases
                )
            if matched:
                action = (
                    ResolutionAction.link if single_token else ResolutionAction.merge
                )
                return ResolutionCandidate(
                    entity_name=entity.name,
                    existing_node_id=ex.node_id,
                    existing_name=ex.name,
                    score=1.0,
                    action=action,
                    method="level_1",
                )

        # --- Level 2: fuzzy scoring ---
        best_candidate: ResolutionCandidate | None = None
        best_score = -1.0

        for ex in existing:
            if _is_cross_type(entity.type, ex.type):
                continue

            name_sim = name_similarity(
                normalize_name(entity.name), normalize_name(ex.name)
            )
            ctx_sim = context_overlap(
                set(entity.source_fragment_ids), set(ex.fragment_ids)
            )
            prop_compat = property_compatibility(entity.properties, ex.properties)
            score = composite_score(name_sim, ctx_sim, prop_compat, self._config)

            if score > best_score:
                best_score = score
                action = self._score_to_action(score, entity.name)
                best_candidate = ResolutionCandidate(
                    entity_name=entity.name,
                    existing_node_id=ex.node_id,
                    existing_name=ex.name,
                    score=score,
                    action=action,
                    method="level_2",
                )

        if best_candidate is None:
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=None,
                existing_name=None,
                score=0.0,
                action=ResolutionAction.create_new,
                method="level_2",
            )

        # --- Level 3: LLM disambiguation (if enabled and ambiguous) ---
        if (
            self._llm is not None
            and self._config.llm_assisted_enabled
            and self._config.create_link_threshold
            <= best_candidate.score
            < self._config.auto_merge_threshold
        ):
            llm_action = await self._llm_disambiguate(entity, best_candidate)
            return ResolutionCandidate(
                entity_name=entity.name,
                existing_node_id=best_candidate.existing_node_id,
                existing_name=best_candidate.existing_name,
                score=best_candidate.score,
                action=llm_action,
                method="level_3",
            )

        return best_candidate

    def _score_to_action(self, score: float, entity_name: str) -> ResolutionAction:
        """Map a composite score to a resolution action."""
        # Single-token guard: never auto-merge
        if _is_single_token(normalize_name(entity_name)):
            if score >= self._config.create_link_threshold:
                return ResolutionAction.link
            return ResolutionAction.create_new

        if score >= self._config.auto_merge_threshold:
            return ResolutionAction.merge
        if score >= self._config.flag_for_review_threshold:
            return ResolutionAction.review
        if score >= self._config.create_link_threshold:
            return ResolutionAction.link
        return ResolutionAction.create_new

    async def _llm_disambiguate(
        self,
        entity: ExtractedEntity,
        candidate: ResolutionCandidate,
    ) -> ResolutionAction:
        """Ask the LLM to resolve ambiguity."""
        assert self._llm is not None
        prompt = build_disambiguation_prompt(
            entity_a_name=entity.name,
            entity_a_context=entity.disambiguating_context or "",
            entity_b_name=candidate.existing_name or "",
            entity_b_context="",
        )
        response = await self._llm.complete(prompt)
        response = response.strip().upper()
        if "SAME" in response:
            return ResolutionAction.merge
        if "DIFFERENT" in response:
            return ResolutionAction.create_new
        return ResolutionAction.link


# ---------------------------------------------------------------------------
# MergeExecutor
# ---------------------------------------------------------------------------


class MergeExecutor:
    """Executes merge decisions by mutating the graph."""

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    async def execute_merge(
        self,
        survivor_id: str,
        absorbed_id: str,
        merge_run_id: str,
    ) -> MergeResult:
        """Merge *absorbed* node into *survivor*.

        1. Get both nodes
        2. Count transferable relationships (for MergeResult)
        3. Add absorbed name + aliases to survivor's aliases
        4. Store merge traceability on survivor properties
        5. Delete absorbed node (DETACH DELETE removes its relationships)
        6. Return MergeResult

        Merge traceability is stored as ``merged_from_ids`` on the survivor
        node rather than as a MERGED_FROM relationship, because
        ``GraphStore.delete_node`` uses ``DETACH DELETE`` which would
        destroy any relationship pointing to the absorbed node.
        """
        survivor_node = await self._graph.get_node(survivor_id)
        absorbed_node = await self._graph.get_node(absorbed_id)

        # --- Count transferable relationships ---
        rels = await self._graph.get_relationships(absorbed_id)
        transferred = 0
        for rel_data in rels:
            rel_type = rel_data["type"]
            from_id = rel_data["from_id"]
            to_id = rel_data["to_id"]
            # Skip POSSIBLY_SAME_AS between survivor and absorbed
            if rel_type == "POSSIBLY_SAME_AS" and {from_id, to_id} == {
                survivor_id,
                absorbed_id,
            }:
                continue
            transferred += 1

        # --- Compute aliases to add ---
        absorbed_name = getattr(absorbed_node, "name", str(absorbed_node))
        absorbed_aliases = getattr(absorbed_node, "aliases", []) or []
        survivor_existing_aliases = getattr(survivor_node, "aliases", []) or []

        new_aliases: list[str] = []
        all_existing = set(survivor_existing_aliases)
        for alias_name in [absorbed_name, *absorbed_aliases]:
            if alias_name and alias_name not in all_existing:
                new_aliases.append(alias_name)
                all_existing.add(alias_name)

        # --- Store merge traceability on survivor node ---
        existing_merged = getattr(survivor_node, "merged_from_ids", None) or []
        merged_from_ids = list(dict.fromkeys([*existing_merged, absorbed_id]))

        await self._graph.update_node(
            survivor_id,
            aliases=list(all_existing),
            merged_from_ids=merged_from_ids,
            last_merge_run_id=merge_run_id,
        )

        # --- Delete absorbed node (DETACH DELETE removes all its rels) ---
        await self._graph.delete_node(absorbed_id)

        return MergeResult(
            survivor_id=survivor_id,
            absorbed_id=absorbed_id,
            aliases_added=new_aliases,
            relations_transferred=transferred,
        )
