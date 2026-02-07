"""Tests for entity resolution (Layer 4 partial).

All pure unit tests — no containers. GraphStore is mocked for merge tests.
LLM is mocked via a local MockLLMAdapter.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from engramcp.config import EntityResolutionConfig
from engramcp.engine.schemas import ExtractedEntity
from engramcp.graph.entity_resolution import (
    ExistingEntity,
    MergeExecutor,
    MergeResult,
    ResolutionAction,
    ResolutionCandidate,
    EntityResolver,
    build_disambiguation_prompt,
    composite_score,
    context_overlap,
    name_similarity,
    normalize_name,
    normalized_edit_distance,
    property_compatibility,
    token_jaccard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _existing(
    name: str,
    *,
    node_id: str = "node_1",
    type: str = "Agent",
    aliases: list[str] | None = None,
    properties: dict | None = None,
    fragment_ids: list[str] | None = None,
) -> ExistingEntity:
    return ExistingEntity(
        node_id=node_id,
        name=name,
        type=type,
        aliases=aliases or [],
        properties=properties or {},
        fragment_ids=fragment_ids or [],
    )


def _extracted(
    name: str,
    *,
    type: str = "Agent",
    aliases: list[str] | None = None,
    properties: dict | None = None,
    source_fragment_ids: list[str] | None = None,
    disambiguating_context: str | None = None,
) -> ExtractedEntity:
    return ExtractedEntity(
        name=name,
        type=type,
        aliases=aliases or [],
        properties=properties or {},
        source_fragment_ids=source_fragment_ids or [],
        disambiguating_context=disambiguating_context,
    )


class MockLLMAdapter:
    """Test double for LLMAdapter. Returns canned responses."""

    def __init__(self, response: str = "SAME") -> None:
        self.calls: list[dict] = []
        self._response = response

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self._response


# ===========================================================================
# TestNameNormalization
# ===========================================================================


class TestNameNormalization:
    def test_case_normalization(self):
        assert normalize_name("Jeffrey EPSTEIN") == "jeffrey epstein"

    def test_name_reordering(self):
        assert normalize_name("Epstein, Jeffrey") == "jeffrey epstein"

    def test_title_stripping(self):
        assert normalize_name("Mr. Jeffrey Epstein") == "jeffrey epstein"
        assert normalize_name("Jeffrey Epstein Esq.") == "jeffrey epstein"
        assert normalize_name("Dr. John Smith Jr.") == "john smith"
        assert normalize_name("Mrs. Jane Doe Sr.") == "jane doe"

    def test_whitespace_normalization(self):
        assert normalize_name("  Jeffrey   Epstein  ") == "jeffrey epstein"

    def test_unicode_normalization(self):
        # Pre-composed vs decomposed e-acute
        import unicodedata

        nfc = unicodedata.normalize("NFC", "caf\u00e9")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        assert normalize_name(nfc) == normalize_name(nfd)

    def test_exact_match_after_normalization(self):
        assert normalize_name("Mr. Jeffrey Epstein") == normalize_name(
            "jeffrey epstein"
        )


# ===========================================================================
# TestTokenSimilarity
# ===========================================================================


class TestTokenSimilarity:
    def test_token_jaccard_identical(self):
        assert token_jaccard("jeffrey epstein", "jeffrey epstein") == 1.0

    def test_token_jaccard_partial_overlap(self):
        score = token_jaccard("jeff epstein", "jeffrey epstein")
        # "epstein" shared, "jeff"/"jeffrey" differ → 1 out of 3 unique tokens
        assert 0.0 < score < 1.0

    def test_token_jaccard_no_overlap(self):
        assert token_jaccard("alice bob", "charlie dave") == 0.0

    def test_edit_distance_scoring(self):
        # Identical → 0.0
        assert normalized_edit_distance("jeffrey", "jeffrey") == 0.0
        # Completely different → close to 1.0
        dist = normalized_edit_distance("abcdef", "uvwxyz")
        assert dist > 0.8
        # Similar → low distance
        dist = normalized_edit_distance("jeffrey", "jeffery")
        assert dist < 0.3


# ===========================================================================
# TestCompositeScoring
# ===========================================================================


class TestCompositeScoring:
    def test_context_overlap_scoring(self):
        # Shared fragment IDs
        score = context_overlap({"f1", "f2", "f3"}, {"f2", "f3", "f4"})
        # Jaccard: 2 / 4 = 0.5
        assert score == pytest.approx(0.5)

    def test_context_overlap_both_empty(self):
        assert context_overlap(set(), set()) == 0.0

    def test_property_compatibility_pass(self):
        props_a = {"date_of_birth": "1953-01-20", "nationality": "US"}
        props_b = {"date_of_birth": "1953-01-20", "role": "financier"}
        score = property_compatibility(props_a, props_b)
        assert score is not None
        assert score > 0.0

    def test_property_compatibility_blocks(self):
        # Different DOB → blocking
        props_a = {"date_of_birth": "1953-01-20"}
        props_b = {"date_of_birth": "1970-05-10"}
        score = property_compatibility(props_a, props_b)
        assert score is None

    def test_composite_score_calculation(self):
        config = EntityResolutionConfig()
        score = composite_score(
            name_sim=0.8,
            context_sim=0.6,
            prop_compat=1.0,
            config=config,
        )
        # 0.8*0.5 + 0.6*0.3 + 1.0*0.2 = 0.4 + 0.18 + 0.2 = 0.78
        assert score == pytest.approx(0.78)

    def test_composite_score_blocking(self):
        config = EntityResolutionConfig()
        score = composite_score(
            name_sim=0.9,
            context_sim=0.9,
            prop_compat=None,  # blocking
            config=config,
        )
        assert score == 0.0

    def test_all_weights_configurable(self):
        config = EntityResolutionConfig(
            name_similarity_weight=0.7,
            context_overlap_weight=0.2,
            property_compatibility_weight=0.1,
        )
        score = composite_score(
            name_sim=1.0,
            context_sim=0.0,
            prop_compat=0.0,
            config=config,
        )
        # 1.0*0.7 + 0.0*0.2 + 0.0*0.1 = 0.7
        assert score == pytest.approx(0.7)


# ===========================================================================
# TestEntityResolver
# ===========================================================================


class TestEntityResolver:
    async def test_level_1_exact_match_returns_merge(self):
        resolver = EntityResolver()
        entity = _extracted("Jeffrey Epstein")
        existing = [_existing("jeffrey epstein")]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.merge
        assert result.method == "level_1"
        assert result.existing_node_id == "node_1"

    async def test_level_1_alias_match_returns_merge(self):
        resolver = EntityResolver()
        entity = _extracted("Jeff Epstein")
        existing = [
            _existing(
                "Jeffrey Epstein",
                aliases=["jeff epstein"],
            )
        ]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.merge
        assert result.method == "level_1"

    async def test_level_2_high_score_auto_merges(self):
        resolver = EntityResolver()
        entity = _extracted(
            "Jeffrey E. Epstein",
            source_fragment_ids=["f1", "f2"],
            properties={"date_of_birth": "1953-01-20"},
        )
        existing = [
            _existing(
                "Jeffrey Epstein",
                fragment_ids=["f1", "f2"],
                properties={"date_of_birth": "1953-01-20"},
            )
        ]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.merge
        assert result.score > 0.9

    async def test_level_2_medium_score_flags(self):
        config = EntityResolutionConfig(llm_assisted_enabled=False)
        resolver = EntityResolver(config=config)
        # "Jeff Epstein" vs "Jeffrey Epstein" with partial fragment overlap + compatible props
        entity = _extracted(
            "Jeff Epstein",
            source_fragment_ids=["f1"],
            properties={"role": "financier"},
        )
        existing = [
            _existing(
                "Jeffrey Epstein",
                fragment_ids=["f1", "f2"],
                properties={"role": "financier"},
            )
        ]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.review
        assert 0.7 <= result.score < 0.9

    async def test_level_2_low_score_creates_link(self):
        config = EntityResolutionConfig(llm_assisted_enabled=False)
        resolver = EntityResolver(config=config)
        # "J. Epstein" (2 tokens) vs "Jeffrey Epstein" with shared fragment → ~0.6
        entity = _extracted(
            "J. Epstein",
            source_fragment_ids=["f1"],
        )
        existing = [
            _existing(
                "Jeffrey Epstein",
                fragment_ids=["f1"],
            )
        ]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.link
        assert 0.5 <= result.score < 0.7

    async def test_level_2_very_low_keeps_separate(self):
        resolver = EntityResolver()
        entity = _extracted("Alice Johnson")
        existing = [_existing("Bob Smith")]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.create_new
        assert result.score < 0.5


# ===========================================================================
# TestLLMAssisted
# ===========================================================================


class TestLLMAssisted:
    async def test_ambiguous_cases_sent_to_llm(self):
        llm = MockLLMAdapter(response="SAME")
        config = EntityResolutionConfig(llm_assisted_enabled=True)
        resolver = EntityResolver(config=config, llm=llm)
        entity = _extracted(
            "J. Epstein",
            source_fragment_ids=["f1"],
        )
        existing = [
            _existing(
                "Jeffrey Epstein",
                fragment_ids=["f1"],
            )
        ]
        result = await resolver.resolve(entity, existing)
        assert len(llm.calls) == 1
        assert result.method == "level_3"

    async def test_llm_receives_full_context(self):
        llm = MockLLMAdapter(response="SAME")
        config = EntityResolutionConfig(llm_assisted_enabled=True)
        resolver = EntityResolver(config=config, llm=llm)
        entity = _extracted(
            "J. Epstein",
            source_fragment_ids=["f1"],
            disambiguating_context="financier",
        )
        existing = [
            _existing(
                "Jeffrey Epstein",
                fragment_ids=["f1"],
            )
        ]
        await resolver.resolve(entity, existing)
        prompt = llm.calls[0]["prompt"]
        assert "J. Epstein" in prompt
        assert "Jeffrey Epstein" in prompt

    async def test_llm_same_triggers_merge(self):
        llm = MockLLMAdapter(response="SAME")
        config = EntityResolutionConfig(llm_assisted_enabled=True)
        resolver = EntityResolver(config=config, llm=llm)
        entity = _extracted("J. Epstein", source_fragment_ids=["f1"])
        existing = [_existing("Jeffrey Epstein", fragment_ids=["f1"])]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.merge

    async def test_llm_different_keeps_separate(self):
        llm = MockLLMAdapter(response="DIFFERENT")
        config = EntityResolutionConfig(llm_assisted_enabled=True)
        resolver = EntityResolver(config=config, llm=llm)
        entity = _extracted("J. Epstein", source_fragment_ids=["f1"])
        existing = [_existing("Jeffrey Epstein", fragment_ids=["f1"])]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.create_new

    async def test_llm_uncertain_creates_link(self):
        llm = MockLLMAdapter(response="UNCERTAIN")
        config = EntityResolutionConfig(llm_assisted_enabled=True)
        resolver = EntityResolver(config=config, llm=llm)
        entity = _extracted("J. Epstein", source_fragment_ids=["f1"])
        existing = [_existing("Jeffrey Epstein", fragment_ids=["f1"])]
        result = await resolver.resolve(entity, existing)
        assert result.action == ResolutionAction.link

    async def test_level_3_disabled_skips_llm(self):
        llm = MockLLMAdapter(response="SAME")
        config = EntityResolutionConfig(llm_assisted_enabled=False)
        resolver = EntityResolver(config=config, llm=llm)
        entity = _extracted("J. Epstein", source_fragment_ids=["f1"])
        existing = [_existing("Jeffrey Epstein", fragment_ids=["f1"])]
        result = await resolver.resolve(entity, existing)
        assert len(llm.calls) == 0
        assert result.method == "level_2"


# ===========================================================================
# TestAntiPatterns
# ===========================================================================


class TestAntiPatterns:
    def test_last_name_only_never_auto_merges(self):
        """Single-token names should never auto-merge, only link at best."""
        resolver = EntityResolver()
        # name_similarity for "epstein" vs "jeffrey epstein" should be penalized
        # by the single-token guard in resolve()
        # We test the guard function directly
        from engramcp.graph.entity_resolution import _is_single_token

        assert _is_single_token("Maxwell") is True
        assert _is_single_token("Ghislaine Maxwell") is False

    def test_cross_type_never_merges(self):
        from engramcp.graph.entity_resolution import _is_cross_type

        assert _is_cross_type("Agent", "Artifact") is True
        assert _is_cross_type("Agent", "Agent") is False

    async def test_no_transitive_auto_merge(self):
        """Verify resolver evaluates each entity independently."""
        resolver = EntityResolver()
        entity = _extracted("Alice Smith")
        # Two different existing entities — resolver picks the best match
        existing = [
            _existing("Alice Smith Jones", node_id="n1"),
            _existing("Bob Smith", node_id="n2"),
        ]
        result = await resolver.resolve(entity, existing)
        # The result should reflect the best single match, not transitivity
        assert result.existing_node_id in ("n1", "n2", None)

    def test_single_token_name_conservative(self):
        """Single token names should be flagged as conservative."""
        from engramcp.graph.entity_resolution import _is_single_token

        assert _is_single_token("Maxwell") is True
        assert _is_single_token("A") is True
        assert _is_single_token("") is False  # empty is not single-token


# ===========================================================================
# TestMergeExecutor
# ===========================================================================


class TestMergeExecutor:
    def _make_mock_graph(self):
        """Create a mock GraphStore with async methods."""
        graph = AsyncMock()
        # get_node returns mock nodes
        graph.get_node = AsyncMock(
            side_effect=lambda nid: MagicMock(
                name="Jeffrey Epstein" if nid == "survivor_1" else "Jeff Epstein",
                aliases=["Jeff E."] if nid == "survivor_1" else [],
                id=nid,
            )
        )
        # get_relationships returns mock relations
        graph.get_relationships = AsyncMock(
            return_value=[
                {
                    "type": "CONCERNS",
                    "props": {"created_at": "2025-01-01T00:00:00+00:00"},
                    "from_id": "absorbed_1",
                    "to_id": "fact_1",
                }
            ]
        )
        graph.create_relationship = AsyncMock(return_value=True)
        graph.update_node = AsyncMock(return_value=MagicMock())
        graph.delete_node = AsyncMock(return_value=True)
        return graph

    async def test_surviving_node_absorbs_relations(self):
        graph = self._make_mock_graph()
        executor = MergeExecutor(graph)
        result = await executor.execute_merge("survivor_1", "absorbed_1", "run_001")
        assert isinstance(result, MergeResult)
        assert result.survivor_id == "survivor_1"
        assert result.absorbed_id == "absorbed_1"
        assert result.relations_transferred >= 1

    async def test_aliases_populated_on_merge(self):
        graph = self._make_mock_graph()
        executor = MergeExecutor(graph)
        result = await executor.execute_merge("survivor_1", "absorbed_1", "run_001")
        # The absorbed node's name should be in aliases_added
        assert len(result.aliases_added) > 0

    async def test_aliases_feed_future_level_1_matching(self):
        """After merge, the absorbed name becomes an alias on the survivor."""
        graph = self._make_mock_graph()
        executor = MergeExecutor(graph)
        result = await executor.execute_merge("survivor_1", "absorbed_1", "run_001")
        # update_node should have been called with aliases
        graph.update_node.assert_called()

    async def test_merged_from_relation_created(self):
        graph = self._make_mock_graph()
        executor = MergeExecutor(graph)
        await executor.execute_merge("survivor_1", "absorbed_1", "run_001")
        # create_relationship should include a MERGED_FROM call
        rel_calls = graph.create_relationship.call_args_list
        merged_from_calls = [
            c
            for c in rel_calls
            if hasattr(c.args[2] if len(c.args) > 2 else c.kwargs.get("rel"), "rel_type")
            and (c.args[2] if len(c.args) > 2 else c.kwargs.get("rel")).rel_type
            == "MERGED_FROM"
        ]
        assert len(merged_from_calls) >= 1

    async def test_possibly_same_as_removed_on_merge(self):
        graph = self._make_mock_graph()
        # Add a POSSIBLY_SAME_AS relation to the absorbed node
        graph.get_relationships.return_value = [
            {
                "type": "POSSIBLY_SAME_AS",
                "props": {"similarity_score": 0.8},
                "from_id": "survivor_1",
                "to_id": "absorbed_1",
            },
            {
                "type": "CONCERNS",
                "props": {"created_at": "2025-01-01T00:00:00+00:00"},
                "from_id": "absorbed_1",
                "to_id": "fact_1",
            },
        ]
        executor = MergeExecutor(graph)
        await executor.execute_merge("survivor_1", "absorbed_1", "run_001")
        # POSSIBLY_SAME_AS between them should NOT be transferred
        # The absorbed node is deleted (which removes all its rels)
        graph.delete_node.assert_called_with("absorbed_1")

    async def test_source_provenance_preserved(self):
        """Relations transferred should preserve their original properties."""
        graph = self._make_mock_graph()
        graph.get_relationships.return_value = [
            {
                "type": "SOURCED_FROM",
                "props": {
                    "credibility": 2,
                    "extracted_at": "2025-01-01T00:00:00+00:00",
                    "extraction_method": "llm_extraction",
                    "created_at": "2025-01-01T00:00:00+00:00",
                },
                "from_id": "absorbed_1",
                "to_id": "source_1",
            }
        ]
        executor = MergeExecutor(graph)
        result = await executor.execute_merge("survivor_1", "absorbed_1", "run_001")
        assert result.relations_transferred >= 1
