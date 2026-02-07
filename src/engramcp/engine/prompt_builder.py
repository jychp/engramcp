"""Prompt construction for LLM-based extraction.

Builds a structured prompt from a list of ``MemoryFragment`` objects,
including the expected JSON output schema. Separate module because the
prompt will evolve independently when concept emergence enriches it.
"""

from __future__ import annotations

import json

from engramcp.engine.schemas import ExtractionResult
from engramcp.memory.schemas import MemoryFragment

# Node types from the ontology (used to guide the LLM)
ENTITY_TYPES = [
    "Agent",
    "Artifact",
    "Fact",
    "Event",
    "Observation",
    "Decision",
    "Outcome",
]

# Relation types from the ontology
RELATION_TYPES = [
    "SOURCED_FROM",
    "DERIVED_FROM",
    "CITES",
    "CAUSED_BY",
    "LEADS_TO",
    "PRECEDED",
    "FOLLOWED",
    "SUPPORTS",
    "CONTRADICTS",
    "PARTICIPATED_IN",
    "DECIDED_BY",
    "OBSERVED_BY",
    "MENTIONS",
    "CONCERNS",
    "GENERALIZES",
    "INSTANCE_OF",
    "POSSIBLY_SAME_AS",
]


def _format_fragment(fragment: MemoryFragment) -> str:
    """Format a single fragment for inclusion in the prompt."""
    lines = [
        f"--- Fragment {fragment.id} ---",
        f"Content: {fragment.content}",
        f"Type: {fragment.type}",
    ]
    if fragment.confidence:
        lines.append(f"Confidence: {fragment.confidence}")
    if fragment.participants:
        lines.append(
            f"Participants: {', '.join(str(p) for p in fragment.participants)}"
        )
    if fragment.sources:
        sources_str = "; ".join(json.dumps(s, default=str) for s in fragment.sources)
        lines.append(f"Sources: {sources_str}")
    lines.append("---")
    return "\n".join(lines)


def build_extraction_prompt(fragments: list[MemoryFragment]) -> str:
    """Build a structured extraction prompt from memory fragments.

    The prompt instructs the LLM to extract entities, relations, and
    claims from the given fragments and return them as JSON conforming
    to the ``ExtractionResult`` schema.
    """
    # 1. System instructions
    system = (
        "You are a knowledge extraction engine. Your task is to analyze "
        "the following memory fragments and extract structured knowledge "
        "from them.\n\n"
        "Extract three types of information:\n"
        "1. **Entities**: people, organizations, systems, documents, or "
        "other named things mentioned in the fragments.\n"
        "2. **Relations**: connections between entities (causal, temporal, "
        "epistemic, or participatory).\n"
        "3. **Claims**: factual assertions, events, observations, or "
        "decisions described in the fragments.\n"
    )

    # 2. Fragment data
    fragment_section = "\n\n".join(_format_fragment(f) for f in fragments)

    # 3. Available types
    entity_types_str = ", ".join(ENTITY_TYPES)
    relation_types_str = ", ".join(RELATION_TYPES)
    types_section = (
        f"Available entity types: {entity_types_str}\n"
        f"Available relation types: {relation_types_str}\n"
    )

    # 4. Output schema
    schema = json.dumps(ExtractionResult.model_json_schema(), indent=2)
    output_section = (
        "Return your response as a single JSON object conforming to "
        "this schema:\n\n"
        f"```json\n{schema}\n```\n\n"
        "Important:\n"
        "- For each entity, include a `source_fragment_ids` list with "
        "the IDs of fragments that mention it.\n"
        "- For each relation and claim, include `source_fragment_ids`.\n"
        "- Use entity names (not IDs) in `from_entity`, `to_entity`, "
        "and `involved_entities`.\n"
        "- If temporal information is available, include it in claims.\n"
        "- If an entity could be confused with another, add "
        "`disambiguating_context`.\n"
        "- Return ONLY the JSON object, no other text.\n"
    )

    return (
        f"{system}\n"
        f"## Fragments\n\n{fragment_section}\n\n"
        f"## Types\n\n{types_section}\n"
        f"## Output Format\n\n{output_section}"
    )
