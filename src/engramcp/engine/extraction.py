"""LLM extraction adapter — Layer 4 (partial).

Takes raw ``MemoryFragment`` objects and uses an LLM (via the
``LLMAdapter`` protocol) to extract structured entities, relations,
and claims.
"""

from __future__ import annotations

import json
import re
from typing import Protocol
from typing import runtime_checkable

from engramcp.config import ConsolidationConfig
from engramcp.config import LLMConfig
from engramcp.engine.prompt_builder import build_extraction_prompt
from engramcp.engine.schemas import ExtractionResult
from engramcp.memory.schemas import MemoryFragment

# ---------------------------------------------------------------------------
# LLM abstraction
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for LLM provider adapters.

    Concrete implementations (OpenAI, Anthropic, etc.) are added when
    wiring real providers. Tests use a ``MockLLMAdapter``.
    """

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
    ) -> str: ...


class LLMError(Exception):
    """Raised by LLM adapters when a call fails."""


# ---------------------------------------------------------------------------
# Extraction engine
# ---------------------------------------------------------------------------

# Regex to strip Markdown code fences wrapping JSON output
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)


class ExtractionEngine:
    """Orchestrates LLM-based extraction from memory fragments."""

    def __init__(
        self,
        llm: LLMAdapter,
        llm_config: LLMConfig | None = None,
        consolidation_config: ConsolidationConfig | None = None,
    ) -> None:
        self._llm = llm
        self._llm_config = llm_config or LLMConfig()
        self._consolidation_config = consolidation_config or ConsolidationConfig()

    async def extract(self, fragments: list[MemoryFragment]) -> ExtractionResult:
        """Extract entities, relations, and claims from *fragments*.

        Splits fragments into sub-batches of
        ``consolidation_config.extraction_batch_size``, processes each
        sequentially, and merges the results. Failed batches record an
        error but do not abort the pipeline.
        """
        if not fragments:
            return ExtractionResult()

        batch_size = self._consolidation_config.extraction_batch_size
        batches = [
            fragments[i : i + batch_size] for i in range(0, len(fragments), batch_size)
        ]

        results: list[ExtractionResult] = []
        for batch in batches:
            result = await self._extract_batch(batch)
            results.append(result)

        return self._merge_results(results)

    async def _extract_batch(self, fragments: list[MemoryFragment]) -> ExtractionResult:
        """Extract from a single batch of fragments."""
        prompt = build_extraction_prompt(fragments)
        fragment_ids = [f.id for f in fragments]

        try:
            raw = await self._llm.complete(
                prompt,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
                timeout_seconds=self._llm_config.timeout_seconds,
            )
        except LLMError as exc:
            return ExtractionResult(
                fragment_ids_processed=fragment_ids,
                errors=[f"LLM call failed: {exc}"],
            )

        result = self._parse_llm_output(raw)
        # Ensure fragment IDs are tracked even if the LLM omits them
        result.fragment_ids_processed = list(
            dict.fromkeys(result.fragment_ids_processed + fragment_ids)
        )
        return result

    @staticmethod
    def _parse_llm_output(raw: str) -> ExtractionResult:
        """Parse raw LLM output into an ``ExtractionResult``.

        Handles code fences and invalid JSON gracefully.
        """
        text = raw.strip()

        # Strip code fences (```json ... ```)
        match = _CODE_FENCE_RE.match(text)
        if match:
            text = match.group(1).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return ExtractionResult(
                errors=[f"Invalid JSON from LLM: {exc}"],
            )

        try:
            return ExtractionResult.model_validate(data)
        except Exception as exc:
            return ExtractionResult(
                errors=[f"Schema validation failed: {exc}"],
            )

    @staticmethod
    def _merge_results(
        results: list[ExtractionResult],
    ) -> ExtractionResult:
        """Merge multiple batch results into a single result."""
        entities = []
        relations = []
        claims = []
        fragment_ids: list[str] = []
        errors: list[str] = []

        for r in results:
            entities.extend(r.entities)
            relations.extend(r.relations)
            claims.extend(r.claims)
            fragment_ids.extend(r.fragment_ids_processed)
            errors.extend(r.errors)

        return ExtractionResult(
            entities=entities,
            relations=relations,
            claims=claims,
            fragment_ids_processed=list(dict.fromkeys(fragment_ids)),
            errors=errors,
        )
