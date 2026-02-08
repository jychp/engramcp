"""Unit tests for server consolidation flush behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import engramcp.server as server_module
from engramcp.engine.consolidation import ConsolidationRunResult
from engramcp.memory.schemas import MemoryFragment


class TestRunConsolidation:
    async def test_errors_without_mutation_raise_and_keep_fragments(self, monkeypatch):
        pipeline = AsyncMock()
        pipeline.run.return_value = ConsolidationRunResult(
            run_id="run-1",
            errors=["extraction failed"],
        )
        wm = AsyncMock()

        monkeypatch.setattr(server_module, "_consolidation_pipeline", pipeline)
        monkeypatch.setattr(server_module, "_wm", wm)

        fragments = [MemoryFragment(id="mem-1", content="fact")]

        with pytest.raises(RuntimeError, match="skipping fragment deletion for retry"):
            await server_module._run_consolidation(fragments)

        wm.delete.assert_not_awaited()

    async def test_errors_with_mutation_delete_fragments(self, monkeypatch):
        pipeline = AsyncMock()
        pipeline.run.return_value = ConsolidationRunResult(
            run_id="run-1",
            claims_created=1,
            errors=["relation unresolved"],
        )
        wm = AsyncMock()

        monkeypatch.setattr(server_module, "_consolidation_pipeline", pipeline)
        monkeypatch.setattr(server_module, "_wm", wm)

        fragments = [
            MemoryFragment(id="mem-1", content="fact one"),
            MemoryFragment(id="mem-2", content="fact two"),
        ]

        await server_module._run_consolidation(fragments)

        assert wm.delete.await_count == 2
        wm.delete.assert_any_await("mem-1")
        wm.delete.assert_any_await("mem-2")
