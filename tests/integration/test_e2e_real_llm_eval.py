"""Optional end-to-end evals with a real OpenAI-compatible LLM.

These tests are intentionally opt-in because they require network access,
credentials, and can incur provider costs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from fastmcp import Client

from engramcp.config import AuditConfig
from engramcp.config import ConsolidationConfig
from engramcp.config import LLMConfig
from engramcp.server import _get_wm
from engramcp.server import configure
from engramcp.server import mcp
from engramcp.server import shutdown

_RUN_REAL_EVALS = os.getenv("ENGRAMCP_RUN_REAL_LLM_EVALS") == "1"
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pytestmark = pytest.mark.skipif(
    not (_RUN_REAL_EVALS and _OPENAI_API_KEY),
    reason=(
        "Real-LLM evals are opt-in. Set ENGRAMCP_RUN_REAL_LLM_EVALS=1 and "
        "OPENAI_API_KEY to enable."
    ),
)


def _parse(result) -> dict:
    return json.loads(result.content[0].text)


async def _wait_until(predicate, *, timeout: float = 20.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.fixture(autouse=True)
async def _setup_server(redis_container, neo4j_container):
    await configure(
        redis_url=redis_container,
        enable_consolidation=True,
        neo4j_url=neo4j_container,
        llm_config=LLMConfig(
            provider="openai",
            model=os.getenv("ENGRAMCP_EVAL_OPENAI_MODEL", "gpt-4o-mini"),
            api_key=_OPENAI_API_KEY,
            temperature=0.0,
            timeout_seconds=60.0,
        ),
        consolidation_config=ConsolidationConfig(
            fragment_threshold=1,
            extraction_max_retries=2,
            extraction_retry_backoff_seconds=0.2,
        ),
        audit_config=AuditConfig(enabled=False),
    )
    yield
    await shutdown()


class TestRealLLME2EEvals:
    async def test_send_consolidate_get_retrieves_relevant_memory(self):
        async with Client(mcp) as client:
            await client.call_tool(
                "send_memory",
                {"content": "The Eiffel Tower is located in Paris."},
            )
            await client.call_tool(
                "send_memory",
                {"content": "The Eiffel Tower opened to the public in 1889."},
            )
            await client.call_tool(
                "send_memory",
                {"content": "Mount Fuji is in Japan."},
            )

            wm = _get_wm()

            async def _wm_empty() -> bool:
                return await wm.count() == 0

            assert await _wait_until(_wm_empty)

            result = await client.call_tool("get_memory", {"query": "Eiffel Tower"})
            data = _parse(result)

        assert data["meta"]["working_memory_hits"] == 0
        assert data["meta"]["graph_hits"] >= 1
        assert any("Eiffel Tower" in memory["content"] for memory in data["memories"])

    async def test_send_consolidate_get_surfaces_conflicting_claims(self):
        async with Client(mcp) as client:
            await client.call_tool(
                "send_memory",
                {"content": "The meeting happened on March 15, 2025."},
            )
            await client.call_tool(
                "send_memory",
                {"content": "The meeting happened on March 17, 2025."},
            )

            wm = _get_wm()

            async def _wm_empty() -> bool:
                return await wm.count() == 0

            assert await _wait_until(_wm_empty)

            result = await client.call_tool("get_memory", {"query": "2025"})
            data = _parse(result)

        assert data["meta"]["graph_hits"] >= 1
        assert len(data["memories"]) >= 1
        assert (
            len(data["contradictions"]) >= 1
            or sum("2025" in memory["content"] for memory in data["memories"]) >= 1
        )
