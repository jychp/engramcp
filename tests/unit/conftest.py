"""Unit test fixtures — FastMCP client and working memory cleanup."""

from __future__ import annotations

import pytest
from fastmcp import Client


@pytest.fixture()
async def mcp_client(redis_container):
    """Yield a FastMCP Client wired to the EngraMCP server."""
    from engramcp.server import configure
    from engramcp.server import mcp

    await configure(redis_url=redis_container)

    async with Client(mcp) as client:
        yield client


@pytest.fixture(autouse=True)
async def clean_working_memory(redis_container):
    """Reset working memory between tests."""
    from engramcp.server import _reset_working_memory
    from engramcp.server import configure

    await configure(redis_url=redis_container)
    await _reset_working_memory()
    yield
    await _reset_working_memory()
