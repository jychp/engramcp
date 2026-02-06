"""Unit test fixtures — FastMCP client and working memory cleanup."""

from __future__ import annotations

import pytest
from fastmcp import Client


@pytest.fixture()
async def mcp_client():
    """Yield a FastMCP Client wired to the EngraMCP server."""
    from engramcp.server import mcp

    async with Client(mcp) as client:
        yield client


@pytest.fixture(autouse=True)
def clean_working_memory():
    """Reset the module-level working memory between tests."""
    from engramcp.server import _reset_working_memory

    _reset_working_memory()
    yield
    _reset_working_memory()
