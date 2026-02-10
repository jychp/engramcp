"""Unit tests for MCP auth helpers."""

from __future__ import annotations

import pytest

from engramcp.auth import APIKeyVerifier
from engramcp.auth import create_mcp_auth
from engramcp.auth import get_mcp_auth_key


class TestAPIKeyVerifier:
    def test_rejects_empty_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key must be a non-empty"):
            APIKeyVerifier("")

    def test_rejects_blank_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key must be a non-empty"):
            APIKeyVerifier("   ")

    async def test_verify_valid_token(self) -> None:
        verifier = APIKeyVerifier("my-secret-key")

        result = await verifier.verify_token("my-secret-key")

        assert result is not None
        assert result.token == "my-secret-key"
        assert result.client_id == "engramcp-client"
        assert result.scopes == ["engramcp:all"]
        assert result.expires_at is None

    async def test_verify_invalid_token(self) -> None:
        verifier = APIKeyVerifier("my-secret-key")

        result = await verifier.verify_token("wrong-key")

        assert result is None

    async def test_verify_empty_token(self) -> None:
        verifier = APIKeyVerifier("my-secret-key")

        result = await verifier.verify_token("")

        assert result is None


class TestGetMcpAuthKey:
    def test_returns_env_value(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTH_KEY", "env-auth-key")

        assert get_mcp_auth_key() == "env-auth-key"

    def test_returns_none_when_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("MCP_AUTH_KEY", raising=False)

        assert get_mcp_auth_key() is None

    def test_returns_none_when_blank(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTH_KEY", "   ")

        assert get_mcp_auth_key() is None


class TestCreateMcpAuth:
    def test_returns_verifier_when_key_present(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTH_KEY", "test-auth-key")

        verifier = create_mcp_auth()

        assert verifier is not None
        assert isinstance(verifier, APIKeyVerifier)

    def test_returns_none_when_key_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("MCP_AUTH_KEY", raising=False)

        assert create_mcp_auth() is None
