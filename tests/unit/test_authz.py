"""Unit tests for MCP authorization policy helpers."""

from __future__ import annotations

from fastmcp.server.auth import AccessToken

from engramcp.authz import authorize_correction_action
from engramcp.authz import authorize_tool
from engramcp.models.schemas import CorrectionAction


def _token(
    scopes: list[str],
    *,
    claims: dict[str, object] | None = None,
) -> AccessToken:
    return AccessToken(
        token="tok",
        client_id="client",
        scopes=scopes,
        expires_at=None,
        claims=claims or {},
    )


class TestAuthorizationToggle:
    def test_allows_when_authz_disabled(self, monkeypatch) -> None:
        monkeypatch.delenv("MCP_AUTHZ_ENABLED", raising=False)

        decision = authorize_tool("send_memory", token=None)
        assert decision.allowed is True


class TestToolAuthorization:
    def test_denies_send_memory_without_token_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")

        decision = authorize_tool("send_memory", token=None)
        assert decision.allowed is False
        assert decision.error_code == "forbidden"

    def test_allows_send_memory_with_write_scope(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token(["engramcp:memory:write"])

        decision = authorize_tool("send_memory", token=token)
        assert decision.allowed is True

    def test_allows_get_memory_via_viewer_role(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token([], claims={"role": "viewer"})

        decision = authorize_tool("get_memory", token=token)
        assert decision.allowed is True

    def test_allows_all_tools_with_wildcard_scope(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token(["engramcp:all"])

        decision = authorize_tool("correct_memory", token=token)
        assert decision.allowed is True

    def test_allows_unmapped_tool_when_requirements_are_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token([])

        decision = authorize_tool("future_tool", token=token)
        assert decision.allowed is True


class TestCorrectionAuthorization:
    def test_denies_split_for_editor_role(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token([], claims={"role": "editor"})

        decision = authorize_correction_action(CorrectionAction.split_entity, token)
        assert decision.allowed is False
        assert decision.error_code == "forbidden"

    def test_allows_split_for_admin_role(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token([], claims={"roles": ["admin"]})

        decision = authorize_correction_action(CorrectionAction.split_entity, token)
        assert decision.allowed is True

    def test_allows_contest_for_editor_role(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_AUTHZ_ENABLED", "1")
        token = _token([], claims={"role": "editor"})

        decision = authorize_correction_action(CorrectionAction.contest, token)
        assert decision.allowed is True
