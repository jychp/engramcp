"""MCP authentication helpers."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from collections.abc import Mapping

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth import TokenVerifier

logger = logging.getLogger(__name__)


class APIKeyVerifier(TokenVerifier):
    """Static bearer-token verifier for MCP requests."""

    def __init__(
        self,
        api_key: str,
        *,
        scopes: list[str] | None = None,
        claims: Mapping[str, object] | None = None,
    ) -> None:
        normalized = api_key.strip()
        if not normalized:
            raise ValueError("api_key must be a non-empty, non-whitespace string")
        super().__init__()
        self._api_key = normalized
        self._scopes = scopes[:] if scopes else ["engramcp:all"]
        self._claims = dict(claims or {})

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return an access token when the provided bearer token is valid."""
        if hmac.compare_digest(token, self._api_key):
            return AccessToken(
                token=token,
                client_id="engramcp-client",
                scopes=self._scopes,
                expires_at=None,
                claims=self._claims,
            )

        token_fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
        logger.debug(
            "Invalid MCP auth token provided (token_len=%d, token_fp=%s)",
            len(token),
            token_fingerprint,
        )
        return None


def get_mcp_auth_key() -> str | None:
    """Get the MCP static auth key from environment."""
    token = os.getenv("MCP_AUTH_KEY")
    if token is None:
        return None
    stripped = token.strip()
    return stripped if stripped else None


def get_mcp_auth_scopes() -> list[str]:
    """Get MCP auth scopes from ``MCP_AUTH_SCOPES`` env var."""
    raw = os.getenv("MCP_AUTH_SCOPES", "")
    parsed = [scope.strip() for scope in raw.split(",") if scope.strip()]
    return parsed if parsed else ["engramcp:all"]


def get_mcp_auth_role() -> str | None:
    """Get an optional role claim for the static MCP auth token."""
    role = os.getenv("MCP_AUTH_ROLE")
    if role is None:
        return None
    normalized = role.strip()
    return normalized if normalized else None


def create_mcp_auth() -> APIKeyVerifier | None:
    """Create an auth verifier when ``MCP_AUTH_KEY`` is configured."""
    api_key = get_mcp_auth_key()
    if api_key:
        claims: dict[str, object] = {}
        role = get_mcp_auth_role()
        if role is not None:
            claims["role"] = role
        return APIKeyVerifier(
            api_key,
            scopes=get_mcp_auth_scopes(),
            claims=claims,
        )
    return None
