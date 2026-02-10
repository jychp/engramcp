"""MCP authentication helpers."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth import TokenVerifier

logger = logging.getLogger(__name__)


class APIKeyVerifier(TokenVerifier):
    """Static bearer-token verifier for MCP requests."""

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return an access token when the provided bearer token is valid."""
        if hmac.compare_digest(token, self._api_key):
            return AccessToken(
                token=token,
                client_id="engramcp-client",
                scopes=["engramcp:all"],
                expires_at=None,
                claims={},
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


def create_mcp_auth() -> APIKeyVerifier | None:
    """Create an auth verifier when ``MCP_AUTH_KEY`` is configured."""
    api_key = get_mcp_auth_key()
    if api_key:
        return APIKeyVerifier(api_key)
    return None
