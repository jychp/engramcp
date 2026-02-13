"""MCP authorization policy helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fastmcp.server.auth import AccessToken

from engramcp.models.schemas import CorrectionAction

_ROLE_SCOPES: dict[str, set[str]] = {
    "viewer": {"engramcp:memory:read"},
    "editor": {
        "engramcp:memory:read",
        "engramcp:memory:write",
        "engramcp:memory:correct",
    },
    "admin": {
        "engramcp:memory:read",
        "engramcp:memory:write",
        "engramcp:memory:correct",
        "engramcp:memory:admin",
    },
}

_TOOL_REQUIRED_SCOPES: dict[str, set[str]] = {
    "send_memory": {"engramcp:memory:write"},
    "get_memory": {"engramcp:memory:read"},
    "correct_memory": {"engramcp:memory:correct", "engramcp:memory:admin"},
}

_CORRECTION_REQUIRED_SCOPES: dict[CorrectionAction, set[str]] = {
    CorrectionAction.contest: {"engramcp:memory:correct"},
    CorrectionAction.annotate: {"engramcp:memory:correct"},
    CorrectionAction.reclassify: {"engramcp:memory:correct"},
    CorrectionAction.merge_entities: {"engramcp:memory:admin"},
    CorrectionAction.split_entity: {"engramcp:memory:admin"},
}

_WILDCARD_SCOPE = "engramcp:all"
_AUTHZ_ENV = "MCP_AUTHZ_ENABLED"


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    error_code: str | None = None
    message: str | None = None


def is_authorization_enabled() -> bool:
    """Return whether MCP authorization checks are enabled."""
    raw = os.getenv(_AUTHZ_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _extract_roles(claims: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    role = claims.get("role")
    if isinstance(role, str) and role.strip():
        roles.add(role.strip())

    role_list = claims.get("roles")
    if isinstance(role_list, list):
        for item in role_list:
            if isinstance(item, str) and item.strip():
                roles.add(item.strip())

    return roles


def _effective_scopes(token: AccessToken | None) -> set[str]:
    if token is None:
        return set()

    scopes = {scope.strip() for scope in token.scopes if scope.strip()}
    if _WILDCARD_SCOPE in scopes:
        return {_WILDCARD_SCOPE}

    claims = token.claims if isinstance(token.claims, dict) else {}
    for role in _extract_roles(claims):
        scopes.update(_ROLE_SCOPES.get(role, set()))
    return scopes


def _has_required_scope(
    token_scopes: set[str],
    required_scopes: set[str],
) -> bool:
    if _WILDCARD_SCOPE in token_scopes:
        return True
    return bool(token_scopes.intersection(required_scopes))


def authorize_tool(
    tool_name: str,
    token: AccessToken | None,
) -> AuthorizationDecision:
    """Authorize access to a top-level MCP tool."""
    if not is_authorization_enabled():
        return AuthorizationDecision(allowed=True)

    required_scopes = _TOOL_REQUIRED_SCOPES.get(tool_name, set())
    token_scopes = _effective_scopes(token)

    if not _has_required_scope(token_scopes, required_scopes):
        required = ", ".join(sorted(required_scopes))
        return AuthorizationDecision(
            allowed=False,
            error_code="forbidden",
            message=f"Insufficient scope for {tool_name}. Required one of: {required}.",
        )
    return AuthorizationDecision(allowed=True)


def authorize_correction_action(
    action: CorrectionAction,
    token: AccessToken | None,
) -> AuthorizationDecision:
    """Authorize a specific ``correct_memory`` action."""
    if not is_authorization_enabled():
        return AuthorizationDecision(allowed=True)

    required_scopes = _CORRECTION_REQUIRED_SCOPES.get(action, set())
    token_scopes = _effective_scopes(token)
    if not _has_required_scope(token_scopes, required_scopes):
        required = ", ".join(sorted(required_scopes))
        return AuthorizationDecision(
            allowed=False,
            error_code="forbidden",
            message=(
                "Insufficient scope for correct_memory action "
                f"{action.value}. Required one of: {required}."
            ),
        )
    return AuthorizationDecision(allowed=True)
