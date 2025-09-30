"""Secrets management client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..context import Context


class SecretsClient:
    """Read-only access to secrets injected for the invocation."""

    def __init__(self, secrets: Mapping[str, Any]):
        self._secrets = dict(secrets)

    def get(self, name: str) -> str:
        if name not in self._secrets:
            raise KeyError(f"Secret '{name}' is not available")
        value = self._secrets[name]
        if not isinstance(value, str):
            raise TypeError(f"Secret '{name}' must be a string, got {type(value).__name__}")
        return value
