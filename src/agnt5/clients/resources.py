"""Resource access client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from ..context import Context


class ResourcesClient:
    """Container for runtime-provided resource handles."""

    def __init__(self, providers: Mapping[str, Any]):
        self._providers = dict(providers)

    def __getattr__(self, name: str) -> Any:
        if name in self._providers:
            return self._providers[name]
        raise AttributeError(f"Resource '{name}' is not configured for this run")

    def items(self):  # pragma: no cover - convenience helper
        return self._providers.items()
