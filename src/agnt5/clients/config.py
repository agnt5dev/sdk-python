"""Configuration access client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from ..context import Context


class ConfigClient:
    """Configuration facade exposing feature flags and variants."""

    def __init__(self, config: Mapping[str, Any]):
        raw = dict(config)
        variants = raw.get("variants")
        if isinstance(variants, Mapping):
            self._variants = dict(variants)
        else:
            self._variants = {}
        self._values = raw

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def variant(self, key: str, default: Any = None) -> Any:
        return self._variants.get(key, default)


__all__ = [
    "AgentCallResult",
    "AgentClient",
    "Context",
    "ConfigClient",
    "EvalClient",
    "MemoryClient",
    "MemoryStateTransition",
    "MemoryStateUpdate",
    "MetricsClient",
    "ResourcesClient",
    "SignalClient",
    "HumanClient",
    "ApprovalResult",
    "SecretsClient",
    "TimerClient",
    "SpanContext",
    "StepCheckpoint",
    "ToolClient",
]
