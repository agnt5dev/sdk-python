"""Configuration primitives for the core-backed Context."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from .function_registry import FunctionRegistry


@dataclass(frozen=True)
class ContextConfig:
    """Identifiers used to bootstrap a Context instance."""

    tenant_id: str
    session_id: str
    run_id: str
    step_id: str
    attempt: int = 0
    invocation_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)

    def with_invocation_id(self, invocation_id: str) -> "ContextConfig":
        return replace(self, invocation_id=invocation_id)

    def with_metadata(self, key: str, value: str) -> "ContextConfig":
        metadata = dict(self.metadata)
        metadata[key] = value
        return replace(self, metadata=metadata)

    def with_function_registry(self, registry: FunctionRegistry) -> "ContextConfig":
        return replace(self, function_registry=registry)
