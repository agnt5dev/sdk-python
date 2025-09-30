"""Public entrypoint for the core-first Context implementation."""

from __future__ import annotations

from typing import Any, Optional

from .config import ContextConfig
from .core import (
    Context,
    ContextNotReadyError,
    FunctionHandle,
    FunctionNamespace,
    FunctionResult,
    FunctionStatus,
    LanguageModelNamespace,
    SignalNamespace,
    TimerNamespace,
)
from .function_registry import FunctionCall, FunctionRegistry

__all__ = [
    "Context",
    "ContextConfig",
    "ContextNotReadyError",
    "FunctionCall",
    "FunctionHandle",
    "FunctionNamespace",
    "FunctionRegistry",
    "FunctionResult",
    "FunctionStatus",
    "LanguageModelNamespace",
    "SignalNamespace",
    "TimerNamespace",
    "create_context",
]


def create_context(
    tenant_id: str,
    session_id: str,
    run_id: str,
    step_id: str,
    *,
    attempt: int = 0,
    invocation_id: Optional[str] = None,
    metadata: Optional[dict[str, str]] = None,
    function_registry: Optional[FunctionRegistry] = None,
    runtime_handle: Any = None,
) -> Context:
    """Convenience helper for constructing a core Context."""

    config = ContextConfig(
        tenant_id=tenant_id,
        session_id=session_id,
        run_id=run_id,
        step_id=step_id,
        attempt=attempt,
        invocation_id=invocation_id,
        metadata=metadata or {},
        function_registry=function_registry or FunctionRegistry(),
    )
    return Context(config=config, runtime_handle=runtime_handle)
