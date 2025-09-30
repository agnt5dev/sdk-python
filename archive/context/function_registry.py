from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


@dataclass(slots=True)
class FunctionCall:
    target_service: str
    handler: str
    payload: Any
    key: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def with_key(self, key: str) -> "FunctionCall":
        self.key = key
        return self

    def with_metadata(self, key: str, value: str) -> "FunctionCall":
        self.metadata[key] = value
        return self


@dataclass(slots=True)
class InvocationContext:
    tenant_id: str
    session_id: str
    run_id: str
    step_id: str
    attempt: int
    parent_invocation_id: Optional[str]


FunctionCallable = Callable[[FunctionCall, InvocationContext], Awaitable[Any]]


class FunctionNotRegisteredError(LookupError):
    """Raised when attempting to invoke an unregistered function."""


class FunctionRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[Tuple[str, str], FunctionCallable] = {}

    def register(
        self,
        service: str,
        handler: str,
        callback: FunctionCallable,
    ) -> None:
        self._handlers[(service, handler)] = callback

    def unregister(self, service: str, handler: str) -> None:
        self._handlers.pop((service, handler), None)

    async def invoke(
        self,
        call: FunctionCall,
        ctx: InvocationContext,
    ) -> Any:
        try:
            callback = self._handlers[(call.target_service, call.handler)]
        except KeyError as exc:  # pragma: no cover - converted below
            raise FunctionNotRegisteredError(
                f"no function registered for {call.target_service}::{call.handler}"
            ) from exc

        return await callback(call, ctx)


__all__ = [
    "FunctionCall",
    "FunctionRegistry",
    "FunctionNotRegisteredError",
    "InvocationContext",
]
