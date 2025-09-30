"""Core-backed Context scaffolding.

The current implementation provides an in-memory execution model so SDK users can
exercise the Context API without a runtime connection. Once the durable runtime
integration lands, these namespaces will delegate to the Rust core.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .config import ContextConfig
from .function_registry import (
    FunctionCall,
    FunctionNotRegisteredError,
    FunctionRegistry,
    InvocationContext,
)


class ContextNotReadyError(RuntimeError):
    """Raised by placeholder namespaces until the core integration lands."""


@dataclass
class Context:
    config: ContextConfig
    runtime_handle: Optional[Any] = None

    @property
    def function_registry(self) -> FunctionRegistry:
        return self.config.function_registry

    def functions(self) -> "FunctionNamespace":
        return FunctionNamespace(self)

    def signals(self) -> "SignalNamespace":
        return SignalNamespace(self)

    def timers(self) -> "TimerNamespace":
        return TimerNamespace(self)

    def language_model(self) -> "LanguageModelNamespace":
        return LanguageModelNamespace(self)


@dataclass
class _BaseNamespace:
    context: Context

    def _not_ready(self, capability: str) -> ContextNotReadyError:
        return ContextNotReadyError(
            f"{capability} is not yet wired to the Rust core runtime"
        )


class FunctionStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class FunctionResult:
    request: FunctionCall
    invocation_id: str
    status: FunctionStatus
    output: Optional[Any]
    error: Optional[str]


@dataclass
class FunctionHandle:
    _result: FunctionResult

    async def result(self) -> FunctionResult:
        return self._result

    def status(self) -> FunctionStatus:
        return self._result.status

    @classmethod
    def succeeded(cls, call: FunctionCall, invocation_id: str, output: Any) -> "FunctionHandle":
        return cls(
            FunctionResult(
                request=call,
                invocation_id=invocation_id,
                status=FunctionStatus.SUCCEEDED,
                output=output,
                error=None,
            )
        )

    @classmethod
    def failed(cls, call: FunctionCall, invocation_id: str, error: str) -> "FunctionHandle":
        return cls(
            FunctionResult(
                request=call,
                invocation_id=invocation_id,
                status=FunctionStatus.FAILED,
                output=None,
                error=error,
            )
        )


class FunctionNamespace(_BaseNamespace):
    async def call(self, call: FunctionCall) -> FunctionHandle:
        invocation_id = uuid.uuid4().hex
        invocation_ctx = InvocationContext(
            tenant_id=self.context.config.tenant_id,
            session_id=self.context.config.session_id,
            run_id=self.context.config.run_id,
            step_id=self.context.config.step_id,
            attempt=self.context.config.attempt,
            parent_invocation_id=self.context.config.invocation_id,
        )

        try:
            output = await self.context.function_registry.invoke(call, invocation_ctx)
        except FunctionNotRegisteredError as exc:
            raise ValueError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            return FunctionHandle.failed(call, invocation_id, str(exc))

        return FunctionHandle.succeeded(call, invocation_id, output)


class SignalNamespace(_BaseNamespace):
    async def wait(self, *args: Any, **kwargs: Any) -> Any:
        raise self._not_ready("Signal coordination")

    async def emit(self, *args: Any, **kwargs: Any) -> None:
        raise self._not_ready("Signal coordination")


class TimerNamespace(_BaseNamespace):
    async def sleep(self, *args: Any, **kwargs: Any) -> None:
        raise self._not_ready("Durable timers")


class LanguageModelNamespace(_BaseNamespace):
    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self._not_ready("Language model integration")


__all__ = [
    "Context",
    "ContextNotReadyError",
    "FunctionHandle",
    "FunctionNamespace",
    "FunctionResult",
    "FunctionStatus",
    "LanguageModelNamespace",
    "SignalNamespace",
    "TimerNamespace",
]
