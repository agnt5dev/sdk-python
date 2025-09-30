"""Core-backed Context scaffolding.

The implementation is intentionally minimal for now. It provides structure and
placeholders that will be backed by the Rust runtime bridge in later steps of the
rollout. Callers can begin integrating against the async surface while the heavy
lifting lives in `sdk-core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import ContextConfig


class ContextNotReadyError(RuntimeError):
    """Raised by placeholder namespaces until the core integration lands."""


@dataclass
class Context:
    config: ContextConfig
    runtime_handle: Optional[Any] = None

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


class FunctionNamespace(_BaseNamespace):
    async def call(self, *args: Any, **kwargs: Any) -> Any:
        raise self._not_ready("Function orchestration")


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
