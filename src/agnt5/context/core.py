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
class CoreContext:
    config: ContextConfig
    runtime_handle: Optional[Any] = None

    def tasks(self) -> "TaskNamespace":
        return TaskNamespace(self)

    def signals(self) -> "SignalNamespace":
        return SignalNamespace(self)

    def timers(self) -> "TimerNamespace":
        return TimerNamespace(self)

    def llm(self) -> "LlmNamespace":
        return LlmNamespace(self)


@dataclass
class _BaseNamespace:
    context: CoreContext

    def _not_ready(self, capability: str) -> ContextNotReadyError:
        return ContextNotReadyError(
            f"{capability} is not yet wired to the Rust core runtime"
        )


class TaskNamespace(_BaseNamespace):
    async def call(self, *args: Any, **kwargs: Any) -> Any:
        raise self._not_ready("Task orchestration")


class SignalNamespace(_BaseNamespace):
    async def wait(self, *args: Any, **kwargs: Any) -> Any:
        raise self._not_ready("Signal coordination")

    async def emit(self, *args: Any, **kwargs: Any) -> None:
        raise self._not_ready("Signal coordination")


class TimerNamespace(_BaseNamespace):
    async def sleep(self, *args: Any, **kwargs: Any) -> None:
        raise self._not_ready("Durable timers")


class LlmNamespace(_BaseNamespace):
    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self._not_ready("LLM integration")
