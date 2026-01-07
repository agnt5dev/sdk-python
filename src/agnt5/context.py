"""Context implementation for AGNT5 SDK."""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Optional,
)

from ._telemetry import ContextLogger
from .events import Event, EventEmitter, EventEnvelope

if TYPE_CHECKING:
    from .memoization import MemoizationManager


# Task-local storage (NOT global) - each asyncio task gets its own copy
_current_context: contextvars.ContextVar[Optional["Context"]] = contextvars.ContextVar(
    "_current_context", default=None
)


class _CorrelationFilter(logging.Filter):
    """Inject correlation IDs (run_id, trace_id, span_id) into log records."""

    def __init__(self, runtime_context: Any) -> None:
        super().__init__()
        self.runtime_context = runtime_context

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.runtime_context.run_id
        if self.runtime_context.trace_id:
            record.trace_id = self.runtime_context.trace_id
        if self.runtime_context.span_id:
            record.span_id = self.runtime_context.span_id
        return True


class Context:
    """Base context providing logging, event emission, and execution metadata.

    Extended by FunctionContext, WorkflowContext, and AgentContext.
    """

    def __init__(
        self,
        run_id: str,
        correlation_id: str,
        parent_correlation_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        session_id: Optional[str] = None,
        enable_memoization: bool = False,
        is_streaming: bool = False,
        worker: Optional[Any] = None,
    ) -> None:
        self._run_id = run_id
        self._attempt = attempt
        self._runtime_context = runtime_context
        self._session_id = session_id
        self._is_streaming = is_streaming
        self._worker = worker

        # Correlation tracking for event hierarchy (required, never null)
        self._correlation_id: str = correlation_id
        self._parent_correlation_id: str = parent_correlation_id
        self._component_name: Optional[str] = None

        self._emitter: Optional[EventEmitter] = None

        if enable_memoization:
            from .memoization import MemoizationManager

            self._memo: Optional["MemoizationManager"] = MemoizationManager(self)
        else:
            self._memo = None

        base_logger = logging.getLogger(f"agnt5.{run_id}")
        from ._telemetry import setup_context_logger

        setup_context_logger(base_logger)
        if runtime_context:
            base_logger.addFilter(_CorrelationFilter(runtime_context))
        self._logger = ContextLogger(base_logger)

    @property
    def run_id(self) -> str:
        """Unique execution identifier."""
        return self._run_id

    @property
    def attempt(self) -> int:
        """Current retry attempt (0-indexed)."""
        return self._attempt

    @property
    def logger(self) -> ContextLogger:
        """Logger with correlation IDs. Supports keyword args as log attributes."""
        return self._logger

    @property
    def session_id(self) -> Optional[str]:
        """Session identifier for multi-turn conversations."""
        return self._session_id

    @property
    def correlation_id(self) -> str:
        """Current correlation ID for event hierarchy."""
        return self._correlation_id

    @correlation_id.setter
    def correlation_id(self, value: str) -> None:
        """Set the current correlation ID."""
        self._correlation_id = value

    @property
    def parent_correlation_id(self) -> str:
        """Parent correlation ID for event hierarchy."""
        return self._parent_correlation_id

    @parent_correlation_id.setter
    def parent_correlation_id(self, value: str) -> None:
        """Set the parent correlation ID."""
        self._parent_correlation_id = value

    @property
    def component_name(self) -> Optional[str]:
        """Component name for events."""
        return self._component_name

    @component_name.setter
    def component_name(self, value: str) -> None:
        """Set the component name."""
        self._component_name = value

    def _get_emitter(self) -> EventEmitter:
        """Get or create the event emitter (lazy initialization)."""
        if self._emitter is None:
            self._emitter = EventEmitter(run_id=self._run_id)
            if self._worker is not None:
                self._emitter.set_worker(self._worker)
                logging.getLogger(__name__).debug(
                    f"[Context._get_emitter] EventEmitter created with worker for run_id={self._run_id}"
                )
            else:
                logging.getLogger(__name__).warning(
                    f"[Context._get_emitter] EventEmitter created WITHOUT worker for run_id={self._run_id}"
                )
        return self._emitter

    def emit(self, event: Event) -> EventEnvelope:
        """Emit a typed event.

        The event already contains correlation_id and parent_correlation_id.
        """
        logging.getLogger(__name__).info(
            f"[Context.emit] Emitting event: type={event.event_type}, "
            f"run_id={self._run_id}, correlation_id={event.correlation_id}"
        )
        return self._get_emitter().emit(event)

    @contextmanager
    def as_parent(self) -> Generator[None, None, None]:
        """Set this context's correlation_id as parent for nested component events."""
        old_parent = self._parent_correlation_id
        self._parent_correlation_id = self._correlation_id
        try:
            yield
        finally:
            self._parent_correlation_id = old_parent

    def get_event_context(self) -> dict[str, str]:
        """Get correlation_id and parent_correlation_id for event hierarchy."""
        return {
            "correlation_id": self._correlation_id,
            "parent_correlation_id": self._parent_correlation_id,
        }

    def get_event_metadata(self) -> dict[str, str]:
        """Get metadata fields for events. Subclasses can override."""
        meta: dict[str, str] = {}
        if self._component_name:
            meta["name"] = self._component_name
        return meta

    def set_as_parent(self, correlation_id: str) -> str:
        """Set correlation_id as parent. Returns previous parent for restoration."""
        original_parent = self._parent_correlation_id
        self._correlation_id = correlation_id
        self._parent_correlation_id = correlation_id
        return original_parent

    def restore_parent(self, original_parent: str) -> None:
        """Restore the parent correlation ID to a previous value."""
        self._parent_correlation_id = original_parent


def get_current_context() -> Optional[Context]:
    """Get the current execution context from task-local storage."""
    return _current_context.get()


def set_current_context(ctx: Context) -> contextvars.Token:
    """Set the current context. Returns token for reset via _current_context.reset(token)."""
    return _current_context.set(ctx)
