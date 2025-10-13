"""Context implementation for AGNT5 SDK."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")


class _CorrelationFilter(logging.Filter):
    """Inject correlation IDs (run_id, trace_id, span_id) into every log record."""

    def __init__(self, runtime_context: Any) -> None:
        super().__init__()
        self.runtime_context = runtime_context

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation IDs as extra fields to the log record."""
        record.run_id = self.runtime_context.run_id
        if self.runtime_context.trace_id:
            record.trace_id = self.runtime_context.trace_id
        if self.runtime_context.span_id:
            record.span_id = self.runtime_context.span_id
        return True


class Context:
    """
    Base context providing common functionality.

    Provides:
    - Logging with correlation IDs
    - Execution metadata (run_id, attempt)
    - Runtime context for tracing

    Extended by:
    - FunctionContext: Minimal context for stateless functions
    - WorkflowContext: Context for durable workflows
    """

    def __init__(
        self,
        run_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
    ) -> None:
        """
        Initialize base context.

        Args:
            run_id: Unique execution identifier
            attempt: Retry attempt number (0-indexed)
            runtime_context: RuntimeContext for trace correlation
        """
        self._run_id = run_id
        self._attempt = attempt
        self._runtime_context = runtime_context

        # Create logger with correlation
        self._logger = logging.getLogger(f"agnt5.{run_id}")
        from ._telemetry import setup_context_logger
        setup_context_logger(self._logger)

        if runtime_context:
            self._logger.addFilter(_CorrelationFilter(runtime_context))

    @property
    def run_id(self) -> str:
        """Unique execution identifier."""
        return self._run_id

    @property
    def attempt(self) -> int:
        """Current retry attempt (0-indexed)."""
        return self._attempt

    @property
    def logger(self) -> logging.Logger:
        """Full logger for .debug(), .warning(), .error(), etc."""
        return self._logger



