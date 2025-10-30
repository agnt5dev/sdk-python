"""Context implementation for AGNT5 SDK."""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")

# Task-local context variable for automatic context propagation
# This is NOT a global variable - contextvars provide task-isolated storage
# Each asyncio task gets its own independent copy, preventing cross-contamination
_current_context: contextvars.ContextVar[Optional["Context"]] = contextvars.ContextVar(
    "_current_context", default=None
)


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


def get_current_context() -> Optional[Context]:
    """
    Get the current execution context from task-local storage.

    This function retrieves the context that was set by the nearest enclosing
    decorator (@function, @workflow) or Agent.run() call in the current asyncio task.

    Returns:
        Current Context if available (WorkflowContext, FunctionContext, AgentContext),
        None if no context is set (e.g., running outside AGNT5 execution)

    Example:
        >>> ctx = get_current_context()
        >>> if ctx:
        ...     ctx.logger.info("Logging from anywhere in the call stack!")
        ...     runtime = ctx._runtime_context  # Access tracing context

    Note:
        This uses Python's contextvars which provide task-local (NOT global) storage.
        Each asyncio task has its own isolated context, preventing cross-contamination
        between concurrent executions.
    """
    return _current_context.get()


def set_current_context(ctx: Context) -> contextvars.Token:
    """
    Set the current execution context in task-local storage.

    This is typically called by decorators and framework code, not by user code.
    Returns a token that can be used to reset the context to its previous value.

    Args:
        ctx: Context to set as current

    Returns:
        Token for resetting the context later (use with contextvars.Token.reset())

    Example:
        >>> token = set_current_context(my_context)
        >>> try:
        ...     # Context is available via get_current_context()
        ...     do_work()
        >>> finally:
        ...     _current_context.reset(token)  # Restore previous context

    Note:
        Always use try/finally to ensure context is properly reset, even if
        an exception occurs. This prevents context leakage between executions.
    """
    return _current_context.set(ctx)


def get_workflow_context() -> Optional["WorkflowContext"]:
    """
    Get the WorkflowContext from the current context or its parent chain.

    This function traverses the context hierarchy to find a WorkflowContext,
    which is needed for emitting workflow checkpoints from nested contexts
    like AgentContext or FunctionContext.

    Returns:
        WorkflowContext if found in the context chain, None otherwise

    Example:
        >>> # Inside an agent called from a workflow
        >>> workflow_ctx = get_workflow_context()
        >>> if workflow_ctx:
        ...     workflow_ctx._send_checkpoint("workflow.lm.started", {...})
    """
    from .workflow import WorkflowContext

    ctx = get_current_context()

    # Traverse up the context chain looking for WorkflowContext
    while ctx is not None:
        if isinstance(ctx, WorkflowContext):
            return ctx
        # Check if this context has a parent_context attribute
        if hasattr(ctx, 'parent_context'):
            ctx = ctx.parent_context
        else:
            break

    return None


