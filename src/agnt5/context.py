"""Context implementation for AGNT5 SDK."""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union, TYPE_CHECKING

from ._telemetry import ContextLogger
from .emit import EventEmitter, JournalEvent

if TYPE_CHECKING:
    from .memoization import MemoizationManager
    from .emit import CheckpointCallback, DeltaCallback

T = TypeVar("T")

# Task-local context variable for automatic context propagation
# This is NOT a global variable - contextvars provide task-isolated storage
# Each asyncio task gets its own independent copy, preventing cross-contamination
_current_context: contextvars.ContextVar[Optional["Context"]] = contextvars.ContextVar(
    "_current_context", default=None
)


class _CorrelationFilter(logging.Filter):
    """Inject correlation IDs (run_id, trace_id, span_id) and streaming context into every log record."""

    def __init__(
        self,
        runtime_context: Any,
        is_streaming: bool = False,
        tenant_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.runtime_context = runtime_context
        self.is_streaming = is_streaming
        self.tenant_id = tenant_id
        self.deployment_id = deployment_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation IDs and streaming context as extra fields to the log record."""
        record.run_id = self.runtime_context.run_id
        if self.runtime_context.trace_id:
            record.trace_id = self.runtime_context.trace_id
        if self.runtime_context.span_id:
            record.span_id = self.runtime_context.span_id
        # Add streaming context for journal export
        record.is_streaming = self.is_streaming
        record.tenant_id = self.tenant_id
        record.deployment_id = self.deployment_id
        return True


class Context:
    """
    Base context providing common functionality.

    Provides:
    - Logging with correlation IDs
    - Execution metadata (run_id, attempt)
    - Runtime context for tracing
    - Optional memoization for LLM/tool calls

    Extended by:
    - FunctionContext: Minimal context for stateless functions
    - WorkflowContext: Context for durable workflows
    - AgentContext: Context for AI agents (memoization enabled by default)
    """

    def __init__(
        self,
        run_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        is_streaming: bool = False,
        tenant_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_memoization: bool = False,
        checkpoint_callback: Optional["CheckpointCallback"] = None,
        delta_callback: Optional["DeltaCallback"] = None,
    ) -> None:
        """
        Initialize base context.

        Args:
            run_id: Unique execution identifier
            attempt: Retry attempt number (0-indexed)
            runtime_context: RuntimeContext for trace correlation
            is_streaming: Whether this is a streaming request (for real-time SSE log delivery)
            tenant_id: Tenant ID for multi-tenant deployments
            deployment_id: Deployment ID for tracking deployments
            session_id: Session ID for multi-turn conversations (optional)
            enable_memoization: Whether to enable LLM/tool call memoization
            checkpoint_callback: Callback for buffered event delivery
            delta_callback: Callback for streaming event delivery
        """
        self._run_id = run_id
        self._attempt = attempt
        self._runtime_context = runtime_context
        self._is_streaming = is_streaming
        self._tenant_id = tenant_id
        self._deployment_id = deployment_id
        self._session_id = session_id
        self._checkpoint_callback = checkpoint_callback
        self._delta_callback = delta_callback

        # Lazily initialized event emitter
        self._emitter: Optional[EventEmitter] = None

        # Initialize memoization if enabled
        if enable_memoization:
            from .memoization import MemoizationManager
            self._memo: Optional["MemoizationManager"] = MemoizationManager(self)
        else:
            self._memo = None

        # Create logger with correlation
        base_logger = logging.getLogger(f"agnt5.{run_id}")
        from ._telemetry import setup_context_logger
        setup_context_logger(base_logger)

        if runtime_context:
            base_logger.addFilter(_CorrelationFilter(runtime_context, is_streaming, tenant_id, deployment_id))

        # Wrap with ContextLogger to support arbitrary keyword arguments
        # This enables: ctx.logger.info("message", attr1="value1", attr2="value2")
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
        """Full logger for .debug(), .warning(), .error(), etc.

        Supports passing arbitrary keyword arguments as log attributes:
            ctx.logger.info("message", attr1="value1", attr2="value2")
        """
        return self._logger

    @property
    def tenant_id(self) -> Optional[str]:
        """Tenant identifier for multi-tenant deployments."""
        return self._tenant_id

    @property
    def deployment_id(self) -> Optional[str]:
        """Deployment identifier for tracking deployments."""
        return self._deployment_id

    @property
    def session_id(self) -> Optional[str]:
        """Session identifier for multi-turn conversations."""
        return self._session_id

    @property
    def emit(self) -> EventEmitter:
        """Unified event emission API.

        Provides a consistent way to emit events throughout the SDK.
        Events are automatically timestamped with nanosecond precision
        and routed to the appropriate delivery mechanism.

        Streaming mode (is_streaming=True):
            Events are delivered immediately via the delta queue.

        Non-streaming mode (is_streaming=False):
            Events are buffered and delivered via the checkpoint queue.

        Example:
            # Emit workflow events
            ctx.emit.emit("workflow.step.started", {"step_name": "process"})

            # Emit agent events
            ctx.emit.emit("agent.completed", {"output": result})

            # Emit LM events with memoization key
            ctx.emit.emit("lm.call.started", {...}, step_key="lm_call_1")

        Returns:
            EventEmitter instance for emitting events
        """
        if self._emitter is None:
            self._emitter = EventEmitter(
                checkpoint_callback=self._checkpoint_callback,
                delta_callback=self._delta_callback,
                is_streaming=self._is_streaming,
            )
        return self._emitter


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


