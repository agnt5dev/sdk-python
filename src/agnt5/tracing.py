"""
User-facing tracing API for AGNT5 SDK.

Provides decorators and context managers for instrumenting Python code with
OpenTelemetry spans. All spans are created via Rust FFI and exported through
the centralized Rust OpenTelemetry system.

This module uses Python's contextvars for async-safe span context propagation,
ensuring proper parent-child relationships even with asyncio.gather() and
other parallel async operations.

Example:
    ```python
    from agnt5.tracing import span

    @span("my_operation")
    async def my_function(ctx, data):
        # Your code here
        return result

    # Or use context manager
    from agnt5.tracing import span_context

    async def process():
        with span_context("processing", user_id="123") as s:
            data = await fetch_data()
            s.set_attribute("records", str(len(data)))
            return data
    ```
"""

import functools
import inspect
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from ._core import create_span as _rust_create_span


@dataclass
class SpanInfo:
    """Information about the current span for context propagation."""
    trace_id: str
    span_id: str


# Async-safe contextvar for tracking the current span
# This is task-local in asyncio, ensuring proper isolation for parallel operations
_current_span: ContextVar[Optional[SpanInfo]] = ContextVar('current_span', default=None)


def get_current_span_info() -> Optional[SpanInfo]:
    """Get the current span info from the contextvar (if any)."""
    return _current_span.get()


class SpanContextManager:
    """
    Wrapper around PySpan that manages the contextvar for proper async-safe
    parent-child span linking.

    When entering this context manager:
    1. Saves the previous span info
    2. Sets this span as the current span in the contextvar

    When exiting:
    1. Restores the previous span info
    2. Calls PySpan.__exit__ to end the span
    """

    def __init__(self, py_span):
        self._py_span = py_span
        self._token = None

    def __enter__(self):
        # Enter the underlying PySpan first
        self._py_span.__enter__()

        # Set this span as the current span in the contextvar
        # The token allows us to restore the previous value on exit
        span_info = SpanInfo(
            trace_id=self._py_span.trace_id,
            span_id=self._py_span.span_id
        )
        self._token = _current_span.set(span_info)

        return self._py_span

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the previous span context first
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None

        # Then exit the underlying PySpan (ends the span, handles exceptions)
        return self._py_span.__exit__(exc_type, exc_val, exc_tb)


def create_span(
    name: str,
    component_type: str = "operation",
    runtime_context: Optional[Any] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> SpanContextManager:
    """
    Create a span with proper async-safe context propagation.

    This function checks the contextvar for the current span and passes
    the parent trace_id/span_id to Rust for proper parent-child linking.

    Args:
        name: Span name
        component_type: Component type (e.g., "function", "task", "agent")
        runtime_context: Optional RuntimeContext for initial trace context
        attributes: Optional span attributes

    Returns:
        SpanContextManager that can be used as a context manager
    """
    # Get the current span from contextvar (async-safe parent lookup)
    current_span = _current_span.get()

    parent_trace_id = None
    parent_span_id = None
    if current_span is not None:
        parent_trace_id = current_span.trace_id
        parent_span_id = current_span.span_id

    # Create the Rust span with parent IDs for proper linking
    py_span = _rust_create_span(
        name,
        component_type,
        runtime_context,
        attributes or {},
        parent_trace_id,
        parent_span_id,
    )

    # Wrap in our context manager for contextvar management
    return SpanContextManager(py_span)


def span(
    name: Optional[str] = None,
    component_type: str = "function",
    runtime_context: Optional[Any] = None,
    **attributes: str
):
    """
    Decorator to automatically create spans for functions.

    Args:
        name: Span name (defaults to function name)
        component_type: Component type (default: "function")
        runtime_context: Optional RuntimeContext for trace linking
        **attributes: Additional span attributes

    Example:
        ```python
        @span("fetch_user_data", user_type="premium")
        async def fetch_user(user_id: str):
            return await db.get_user(user_id)
        ```
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Try to extract runtime_context from first arg if it's a Context
                ctx = runtime_context
                if ctx is None and args:
                    from .context import Context
                    if isinstance(args[0], Context):
                        ctx = args[0]._runtime_context

                with create_span(span_name, component_type, ctx, attributes) as s:
                    try:
                        result = await func(*args, **kwargs)
                        # Span automatically marked as OK on success
                        return result
                    except Exception as e:
                        # Exception automatically recorded by PySpan.__exit__
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Try to extract runtime_context from first arg if it's a Context
                ctx = runtime_context
                if ctx is None and args:
                    from .context import Context
                    if isinstance(args[0], Context):
                        ctx = args[0]._runtime_context

                with create_span(span_name, component_type, ctx, attributes) as s:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        raise
            return sync_wrapper

    return decorator


@contextmanager
def span_context(
    name: str,
    component_type: str = "operation",
    runtime_context: Optional[Any] = None,
    **attributes: str
):
    """
    Context manager for creating spans around code blocks.

    Args:
        name: Span name
        component_type: Component type (default: "operation")
        runtime_context: Optional RuntimeContext for trace linking
        **attributes: Span attributes

    Yields:
        PySpan object with set_attribute() and record_exception() methods

    Example:
        ```python
        with span_context("db_query", runtime_context=ctx._runtime_context, table="users") as s:
            results = query_database()
            s.set_attribute("result_count", str(len(results)))
        ```
    """
    with create_span(name, component_type, runtime_context, attributes) as s:
        yield s


def create_task_span(name: str, runtime_context: Optional[Any] = None, **attributes: str):
    """
    Create a span for task execution.

    Args:
        name: Task name
        runtime_context: Optional RuntimeContext for trace linking
        **attributes: Task attributes

    Returns:
        SpanContextManager to use as context manager

    Example:
        ```python
        with create_task_span("process_data", runtime_context=ctx._runtime_context, batch_size="100") as s:
            result = await process()
        ```
    """
    return create_span(name, "task", runtime_context, attributes)


def create_workflow_span(name: str, runtime_context: Optional[Any] = None, **attributes: str):
    """
    Create a span for workflow execution.

    Args:
        name: Workflow name
        runtime_context: Optional RuntimeContext for trace linking
        **attributes: Workflow attributes

    Returns:
        SpanContextManager to use as context manager
    """
    return create_span(name, "workflow", runtime_context, attributes)


def create_agent_span(name: str, runtime_context: Optional[Any] = None, **attributes: str):
    """
    Create a span for agent execution.

    Args:
        name: Agent name
        runtime_context: Optional RuntimeContext for trace linking
        **attributes: Agent attributes

    Returns:
        SpanContextManager to use as context manager
    """
    return create_span(name, "agent", runtime_context, attributes)


__all__ = [
    "span",
    "span_context",
    "create_span",
    "create_task_span",
    "create_workflow_span",
    "create_agent_span",
    "get_current_span_info",
    "SpanInfo",
]
