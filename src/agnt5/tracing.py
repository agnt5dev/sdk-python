"""
User-facing tracing API for AGNT5 SDK.

Provides decorators and context managers for instrumenting Python code with
OpenTelemetry spans. All spans are created via Rust FFI and exported through
the centralized Rust OpenTelemetry system.

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
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

from ._core import create_span as _create_span


def span(
    name: Optional[str] = None,
    component_type: str = "function",
    **attributes: str
):
    """
    Decorator to automatically create spans for functions.

    Args:
        name: Span name (defaults to function name)
        component_type: Component type (default: "function")
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
                with _create_span(span_name, component_type, attributes) as s:
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
                with _create_span(span_name, component_type, attributes) as s:
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
    **attributes: str
):
    """
    Context manager for creating spans around code blocks.

    Args:
        name: Span name
        component_type: Component type (default: "operation")
        **attributes: Span attributes

    Yields:
        PySpan object with set_attribute() and record_exception() methods

    Example:
        ```python
        with span_context("db_query", table="users") as s:
            results = query_database()
            s.set_attribute("result_count", str(len(results)))
        ```
    """
    s = _create_span(name, component_type, attributes)
    try:
        yield s
        # Context manager automatically calls s.__exit__ which sets status
    except Exception as e:
        # Exception will be recorded by __exit__
        raise
    finally:
        # PySpan's __exit__ is called automatically when context ends
        pass


def create_task_span(name: str, **attributes: str):
    """
    Create a span for task execution.

    Args:
        name: Task name
        **attributes: Task attributes

    Returns:
        PySpan object to use as context manager

    Example:
        ```python
        with create_task_span("process_data", batch_size="100") as s:
            result = await process()
        ```
    """
    return _create_span(name, "task", attributes)


def create_workflow_span(name: str, **attributes: str):
    """
    Create a span for workflow execution.

    Args:
        name: Workflow name
        **attributes: Workflow attributes

    Returns:
        PySpan object to use as context manager
    """
    return _create_span(name, "workflow", attributes)


def create_agent_span(name: str, **attributes: str):
    """
    Create a span for agent execution.

    Args:
        name: Agent name
        **attributes: Agent attributes

    Returns:
        PySpan object to use as context manager
    """
    return _create_span(name, "agent", attributes)


__all__ = [
    "span",
    "span_context",
    "create_task_span",
    "create_workflow_span",
    "create_agent_span",
]
