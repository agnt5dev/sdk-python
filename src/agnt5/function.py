"""Function component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union, cast

from .context import Context
from .exceptions import RetryError
from .types import BackoffPolicy, BackoffType, FunctionConfig, HandlerFunc, RetryPolicy

T = TypeVar("T")

# Global function registry
_FUNCTION_REGISTRY: Dict[str, FunctionConfig] = {}


class FunctionRegistry:
    """Registry for function handlers."""

    @staticmethod
    def register(config: FunctionConfig) -> None:
        """Register a function handler."""
        _FUNCTION_REGISTRY[config.name] = config

    @staticmethod
    def get(name: str) -> Optional[FunctionConfig]:
        """Get function configuration by name."""
        return _FUNCTION_REGISTRY.get(name)

    @staticmethod
    def all() -> Dict[str, FunctionConfig]:
        """Get all registered functions."""
        return _FUNCTION_REGISTRY.copy()

    @staticmethod
    def clear() -> None:
        """Clear all registered functions."""
        _FUNCTION_REGISTRY.clear()


def _calculate_backoff_delay(
    attempt: int,
    retry_policy: RetryPolicy,
    backoff_policy: BackoffPolicy,
) -> float:
    """Calculate backoff delay in seconds based on attempt number."""
    if backoff_policy.type == BackoffType.CONSTANT:
        delay_ms = retry_policy.initial_interval_ms
    elif backoff_policy.type == BackoffType.LINEAR:
        delay_ms = retry_policy.initial_interval_ms * (attempt + 1)
    else:  # EXPONENTIAL
        delay_ms = retry_policy.initial_interval_ms * (backoff_policy.multiplier**attempt)

    # Cap at max_interval_ms
    delay_ms = min(delay_ms, retry_policy.max_interval_ms)
    return delay_ms / 1000.0  # Convert to seconds


async def _execute_with_retry(
    handler: HandlerFunc,
    ctx: Context,
    retry_policy: RetryPolicy,
    backoff_policy: BackoffPolicy,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute handler with retry logic."""
    last_error: Optional[Exception] = None

    for attempt in range(retry_policy.max_attempts):
        try:
            # Update context attempt number
            ctx._attempt = attempt

            # Execute handler
            result = await handler(ctx, *args, **kwargs)
            return result

        except Exception as e:
            last_error = e
            ctx.logger.warning(
                f"Function execution failed (attempt {attempt + 1}/{retry_policy.max_attempts}): {e}"
            )

            # If this was the last attempt, raise RetryError
            if attempt == retry_policy.max_attempts - 1:
                raise RetryError(
                    f"Function failed after {retry_policy.max_attempts} attempts",
                    attempts=retry_policy.max_attempts,
                    last_error=e,
                )

            # Calculate backoff delay
            delay = _calculate_backoff_delay(attempt, retry_policy, backoff_policy)
            ctx.logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    assert last_error is not None
    raise RetryError(
        f"Function failed after {retry_policy.max_attempts} attempts",
        attempts=retry_policy.max_attempts,
        last_error=last_error,
    )


def function(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    retries: Optional[RetryPolicy] = None,
    backoff: Optional[BackoffPolicy] = None,
) -> Callable[..., Any]:
    """
    Decorator to mark a function as an AGNT5 durable function.

    Args:
        name: Custom function name (default: function's __name__)
        retries: Retry policy configuration
        backoff: Backoff policy for retries

    Example:
        @function
        async def greet(ctx: Context, name: str) -> str:
            return f"Hello, {name}!"

        @function(name="add_numbers", retries=RetryPolicy(max_attempts=5))
        async def add(ctx: Context, a: int, b: int) -> int:
            return a + b
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get function name
        func_name = name or func.__name__

        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params or params[0].name != "ctx":
            raise ValueError(
                f"Function '{func_name}' must have 'ctx: Context' as first parameter"
            )

        # Convert sync to async if needed
        if inspect.iscoroutinefunction(func):
            handler_func = cast(HandlerFunc, func)
        else:
            # Wrap sync function in async
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            handler_func = cast(HandlerFunc, async_wrapper)

        # Register function
        config = FunctionConfig(
            name=func_name,
            handler=handler_func,
            retries=retries or RetryPolicy(),
            backoff=backoff or BackoffPolicy(),
        )
        FunctionRegistry.register(config)

        # Create wrapper with retry logic
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create context if not provided
            if not args or not isinstance(args[0], Context):
                # Auto-create context for direct function calls
                ctx = Context(
                    run_id=f"local-{uuid.uuid4().hex[:8]}",
                    component_type="function",
                )
                # Execute with retry
                return await _execute_with_retry(
                    handler_func,
                    ctx,
                    config.retries or RetryPolicy(),
                    config.backoff or BackoffPolicy(),
                    *args,
                    **kwargs,
                )
            else:
                # Context provided - use it
                ctx = args[0]
                return await _execute_with_retry(
                    handler_func,
                    ctx,
                    config.retries or RetryPolicy(),
                    config.backoff or BackoffPolicy(),
                    *args[1:],
                    **kwargs,
                )

        # Store config on wrapper for introspection
        wrapper._agnt5_config = config  # type: ignore
        return wrapper

    # Handle both @function and @function(...) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
