"""Function component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union, cast, get_type_hints

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


def _type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """Convert Python type hint to JSON Schema."""
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    if python_type is str:
        return {"type": "string"}
    if python_type is int:
        return {"type": "integer"}
    if python_type is float:
        return {"type": "number"}
    if python_type is bool:
        return {"type": "boolean"}

    # Handle typing module types
    origin = getattr(python_type, "__origin__", None)

    if origin is list:
        args = getattr(python_type, "__args__", ())
        if args:
            return {"type": "array", "items": _type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if origin is Union:
        args = getattr(python_type, "__args__", ())
        # Handle Optional[T] (Union[T, None])
        if len(args) == 2 and type(None) in args:
            non_none = args[0] if args[1] is type(None) else args[1]
            schema = _type_to_json_schema(non_none)
            # Mark as nullable in JSON Schema
            return {**schema, "nullable": True}
        # Handle other unions as anyOf
        return {"anyOf": [_type_to_json_schema(arg) for arg in args]}

    # Default to object for unknown types
    return {"type": "object"}


def _extract_function_schemas(func: Callable[..., Any]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract input and output schemas from function type hints.

    Returns:
        Tuple of (input_schema, output_schema) where each is a JSON Schema dict or None
    """
    try:
        # Get type hints
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Build input schema from parameters (excluding 'ctx')
        input_properties = {}
        required_params = []

        for param_name, param in sig.parameters.items():
            if param_name == "ctx":
                continue

            # Get type hint for this parameter
            if param_name in hints:
                param_type = hints[param_name]
                input_properties[param_name] = _type_to_json_schema(param_type)
            else:
                # No type hint, use generic object
                input_properties[param_name] = {"type": "object"}

            # Check if parameter is required (no default value)
            if param.default is inspect.Parameter.empty:
                required_params.append(param_name)

        input_schema = None
        if input_properties:
            input_schema = {
                "type": "object",
                "properties": input_properties,
            }
            if required_params:
                input_schema["required"] = required_params

            # Add description from docstring if available
            if func.__doc__:
                docstring = inspect.cleandoc(func.__doc__)
                first_line = docstring.split('\n')[0].strip()
                if first_line:
                    input_schema["description"] = first_line

        # Build output schema from return type hint
        output_schema = None
        if "return" in hints:
            return_type = hints["return"]
            output_schema = _type_to_json_schema(return_type)

        return input_schema, output_schema

    except Exception:
        # If schema extraction fails, return None schemas
        return None, None


def _extract_function_metadata(func: Callable[..., Any]) -> Dict[str, str]:
    """Extract metadata from function including description from docstring.

    Returns:
        Dictionary with metadata fields like 'description'
    """
    metadata = {}

    # Extract description from docstring
    if func.__doc__:
        # Get first line of docstring as description
        docstring = inspect.cleandoc(func.__doc__)
        first_line = docstring.split('\n')[0].strip()
        if first_line:
            metadata["description"] = first_line

    return metadata


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
        # Note: Async generators should NOT be wrapped - they need to be returned as-is
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            handler_func = cast(HandlerFunc, func)
        else:
            # Wrap sync function in async
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            handler_func = cast(HandlerFunc, async_wrapper)

        # Extract schemas from type hints
        input_schema, output_schema = _extract_function_schemas(func)

        # Extract metadata (description, etc.)
        metadata = _extract_function_metadata(func)

        # Register function
        config = FunctionConfig(
            name=func_name,
            handler=handler_func,
            retries=retries or RetryPolicy(),
            backoff=backoff or BackoffPolicy(),
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
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
