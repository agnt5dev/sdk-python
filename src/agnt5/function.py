"""Function component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import secrets
import time
import uuid
from typing import Any, Callable, Optional, TypeVar, Union, cast

from ._retry_utils import execute_with_retry, parse_backoff_policy, parse_retry_policy
from ._schema_utils import extract_function_metadata, extract_function_schemas
from .context import Context, set_current_context
from .events import Completed, ComponentType, Failed, Started
from .types import BackoffPolicy, FunctionConfig, HandlerFunc, RetryPolicy

T = TypeVar("T")

# Global function registry
_FUNCTION_REGISTRY: dict[str, FunctionConfig] = {}


class FunctionContext(Context):
    """Context for stateless functions. Use workflows for orchestration/checkpointing."""

    def __init__(
        self,
        run_id: str,
        correlation_id: str,
        parent_correlation_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        retry_policy: Optional[Any] = None,
        is_streaming: bool = False,
        worker: Optional[Any] = None,
    ) -> None:
        super().__init__(
            run_id,
            correlation_id,
            parent_correlation_id,
            attempt,
            runtime_context,
            is_streaming=is_streaming,
            worker=worker,
        )
        self._retry_policy = retry_policy

    def log(self, message: str, **extra: Any) -> None:
        """Log with structured data: ctx.log("msg", key=value)"""
        self._logger.info(message, extra=extra)

    def should_retry(self, error: Exception) -> bool:
        """Check if error is retryable based on configured policy."""
        # TODO: Implement retry policy checks
        return True

    async def sleep(self, seconds: float) -> None:
        """Non-durable async sleep. Use workflows for durable sleep."""
        await asyncio.sleep(seconds)


class FunctionRegistry:
    """Registry for function handlers."""

    @staticmethod
    def register(config: FunctionConfig) -> None:
        """Register a function handler.

        Args:
            config: Function configuration to register

        Raises:
            ValueError: If a function with the same name is already registered
        """
        # Check for name collision
        if config.name in _FUNCTION_REGISTRY:
            existing_config = _FUNCTION_REGISTRY[config.name]
            existing_module = existing_config.handler.__module__
            new_module = config.handler.__module__

            raise ValueError(
                f"Function name collision: '{config.name}' is already registered.\n"
                f"  Existing: {existing_module}.{existing_config.handler.__name__}\n"
                f"  New:      {new_module}.{config.handler.__name__}\n"
                f"Please use a different function name or use name= parameter to specify a unique name."
            )

        _FUNCTION_REGISTRY[config.name] = config

    @staticmethod
    def get(name: str) -> Optional[FunctionConfig]:
        """Get function configuration by name."""
        return _FUNCTION_REGISTRY.get(name)

    @staticmethod
    def all() -> dict[str, FunctionConfig]:
        """Get all registered functions."""
        return _FUNCTION_REGISTRY.copy()

    @staticmethod
    def clear() -> None:
        """Clear all registered functions."""
        _FUNCTION_REGISTRY.clear()


def function(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    retries: Optional[Union[int, dict[str, Any], RetryPolicy]] = None,
    backoff: Optional[Union[str, dict[str, Any], BackoffPolicy]] = None,
    timeout_ms: Optional[int] = None,
) -> Callable[..., Any]:
    """Decorator to mark a function as an AGNT5 durable function.

    Args:
        name: Custom function name (default: function's __name__)
        retries: int, dict, or RetryPolicy
        backoff: str ("constant"/"linear"/"exponential"), dict, or BackoffPolicy
        timeout_ms: Maximum execution time in milliseconds

    Note:
        Sync functions are automatically wrapped to run in a thread pool.

    Example:
        @function
        async def greet(ctx: FunctionContext, name: str) -> str:
            return f"Hello, {name}!"

        @function(retries=5, backoff="exponential")
        async def with_retries(data: str) -> str:
            return data.upper()
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = name or func.__name__
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        needs_context = bool(params) and params[0].name == "ctx"

        # Async generators should NOT be wrapped - they need to be returned as-is
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            handler_func = cast(HandlerFunc, func)
        else:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

            handler_func = cast(HandlerFunc, async_wrapper)

        input_schema, output_schema = extract_function_schemas(func)
        metadata = extract_function_metadata(func)
        retry_policy = parse_retry_policy(retries)
        backoff_policy = parse_backoff_policy(backoff)

        config = FunctionConfig(
            name=func_name,
            handler=handler_func,
            retries=retry_policy,
            backoff=backoff_policy,
            timeout_ms=timeout_ms,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
        )
        FunctionRegistry.register(config)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if needs_context:
                if not args or not isinstance(args[0], FunctionContext):
                    raise TypeError(
                        f"Function '{func_name}' requires FunctionContext as first argument. "
                        f"Usage: await {func_name}(ctx, ...)"
                    )
                ctx = args[0]
                func_args = args[1:]
            else:
                if args and isinstance(args[0], FunctionContext):
                    ctx = args[0]
                    func_args = args[1:]
                else:
                    # Local execution without parent context (dev/testing)
                    run_id = f"local-{uuid.uuid4().hex[:8]}"
                    correlation_id = f"fn-{secrets.token_hex(4)}"
                    ctx = FunctionContext(
                        run_id=run_id,
                        correlation_id=correlation_id,
                        parent_correlation_id=f"run-{run_id}",
                        retry_policy=retry_policy,
                    )
                    func_args = args

            token = set_current_context(ctx)
            start_time_ns = time.time_ns()
            # Use context's correlation_id (set above for local, or from parent context)
            correlation_id = ctx.correlation_id

            # Emit function.started
            bound = sig.bind_partial(*func_args, **kwargs)
            ctx.emit(
                Started(
                    name=func_name,
                    correlation_id=correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    attempt=ctx.attempt,
                    input_data=dict(bound.arguments),
                )
            )

            try:
                result = await execute_with_retry(
                    handler_func,
                    ctx,
                    config.retries or RetryPolicy(),
                    config.backoff or BackoffPolicy(),
                    needs_context,
                    config.timeout_ms,
                    *func_args,
                    **kwargs,
                )

                # Emit function.completed
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                ctx.emit(
                    Completed(
                        name=func_name,
                        correlation_id=correlation_id,
                        parent_correlation_id=ctx.parent_correlation_id,
                        component_type=ComponentType.FUNCTION,
                        output_data=result,
                        duration_ms=duration_ms,
                    )
                )

                return result

            except Exception as e:
                # Emit function.failed
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                ctx.emit(
                    Failed(
                        name=func_name,
                        correlation_id=correlation_id,
                        parent_correlation_id=ctx.parent_correlation_id or ctx.run_id,
                        component_type=ComponentType.FUNCTION,
                        error_code=type(e).__name__,
                        error_message=str(e),
                        duration_ms=duration_ms,
                    )
                )
                raise

            finally:
                from .context import _current_context

                _current_context.reset(token)

        wrapper._agnt5_config = config  # type: ignore
        return wrapper

    # Handle both @function and @function(...) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
