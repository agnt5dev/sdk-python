"""Function component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union, cast

from ._retry_utils import execute_with_retry, parse_backoff_policy, parse_retry_policy
from ._schema_utils import extract_function_metadata, extract_function_schemas
from .context import Context, set_current_context
from .exceptions import RetryError
from .types import BackoffPolicy, BackoffType, FunctionConfig, HandlerFunc, RetryPolicy

T = TypeVar("T")

# Global function registry
_FUNCTION_REGISTRY: Dict[str, FunctionConfig] = {}

class FunctionContext(Context):
    """
    Lightweight context for stateless functions.

    AGNT5 Philosophy: Context is a convenience, not a requirement.
    The best function is one that doesn't need context at all!

    Provides only:
    - Quick logging (ctx.log())
    - Execution metadata (run_id, attempt)
    - Smart retry helper (should_retry())
    - Non-durable sleep

    Does NOT provide:
    - Orchestration (task, parallel, gather) - use workflows
    - State management (get, set, delete) - functions are stateless
    - Checkpointing (step) - functions are atomic
    """

    def __init__(
        self,
        run_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        retry_policy: Optional[Any] = None,
    ) -> None:
        """
        Initialize function context.

        Args:
            run_id: Unique execution identifier
            attempt: Retry attempt number (0-indexed)
            runtime_context: RuntimeContext for trace correlation
            retry_policy: RetryPolicy for should_retry() checks
        """
        super().__init__(run_id, attempt, runtime_context)
        self._retry_policy = retry_policy

    # === Quick Logging ===

    def log(self, message: str, **extra) -> None:
        """
        Quick logging shorthand with structured data.

        Example:
            ctx.log("Processing payment", amount=100.50, user_id="123")
        """
        self._logger.info(message, extra=extra)

    # === Smart Execution ===

    def should_retry(self, error: Exception) -> bool:
        """
        Check if error is retryable based on configured policy.

        Example:
            try:
                result = await external_api()
            except Exception as e:
                if not ctx.should_retry(e):
                    raise  # Fail fast for non-retryable errors
                # Otherwise let retry policy handle it
                raise

        Returns:
            True if error is retryable, False otherwise
        """
        # TODO: Implement retry policy checks
        # For now, all errors are retryable (let retry policy handle it)
        return True

    async def sleep(self, seconds: float) -> None:
        """
        Non-durable async sleep.

        For durable sleep across failures, use workflows.

        Args:
            seconds: Number of seconds to sleep
        """
        import asyncio
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
    def all() -> Dict[str, FunctionConfig]:
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
    retries: Optional[Union[int, Dict[str, Any], RetryPolicy]] = None,
    backoff: Optional[Union[str, Dict[str, Any], BackoffPolicy]] = None,
) -> Callable[..., Any]:
    """
    Decorator to mark a function as an AGNT5 durable function.

    Args:
        name: Custom function name (default: function's __name__)
        retries: Retry policy configuration. Can be:
            - int: max attempts (e.g., 5)
            - dict: RetryPolicy params (e.g., {"max_attempts": 5, "initial_interval_ms": 1000})
            - RetryPolicy: full policy object
        backoff: Backoff policy for retries. Can be:
            - str: backoff type ("constant", "linear", "exponential")
            - dict: BackoffPolicy params (e.g., {"type": "exponential", "multiplier": 2.0})
            - BackoffPolicy: full policy object

    Note:
        Sync Functions: Synchronous functions are automatically executed in a thread pool
        to prevent blocking the event loop. This is ideal for I/O-bound operations
        (requests.get(), file I/O, etc.). For CPU-bound operations or when you need
        explicit control over concurrency, use async functions instead.

    Example:
        # Basic function with context
        @function
        async def greet(ctx: FunctionContext, name: str) -> str:
            ctx.log(f"Greeting {name}")  # AGNT5 shorthand!
            return f"Hello, {name}!"

        # Simple function without context (optional)
        @function
        async def add(a: int, b: int) -> int:
            return a + b

        # With Pydantic models (automatic validation + rich schemas)
        from pydantic import BaseModel

        class UserInput(BaseModel):
            name: str
            age: int

        class UserOutput(BaseModel):
            greeting: str
            is_adult: bool

        @function
        async def process_user(ctx: FunctionContext, user: UserInput) -> UserOutput:
            ctx.log(f"Processing user {user.name}")
            return UserOutput(
                greeting=f"Hello, {user.name}!",
                is_adult=user.age >= 18
            )

        # Simple retry count
        @function(retries=5)
        async def with_retries(data: str) -> str:
            return data.upper()

        # Dict configuration
        @function(retries={"max_attempts": 5}, backoff="exponential")
        async def advanced(a: int, b: int) -> int:
            return a + b
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get function name
        func_name = name or func.__name__

        # Validate function signature and check if context is needed
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # Check if function declares 'ctx' parameter
        needs_context = params and params[0].name == "ctx"

        # Convert sync to async if needed
        # Note: Async generators should NOT be wrapped - they need to be returned as-is
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            handler_func = cast(HandlerFunc, func)
        else:
            # Wrap sync function to run in thread pool (prevents blocking event loop)
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                loop = asyncio.get_running_loop()
                # Run sync function in thread pool executor to prevent blocking
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

            handler_func = cast(HandlerFunc, async_wrapper)

        # Extract schemas from type hints
        input_schema, output_schema = extract_function_schemas(func)

        # Extract metadata (description, etc.)
        metadata = extract_function_metadata(func)

        # Parse retry and backoff policies from flexible formats
        retry_policy = parse_retry_policy(retries)
        backoff_policy = parse_backoff_policy(backoff)

        # Register function
        config = FunctionConfig(
            name=func_name,
            handler=handler_func,
            retries=retry_policy,
            backoff=backoff_policy,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
        )
        FunctionRegistry.register(config)

        # Create wrapper with retry logic
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract or create context based on function signature
            if needs_context:
                # Function declares ctx parameter - first argument must be FunctionContext
                if not args or not isinstance(args[0], FunctionContext):
                    raise TypeError(
                        f"Function '{func_name}' requires FunctionContext as first argument. "
                        f"Usage: await {func_name}(ctx, ...)"
                    )
                ctx = args[0]
                func_args = args[1:]
            else:
                # Function doesn't use context - create a minimal one for internal use
                # But first check if a context was passed anyway (for Worker execution)
                if args and isinstance(args[0], FunctionContext):
                    # Context was provided by Worker - use it but don't pass to function
                    ctx = args[0]
                    func_args = args[1:]
                else:
                    # No context provided - create a default one
                    ctx = FunctionContext(
                        run_id=f"local-{uuid.uuid4().hex[:8]}",
                        retry_policy=retry_policy
                    )
                    func_args = args

            # Set context in task-local storage for automatic propagation
            token = set_current_context(ctx)
            try:
                # Execute with retry
                return await execute_with_retry(
                    handler_func,
                    ctx,
                    config.retries or RetryPolicy(),
                    config.backoff or BackoffPolicy(),
                    needs_context,
                    *func_args,
                    **kwargs,
                )
            finally:
                # Always reset context to prevent leakage
                from .context import _current_context
                _current_context.reset(token)

        # Store config on wrapper for introspection
        wrapper._agnt5_config = config  # type: ignore
        return wrapper

    # Handle both @function and @function(...) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
