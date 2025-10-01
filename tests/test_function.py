"""Tests for Function decorator and retry logic."""

import asyncio

import pytest

from agnt5 import BackoffPolicy, BackoffType, Context, FunctionRegistry, RetryPolicy, function
from agnt5.exceptions import RetryError


class TestFunctionDecorator:
    """Test @function decorator."""

    def test_function_basic(self) -> None:
        @function
        async def my_func(ctx: Context, value: str) -> str:
            return f"Result: {value}"

        # Check registration
        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.name == "my_func"

    def test_function_custom_name(self) -> None:
        @function(name="custom_function")
        async def my_func(ctx: Context) -> str:
            return "test"

        # Check registration with custom name
        config = FunctionRegistry.get("custom_function")
        assert config is not None
        assert config.name == "custom_function"

    def test_function_with_retry_policy(self) -> None:
        retry_policy = RetryPolicy(max_attempts=5, initial_interval_ms=500)

        @function(retries=retry_policy)
        async def my_func(ctx: Context) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.retries is not None
        assert config.retries.max_attempts == 5
        assert config.retries.initial_interval_ms == 500

    def test_function_with_backoff_policy(self) -> None:
        backoff_policy = BackoffPolicy(type=BackoffType.LINEAR, multiplier=1.5)

        @function(backoff=backoff_policy)
        async def my_func(ctx: Context) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.backoff is not None
        assert config.backoff.type == BackoffType.LINEAR
        assert config.backoff.multiplier == 1.5

    def test_function_requires_ctx_parameter(self) -> None:
        with pytest.raises(ValueError, match="must have 'ctx: Context' as first parameter"):

            @function
            async def bad_func(value: str) -> str:
                return value

    def test_sync_function_converted_to_async(self) -> None:
        @function
        def sync_func(ctx: Context, value: int) -> int:
            return value * 2

        config = FunctionRegistry.get("sync_func")
        assert config is not None

        # Should be callable as async
        ctx = Context(run_id="test")
        result = asyncio.run(sync_func(ctx, 5))
        assert result == 10


class TestFunctionExecution:
    """Test function execution."""

    @pytest.mark.asyncio
    async def test_basic_execution(self) -> None:
        @function
        async def add(ctx: Context, a: int, b: int) -> int:
            return a + b

        ctx = Context(run_id="test-123")
        result = await add(ctx, 3, 5)
        assert result == 8

    @pytest.mark.asyncio
    async def test_execution_with_context_auto_creation(self) -> None:
        @function
        async def greet(ctx: Context, name: str) -> str:
            return f"Hello, {name}"

        # Call without providing context - should auto-create
        result = await greet(name="Alice")  # type: ignore
        assert result == "Hello, Alice"

    @pytest.mark.asyncio
    async def test_context_metadata_in_function(self) -> None:
        @function
        async def get_run_id(ctx: Context) -> str:
            return ctx.run_id

        ctx = Context(run_id="test-456")
        result = await get_run_id(ctx)
        assert result == "test-456"


class TestRetryLogic:
    """Test retry behavior."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self) -> None:
        call_count = 0

        @function
        async def success_func(ctx: Context) -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        ctx = Context(run_id="test")
        result = await success_func(ctx)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_until_success(self) -> None:
        call_count = 0

        @function(retries=RetryPolicy(max_attempts=3, initial_interval_ms=10))
        async def flaky_func(ctx: Context) -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise Exception("Transient error")
            return "success"

        ctx = Context(run_id="test")
        result = await flaky_func(ctx)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exceeds_max_attempts(self) -> None:
        call_count = 0

        @function(retries=RetryPolicy(max_attempts=3, initial_interval_ms=10))
        async def always_fails(ctx: Context) -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent error")

        ctx = Context(run_id="test")

        with pytest.raises(RetryError) as exc_info:
            await always_fails(ctx)

        assert call_count == 3
        assert exc_info.value.attempts == 3
        assert "Permanent error" in str(exc_info.value.last_error)

    @pytest.mark.asyncio
    async def test_context_attempt_increments(self) -> None:
        attempts_seen = []

        @function(retries=RetryPolicy(max_attempts=3, initial_interval_ms=10))
        async def track_attempts(ctx: Context) -> str:
            attempts_seen.append(ctx.attempt)
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = Context(run_id="test")
        await track_attempts(ctx)

        assert attempts_seen == [0, 1, 2]


class TestBackoffCalculation:
    """Test backoff delay calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        import time

        @function(
            retries=RetryPolicy(max_attempts=3, initial_interval_ms=100, max_interval_ms=1000),
            backoff=BackoffPolicy(type=BackoffType.EXPONENTIAL, multiplier=2.0),
        )
        async def exponential_func(ctx: Context) -> str:
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = Context(run_id="test")
        start = time.time()
        await exponential_func(ctx)
        duration = time.time() - start

        # Should take roughly: 100ms + 200ms = 300ms (with some tolerance)
        assert duration >= 0.25  # At least 250ms
        assert duration < 0.5  # Less than 500ms

    @pytest.mark.asyncio
    async def test_linear_backoff(self) -> None:
        import time

        @function(
            retries=RetryPolicy(max_attempts=3, initial_interval_ms=100),
            backoff=BackoffPolicy(type=BackoffType.LINEAR, multiplier=1.0),
        )
        async def linear_func(ctx: Context) -> str:
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = Context(run_id="test")
        start = time.time()
        await linear_func(ctx)
        duration = time.time() - start

        # Should take roughly: 100ms + 200ms = 300ms
        assert duration >= 0.25
        assert duration < 0.5

    @pytest.mark.asyncio
    async def test_constant_backoff(self) -> None:
        import time

        @function(
            retries=RetryPolicy(max_attempts=3, initial_interval_ms=100),
            backoff=BackoffPolicy(type=BackoffType.CONSTANT),
        )
        async def constant_func(ctx: Context) -> str:
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = Context(run_id="test")
        start = time.time()
        await constant_func(ctx)
        duration = time.time() - start

        # Should take roughly: 100ms + 100ms = 200ms
        assert duration >= 0.15
        assert duration < 0.4


class TestFunctionRegistry:
    """Test function registry."""

    def test_registry_get(self) -> None:
        @function
        async def test_func(ctx: Context) -> str:
            return "test"

        config = FunctionRegistry.get("test_func")
        assert config is not None
        assert config.name == "test_func"

    def test_registry_get_missing(self) -> None:
        result = FunctionRegistry.get("nonexistent")
        assert result is None

    def test_registry_all(self) -> None:
        # Clear registry first
        FunctionRegistry.clear()

        @function
        async def func1(ctx: Context) -> str:
            return "1"

        @function
        async def func2(ctx: Context) -> str:
            return "2"

        all_funcs = FunctionRegistry.all()
        assert "func1" in all_funcs
        assert "func2" in all_funcs
        assert len(all_funcs) >= 2

    def test_registry_clear(self) -> None:
        @function
        async def temp_func(ctx: Context) -> str:
            return "temp"

        assert FunctionRegistry.get("temp_func") is not None

        FunctionRegistry.clear()

        assert FunctionRegistry.get("temp_func") is None
