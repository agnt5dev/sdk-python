"""Tests for Function decorator.

Note: Retry execution is now handled by the platform (Execution Engine).
The SDK executes functions once and reports results; the platform orchestrates retries.
"""

import asyncio

import pytest

from agnt5 import BackoffPolicy, BackoffType, FunctionContext, FunctionRegistry, RetryPolicy, function

# Test constants
MAX_TEST_RETRIES = 3
TEST_INTERVAL_MS = 10
BACKOFF_INTERVAL_MS = 100
BACKOFF_MAX_INTERVAL_MS = 1000


# Module-level fixtures
@pytest.fixture(autouse=True)
def clear_registry():
    """Clear function registry before and after each test."""
    FunctionRegistry.clear()
    yield
    FunctionRegistry.clear()


@pytest.fixture
def test_context():
    """Reusable test context."""
    return FunctionContext(
        run_id="test-123",
        correlation_id="corr-123",
        parent_correlation_id="parent-123",
    )


class TestFunctionDecorator:
    """Test @function decorator."""

    def test_function_basic(self) -> None:
        @function
        async def my_func(ctx: FunctionContext, value: str) -> str:
            return f"Result: {value}"

        # Check registration
        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.name == "my_func"

    def test_function_custom_name(self) -> None:
        @function(name="custom_function")
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        # Check registration with custom name
        config = FunctionRegistry.get("custom_function")
        assert config is not None
        assert config.name == "custom_function"

    def test_function_with_retry_policy_object(self) -> None:
        retry_policy = RetryPolicy(max_attempts=5, initial_interval_ms=500)

        @function(retries=retry_policy)
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.retries is not None
        assert config.retries.max_attempts == 5
        assert config.retries.initial_interval_ms == 500

    def test_function_with_retry_policy_int(self) -> None:
        """Test simplified retry configuration with just max attempts."""
        @function(retries=5)
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.retries is not None
        assert config.retries.max_attempts == 5

    def test_function_with_retry_policy_dict(self) -> None:
        """Test retry configuration with dict."""
        @function(retries={"max_attempts": 5, "initial_interval_ms": 1000})
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.retries is not None
        assert config.retries.max_attempts == 5
        assert config.retries.initial_interval_ms == 1000

    def test_function_with_backoff_policy_object(self) -> None:
        backoff_policy = BackoffPolicy(type=BackoffType.LINEAR, multiplier=1.5)

        @function(backoff=backoff_policy)
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.backoff is not None
        assert config.backoff.type == BackoffType.LINEAR
        assert config.backoff.multiplier == 1.5

    def test_function_with_backoff_policy_string(self) -> None:
        """Test simplified backoff configuration with string."""
        @function(backoff="exponential")
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.backoff is not None
        assert config.backoff.type == BackoffType.EXPONENTIAL

    def test_function_with_backoff_policy_dict(self) -> None:
        """Test backoff configuration with dict."""
        @function(backoff={"type": "linear", "multiplier": 2.0})
        async def my_func(ctx: FunctionContext) -> str:
            return "test"

        config = FunctionRegistry.get("my_func")
        assert config is not None
        assert config.backoff is not None
        assert config.backoff.type == BackoffType.LINEAR
        assert config.backoff.multiplier == 2.0

    def test_function_without_context_parameter(self) -> None:
        """Test that functions can omit the context parameter."""
        @function
        async def add(a: int, b: int) -> int:
            return a + b

        config = FunctionRegistry.get("add")
        assert config is not None
        assert config.name == "add"

    def test_sync_function_converted_to_async(self) -> None:
        @function
        def sync_func(ctx: FunctionContext, value: int) -> int:
            return value * 2

        config = FunctionRegistry.get("sync_func")
        assert config is not None

        # Should be callable as async
        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        result = asyncio.run(sync_func(ctx, 5))
        assert result == 10


class TestFunctionExecution:
    """Test function execution."""

    @pytest.mark.asyncio
    async def test_basic_execution_with_context(self) -> None:
        @function
        async def add(ctx: FunctionContext, a: int, b: int) -> int:
            return a + b

        ctx = FunctionContext(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")
        result = await add(ctx, 3, 5)
        assert result == 8

    @pytest.mark.asyncio
    async def test_basic_execution_without_context(self) -> None:
        """Test functions without context parameter."""
        @function
        async def add(a: int, b: int) -> int:
            return a + b

        # Call without providing context - should auto-create internally
        result = await add(3, 5)
        assert result == 8

    @pytest.mark.asyncio
    async def test_function_without_context_requires_no_context_arg(self) -> None:
        """Functions without ctx parameter should not accept context as first arg."""
        @function
        async def multiply(x: int, y: int) -> int:
            return x * y

        # Call with values only
        result = await multiply(4, 5)
        assert result == 20

    @pytest.mark.asyncio
    async def test_function_with_context_requires_context_arg(self) -> None:
        """Functions with ctx parameter must receive context."""
        @function
        async def greet(ctx: FunctionContext, name: str) -> str:
            ctx.logger.info(f"Greeting {name}")
            return f"Hello, {name}"

        # Must provide context
        with pytest.raises(TypeError, match="requires FunctionContext as first argument"):
            await greet("Alice")  # type: ignore

        # Should work with context
        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        result = await greet(ctx, "Alice")
        assert result == "Hello, Alice"

    @pytest.mark.asyncio
    async def test_context_metadata_in_function(self) -> None:
        @function
        async def get_run_id(ctx: FunctionContext) -> str:
            return ctx.run_id

        ctx = FunctionContext(run_id="test-456", correlation_id="corr-456", parent_correlation_id="parent-456")
        result = await get_run_id(ctx)
        assert result == "test-456"


class TestFunctionExecution_PlatformRetry:
    """Test function execution with platform-level retry.

    Note: Retry execution is now handled by the platform (Execution Engine).
    The SDK executes functions once and reports results; the platform orchestrates retries.
    These tests verify:
    - Functions execute exactly once per invocation
    - Exceptions are propagated to caller (platform will decide whether to retry)
    - ctx.attempt reflects the platform-provided attempt number
    """

    @pytest.mark.asyncio
    async def test_successful_execution_single_call(self) -> None:
        """Test that successful function executes exactly once."""
        call_count = 0

        @function
        async def success_func(ctx: FunctionContext) -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        result = await success_func(ctx)

        assert result == "success"
        assert call_count == 1  # Executes exactly once

    @pytest.mark.asyncio
    async def test_exception_propagated_to_platform(self) -> None:
        """Test that exceptions are propagated (platform handles retry decisions)."""
        call_count = 0

        @function(retries=MAX_TEST_RETRIES)  # Config still flows to platform
        async def failing_func(ctx: FunctionContext) -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Simulated failure")

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")

        # Exception is propagated directly - no RetryError wrapping
        with pytest.raises(ValueError, match="Simulated failure"):
            await failing_func(ctx)

        # SDK executes only once - platform will call again for retries
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_context_attempt_from_platform(self) -> None:
        """Test that ctx.attempt reflects platform-provided attempt number."""
        attempt_seen = None

        @function
        async def check_attempt(ctx: FunctionContext) -> int:
            nonlocal attempt_seen
            attempt_seen = ctx.attempt
            return ctx.attempt

        # Simulate platform's retry attempt (attempt=2 means third try)
        ctx = FunctionContext(
            run_id="test",
            correlation_id="corr",
            parent_correlation_id="parent",
            attempt=2,  # Platform provides this on retry
        )
        result = await check_attempt(ctx)

        assert attempt_seen == 2
        assert result == 2

    @pytest.mark.asyncio
    async def test_retry_config_flows_to_registry(self) -> None:
        """Test that retry config is captured and available for platform."""
        @function(retries={"max_attempts": 5, "initial_interval_ms": 1000})
        async def configured_func(ctx: FunctionContext) -> str:
            return "done"

        config = FunctionRegistry.get("configured_func")
        assert config is not None
        assert config.retries is not None
        assert config.retries.max_attempts == 5
        assert config.retries.initial_interval_ms == 1000

    @pytest.mark.asyncio
    async def test_function_without_context_executes_once(self) -> None:
        """Test functions without context also execute exactly once."""
        call_count = 0

        @function(retries=MAX_TEST_RETRIES)
        async def simple_add(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        result = await simple_add(3, 5)
        assert result == 8
        assert call_count == 1


class TestBackoffCalculation:
    """Test backoff delay calculation utility.

    Note: The calculate_backoff_delay function is used by the platform for retry scheduling.
    These tests verify the utility function works correctly.
    """

    def test_exponential_backoff_calculation(self) -> None:
        """Test exponential backoff delay calculation."""
        from agnt5._retry_utils import calculate_backoff_delay

        retry_policy = RetryPolicy(initial_interval_ms=100, max_interval_ms=10000)
        backoff_policy = BackoffPolicy(type=BackoffType.EXPONENTIAL, multiplier=2.0)

        # Attempt 0: 100ms * 2^0 = 100ms
        delay = calculate_backoff_delay(0, retry_policy, backoff_policy)
        assert delay == 0.1  # 100ms in seconds

        # Attempt 1: 100ms * 2^1 = 200ms
        delay = calculate_backoff_delay(1, retry_policy, backoff_policy)
        assert delay == 0.2

        # Attempt 2: 100ms * 2^2 = 400ms
        delay = calculate_backoff_delay(2, retry_policy, backoff_policy)
        assert delay == 0.4

    def test_linear_backoff_calculation(self) -> None:
        """Test linear backoff delay calculation."""
        from agnt5._retry_utils import calculate_backoff_delay

        retry_policy = RetryPolicy(initial_interval_ms=100, max_interval_ms=10000)
        backoff_policy = BackoffPolicy(type=BackoffType.LINEAR)

        # Attempt 0: 100ms * (0+1) = 100ms
        delay = calculate_backoff_delay(0, retry_policy, backoff_policy)
        assert delay == 0.1

        # Attempt 1: 100ms * (1+1) = 200ms
        delay = calculate_backoff_delay(1, retry_policy, backoff_policy)
        assert delay == 0.2

        # Attempt 2: 100ms * (2+1) = 300ms
        delay = calculate_backoff_delay(2, retry_policy, backoff_policy)
        assert delay == 0.3

    def test_constant_backoff_calculation(self) -> None:
        """Test constant backoff delay calculation."""
        from agnt5._retry_utils import calculate_backoff_delay

        retry_policy = RetryPolicy(initial_interval_ms=100, max_interval_ms=10000)
        backoff_policy = BackoffPolicy(type=BackoffType.CONSTANT)

        # All attempts should have same delay
        for attempt in range(5):
            delay = calculate_backoff_delay(attempt, retry_policy, backoff_policy)
            assert delay == 0.1  # Always 100ms

    def test_backoff_respects_max_interval(self) -> None:
        """Test that backoff delay is capped at max_interval_ms."""
        from agnt5._retry_utils import calculate_backoff_delay

        retry_policy = RetryPolicy(initial_interval_ms=100, max_interval_ms=500)
        backoff_policy = BackoffPolicy(type=BackoffType.EXPONENTIAL, multiplier=2.0)

        # Attempt 10: would be 100ms * 2^10 = 102400ms, but capped at 500ms
        delay = calculate_backoff_delay(10, retry_policy, backoff_policy)
        assert delay == 0.5  # Capped at 500ms


class TestFunctionRegistry:
    """Test function registry."""

    def test_registry_get(self) -> None:
        @function
        async def test_func(ctx: FunctionContext) -> str:
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
        async def func1(ctx: FunctionContext) -> str:
            return "1"

        @function
        async def func2(ctx: FunctionContext) -> str:
            return "2"

        all_funcs = FunctionRegistry.all()
        assert "func1" in all_funcs
        assert "func2" in all_funcs
        assert len(all_funcs) >= 2

    def test_registry_clear(self) -> None:
        @function
        async def temp_func(ctx: FunctionContext) -> str:
            return "temp"

        assert FunctionRegistry.get("temp_func") is not None

        FunctionRegistry.clear()

        assert FunctionRegistry.get("temp_func") is None

    def test_name_collision_detection(self) -> None:
        """Test that registering duplicate names raises error."""
        FunctionRegistry.clear()

        @function
        async def my_function(ctx: FunctionContext) -> str:
            return "first"

        # Trying to register another function with same name should fail
        with pytest.raises(ValueError, match="name collision"):
            @function
            async def my_function(ctx: FunctionContext) -> str:  # noqa: F811
                return "second"

    def test_name_collision_with_custom_name(self) -> None:
        """Test name collision with custom names."""
        FunctionRegistry.clear()

        @function(name="duplicate_name")
        async def func1(ctx: FunctionContext) -> str:
            return "first"

        # Using same custom name should fail
        with pytest.raises(ValueError, match="name collision"):
            @function(name="duplicate_name")
            async def func2(ctx: FunctionContext) -> str:
                return "second"


class TestPydanticIntegration:
    """Test Pydantic model support."""

    @pytest.mark.asyncio
    async def test_function_with_pydantic_input(self) -> None:
        """Test functions can use Pydantic models for input."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class UserInput(BaseModel):
            name: str
            age: int

        @function
        async def process_user(ctx: FunctionContext, user: UserInput) -> str:
            return f"{user.name} is {user.age} years old"

        # Check schema was extracted
        config = FunctionRegistry.get("process_user")
        assert config is not None
        assert config.input_schema is not None
        assert "properties" in config.input_schema

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        user = UserInput(name="Alice", age=30)
        result = await process_user(ctx, user)
        assert result == "Alice is 30 years old"

    @pytest.mark.asyncio
    async def test_function_with_pydantic_output(self) -> None:
        """Test functions can use Pydantic models for output."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class UserOutput(BaseModel):
            greeting: str
            user_id: str

        @function
        async def create_greeting(ctx: FunctionContext, name: str) -> UserOutput:
            return UserOutput(
                greeting=f"Hello, {name}!",
                user_id=f"user_{name.lower()}"
            )

        # Check output schema was extracted
        config = FunctionRegistry.get("create_greeting")
        assert config is not None
        assert config.output_schema is not None

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        result = await create_greeting(ctx, "Bob")
        assert result.greeting == "Hello, Bob!"
        assert result.user_id == "user_bob"


class TestSchemaExtraction:
    """Test automatic schema extraction from type hints."""

    def test_basic_type_hints_extracted(self) -> None:
        """Test that basic Python type hints are extracted."""
        @function
        async def typed_function(ctx: FunctionContext, name: str, age: int, active: bool) -> dict:
            return {"name": name, "age": age, "active": active}

        config = FunctionRegistry.get("typed_function")
        assert config is not None
        assert config.input_schema is not None

        # Check properties were extracted
        props = config.input_schema.get("properties", {})
        assert "name" in props
        assert "age" in props
        assert "active" in props

        # Check types
        assert props["name"]["type"] == "string"
        assert props["age"]["type"] == "integer"
        assert props["active"]["type"] == "boolean"

    def test_optional_parameters_not_required(self) -> None:
        """Test that parameters with defaults are not in required list."""
        @function
        async def with_defaults(ctx: FunctionContext, required: str, optional: str = "default") -> str:
            return f"{required}-{optional}"

        config = FunctionRegistry.get("with_defaults")
        assert config is not None
        assert config.input_schema is not None

        # Only 'required' should be in required list
        required_params = config.input_schema.get("required", [])
        assert "required" in required_params
        assert "optional" not in required_params

    def test_return_type_extracted(self) -> None:
        """Test that return type hints are extracted."""
        @function
        async def returns_dict(ctx: FunctionContext, x: int) -> dict:
            return {"value": x}

        config = FunctionRegistry.get("returns_dict")
        assert config is not None
        assert config.output_schema is not None
        assert config.output_schema["type"] == "object"
