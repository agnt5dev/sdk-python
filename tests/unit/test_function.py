"""Tests for Function decorator and retry logic."""

import asyncio
import time

import pytest

from agnt5 import BackoffPolicy, BackoffType, FunctionContext, FunctionRegistry, RetryPolicy, function
from agnt5.exceptions import RetryError

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


class TestRetryLogic:
    """Test retry behavior."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self) -> None:
        call_count = 0

        @function
        async def success_func(ctx: FunctionContext) -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        result = await success_func(ctx)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_until_success(self) -> None:
        call_count = 0

        @function(retries={"max_attempts": MAX_TEST_RETRIES, "initial_interval_ms": TEST_INTERVAL_MS})
        async def flaky_func(ctx: FunctionContext) -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise Exception("Transient error")
            return "success"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        result = await flaky_func(ctx)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exceeds_max_attempts(self) -> None:
        call_count = 0

        @function(retries=MAX_TEST_RETRIES)  # Simple form
        async def always_fails(ctx: FunctionContext) -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent error")

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")

        with pytest.raises(RetryError) as exc_info:
            await always_fails(ctx)

        assert call_count == MAX_TEST_RETRIES
        assert exc_info.value.attempts == MAX_TEST_RETRIES
        assert "Permanent error" in str(exc_info.value.last_error)

    @pytest.mark.asyncio
    async def test_context_attempt_increments(self) -> None:
        attempts_seen = []

        @function(retries={"max_attempts": MAX_TEST_RETRIES, "initial_interval_ms": TEST_INTERVAL_MS})
        async def track_attempts(ctx: FunctionContext) -> str:
            attempts_seen.append(ctx.attempt)
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        await track_attempts(ctx)

        assert attempts_seen == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_retry_with_function_without_context(self) -> None:
        """Test retry logic works for functions without context."""
        call_count = 0

        @function(retries=MAX_TEST_RETRIES)
        async def flaky_add(a: int, b: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Transient error")
            return a + b

        result = await flaky_add(3, 5)
        assert result == 8
        assert call_count == 2


class TestBackoffCalculation:
    """Test backoff delay calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        @function(
            retries={"max_attempts": MAX_TEST_RETRIES, "initial_interval_ms": BACKOFF_INTERVAL_MS, "max_interval_ms": BACKOFF_MAX_INTERVAL_MS},
            backoff="exponential",  # Simple string form
        )
        async def exponential_func(ctx: FunctionContext) -> str:
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        start = time.time()
        await exponential_func(ctx)
        duration = time.time() - start

        # Should take roughly: 100ms + 200ms = 300ms (with some tolerance)
        assert duration >= 0.25  # At least 250ms
        assert duration < 0.5  # Less than 500ms

    @pytest.mark.asyncio
    async def test_linear_backoff(self) -> None:
        @function(
            retries={"max_attempts": MAX_TEST_RETRIES, "initial_interval_ms": BACKOFF_INTERVAL_MS},
            backoff={"type": "linear", "multiplier": 1.0},  # Dict form
        )
        async def linear_func(ctx: FunctionContext) -> str:
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
        start = time.time()
        await linear_func(ctx)
        duration = time.time() - start

        # Should take roughly: 100ms + 200ms = 300ms
        assert duration >= 0.25
        assert duration < 0.5

    @pytest.mark.asyncio
    async def test_constant_backoff(self) -> None:
        @function(
            retries={"max_attempts": MAX_TEST_RETRIES, "initial_interval_ms": BACKOFF_INTERVAL_MS},
            backoff="constant",  # Simple string form
        )
        async def constant_func(ctx: FunctionContext) -> str:
            if ctx.attempt < 2:
                raise Exception("Retry")
            return "done"

        ctx = FunctionContext(run_id="test", correlation_id="corr", parent_correlation_id="parent")
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
