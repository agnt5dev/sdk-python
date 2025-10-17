"""Tests for Context implementation."""

import asyncio

import pytest

from agnt5 import Context
from agnt5.exceptions import StateError


class TestContextMetadata:
    """Test Context metadata properties."""

    def test_run_id(self) -> None:
        ctx = Context(run_id="test-123")
        assert ctx.run_id == "test-123"

    def test_component_type_default(self) -> None:
        ctx = Context(run_id="test-123")
        assert ctx.component_type == "function"

    def test_component_type_custom(self) -> None:
        ctx = Context(run_id="test-123", component_type="workflow")
        assert ctx.component_type == "workflow"

    def test_step_id(self) -> None:
        ctx = Context(run_id="test-123", step_id="step-1")
        assert ctx.step_id == "step-1"

    def test_attempt_default(self) -> None:
        ctx = Context(run_id="test-123")
        assert ctx.attempt == 0

    def test_attempt_custom(self) -> None:
        ctx = Context(run_id="test-123", attempt=2)
        assert ctx.attempt == 2


class TestStateManagement:
    """Test Context state management (async API)."""

    @pytest.mark.asyncio
    async def test_get_set(self) -> None:
        ctx = Context(run_id="test-123")

        # Set value
        await ctx.set("key1", "value1")

        # Get value
        result = await ctx.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_default(self) -> None:
        ctx = Context(run_id="test-123")

        # Get non-existent key with default
        result = await ctx.get("missing_key", default="default_value")
        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_get_no_default(self) -> None:
        ctx = Context(run_id="test-123")

        # Get non-existent key without default
        result = await ctx.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing_key(self) -> None:
        ctx = Context(run_id="test-123")

        # Set and delete
        await ctx.set("key1", "value1")
        await ctx.delete("key1")

        # Verify deleted
        result = await ctx.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_missing_key(self) -> None:
        ctx = Context(run_id="test-123")

        # Delete non-existent key should raise
        with pytest.raises(StateError, match="Key 'missing' not found"):
            await ctx.delete("missing")

    @pytest.mark.asyncio
    async def test_state_types(self) -> None:
        ctx = Context(run_id="test-123")

        # Test various types
        await ctx.set("string", "hello")
        await ctx.set("int", 42)
        await ctx.set("float", 3.14)
        await ctx.set("bool", True)
        await ctx.set("list", [1, 2, 3])
        await ctx.set("dict", {"a": 1, "b": 2})

        assert await ctx.get("string") == "hello"
        assert await ctx.get("int") == 42
        assert await ctx.get("float") == 3.14
        assert await ctx.get("bool") is True
        assert await ctx.get("list") == [1, 2, 3]
        assert await ctx.get("dict") == {"a": 1, "b": 2}


class TestCheckpointing:
    """Test Context checkpointing (improved API)."""

    @pytest.mark.asyncio
    async def test_step_basic(self) -> None:
        ctx = Context(run_id="test-123")
        call_count = 0

        async def expensive_op() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        # First call executes
        result1 = await ctx.step("step1", expensive_op())
        assert result1 == "result"
        assert call_count == 1

        # Second call returns cached result (pass function reference to avoid creating coroutine)
        result2 = await ctx.step("step1", expensive_op)
        assert result2 == "result"
        assert call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_step_with_function_reference(self) -> None:
        """Test step() can accept function reference (called automatically)."""
        ctx = Context(run_id="test-123")
        call_count = 0

        async def operation() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        # Pass function reference - should be called automatically
        result = await ctx.step("op1", operation)
        assert result == 42
        assert call_count == 1

        # Second call should use cached result
        result2 = await ctx.step("op1", operation)
        assert result2 == 42
        assert call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_step_with_coroutine(self) -> None:
        """Test step() accepts coroutines directly (recommended)."""
        ctx = Context(run_id="test-123")
        call_count = 0

        async def counting_op() -> int:
            nonlocal call_count
            call_count += 1
            return 100

        # Pass coroutine directly (recommended)
        result1 = await ctx.step("op2", counting_op())
        assert result1 == 100
        assert call_count == 1

        # Second call should use cached result (pass function reference)
        result2 = await ctx.step("op2", counting_op)
        assert result2 == 100
        assert call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_multiple_steps(self) -> None:
        ctx = Context(run_id="test-123")

        async def step1_func() -> str:
            return "a"

        async def step2_func() -> str:
            return "b"

        async def step3_func() -> str:
            return "c"

        # Pass coroutines directly
        result1 = await ctx.step("step1", step1_func())
        result2 = await ctx.step("step2", step2_func())
        result3 = await ctx.step("step3", step3_func())

        assert result1 == "a"
        assert result2 == "b"
        assert result3 == "c"

    @pytest.mark.asyncio
    async def test_step_backward_compatible_with_lambda(self) -> None:
        """Test that lambda wrapping still works (backward compatibility)."""
        ctx = Context(run_id="test-123")

        async def operation() -> str:
            return "legacy"

        # Old style with lambda (still supported)
        result = await ctx.step("legacy_op", lambda: operation())
        assert result == "legacy"


class TestLogging:
    """Test Context logging."""

    def test_log_returns_logger(self) -> None:
        ctx = Context(run_id="test-123")
        logger = ctx.log()
        assert logger is not None
        assert "test-123" in logger.name

    def test_logger_property(self) -> None:
        ctx = Context(run_id="test-123")
        assert ctx.logger is ctx.log()


class TestOrchestration:
    """Test orchestration features (Phase 1: in-memory implementation)."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        ctx = Context(run_id="test-123")

        async def task1() -> int:
            await asyncio.sleep(0.01)
            return 1

        async def task2() -> int:
            await asyncio.sleep(0.01)
            return 2

        results = await ctx.parallel(task1(), task2())
        assert results == [1, 2]

    @pytest.mark.asyncio
    async def test_gather_named_results(self) -> None:
        ctx = Context(run_id="test-123")

        async def task_a() -> str:
            await asyncio.sleep(0.01)
            return "result_a"

        async def task_b() -> str:
            await asyncio.sleep(0.01)
            return "result_b"

        results = await ctx.gather(a=task_a(), b=task_b())
        assert results == {"a": "result_a", "b": "result_b"}

    @pytest.mark.asyncio
    async def test_sleep(self) -> None:
        ctx = Context(run_id="test-123")
        import time

        start = time.time()
        await ctx.sleep(0.05)
        elapsed = time.time() - start
        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_timer_delay(self) -> None:
        ctx = Context(run_id="test-123")
        import time

        start = time.time()
        await ctx.timer(delay_ms=50)
        elapsed = time.time() - start
        assert elapsed >= 0.05


class TestSignaling:
    """Test signal coordination primitives."""

    @pytest.mark.asyncio
    async def test_signal_basic(self) -> None:
        """Test basic signal send and receive."""
        ctx = Context(run_id="test-123")

        # Send signal
        ctx.signal_send("test_signal", {"data": "value"})

        # Receive signal
        result = await ctx.signal("test_signal")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_signal_with_timeout(self) -> None:
        """Test signal with timeout returns default."""
        ctx = Context(run_id="test-123")

        # No signal sent - should timeout and return default
        result = await ctx.signal("missing_signal", timeout_ms=50, default="timeout_value")
        assert result == "timeout_value"

    @pytest.mark.asyncio
    async def test_signal_concurrent(self) -> None:
        """Test signal coordination between concurrent tasks."""
        ctx = Context(run_id="test-123")
        results = []

        async def waiter():
            # Wait for signal
            value = await ctx.signal("go_signal")
            results.append(value)

        async def sender():
            # Small delay then send signal
            await asyncio.sleep(0.05)
            ctx.signal_send("go_signal", "proceed")

        # Run both concurrently
        await asyncio.gather(waiter(), sender())

        assert results == ["proceed"]


class TestTypeSafeFunctionCalls:
    """Test type-safe function calls via ctx.task()."""

    @pytest.mark.asyncio
    async def test_task_with_function_reference(self) -> None:
        """Test calling functions with type-safe references."""
        from agnt5 import function

        # Define a test function
        @function
        async def helper_func(ctx: Context, value: int) -> int:
            return value * 2

        # Create context
        ctx = Context(run_id="test-123", component_type="workflow")

        # Call function with type-safe reference
        result = await ctx.task(helper_func, input=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_task_with_string_backward_compatible(self) -> None:
        """Test that string-based function calls still work."""
        from agnt5 import function

        @function
        async def my_function(ctx: Context, data: str) -> str:
            return data.upper()

        ctx = Context(run_id="test-123", component_type="workflow")

        # String-based call (backward compatible)
        result = await ctx.task("my_function", input="hello")
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_task_with_unregistered_function_raises(self) -> None:
        """Test that calling unregistered function raises error."""
        ctx = Context(run_id="test-123", component_type="workflow")

        # Should raise error for non-existent function
        with pytest.raises(ValueError, match="not found in registry"):
            await ctx.task("nonexistent_function", input="data")

    @pytest.mark.asyncio
    async def test_task_with_non_decorated_function_raises(self) -> None:
        """Test that passing non-decorated function raises error."""
        ctx = Context(run_id="test-123", component_type="workflow")

        # Plain function without @function decorator
        async def plain_func(ctx: Context, x: int) -> int:
            return x

        # Should raise error because it's not decorated
        with pytest.raises(ValueError, match="not a registered @function"):
            await ctx.task(plain_func, input=5)
