"""Tests for Context implementation."""

import asyncio

import pytest

from agnt5 import Context
from agnt5.exceptions import NotImplementedError as AGNT5NotImplementedError
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
    """Test Context state management."""

    def test_get_set(self) -> None:
        ctx = Context(run_id="test-123")

        # Set value
        ctx.set("key1", "value1")

        # Get value
        result = ctx.get("key1")
        assert result == "value1"

    def test_get_default(self) -> None:
        ctx = Context(run_id="test-123")

        # Get non-existent key with default
        result = ctx.get("missing_key", default="default_value")
        assert result == "default_value"

    def test_get_no_default(self) -> None:
        ctx = Context(run_id="test-123")

        # Get non-existent key without default
        result = ctx.get("missing_key")
        assert result is None

    def test_delete_existing_key(self) -> None:
        ctx = Context(run_id="test-123")

        # Set and delete
        ctx.set("key1", "value1")
        ctx.delete("key1")

        # Verify deleted
        result = ctx.get("key1")
        assert result is None

    def test_delete_missing_key(self) -> None:
        ctx = Context(run_id="test-123")

        # Delete non-existent key should raise
        with pytest.raises(StateError, match="Key 'missing' not found"):
            ctx.delete("missing")

    def test_state_types(self) -> None:
        ctx = Context(run_id="test-123")

        # Test various types
        ctx.set("string", "hello")
        ctx.set("int", 42)
        ctx.set("float", 3.14)
        ctx.set("bool", True)
        ctx.set("list", [1, 2, 3])
        ctx.set("dict", {"a": 1, "b": 2})

        assert ctx.get("string") == "hello"
        assert ctx.get("int") == 42
        assert ctx.get("float") == 3.14
        assert ctx.get("bool") is True
        assert ctx.get("list") == [1, 2, 3]
        assert ctx.get("dict") == {"a": 1, "b": 2}


class TestCheckpointing:
    """Test Context checkpointing."""

    @pytest.mark.asyncio
    async def test_step_basic(self) -> None:
        ctx = Context(run_id="test-123")
        call_count = 0

        async def expensive_op() -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        # First call executes
        result1 = await ctx.step("step1", expensive_op)
        assert result1 == "result"
        assert call_count == 1

        # Second call returns cached result
        result2 = await ctx.step("step1", expensive_op)
        assert result2 == "result"
        assert call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_run_alias(self) -> None:
        ctx = Context(run_id="test-123")

        async def operation() -> int:
            return 42

        result = await ctx.step("op1", operation)
        assert result == 42

        # Verify it's checkpointed
        call_count = 0

        async def counting_op() -> int:
            nonlocal call_count
            call_count += 1
            return 100

        result1 = await ctx.step("op2", counting_op)
        result2 = await ctx.step("op2", counting_op)
        assert result1 == result2 == 100
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_steps(self) -> None:
        ctx = Context(run_id="test-123")

        async def step1_func() -> str:
            return "a"

        async def step2_func() -> str:
            return "b"

        async def step3_func() -> str:
            return "c"

        result1 = await ctx.step("step1", step1_func)
        result2 = await ctx.step("step2", step2_func)
        result3 = await ctx.step("step3", step3_func)

        assert result1 == "a"
        assert result2 == "b"
        assert result3 == "c"


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

    @pytest.mark.asyncio
    async def test_timer_cron_not_implemented(self) -> None:
        ctx = Context(run_id="test-123")
        with pytest.raises(AGNT5NotImplementedError, match="cron"):
            await ctx.timer(cron="*/5 * * * *")

    def test_spawn_not_implemented(self) -> None:
        ctx = Context(run_id="test-123")

        async def dummy() -> int:
            return 1

        with pytest.raises(AGNT5NotImplementedError, match="ctx.spawn"):
            ctx.spawn(dummy)

    def test_llm_not_implemented(self) -> None:
        ctx = Context(run_id="test-123")
        with pytest.raises(AGNT5NotImplementedError, match="ctx.llm"):
            _ = ctx.llm

    def test_entity_not_implemented(self) -> None:
        ctx = Context(run_id="test-123")
        with pytest.raises(AGNT5NotImplementedError, match="ctx.entity"):
            ctx.entity("MyEntity", "key-123")
