"""
Tests for Workflow component.

Tests cover:
- Workflow registration with @workflow decorator
- Workflow execution with context
- Sequential and parallel orchestration
- Task invocation (ctx.task)
- Signal coordination
- Workflow registry
"""

import asyncio

import pytest

from agnt5 import Context, FunctionRegistry, WorkflowRegistry, function, workflow


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear registries before each test."""
    WorkflowRegistry.clear()
    FunctionRegistry.clear()
    yield
    WorkflowRegistry.clear()
    FunctionRegistry.clear()


# Test Workflow Registration


def test_workflow_decorator():
    """Test @workflow decorator registers workflow."""

    @workflow
    async def simple_workflow(ctx: Context, name: str) -> str:
        """Simple workflow."""
        return f"Hello, {name}"

    assert "simple_workflow" in WorkflowRegistry.list_names()
    config = WorkflowRegistry.get("simple_workflow")
    assert config is not None
    assert config.name == "simple_workflow"


def test_workflow_custom_name():
    """Test @workflow with custom name."""

    @workflow(name="custom_name")
    async def my_workflow(ctx: Context) -> None:
        pass

    assert "custom_name" in WorkflowRegistry.list_names()
    assert WorkflowRegistry.get("custom_name") is not None


def test_workflow_decorator_wrong_signature():
    """Test @workflow fails without Context parameter."""
    with pytest.raises(ValueError, match="must have 'ctx: Context'"):

        @workflow
        def bad_workflow(name: str):
            pass


def test_workflow_sync_function_converted():
    """Test sync workflow is converted to async."""

    @workflow
    def sync_workflow(ctx: Context) -> str:
        """Sync workflow."""
        return "sync"

    config = WorkflowRegistry.get("sync_workflow")
    assert config is not None
    assert asyncio.iscoroutinefunction(config.handler)


# Test Workflow Execution


@pytest.mark.asyncio
async def test_workflow_execution_with_context():
    """Test workflow execution with provided context."""

    @workflow
    async def echo_workflow(ctx: Context, message: str) -> str:
        """Echo workflow."""
        return f"Echo: {message}"

    ctx = Context(run_id="test-123", component_type="workflow")
    result = await echo_workflow(ctx, "Hello")
    assert result == "Echo: Hello"


@pytest.mark.asyncio
async def test_workflow_execution_auto_context():
    """Test workflow auto-creates context if not provided."""

    @workflow
    async def auto_ctx_workflow(ctx: Context) -> str:
        """Workflow with auto context."""
        assert ctx.run_id.startswith("workflow-")
        assert ctx.component_type == "workflow"
        return "done"

    # Call without context - should auto-create
    result = await auto_ctx_workflow()
    assert result == "done"


@pytest.mark.asyncio
async def test_workflow_state_management():
    """Test workflow can use context state."""

    @workflow
    async def stateful_workflow(ctx: Context, value: int) -> int:
        """Workflow with state."""
        ctx.set("value", value)
        ctx.set("doubled", value * 2)
        return ctx.get("doubled")

    ctx = Context(run_id="test-123")
    result = await stateful_workflow(ctx, 21)
    assert result == 42
    assert ctx.get("value") == 21


# Test Sequential Execution


@pytest.mark.asyncio
async def test_workflow_sequential_tasks():
    """Test workflow with sequential task execution."""

    @function
    async def step1(ctx: Context) -> int:
        """First step."""
        return 1

    @function
    async def step2(ctx: Context, input: int) -> int:
        """Second step."""
        return input + 1

    @workflow
    async def sequential_workflow(ctx: Context) -> int:
        """Sequential workflow."""
        result1 = await ctx.task("service", "step1")
        result2 = await ctx.task("service", "step2", input=result1)
        return result2

    result = await sequential_workflow()
    assert result == 2


# Test Parallel Execution


@pytest.mark.asyncio
async def test_workflow_parallel_tasks():
    """Test workflow with parallel execution using ctx.parallel()."""

    @function
    async def fast_task(ctx: Context) -> str:
        """Fast task."""
        await asyncio.sleep(0.01)
        return "fast"

    @function
    async def slow_task(ctx: Context) -> str:
        """Slow task."""
        await asyncio.sleep(0.02)
        return "slow"

    @workflow
    async def parallel_workflow(ctx: Context) -> list:
        """Parallel workflow."""
        results = await ctx.parallel(
            ctx.task("service", "fast_task"), ctx.task("service", "slow_task")
        )
        return results

    import time

    start = time.time()
    results = await parallel_workflow()
    elapsed = time.time() - start

    assert results == ["fast", "slow"]
    # Should take ~0.02s (slow task time), not 0.03s (sequential)
    assert elapsed < 0.04


@pytest.mark.asyncio
async def test_workflow_gather_named():
    """Test workflow with named parallel execution using ctx.gather()."""

    @function
    async def task_a(ctx: Context) -> str:
        """Task A."""
        await asyncio.sleep(0.01)
        return "A"

    @function
    async def task_b(ctx: Context) -> str:
        """Task B."""
        await asyncio.sleep(0.01)
        return "B"

    @workflow
    async def gather_workflow(ctx: Context) -> dict:
        """Workflow with gather."""
        results = await ctx.gather(
            first=ctx.task("service", "task_a"), second=ctx.task("service", "task_b")
        )
        return results

    results = await gather_workflow()
    assert results == {"first": "A", "second": "B"}


# Test Complex Workflows


@pytest.mark.asyncio
async def test_workflow_conditional_logic():
    """Test workflow with conditional logic."""

    @function
    async def check_value(ctx: Context, value: int) -> bool:
        """Check if value is positive."""
        return value > 0

    @function
    async def process_positive(ctx: Context) -> str:
        """Process positive value."""
        return "positive"

    @function
    async def process_negative(ctx: Context) -> str:
        """Process negative value."""
        return "negative"

    @workflow
    async def conditional_workflow(ctx: Context, value: int) -> str:
        """Conditional workflow."""
        is_positive = await ctx.task("service", "check_value", input=value)

        if is_positive:
            result = await ctx.task("service", "process_positive")
        else:
            result = await ctx.task("service", "process_negative")

        return result

    assert await conditional_workflow(value=10) == "positive"
    assert await conditional_workflow(value=-5) == "negative"


@pytest.mark.asyncio
async def test_workflow_with_loops():
    """Test workflow with loops."""

    @function
    async def increment(ctx: Context, value: int) -> int:
        """Increment value."""
        return value + 1

    @workflow
    async def loop_workflow(ctx: Context, iterations: int) -> int:
        """Workflow with loop."""
        value = 0
        for i in range(iterations):
            value = await ctx.task("service", "increment", input=value)
        return value

    result = await loop_workflow(iterations=5)
    assert result == 5


# Test Signal Coordination


@pytest.mark.asyncio
async def test_workflow_with_signals():
    """Test workflow signal coordination."""

    @workflow
    async def signal_workflow(ctx: Context) -> dict:
        """Workflow with signal."""
        ctx.set("status", "waiting")

        # Send signal after delay
        async def send_signal():
            await asyncio.sleep(0.05)
            ctx.signal_send("approval", {"approved": True})

        # Start background task
        asyncio.create_task(send_signal())

        # Wait for signal
        approval = await ctx.signal("approval", timeout_ms=1000)
        ctx.set("status", "received")

        return {"approved": approval["approved"]}

    result = await signal_workflow()
    assert result["approved"] is True


@pytest.mark.asyncio
async def test_workflow_signal_timeout():
    """Test workflow signal timeout."""

    @workflow
    async def timeout_workflow(ctx: Context) -> dict:
        """Workflow with signal timeout."""
        # Wait for signal that never comes
        result = await ctx.signal("missing_signal", timeout_ms=10, default={"timeout": True})
        return result

    result = await timeout_workflow()
    assert result["timeout"] is True


# Test Workflow Registry


def test_workflow_registry_list():
    """Test WorkflowRegistry.list_names()."""

    @workflow
    def wf1(ctx: Context):
        pass

    @workflow
    def wf2(ctx: Context):
        pass

    names = WorkflowRegistry.list_names()
    assert len(names) == 2
    assert "wf1" in names
    assert "wf2" in names


def test_workflow_registry_get():
    """Test WorkflowRegistry.get()."""

    @workflow
    def my_workflow(ctx: Context):
        """My workflow."""
        pass

    config = WorkflowRegistry.get("my_workflow")
    assert config is not None
    assert config.name == "my_workflow"

    missing = WorkflowRegistry.get("nonexistent")
    assert missing is None


def test_workflow_registry_all():
    """Test WorkflowRegistry.all()."""

    @workflow
    def wf1(ctx: Context):
        pass

    @workflow
    def wf2(ctx: Context):
        pass

    all_workflows = WorkflowRegistry.all()
    assert len(all_workflows) == 2
    assert "wf1" in all_workflows
    assert "wf2" in all_workflows


def test_workflow_registry_clear():
    """Test WorkflowRegistry.clear()."""

    @workflow
    def temp_workflow(ctx: Context):
        pass

    assert len(WorkflowRegistry.list_names()) > 0

    WorkflowRegistry.clear()
    assert len(WorkflowRegistry.list_names()) == 0


# Test Delays


@pytest.mark.asyncio
async def test_workflow_with_sleep():
    """Test workflow with ctx.sleep()."""

    @workflow
    async def sleep_workflow(ctx: Context) -> str:
        """Workflow with sleep."""
        await ctx.sleep(0.05)
        return "done"

    import time

    start = time.time()
    result = await sleep_workflow()
    elapsed = time.time() - start

    assert result == "done"
    assert elapsed >= 0.05


@pytest.mark.asyncio
async def test_workflow_with_timer():
    """Test workflow with ctx.timer()."""

    @workflow
    async def timer_workflow(ctx: Context) -> str:
        """Workflow with timer."""
        await ctx.timer(delay_ms=50)
        return "done"

    import time

    start = time.time()
    result = await timer_workflow()
    elapsed = time.time() - start

    assert result == "done"
    assert elapsed >= 0.05


# Test Error Handling


@pytest.mark.asyncio
async def test_workflow_function_not_found():
    """Test workflow fails when function not found."""

    @workflow
    async def broken_workflow(ctx: Context) -> None:
        """Workflow calling missing function."""
        await ctx.task("service", "missing_function")

    with pytest.raises(ValueError, match="not found in registry"):
        await broken_workflow()


@pytest.mark.asyncio
async def test_workflow_function_error_propagation():
    """Test errors in functions propagate to workflow."""

    @function
    async def failing_function(ctx: Context) -> None:
        """Function that raises error."""
        raise ValueError("Function failed")

    @workflow
    async def error_workflow(ctx: Context) -> None:
        """Workflow calling failing function."""
        await ctx.task("service", "failing_function")

    with pytest.raises(ValueError, match="Function failed"):
        await error_workflow()
