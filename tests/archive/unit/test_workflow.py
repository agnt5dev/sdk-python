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

from agnt5 import FunctionContext, WorkflowContext, FunctionRegistry, WorkflowRegistry, function, workflow
from agnt5.workflow import WorkflowEntity


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
    async def simple_workflow(ctx: WorkflowContext, name: str) -> str:
        """Simple workflow."""
        return f"Hello, {name}"

    assert "simple_workflow" in WorkflowRegistry.list_names()
    config = WorkflowRegistry.get("simple_workflow")
    assert config is not None
    assert config.name == "simple_workflow"


def test_workflow_custom_name():
    """Test @workflow with custom name."""

    @workflow(name="custom_name")
    async def my_workflow(ctx: WorkflowContext) -> None:
        pass

    assert "custom_name" in WorkflowRegistry.list_names()
    assert WorkflowRegistry.get("custom_name") is not None


def test_workflow_decorator_wrong_signature():
    """Test @workflow fails without ctx parameter."""
    with pytest.raises(ValueError, match="must have 'ctx"):

        @workflow
        def bad_workflow(name: str):
            pass


def test_workflow_sync_function_converted():
    """Test sync workflow is converted to async."""

    @workflow
    def sync_workflow(ctx: WorkflowContext) -> str:
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
    async def echo_workflow(ctx: WorkflowContext, message: str) -> str:
        """Echo workflow."""
        return f"Echo: {message}"

    # Create WorkflowEntity and WorkflowContext
    workflow_entity = WorkflowEntity(run_id="test-123")
    ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="test-123")
    result = await echo_workflow(ctx, "Hello")
    assert result == "Echo: Hello"


@pytest.mark.asyncio
async def test_workflow_execution_auto_context():
    """Test workflow auto-creates context if not provided."""

    @workflow
    async def auto_ctx_workflow(ctx: WorkflowContext) -> str:
        """Workflow with auto context."""
        assert ctx.run_id.startswith("workflow-")
        # WorkflowContext doesn't have component_type anymore
        return "done"

    # Call without context - should auto-create
    result = await auto_ctx_workflow()
    assert result == "done"


@pytest.mark.asyncio
async def test_workflow_state_management():
    """Test workflow can use context state."""
    from agnt5.entity import with_entity_context

    @with_entity_context
    async def run_test():
        @workflow
        async def stateful_workflow(ctx: WorkflowContext, value: int) -> int:
            """Workflow with state."""
            ctx.state.set("value", value)
            ctx.state.set("doubled", value * 2)
            return ctx.state.get("doubled")

        # Create WorkflowEntity and WorkflowContext
        workflow_entity = WorkflowEntity(run_id="test-123")
        ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="test-123")
        result = await stateful_workflow(ctx, 21)
        assert result == 42
        assert ctx.state.get("value") == 21

    await run_test()


# Test Sequential Execution


@pytest.mark.asyncio
async def test_workflow_sequential_tasks():
    """Test workflow with sequential task execution."""

    @function
    async def step1(ctx: FunctionContext) -> int:
        """First step."""
        return 1

    @function
    async def step2(ctx: FunctionContext, input: int) -> int:
        """Second step."""
        return input + 1

    @workflow
    async def sequential_workflow(ctx: WorkflowContext) -> int:
        """Sequential workflow."""
        result1 = await ctx.task(step1)
        result2 = await ctx.task(step2, input=result1)
        return result2

    result = await sequential_workflow()
    assert result == 2


# Test Parallel Execution


@pytest.mark.asyncio
async def test_workflow_parallel_tasks():
    """Test workflow with parallel execution using ctx.parallel()."""

    @function
    async def fast_task(ctx: FunctionContext) -> str:
        """Fast task."""
        await asyncio.sleep(0.01)
        return "fast"

    @function
    async def slow_task(ctx: FunctionContext) -> str:
        """Slow task."""
        await asyncio.sleep(0.02)
        return "slow"

    @workflow
    async def parallel_workflow(ctx: WorkflowContext) -> list:
        """Parallel workflow."""
        results = await ctx.parallel(
            ctx.task(fast_task), ctx.task(slow_task)
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
    async def task_a(ctx: FunctionContext) -> str:
        """Task A."""
        await asyncio.sleep(0.01)
        return "A"

    @function
    async def task_b(ctx: FunctionContext) -> str:
        """Task B."""
        await asyncio.sleep(0.01)
        return "B"

    @workflow
    async def gather_workflow(ctx: WorkflowContext) -> dict:
        """Workflow with gather."""
        results = await ctx.gather(
            first=ctx.task(task_a), second=ctx.task(task_b)
        )
        return results

    results = await gather_workflow()
    assert results == {"first": "A", "second": "B"}


# Test Complex Workflows


@pytest.mark.asyncio
async def test_workflow_conditional_logic():
    """Test workflow with conditional logic."""

    @function
    async def check_value(ctx: FunctionContext, value: int) -> bool:
        """Check if value is positive."""
        return value > 0

    @function
    async def process_positive(ctx: FunctionContext) -> str:
        """Process positive value."""
        return "positive"

    @function
    async def process_negative(ctx: FunctionContext) -> str:
        """Process negative value."""
        return "negative"

    @workflow
    async def conditional_workflow(ctx: WorkflowContext, value: int) -> str:
        """Conditional workflow."""
        is_positive = await ctx.task(check_value, input=value)

        if is_positive:
            result = await ctx.task(process_positive)
        else:
            result = await ctx.task(process_negative)

        return result

    assert await conditional_workflow(value=10) == "positive"
    assert await conditional_workflow(value=-5) == "negative"


@pytest.mark.asyncio
async def test_workflow_with_loops():
    """Test workflow with loops."""

    @function
    async def increment(ctx: FunctionContext, value: int) -> int:
        """Increment value."""
        return value + 1

    @workflow
    async def loop_workflow(ctx: WorkflowContext, iterations: int) -> int:
        """Workflow with loop."""
        value = 0
        for i in range(iterations):
            value = await ctx.task(increment, input=value)
        return value

    result = await loop_workflow(iterations=5)
    assert result == 5


# Test Signal Coordination
# TODO: Implement signal() and signal_send() in WorkflowContext

# @pytest.mark.asyncio
# async def test_workflow_with_signals():
#     """Test workflow signal coordination."""
#
#     @workflow
#     async def signal_workflow(ctx: WorkflowContext) -> dict:
#         """Workflow with signal."""
#         ctx.state.set("status", "waiting")
#
#         # Send signal after delay
#         async def send_signal():
#             await asyncio.sleep(0.05)
#             ctx.signal_send("approval", {"approved": True})
#
#         # Start background task
#         asyncio.create_task(send_signal())
#
#         # Wait for signal
#         approval = await ctx.signal("approval", timeout_ms=1000)
#         ctx.state.set("status", "received")
#
#         return {"approved": approval["approved"]}
#
#     result = await signal_workflow()
#     assert result["approved"] is True


# @pytest.mark.asyncio
# async def test_workflow_signal_timeout():
#     """Test workflow signal timeout."""
#
#     @workflow
#     async def timeout_workflow(ctx: WorkflowContext) -> dict:
#         """Workflow with signal timeout."""
#         # Wait for signal that never comes
#         result = await ctx.signal("missing_signal", timeout_ms=10, default={"timeout": True})
#         return result
#
#     result = await timeout_workflow()
#     assert result["timeout"] is True


# Test Workflow Registry


def test_workflow_registry_list():
    """Test WorkflowRegistry.list_names()."""

    @workflow
    def wf1(ctx: WorkflowContext):
        pass

    @workflow
    def wf2(ctx: WorkflowContext):
        pass

    names = WorkflowRegistry.list_names()
    assert len(names) == 2
    assert "wf1" in names
    assert "wf2" in names


def test_workflow_registry_get():
    """Test WorkflowRegistry.get()."""

    @workflow
    def my_workflow(ctx: WorkflowContext):
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
    def wf1(ctx: WorkflowContext):
        pass

    @workflow
    def wf2(ctx: WorkflowContext):
        pass

    all_workflows = WorkflowRegistry.all()
    assert len(all_workflows) == 2
    assert "wf1" in all_workflows
    assert "wf2" in all_workflows


def test_workflow_registry_clear():
    """Test WorkflowRegistry.clear()."""

    @workflow
    def temp_workflow(ctx: WorkflowContext):
        pass

    assert len(WorkflowRegistry.list_names()) > 0

    WorkflowRegistry.clear()
    assert len(WorkflowRegistry.list_names()) == 0


def test_workflow_name_collision_detection():
    """Test that workflow name collisions are detected and raise error."""

    @workflow
    def process_order(ctx: WorkflowContext) -> None:
        """First workflow."""
        pass

    # Attempt to register another workflow with same name should fail
    with pytest.raises(ValueError, match="already registered"):
        @workflow
        def process_order(ctx: WorkflowContext) -> None:
            """Second workflow - should fail."""
            pass


def test_workflow_name_collision_with_custom_name():
    """Test collision detection with custom names."""

    @workflow(name="my_custom_workflow")
    def workflow_a(ctx: WorkflowContext) -> None:
        pass

    # Different function name but same workflow name should fail
    with pytest.raises(ValueError, match="already registered"):
        @workflow(name="my_custom_workflow")
        def workflow_b(ctx: WorkflowContext) -> None:
            pass


# Test Delays
# TODO: Implement sleep() and timer() in WorkflowContext for durable delays

# @pytest.mark.asyncio
# async def test_workflow_with_sleep():
#     """Test workflow with ctx.sleep()."""
#
#     @workflow
#     async def sleep_workflow(ctx: WorkflowContext) -> str:
#         """Workflow with sleep."""
#         await ctx.sleep(0.05)
#         return "done"
#
#     import time
#
#     start = time.time()
#     result = await sleep_workflow()
#     elapsed = time.time() - start
#
#     assert result == "done"
#     assert elapsed >= 0.05


# @pytest.mark.asyncio
# async def test_workflow_with_timer():
#     """Test workflow with ctx.timer()."""
#
#     @workflow
#     async def timer_workflow(ctx: WorkflowContext) -> str:
#         """Workflow with timer."""
#         await ctx.timer(delay_ms=50)
#         return "done"
#
#     import time
#
#     start = time.time()
#     result = await timer_workflow()
#     elapsed = time.time() - start
#
#     assert result == "done"
#     assert elapsed >= 0.05


# Test Type-Safe Task Execution


@pytest.mark.asyncio
async def test_workflow_type_safe_task_call():
    """Test type-safe task execution with function reference."""

    @function
    async def multiply(ctx: FunctionContext, numbers: list, factor: int = 2) -> list:
        """Multiply numbers by factor."""
        return [n * factor for n in numbers]

    @workflow
    async def type_safe_workflow(ctx: WorkflowContext) -> list:
        """Workflow using type-safe task calls."""
        # Call with positional and keyword arguments
        result = await ctx.task(multiply, [1, 2, 3], factor=3)
        return result

    result = await type_safe_workflow()
    assert result == [3, 6, 9]


@pytest.mark.asyncio
async def test_workflow_legacy_string_task_call():
    """Test backward-compatible string-based task execution."""

    @function
    async def process(ctx: FunctionContext, data: dict) -> str:
        """Process data."""
        return f"Processed: {data['value']}"

    @workflow
    async def legacy_workflow(ctx: WorkflowContext) -> str:
        """Workflow using legacy string-based calls."""
        # Legacy pattern with input parameter
        result = await ctx.task("process", input={"value": "test"})
        return result

    result = await legacy_workflow()
    assert result == "Processed: test"


@pytest.mark.asyncio
async def test_workflow_task_with_non_decorated_function():
    """Test that calling non-decorated function raises clear error."""

    async def not_decorated(ctx: FunctionContext) -> str:
        return "not decorated"

    @workflow
    async def broken_workflow(ctx: WorkflowContext) -> None:
        """Workflow calling non-decorated function."""
        await ctx.task(not_decorated)

    with pytest.raises(ValueError, match="not a registered @function"):
        await broken_workflow()


# Test Error Handling


@pytest.mark.asyncio
async def test_workflow_function_not_found():
    """Test workflow fails when function not found."""

    @workflow
    async def broken_workflow(ctx: WorkflowContext) -> None:
        """Workflow calling missing function."""
        await ctx.task("missing_function")

    with pytest.raises(ValueError, match="not found in registry"):
        await broken_workflow()


@pytest.mark.asyncio
async def test_workflow_function_error_propagation():
    """Test errors in functions propagate to workflow."""

    @function
    async def failing_function(ctx: FunctionContext) -> None:
        """Function that raises error."""
        raise ValueError("Function failed")

    @workflow
    async def error_workflow(ctx: WorkflowContext) -> None:
        """Workflow calling failing function."""
        await ctx.task(failing_function)

    with pytest.raises(ValueError, match="Function failed"):
        await error_workflow()
