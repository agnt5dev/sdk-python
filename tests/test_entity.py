"""
Tests for Entity component.

Tests cover:
- Entity type creation and method registration
- Entity instance creation and method invocation
- Single-writer consistency guarantees
- State isolation between entity instances
- Parallel execution for different keys
- Error handling
"""

import asyncio

import pytest

from agnt5 import Context, entity
from agnt5.entity import _clear_entity_state, _get_all_entity_keys, _get_entity_state
from agnt5.exceptions import ConfigurationError, ExecutionError


@pytest.fixture(autouse=True)
def clear_state():
    """Clear entity state before each test."""
    _clear_entity_state()
    yield
    _clear_entity_state()


# Test Entity Type Creation

def test_entity_type_creation():
    """Test creating entity types."""
    Counter = entity("Counter")
    assert Counter.name == "Counter"
    assert len(Counter._methods) == 0


def test_entity_multiple_types():
    """Test creating multiple entity types."""
    Counter = entity("Counter")
    UserAccount = entity("UserAccount")

    assert Counter.name == "Counter"
    assert UserAccount.name == "UserAccount"
    assert Counter is not UserAccount


# Test Method Registration

def test_register_async_method():
    """Test registering async method."""
    Counter = entity("Counter")

    @Counter.method
    async def increment(ctx: Context, amount: int = 1) -> int:
        current = ctx.get("count", 0)
        new_count = current + amount
        ctx.set("count", new_count)
        return new_count

    assert "increment" in Counter._methods
    assert asyncio.iscoroutinefunction(Counter._methods["increment"])


def test_register_sync_method():
    """Test registering sync method (auto-converted to async)."""
    Calc = entity("Calc")

    @Calc.method
    def add(ctx: Context, a: int, b: int) -> int:
        return a + b

    assert "add" in Calc._methods
    assert asyncio.iscoroutinefunction(Calc._methods["add"])


def test_method_without_context_param_fails():
    """Test that methods without Context parameter fail."""
    Counter = entity("Counter")

    with pytest.raises(ConfigurationError, match="must have at least one parameter"):

        @Counter.method
        async def bad_method():
            pass


def test_method_with_wrong_context_type_fails():
    """Test that methods with wrong first parameter type fail."""
    Counter = entity("Counter")

    with pytest.raises(ConfigurationError, match="first parameter must be 'ctx: Context'"):

        @Counter.method
        async def bad_method(wrong_type: str):
            pass


def test_multiple_methods_same_entity():
    """Test registering multiple methods on same entity."""
    Counter = entity("Counter")

    @Counter.method
    async def increment(ctx: Context) -> int:
        count = ctx.get("count", 0)
        ctx.set("count", count + 1)
        return count + 1

    @Counter.method
    async def decrement(ctx: Context) -> int:
        count = ctx.get("count", 0)
        ctx.set("count", count - 1)
        return count - 1

    @Counter.method
    async def get_count(ctx: Context) -> int:
        return ctx.get("count", 0)

    assert len(Counter._methods) == 3
    assert "increment" in Counter._methods
    assert "decrement" in Counter._methods
    assert "get_count" in Counter._methods


# Test Entity Instance Creation and Invocation

@pytest.mark.asyncio
async def test_entity_instance_creation():
    """Test creating entity instances."""
    Counter = entity("Counter")

    @Counter.method
    async def get_count(ctx: Context) -> int:
        return ctx.get("count", 0)

    counter1 = Counter(key="counter-1")
    counter2 = Counter(key="counter-2")

    assert counter1.entity_type == "Counter"
    assert counter1.key == "counter-1"
    assert counter2.key == "counter-2"
    assert counter1 is not counter2


@pytest.mark.asyncio
async def test_entity_method_invocation():
    """Test invoking entity methods."""
    Counter = entity("Counter")

    @Counter.method
    async def increment(ctx: Context, amount: int = 1) -> int:
        current = ctx.get("count", 0)
        new_count = current + amount
        ctx.set("count", new_count)
        return new_count

    counter = Counter(key="test-counter")

    result1 = await counter.increment()
    assert result1 == 1

    result2 = await counter.increment(amount=5)
    assert result2 == 6

    result3 = await counter.increment(amount=3)
    assert result3 == 9


@pytest.mark.asyncio
async def test_entity_state_persistence():
    """Test that entity state persists across method calls."""
    UserAccount = entity("UserAccount")

    @UserAccount.method
    async def deposit(ctx: Context, amount: float) -> float:
        balance = ctx.get("balance", 0.0)
        new_balance = balance + amount
        ctx.set("balance", new_balance)
        return new_balance

    @UserAccount.method
    async def get_balance(ctx: Context) -> float:
        return ctx.get("balance", 0.0)

    account = UserAccount(key="user-123")

    await account.deposit(amount=100.0)
    await account.deposit(amount=50.0)

    balance = await account.get_balance()
    assert balance == 150.0


# Test State Isolation

@pytest.mark.asyncio
async def test_state_isolation_between_keys():
    """Test that different keys have isolated state."""
    Counter = entity("Counter")

    @Counter.method
    async def increment(ctx: Context) -> int:
        count = ctx.get("count", 0)
        ctx.set("count", count + 1)
        return count + 1

    @Counter.method
    async def get_count(ctx: Context) -> int:
        return ctx.get("count", 0)

    counter1 = Counter(key="counter-1")
    counter2 = Counter(key="counter-2")

    await counter1.increment()
    await counter1.increment()
    await counter2.increment()

    count1 = await counter1.get_count()
    count2 = await counter2.get_count()

    assert count1 == 2
    assert count2 == 1


@pytest.mark.asyncio
async def test_same_entity_type_different_keys():
    """Test multiple instances of same entity type with different keys."""
    Account = entity("Account")

    @Account.method
    async def set_balance(ctx: Context, amount: float):
        ctx.set("balance", amount)

    @Account.method
    async def get_balance(ctx: Context) -> float:
        return ctx.get("balance", 0.0)

    alice = Account(key="alice")
    bob = Account(key="bob")

    await alice.set_balance(amount=100.0)
    await bob.set_balance(amount=200.0)

    alice_balance = await alice.get_balance()
    bob_balance = await bob.get_balance()

    assert alice_balance == 100.0
    assert bob_balance == 200.0


# Test Single-Writer Consistency

@pytest.mark.asyncio
async def test_single_writer_consistency():
    """Test that single-writer consistency prevents lost updates."""
    Counter = entity("Counter")

    @Counter.method
    async def increment(ctx: Context):
        # Simulate read-modify-write with delay
        current = ctx.get("count", 0)
        await asyncio.sleep(0.01)  # Allow time for race condition
        ctx.set("count", current + 1)

    counter = Counter(key="test")

    # Run 10 concurrent increments
    await asyncio.gather(*[counter.increment() for _ in range(10)])

    # Verify count is exactly 10 (no lost updates)
    final_state = _get_entity_state("Counter", "test")
    assert final_state["count"] == 10


@pytest.mark.asyncio
async def test_parallel_execution_different_keys():
    """Test that different keys execute in parallel."""
    SlowCounter = entity("SlowCounter")

    execution_log = []

    @SlowCounter.method
    async def increment(ctx: Context, key_name: str):
        execution_log.append(f"{key_name}_start")
        await asyncio.sleep(0.05)
        count = ctx.get("count", 0)
        ctx.set("count", count + 1)
        execution_log.append(f"{key_name}_end")
        return count + 1

    counter1 = SlowCounter(key="counter-1")
    counter2 = SlowCounter(key="counter-2")

    # Execute in parallel
    await asyncio.gather(
        counter1.increment(key_name="c1"),
        counter2.increment(key_name="c2")
    )

    # Check that executions overlapped (parallel)
    # If serial, would be: c1_start, c1_end, c2_start, c2_end
    # Parallel should interleave: c1_start, c2_start, c1_end, c2_end (or similar)
    assert execution_log[0] in ["c1_start", "c2_start"]
    assert execution_log[1] in ["c1_start", "c2_start"]
    assert execution_log[0] != execution_log[1]  # Both started before either finished


# Test Error Handling

@pytest.mark.asyncio
async def test_method_error_propagates():
    """Test that errors in methods propagate correctly."""
    Faulty = entity("Faulty")

    @Faulty.method
    async def failing_method(ctx: Context):
        raise ValueError("Something went wrong")

    instance = Faulty(key="test")

    with pytest.raises(ExecutionError, match="Something went wrong"):
        await instance.failing_method()


@pytest.mark.asyncio
async def test_nonexistent_method_fails():
    """Test that calling non-existent method raises AttributeError."""
    Counter = entity("Counter")

    @Counter.method
    async def increment(ctx: Context):
        pass

    counter = Counter(key="test")

    with pytest.raises(AttributeError, match="has no method 'does_not_exist'"):
        await counter.does_not_exist()


# Test Context Integration

@pytest.mark.asyncio
async def test_context_properties_in_entity():
    """Test that Context has correct properties in entity methods."""
    Tracker = entity("Tracker")

    captured_context = {}

    @Tracker.method
    async def capture_context(ctx: Context):
        captured_context["run_id"] = ctx.run_id
        captured_context["component_type"] = ctx.component_type
        captured_context["object_id"] = ctx.object_id
        captured_context["method_name"] = ctx.method_name

    tracker = Tracker(key="test-key")
    await tracker.capture_context()

    assert captured_context["run_id"] == "Tracker:test-key:capture_context"
    assert captured_context["component_type"] == "entity"
    assert captured_context["object_id"] == "test-key"
    assert captured_context["method_name"] == "capture_context"


@pytest.mark.asyncio
async def test_context_state_operations():
    """Test all Context state operations work in entities."""
    StateTest = entity("StateTest")

    @StateTest.method
    async def test_all_operations(ctx: Context) -> dict:
        # Set
        ctx.set("key1", "value1")
        ctx.set("key2", {"nested": "data"})

        # Get
        val1 = ctx.get("key1")
        val2 = ctx.get("key2")
        val3 = ctx.get("nonexistent", "default")

        # Delete
        ctx.delete("key1")

        # Verify deleted
        val4 = ctx.get("key1")  # Should be None

        return {
            "val1": val1,
            "val2": val2,
            "val3": val3,
            "val4": val4,
        }

    instance = StateTest(key="test")
    result = await instance.test_all_operations()

    assert result["val1"] == "value1"
    assert result["val2"] == {"nested": "data"}
    assert result["val3"] == "default"
    assert result["val4"] is None


# Test Complex Scenarios

@pytest.mark.asyncio
async def test_conversation_entity_pattern():
    """Test implementing conversation/session pattern with entity."""
    Conversation = entity("Conversation")

    @Conversation.method
    async def add_message(ctx: Context, role: str, content: str):
        messages = ctx.get("messages", [])
        messages.append({"role": role, "content": content})
        ctx.set("messages", messages)

    @Conversation.method
    async def get_messages(ctx: Context) -> list:
        return ctx.get("messages", [])

    @Conversation.method
    async def clear_history(ctx: Context):
        ctx.set("messages", [])

    conv = Conversation(key="chat-123")

    await conv.add_message(role="user", content="Hello!")
    await conv.add_message(role="assistant", content="Hi there!")
    await conv.add_message(role="user", content="How are you?")

    messages = await conv.get_messages()
    assert len(messages) == 3
    assert messages[0]["content"] == "Hello!"
    assert messages[1]["role"] == "assistant"

    await conv.clear_history()
    messages = await conv.get_messages()
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_workflow_state_pattern():
    """Test implementing workflow state tracking with entity."""
    WorkflowState = entity("WorkflowState")

    @WorkflowState.method
    async def start(ctx: Context, workflow_type: str):
        ctx.set("status", "running")
        ctx.set("type", workflow_type)
        ctx.set("steps_completed", [])

    @WorkflowState.method
    async def complete_step(ctx: Context, step_name: str, result: any):
        steps = ctx.get("steps_completed", [])
        steps.append({"name": step_name, "result": result})
        ctx.set("steps_completed", steps)

    @WorkflowState.method
    async def finish(ctx: Context):
        ctx.set("status", "completed")

    @WorkflowState.method
    async def get_status(ctx: Context) -> dict:
        return {
            "status": ctx.get("status"),
            "type": ctx.get("type"),
            "steps": ctx.get("steps_completed", [])
        }

    workflow = WorkflowState(key="workflow-789")

    await workflow.start(workflow_type="data_pipeline")
    await workflow.complete_step(step_name="extract", result={"rows": 1000})
    await workflow.complete_step(step_name="transform", result={"rows": 950})
    await workflow.finish()

    status = await workflow.get_status()
    assert status["status"] == "completed"
    assert status["type"] == "data_pipeline"
    assert len(status["steps"]) == 2


# Test Utility Functions

def test_get_entity_state():
    """Test getting entity state for debugging."""
    Counter = entity("Counter")

    @Counter.method
    async def set_value(ctx: Context, value: int):
        ctx.set("count", value)

    counter = Counter(key="test")

    # Initially no state
    state = _get_entity_state("Counter", "test")
    assert state is None

    # After method call, state exists
    asyncio.run(counter.set_value(value=42))

    state = _get_entity_state("Counter", "test")
    assert state is not None
    assert state["count"] == 42


def test_get_all_entity_keys():
    """Test getting all keys for an entity type."""
    Account = entity("Account")

    @Account.method
    async def init(ctx: Context):
        ctx.set("initialized", True)

    # Create multiple instances
    asyncio.run(Account(key="alice").init())
    asyncio.run(Account(key="bob").init())
    asyncio.run(Account(key="charlie").init())

    keys = _get_all_entity_keys("Account")
    assert set(keys) == {"alice", "bob", "charlie"}
