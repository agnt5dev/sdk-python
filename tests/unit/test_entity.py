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

from agnt5 import Entity
from agnt5.entity import EntityRegistry, EntityStateManager, create_entity_context, _entity_state_manager_ctx
from agnt5.exceptions import ExecutionError


@pytest.fixture(autouse=True)
def entity_state_manager():
    """Create and set EntityStateManager for each test."""
    # Use the new helper function
    manager, token = create_entity_context()

    yield manager

    # Cleanup
    _entity_state_manager_ctx.reset(token)
    manager.clear_all()
    # Clear entity registry between tests to avoid conflicts
    EntityRegistry.clear()


# Test Entity Type Creation

def test_entity_type_creation():
    """Test creating entity types."""
    class Counter(Entity):
        pass

    # Entity should be auto-registered
    entity_type = EntityRegistry.get("Counter")
    assert entity_type is not None
    assert entity_type.name == "Counter"


def test_entity_multiple_types():
    """Test creating multiple entity types."""
    class Counter(Entity):
        pass

    class UserAccount(Entity):
        pass

    counter_type = EntityRegistry.get("Counter")
    account_type = EntityRegistry.get("UserAccount")

    assert counter_type.name == "Counter"
    assert account_type.name == "UserAccount"
    assert counter_type is not account_type


# Test Method Registration

def test_register_async_method():
    """Test registering async method."""
    class Counter(Entity):
        async def increment(self, amount: int = 1) -> int:
            current = self.state.get("count", 0)
            new_count = current + amount
            self.state.set("count", new_count)
            return new_count

    entity_type = EntityRegistry.get("Counter")
    assert "increment" in entity_type._method_schemas

    # Check that method is wrapped
    counter = Counter(key="test")
    assert asyncio.iscoroutinefunction(counter.increment)


def test_multiple_methods_same_entity():
    """Test registering multiple methods on same entity."""
    class Counter(Entity):
        async def increment(self) -> int:
            count = self.state.get("count", 0)
            self.state.set("count", count + 1)
            return count + 1

        async def decrement(self) -> int:
            count = self.state.get("count", 0)
            self.state.set("count", count - 1)
            return count - 1

        async def get_count(self) -> int:
            return self.state.get("count", 0)

    entity_type = EntityRegistry.get("Counter")
    assert len(entity_type._method_schemas) == 3
    assert "increment" in entity_type._method_schemas
    assert "decrement" in entity_type._method_schemas
    assert "get_count" in entity_type._method_schemas


# Test Entity Instance Creation and Invocation

@pytest.mark.asyncio
async def test_entity_instance_creation():
    """Test creating entity instances."""
    class Counter(Entity):
        async def get_count(self) -> int:
            return self.state.get("count", 0)

    counter1 = Counter(key="counter-1")
    counter2 = Counter(key="counter-2")

    assert counter1.entity_type == "Counter"
    assert counter1.key == "counter-1"
    assert counter2.key == "counter-2"
    assert counter1 is not counter2


@pytest.mark.asyncio
async def test_entity_method_invocation():
    """Test invoking entity methods."""
    class Counter(Entity):
        async def increment(self, amount: int = 1) -> int:
            current = self.state.get("count", 0)
            new_count = current + amount
            self.state.set("count", new_count)
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
    class UserAccount(Entity):
        async def deposit(self, amount: float) -> float:
            balance = self.state.get("balance", 0.0)
            new_balance = balance + amount
            self.state.set("balance", new_balance)
            return new_balance

        async def get_balance(self) -> float:
            return self.state.get("balance", 0.0)

    account = UserAccount(key="user-123")

    await account.deposit(amount=100.0)
    await account.deposit(amount=50.0)

    balance = await account.get_balance()
    assert balance == 150.0


# Test State Isolation

@pytest.mark.asyncio
async def test_state_isolation_between_keys():
    """Test that different keys have isolated state."""
    class Counter(Entity):
        async def increment(self) -> int:
            count = self.state.get("count", 0)
            self.state.set("count", count + 1)
            return count + 1

        async def get_count(self) -> int:
            return self.state.get("count", 0)

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
    class Account(Entity):
        async def set_balance(self, amount: float):
            self.state.set("balance", amount)

        async def get_balance(self) -> float:
            return self.state.get("balance", 0.0)

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
async def test_single_writer_consistency(entity_state_manager):
    """Test that single-writer consistency prevents lost updates."""
    class Counter(Entity):
        async def increment(self):
            # Simulate read-modify-write with delay
            current = self.state.get("count", 0)
            await asyncio.sleep(0.01)  # Allow time for race condition
            self.state.set("count", current + 1)

    counter = Counter(key="test")

    # Run 10 concurrent increments
    await asyncio.gather(*[counter.increment() for _ in range(10)])

    # Verify count is exactly 10 (no lost updates)
    final_state = entity_state_manager.get_state("Counter", "test")
    assert final_state["count"] == 10


@pytest.mark.asyncio
async def test_parallel_execution_different_keys():
    """Test that different keys execute in parallel."""
    execution_log = []

    class SlowCounter(Entity):
        async def increment(self, key_name: str):
            execution_log.append(f"{key_name}_start")
            await asyncio.sleep(0.05)
            count = self.state.get("count", 0)
            self.state.set("count", count + 1)
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
    class Faulty(Entity):
        async def failing_method(self):
            raise ValueError("Something went wrong")

    instance = Faulty(key="test")

    with pytest.raises(ExecutionError, match="Something went wrong"):
        await instance.failing_method()


@pytest.mark.asyncio
async def test_nonexistent_method_fails():
    """Test that calling non-existent method raises AttributeError."""
    class Counter(Entity):
        async def increment(self):
            pass

    counter = Counter(key="test")

    with pytest.raises(AttributeError):
        await counter.does_not_exist()


# Test State Operations

@pytest.mark.asyncio
async def test_state_operations():
    """Test all state operations work in entities."""
    class StateTest(Entity):
        async def test_all_operations(self) -> dict:
            # Set
            self.state.set("key1", "value1")
            self.state.set("key2", {"nested": "data"})

            # Get
            val1 = self.state.get("key1")
            val2 = self.state.get("key2")
            val3 = self.state.get("nonexistent", "default")

            # Delete
            self.state.delete("key1")

            # Verify deleted
            val4 = self.state.get("key1")  # Should be None

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
    class Conversation(Entity):
        async def add_message(self, role: str, content: str):
            messages = self.state.get("messages", [])
            messages.append({"role": role, "content": content})
            self.state.set("messages", messages)

        async def get_messages(self) -> list:
            return self.state.get("messages", [])

        async def clear_history(self):
            self.state.set("messages", [])

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
    class WorkflowState(Entity):
        async def start(self, workflow_type: str):
            self.state.set("status", "running")
            self.state.set("type", workflow_type)
            self.state.set("steps_completed", [])

        async def complete_step(self, step_name: str, result: any):
            steps = self.state.get("steps_completed", [])
            steps.append({"name": step_name, "result": result})
            self.state.set("steps_completed", steps)

        async def finish(self):
            self.state.set("status", "completed")

        async def get_status(self) -> dict:
            return {
                "status": self.state.get("status"),
                "type": self.state.get("type"),
                "steps": self.state.get("steps_completed", [])
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

@pytest.mark.asyncio
async def test_get_entity_state(entity_state_manager):
    """Test getting entity state for debugging."""
    class Counter(Entity):
        async def set_value(self, value: int):
            self.state.set("count", value)

    counter = Counter(key="test")

    # Initially no state
    state = entity_state_manager.get_state("Counter", "test")
    assert state is None

    # After method call, state exists
    await counter.set_value(value=42)

    state = entity_state_manager.get_state("Counter", "test")
    assert state is not None
    assert state["count"] == 42


@pytest.mark.asyncio
async def test_get_all_entity_keys(entity_state_manager):
    """Test getting all keys for an entity type."""
    class Account(Entity):
        async def init(self):
            self.state.set("initialized", True)

    # Create multiple instances
    await Account(key="alice").init()
    await Account(key="bob").init()
    await Account(key="charlie").init()

    keys = entity_state_manager.get_all_keys("Account")
    assert set(keys) == {"alice", "bob", "charlie"}


# Test Shopping Cart Example (from docstring)

@pytest.mark.asyncio
async def test_shopping_cart_example():
    """Test the shopping cart example from Entity docstring."""
    class ShoppingCart(Entity):
        async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
            items = self.state.get("items", {})
            items[item_id] = {"quantity": quantity, "price": price}
            self.state.set("items", items)
            return {"total_items": len(items)}

        async def get_total(self) -> float:
            items = self.state.get("items", {})
            return sum(item["quantity"] * item["price"] for item in items.values())

    cart = ShoppingCart(key="user-123")
    result = await cart.add_item("item-abc", quantity=2, price=29.99)

    assert result["total_items"] == 1

    await cart.add_item("item-xyz", quantity=1, price=15.00)
    total = await cart.get_total()

    assert total == 2 * 29.99 + 1 * 15.00  # 74.98


# Edge Case Tests

@pytest.mark.asyncio
async def test_state_access_outside_method():
    """Test that accessing state outside of method raises clear error."""
    class Counter(Entity):
        async def increment(self):
            count = self.state.get("count", 0)
            self.state.set("count", count + 1)

    counter = Counter(key="test")

    # Trying to access state outside of method should raise clear error
    with pytest.raises(RuntimeError, match="Entity state can only be accessed within entity methods"):
        _ = counter.state.get("count")


@pytest.mark.asyncio
async def test_multiple_entity_types_same_method_names():
    """Test different entity types can have methods with same names."""
    class Counter(Entity):
        async def reset(self):
            self.state.set("count", 0)
            return "counter reset"

        async def get_value(self) -> int:
            return self.state.get("count", 0)

    class Timer(Entity):
        async def reset(self):
            self.state.set("seconds", 0)
            return "timer reset"

        async def get_value(self) -> int:
            return self.state.get("seconds", 0)

    counter = Counter(key="c1")
    timer = Timer(key="t1")

    # Set different values
    counter_result = await counter.reset()
    timer_result = await timer.reset()

    assert counter_result == "counter reset"
    assert timer_result == "timer reset"

    # Verify they maintain separate state
    counter_val = await counter.get_value()
    timer_val = await timer.get_value()

    assert counter_val == 0
    assert timer_val == 0


@pytest.mark.asyncio
async def test_entity_with_complex_state_types():
    """Test entity with complex nested state structures."""
    class ComplexEntity(Entity):
        async def set_complex_data(self) -> dict:
            # Store various complex types
            self.state.set("nested_dict", {
                "level1": {
                    "level2": {
                        "value": 123
                    }
                }
            })
            self.state.set("list_of_dicts", [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"}
            ])
            self.state.set("mixed_types", {
                "string": "test",
                "number": 42,
                "float": 3.14,
                "bool": True,
                "none": None,
                "list": [1, 2, 3]
            })

            return {
                "nested": self.state.get("nested_dict"),
                "list": self.state.get("list_of_dicts"),
                "mixed": self.state.get("mixed_types")
            }

    entity = ComplexEntity(key="complex")
    result = await entity.set_complex_data()

    assert result["nested"]["level1"]["level2"]["value"] == 123
    assert len(result["list"]) == 2
    assert result["list"][0]["name"] == "first"
    assert result["mixed"]["string"] == "test"
    assert result["mixed"]["bool"] is True
    assert result["mixed"]["none"] is None


@pytest.mark.asyncio
async def test_entity_state_clear():
    """Test clearing all entity state."""
    class DataStore(Entity):
        async def set_multiple_keys(self):
            self.state.set("key1", "value1")
            self.state.set("key2", "value2")
            self.state.set("key3", "value3")

        async def clear_all(self):
            self.state.clear()

        async def count_keys(self) -> int:
            # Count non-None values (since get returns None for missing keys)
            count = 0
            for key in ["key1", "key2", "key3"]:
                if self.state.get(key) is not None:
                    count += 1
            return count

    store = DataStore(key="store1")

    await store.set_multiple_keys()
    count_before = await store.count_keys()
    assert count_before == 3

    await store.clear_all()
    count_after = await store.count_keys()
    assert count_after == 0


@pytest.mark.asyncio
async def test_entity_method_with_pydantic_model():
    """Test entity method with Pydantic model input/output."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Pydantic not installed")

    class UserData(BaseModel):
        name: str
        age: int
        email: str

    class UserStore(Entity):
        async def save_user(self, user: UserData) -> dict:
            self.state.set("user", {
                "name": user.name,
                "age": user.age,
                "email": user.email
            })
            return {"saved": True, "name": user.name}

        async def get_user(self) -> dict:
            return self.state.get("user", {})

    store = UserStore(key="users")
    user = UserData(name="Alice", age=30, email="alice@example.com")

    result = await store.save_user(user)
    assert result["saved"] is True
    assert result["name"] == "Alice"

    retrieved = await store.get_user()
    assert retrieved["name"] == "Alice"
    assert retrieved["age"] == 30
    assert retrieved["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_entity_registry_operations():
    """Test EntityRegistry utility operations."""
    # Clear registry first
    EntityRegistry.clear()

    class Entity1(Entity):
        pass

    class Entity2(Entity):
        pass

    class Entity3(Entity):
        pass

    # Test .all() returns all entities
    all_entities = EntityRegistry.all()
    assert len(all_entities) == 3
    assert "Entity1" in all_entities
    assert "Entity2" in all_entities
    assert "Entity3" in all_entities

    # Test .get() for specific entity
    entity1_type = EntityRegistry.get("Entity1")
    assert entity1_type is not None
    assert entity1_type.name == "Entity1"

    # Test .get() for non-existent entity
    missing = EntityRegistry.get("NonExistent")
    assert missing is None

    # Test .clear()
    EntityRegistry.clear()
    all_after_clear = EntityRegistry.all()
    assert len(all_after_clear) == 0


@pytest.mark.asyncio
async def test_concurrent_updates_same_key_serialized(entity_state_manager):
    """Test that concurrent updates to same key are properly serialized."""
    update_order = []

    class OrderTracker(Entity):
        async def track_update(self, update_id: str):
            update_order.append(f"{update_id}_start")
            await asyncio.sleep(0.02)  # Simulate slow operation
            count = self.state.get("count", 0)
            self.state.set("count", count + 1)
            update_order.append(f"{update_id}_end")

    tracker = OrderTracker(key="shared")

    # Run 3 concurrent updates on same key
    await asyncio.gather(
        tracker.track_update("A"),
        tracker.track_update("B"),
        tracker.track_update("C")
    )

    # Verify count is correct (no lost updates)
    final_state = entity_state_manager.get_state("OrderTracker", "shared")
    assert final_state["count"] == 3

    # Verify operations were serialized (no interleaving of start/end for same operation)
    # Pattern should be: A_start, A_end, B_start, B_end, C_start, C_end (or any order)
    # But NOT: A_start, B_start, A_end (this would indicate interleaving)
    for i in range(0, len(update_order), 2):
        start = update_order[i]
        end = update_order[i + 1]
        # Each start should be followed by its corresponding end
        assert start.replace("_start", "") == end.replace("_end", "")


@pytest.mark.asyncio
async def test_entity_definition_with_state_schema():
    """Test entity with explicit state schema definition."""
    class InventoryItem(Entity):
        _state_schema = {
            "type": "object",
            "properties": {
                "quantity": {"type": "integer", "description": "Item quantity"},
                "price": {"type": "number", "description": "Item price"},
                "name": {"type": "string", "description": "Item name"}
            },
            "description": "Inventory item state"
        }

        async def set_item(self, name: str, quantity: int, price: float):
            self.state.set("name", name)
            self.state.set("quantity", quantity)
            self.state.set("price", price)

    # Check that state schema was registered
    entity_type = EntityRegistry.get("InventoryItem")
    assert entity_type is not None

    definition = entity_type.build_entity_definition()
    assert "state_schema" in definition
    assert definition["state_schema"]["type"] == "object"
    assert "quantity" in definition["state_schema"]["properties"]
