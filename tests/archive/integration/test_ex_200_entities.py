"""
Integration Tests for ex_02_entities.py

Tests entity features through the platform:
- Counter - Basic counter with increment/decrement
- ConversationMemory - AI conversation history management
- KeyValueStore - Simple key-value storage

Run with:
    # Local mode (requires running dev server with examples worker)
    pytest tests/integration/test_ex_02_entities.py -v

    # Subprocess mode (for CI)
    pytest tests/integration/test_ex_02_entities.py -v --use-subprocess
"""

import uuid

import pytest


def unique_key(prefix: str) -> str:
    """Generate a unique entity key to ensure test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# =============================================================================
# COUNTER ENTITY
# =============================================================================


@pytest.mark.integration
def test_counter_increment(client, worker_process, platform):
    """Test counter increment operation."""
    counter = client.entity("Counter", unique_key("counter"))

    result = counter.increment(amount=5)
    assert result == 5

    result = counter.increment(amount=3)
    assert result == 8


@pytest.mark.integration
def test_counter_decrement(client, worker_process):
    """Test counter decrement operation."""
    counter = client.entity("Counter", unique_key("counter"))

    counter.increment(amount=10)
    result = counter.decrement(amount=3)
    assert result == 7


@pytest.mark.integration
def test_counter_get_count(client, worker_process):
    """Test counter get_count operation."""
    counter = client.entity("Counter", unique_key("counter"))

    # Initial count should be 0
    result = counter.get_count()
    assert result == 0

    counter.increment(amount=5)
    result = counter.get_count()
    assert result == 5


@pytest.mark.integration
def test_counter_reset(client, worker_process):
    """Test counter reset operation."""
    counter = client.entity("Counter", unique_key("counter"))

    counter.increment(amount=100)
    result = counter.reset()
    assert result == 0

    result = counter.get_count()
    assert result == 0


@pytest.mark.integration
def test_counter_state_isolation(client, worker_process):
    """Test that different counter keys have isolated state."""
    counter_a = client.entity("Counter", unique_key("counter-a"))
    counter_b = client.entity("Counter", unique_key("counter-b"))

    counter_a.increment(amount=10)
    counter_b.increment(amount=5)

    assert counter_a.get_count() == 10
    assert counter_b.get_count() == 5


@pytest.mark.integration
def test_counter_concurrent_increments(client, worker_process):
    """Test concurrent increments maintain consistency.

    This test validates that the entity's single-writer semantics
    correctly serialize concurrent operations.

    10 threads × 50 increments each = 500 total expected count.
    """
    from concurrent.futures import ThreadPoolExecutor

    counter = client.entity("Counter", unique_key("counter-concurrent"))

    num_threads = 10
    increments_per_thread = 50

    def increment_many():
        """Each thread increments 50 times."""
        for _ in range(increments_per_thread):
            counter.increment(amount=1)

    # Run concurrent increments
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(increment_many) for _ in range(num_threads)]
        # Wait for all to complete
        for future in futures:
            future.result()

    # Verify final count
    final_count = counter.get_count()
    expected_count = num_threads * increments_per_thread  # 10 * 50 = 500

    assert final_count == expected_count, (
        f"Expected {expected_count} after {num_threads} threads × {increments_per_thread} increments, "
        f"got {final_count} (lost {expected_count - final_count} increments)"
    )


# =============================================================================
# CONVERSATION MEMORY ENTITY
# =============================================================================


@pytest.mark.integration
def test_memory_add_message(client, worker_process):
    """Test adding messages to conversation."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    result = memory.add_message(role="user", content="Hello!")
    assert result["action"] == "added"
    assert result["role"] == "user"
    assert result["message_count"] == 1

    result = memory.add_message(role="assistant", content="Hi there!")
    assert result["message_count"] == 2


@pytest.mark.integration
def test_memory_get_messages(client, worker_process):
    """Test getting all messages."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="Question 1")
    memory.add_message(role="assistant", content="Answer 1")
    memory.add_message(role="user", content="Question 2")

    messages = memory.get_messages()
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Question 1"


@pytest.mark.integration
def test_memory_get_messages_with_limit(client, worker_process):
    """Test getting messages with limit."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="Message 1")
    memory.add_message(role="assistant", content="Message 2")
    memory.add_message(role="user", content="Message 3")
    memory.add_message(role="assistant", content="Message 4")

    messages = memory.get_messages(limit=2)
    assert len(messages) == 2
    # Should return most recent
    assert messages[0]["content"] == "Message 3"
    assert messages[1]["content"] == "Message 4"


@pytest.mark.integration
def test_memory_get_messages_by_role(client, worker_process):
    """Test filtering messages by role."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="User message 1")
    memory.add_message(role="assistant", content="Assistant message")
    memory.add_message(role="user", content="User message 2")

    user_messages = memory.get_messages(roles=["user"])
    assert len(user_messages) == 2
    assert all(m["role"] == "user" for m in user_messages)


@pytest.mark.integration
def test_memory_get_last_message(client, worker_process):
    """Test getting the last message."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="First")
    memory.add_message(role="assistant", content="Second")
    memory.add_message(role="user", content="Third")

    last = memory.get_last_message()
    assert last["content"] == "Third"

    last_assistant = memory.get_last_message(role="assistant")
    assert last_assistant["content"] == "Second"


@pytest.mark.integration
def test_memory_get_context_window(client, worker_process):
    """Test getting context window for LLM."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="Hello")
    memory.add_message(role="assistant", content="Hi!")
    memory.add_message(role="user", content="How are you?")

    context = memory.get_context_window(max_messages=10)
    assert len(context) == 3
    # Should be in LLM format (role + content only)
    assert "timestamp" not in context[0]
    assert context[0] == {"role": "user", "content": "Hello"}


@pytest.mark.integration
def test_memory_summarize(client, worker_process):
    """Test conversation summary."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="Hello")
    memory.add_message(role="assistant", content="Hi there!")
    memory.add_message(role="user", content="Thanks")

    summary = memory.summarize()
    assert summary["message_count"] == 3
    assert summary["role_counts"]["user"] == 2
    assert summary["role_counts"]["assistant"] == 1
    assert summary["total_characters"] > 0
    assert summary["has_system_message"] is False


@pytest.mark.integration
def test_memory_set_system_message(client, worker_process):
    """Test setting system message."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="Hello")
    result = memory.set_system_message(content="You are a helpful assistant.")
    assert result["action"] == "system_message_set"

    messages = memory.get_messages()
    # System message should be first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    summary = memory.summarize()
    assert summary["has_system_message"] is True


@pytest.mark.integration
def test_memory_clear(client, worker_process):
    """Test clearing conversation."""
    memory = client.entity("ConversationMemory", unique_key("memory"))

    memory.add_message(role="user", content="Hello")
    memory.add_message(role="assistant", content="Hi!")

    result = memory.clear()
    assert result["action"] == "cleared"

    messages = memory.get_messages()
    assert len(messages) == 0


@pytest.mark.integration
def test_memory_state_isolation(client, worker_process):
    """Test that different sessions have isolated state."""
    memory_a = client.entity("ConversationMemory", unique_key("session-a"))
    memory_b = client.entity("ConversationMemory", unique_key("session-b"))

    memory_a.add_message(role="user", content="Message in A")
    memory_b.add_message(role="user", content="Message in B")

    assert len(memory_a.get_messages()) == 1
    assert len(memory_b.get_messages()) == 1
    assert memory_a.get_messages()[0]["content"] == "Message in A"
    assert memory_b.get_messages()[0]["content"] == "Message in B"


@pytest.mark.integration
def test_memory_unicode_content(client, worker_process):
    """Test ConversationMemory with Unicode content (P1.5).

    Validates:
    - CJK characters in messages
    - Arabic script
    - Emoji
    - Mixed scripts
    - Retrieval preserves exact Unicode
    """
    memory = client.entity("ConversationMemory", unique_key("memory-unicode"))

    # Add messages with various Unicode content
    unicode_messages = [
        {"role": "user", "content": "你好，世界！"},  # Chinese
        {"role": "assistant", "content": "こんにちは！お手伝いできることはありますか？"},  # Japanese
        {"role": "user", "content": "مرحبا، كيف حالك؟"},  # Arabic
        {"role": "assistant", "content": "Hello! 🌍🚀 I can help with café recommendations"},  # Mixed
        {"role": "user", "content": "Привет! Спасибо за помощь"},  # Russian
    ]

    for msg in unicode_messages:
        memory.add_message(role=msg["role"], content=msg["content"])

    # Retrieve and verify
    messages = memory.get_messages()
    assert len(messages) == len(unicode_messages), (
        f"Expected {len(unicode_messages)} messages, got {len(messages)}"
    )

    # Verify each message content is preserved exactly
    for i, expected in enumerate(unicode_messages):
        assert messages[i]["role"] == expected["role"], (
            f"Message {i}: role mismatch"
        )
        assert messages[i]["content"] == expected["content"], (
            f"Message {i}: Unicode content not preserved. "
            f"Expected '{expected['content']}', got '{messages[i]['content']}'"
        )


# =============================================================================
# KEY-VALUE STORE ENTITY
# =============================================================================


@pytest.mark.integration
def test_kv_set_get(client, worker_process):
    """Test setting and getting values."""
    store = client.entity("KeyValueStore", unique_key("store"))

    result = store.set_value(item_key="theme", value="dark")
    assert result["action"] == "set"
    assert result["key"] == "theme"

    value = store.get_value(item_key="theme")
    assert value == "dark"


@pytest.mark.integration
def test_kv_get_default(client, worker_process):
    """Test getting nonexistent key with default."""
    store = client.entity("KeyValueStore", unique_key("store"))

    value = store.get_value(item_key="nonexistent", default="default_value")
    assert value == "default_value"


@pytest.mark.integration
def test_kv_delete(client, worker_process):
    """Test deleting a key."""
    store = client.entity("KeyValueStore", unique_key("store"))

    store.set_value(item_key="temp", value="data")
    result = store.delete_key(item_key="temp")
    assert result["action"] == "deleted"

    value = store.get_value(item_key="temp")
    assert value is None


@pytest.mark.integration
def test_kv_delete_nonexistent(client, worker_process):
    """Test deleting nonexistent key."""
    store = client.entity("KeyValueStore", unique_key("store"))

    result = store.delete_key(item_key="nonexistent")
    assert result["action"] == "not_found"


@pytest.mark.integration
def test_kv_keys(client, worker_process):
    """Test listing all keys."""
    store = client.entity("KeyValueStore", unique_key("store"))

    store.set_value(item_key="theme", value="dark")
    store.set_value(item_key="language", value="en")
    store.set_value(item_key="timezone", value="UTC")

    keys = store.list_keys()
    assert set(keys) == {"theme", "language", "timezone"}


@pytest.mark.integration
def test_kv_clear(client, worker_process):
    """Test clearing all data."""
    store = client.entity("KeyValueStore", unique_key("store"))

    store.set_value(item_key="key1", value="value1")
    store.set_value(item_key="key2", value="value2")

    result = store.clear()
    assert result["action"] == "cleared"

    keys = store.list_keys()
    assert len(keys) == 0


@pytest.mark.integration
def test_kv_complex_values(client, worker_process):
    """Test storing complex values (dicts, lists)."""
    store = client.entity("KeyValueStore", unique_key("store"))

    complex_value = {
        "nested": {"key": "value"},
        "list": [1, 2, 3],
        "number": 42,
    }

    store.set_value(item_key="complex", value=complex_value)
    result = store.get_value(item_key="complex")

    assert result["nested"]["key"] == "value"
    assert result["list"] == [1, 2, 3]
    assert result["number"] == 42


# =============================================================================
# INVALID INPUT HANDLING (P2.3)
# =============================================================================


@pytest.mark.integration
def test_counter_large_increment(client, worker_process):
    """Test counter with very large increment values.

    Validates:
    - Large numbers are handled correctly
    - No overflow or precision loss
    """
    counter = client.entity("Counter", unique_key("counter-large"))

    # Test with large number
    large_value = 10**12  # 1 trillion
    result = counter.increment(amount=large_value)
    assert result == large_value, f"Large increment failed: expected {large_value}, got {result}"

    # Increment again
    result = counter.increment(amount=large_value)
    assert result == 2 * large_value, f"Double increment failed: expected {2 * large_value}, got {result}"


@pytest.mark.integration
def test_counter_negative_increment(client, worker_process):
    """Test counter with negative increment (effectively decrement).

    Validates:
    - Negative increments work as expected
    - Counter can go below zero
    """
    counter = client.entity("Counter", unique_key("counter-negative"))

    counter.increment(amount=10)
    result = counter.increment(amount=-15)

    # Should be 10 + (-15) = -5
    assert result == -5, f"Negative increment failed: expected -5, got {result}"


@pytest.mark.integration
def test_memory_empty_content(client, worker_process):
    """Test conversation memory with empty content.

    Validates:
    - Empty messages are handled
    - Message count is correct
    """
    memory = client.entity("ConversationMemory", unique_key("memory-empty"))

    result = memory.add_message(role="user", content="")
    assert result["message_count"] == 1

    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == ""


@pytest.mark.integration
def test_memory_very_long_content(client, worker_process):
    """Test conversation memory with very long content.

    Validates:
    - Long messages are stored completely
    - No truncation occurs
    """
    memory = client.entity("ConversationMemory", unique_key("memory-long"))

    long_content = "word " * 10000  # ~50KB of text
    result = memory.add_message(role="user", content=long_content)
    assert result["message_count"] == 1

    messages = memory.get_messages()
    assert len(messages) == 1
    assert len(messages[0]["content"]) == len(long_content), (
        "Long content was truncated"
    )


@pytest.mark.integration
def test_kv_special_key_characters(client, worker_process):
    """Test key-value store with special characters in keys.

    Validates:
    - Keys with special characters work
    - No encoding issues
    """
    store = client.entity("KeyValueStore", unique_key("store-special"))

    special_keys = [
        "key-with-dash",
        "key.with.dots",
        "key_with_underscore",
        "key:with:colons",
        "key/with/slashes",
        "key@with@at",
    ]

    for key in special_keys:
        store.set_value(item_key=key, value=f"value_for_{key}")

    for key in special_keys:
        value = store.get_value(item_key=key)
        assert value == f"value_for_{key}", (
            f"Failed for special key '{key}': got {value}"
        )


@pytest.mark.integration
def test_kv_large_value(client, worker_process):
    """Test storing large nested object (P1.4).

    Validates:
    - Large nested structures can be stored and retrieved
    - Deep nesting is handled correctly
    - Large arrays are preserved
    """
    store = client.entity("KeyValueStore", unique_key("store-large"))

    # Create a large nested object (~100KB when serialized)
    large_value = {
        "metadata": {
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        # Large array of items
        "items": [
            {
                "id": i,
                "name": f"Item {i}",
                "description": "x" * 100,  # 100 chars each
                "tags": [f"tag{j}" for j in range(10)],
            }
            for i in range(500)  # 500 items
        ],
        # Large string field
        "raw_data": "y" * 10_000,
    }

    store.set_value(item_key="large_object", value=large_value)
    result = store.get_value(item_key="large_object")

    # Verify structure preserved
    assert result["metadata"]["version"] == "1.0"
    assert len(result["items"]) == 500
    assert result["items"][0]["id"] == 0
    assert result["items"][499]["id"] == 499
    assert len(result["items"][0]["tags"]) == 10
    assert len(result["raw_data"]) == 10_000


@pytest.mark.integration
def test_kv_deep_nesting(client, worker_process):
    """Test storing deeply nested object (P1.4).

    Validates:
    - 10+ levels of nesting are handled
    - Values at deep levels are preserved
    """
    store = client.entity("KeyValueStore", unique_key("store-deep"))

    # Create deeply nested structure (10 levels)
    deep_value = {"level": 0, "value": "root"}
    current = deep_value
    for i in range(1, 11):
        current["child"] = {"level": i, "value": f"level_{i}"}
        current = current["child"]

    store.set_value(item_key="deep_object", value=deep_value)
    result = store.get_value(item_key="deep_object")

    # Traverse and verify
    assert result["level"] == 0
    current = result
    for i in range(1, 11):
        assert "child" in current, f"Missing child at level {i-1}"
        current = current["child"]
        assert current["level"] == i
        assert current["value"] == f"level_{i}"
