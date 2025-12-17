"""
Entity Durability Tests

Tests that verify entity state is actually persisted to the database:
1. State written to database after mutations
2. Version increments on each save
3. @query methods don't trigger unnecessary persistence
4. State survives worker restart (subprocess mode only)

These tests complement test_ex_02_entities.py which tests functionality.
This file tests durability - that data actually reaches the database.

Run with:
    # Local mode (requires running dev server)
    pytest tests/integration/test_ex_02_entities_durability.py -v

    # Subprocess mode (for CI)
    pytest tests/integration/test_ex_02_entities_durability.py -v --use-subprocess
"""

import json
import sqlite3
import uuid
from typing import Dict, Optional

import pytest


# =============================================================================
# HELPER FUNCTIONS (inline to avoid import issues with pytest)
# =============================================================================


def get_db_path_for_platform(platform: Dict) -> str:
    """
    Get the correct database path for the current platform mode.

    Handles the different database path patterns across modes:
    - Local: /tmp/agnt5/<service-name>/agnt5.db
    - Subprocess: {data_dir}/orchestration.db
    """
    mode = platform.get("mode")

    if mode == "local":
        # Local mode: DB is at /tmp/agnt5/<service>/agnt5.db
        # The service name is "agnt5-python-examples" when running with `just platform standalone python`
        service_name = "agnt5-python-examples"
        return f"/tmp/agnt5/{service_name}/agnt5.db"
    elif mode == "subprocess":
        return platform.get("host_db_path") or platform.get("db_url")
    else:
        return platform.get("db_url")


def get_entity_state_from_platform(
    db_url: str,
    entity_type: str,
    key: str
) -> Optional[Dict]:
    """
    Query database directly for entity state.

    This verifies state was actually persisted to platform,
    not just stored in-memory in the worker.

    Note: Due to a known platform issue with NULL scope_id uniqueness,
    we query for the latest version by state_version DESC.
    """
    conn = sqlite3.connect(db_url)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT current_state, state_version, updated_at
            FROM entities
            WHERE entity_type = ? AND entity_key = ?
            ORDER BY state_version DESC
            LIMIT 1
            """,
            (entity_type, key)
        )

        row = cursor.fetchone()

        if row:
            current_state, state_version, updated_at = row
            # Handle both string and bytes (SQLite can return either depending on how data was stored)
            if isinstance(current_state, bytes):
                current_state = current_state.decode('utf-8')
            return {
                "state": json.loads(current_state) if isinstance(current_state, str) else current_state,
                "version": state_version,
                "updated_at": updated_at
            }

        return None

    finally:
        cursor.close()
        conn.close()


def unique_key(prefix: str) -> str:
    """Generate a unique entity key to ensure test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# =============================================================================
# DATABASE PERSISTENCE TESTS
# =============================================================================


@pytest.mark.integration
def test_counter_state_persisted_to_database(client, worker_process, platform):
    """Verify Counter state is written to database."""
    key = unique_key("counter-db")
    counter = client.entity("Counter", key)

    # Mutate state
    counter.increment(amount=42)

    # Query database directly
    db_path = get_db_path_for_platform(platform)
    db_state = get_entity_state_from_platform(db_path, "Counter", key)

    assert db_state is not None, "Entity should exist in database"
    assert db_state["state"]["count"] == 42
    assert db_state["version"] >= 1


@pytest.mark.integration
def test_counter_version_increments(client, worker_process, platform):
    """Verify version increments on each mutation."""
    key = unique_key("counter-version")
    counter = client.entity("Counter", key)

    db_path = get_db_path_for_platform(platform)

    # First mutation
    counter.increment(amount=1)
    state1 = get_entity_state_from_platform(db_path, "Counter", key)
    version1 = state1["version"]

    # Second mutation
    counter.increment(amount=1)
    state2 = get_entity_state_from_platform(db_path, "Counter", key)
    version2 = state2["version"]

    assert version2 > version1, "Version should increment on mutation"


@pytest.mark.integration
def test_query_methods_dont_increment_version(client, worker_process, platform):
    """Verify @query methods don't trigger state persistence."""
    key = unique_key("counter-query")
    counter = client.entity("Counter", key)

    db_path = get_db_path_for_platform(platform)

    # Mutate to establish initial state
    counter.increment(amount=10)
    state_before = get_entity_state_from_platform(db_path, "Counter", key)
    version_before = state_before["version"]

    # Call query method multiple times
    counter.get_count()
    counter.get_count()
    counter.get_count()

    # Version should not change
    state_after = get_entity_state_from_platform(db_path, "Counter", key)
    assert state_after["version"] == version_before, "@query methods should not increment version"


@pytest.mark.integration
def test_keyvaluestore_persisted_to_database(client, worker_process, platform):
    """Verify KeyValueStore state is written to database."""
    key = unique_key("kv-db")
    store = client.entity("KeyValueStore", key)

    store.set_value(item_key="theme", value="dark")
    store.set_value(item_key="language", value="en")

    db_path = get_db_path_for_platform(platform)
    db_state = get_entity_state_from_platform(db_path, "KeyValueStore", key)

    assert db_state is not None, "Entity should exist in database"
    assert db_state["state"]["data"]["theme"] == "dark"
    assert db_state["state"]["data"]["language"] == "en"


@pytest.mark.integration
def test_conversation_memory_persisted_to_database(client, worker_process, platform):
    """Verify ConversationMemory state is written to database."""
    key = unique_key("memory-db")
    memory = client.entity("ConversationMemory", key)

    memory.add_message(role="user", content="Hello!")
    memory.add_message(role="assistant", content="Hi there!")

    db_path = get_db_path_for_platform(platform)
    db_state = get_entity_state_from_platform(db_path, "ConversationMemory", key)

    assert db_state is not None, "Entity should exist in database"
    messages = db_state["state"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello!"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there!"


# =============================================================================
# WORKER RESTART DURABILITY TESTS (subprocess mode only)
# =============================================================================


@pytest.mark.integration
def test_counter_survives_worker_restart(client, worker_process, platform):
    """
    Verify entity state survives worker restart.

    This test is only meaningful in subprocess mode where we can
    restart the worker. In local mode, we just verify the state
    persists across multiple reads.
    """
    key = unique_key("counter-restart")
    counter = client.entity("Counter", key)

    # Set initial state
    counter.increment(amount=100)
    assert counter.get_count() == 100

    if platform["mode"] == "local":
        # In local mode, just verify state persists in DB
        db_path = get_db_path_for_platform(platform)
        db_state = get_entity_state_from_platform(db_path, "Counter", key)
        assert db_state is not None
        assert db_state["state"]["count"] == 100
        print("✓ Local mode: State verified in database")
    else:
        # In subprocess mode, actually restart the worker
        # Import here to avoid module resolution issues in local mode
        from conftest import restart_worker
        new_worker = restart_worker(worker_process, platform)

        try:
            # Verify state persisted after restart
            counter2 = client.entity("Counter", key)
            assert counter2.get_count() == 100, "State should survive worker restart"
            print("✓ Subprocess mode: State survived worker restart")
        finally:
            new_worker.terminate()


# =============================================================================
# EDGE CASES
# =============================================================================


@pytest.mark.integration
def test_entity_not_in_db_before_mutation(client, worker_process, platform):
    """Verify entity doesn't exist in DB until first mutation."""
    key = unique_key("counter-empty")
    db_path = get_db_path_for_platform(platform)

    # Before any operations, entity shouldn't exist
    db_state = get_entity_state_from_platform(db_path, "Counter", key)
    assert db_state is None, "Entity shouldn't exist before first mutation"


@pytest.mark.integration
def test_complex_state_serialization(client, worker_process, platform):
    """Verify complex nested state is correctly serialized/deserialized."""
    key = unique_key("kv-complex")
    store = client.entity("KeyValueStore", key)

    complex_value = {
        "nested": {"deep": {"value": 42}},
        "array": [1, 2, {"x": 3}],
        "unicode": "Hello 世界 🌍",
        "numbers": [1.5, -2.7, 0],
        "boolean": True,
        "null_value": None,
    }

    store.set_value(item_key="complex", value=complex_value)

    db_path = get_db_path_for_platform(platform)
    db_state = get_entity_state_from_platform(db_path, "KeyValueStore", key)

    assert db_state is not None
    stored_value = db_state["state"]["data"]["complex"]

    assert stored_value["nested"]["deep"]["value"] == 42
    assert stored_value["array"] == [1, 2, {"x": 3}]
    assert stored_value["unicode"] == "Hello 世界 🌍"
    assert stored_value["numbers"] == [1.5, -2.7, 0]
    assert stored_value["boolean"] is True
    assert stored_value["null_value"] is None
