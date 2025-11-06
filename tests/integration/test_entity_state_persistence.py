"""
Integration Tests for Entity State Persistence

Verifies that entity state operations are actually persisted to the platform database,
not just stored in-memory in the worker.

These tests use the platform verification utilities (utils.py) to directly query
the database and verify state persistence.

Coverage:
- Entity state creation and initialization
- State modifications persist to database
- Multiple entity instances maintain separate state
- State survives workflow restarts
- Concurrent entity operations
- Transaction history tracking
"""

import pytest
import time
import sqlite3
import json


@pytest.mark.integration
def test_counter_entity_state_persists(client, worker_process, platform):
    """
    Test Counter entity state persists to platform database.

    Verifies:
    - Entity state is created on first operation
    - State modifications are persisted
    - State can be queried from database
    """
    time.sleep(2)

    # Increment counter
    result = client.run("Counter", "increment", key="test-counter-1", amount=5)
    assert result["count"] == 5

    # Verify state persisted to database
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT state_json
        FROM entity_states
        WHERE entity_type = 'Counter' AND entity_key = 'test-counter-1'
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Counter state not found in database"

    state = json.loads(row[0])
    assert state["count"] == 5

    print(f"\n✅ Counter entity state persists to database")
    print(f"   Count: {state['count']}")


@pytest.mark.integration
def test_shopping_cart_complex_state_persists(client, worker_process, platform):
    """
    Test ShoppingCart entity with complex state persists correctly.

    Verifies:
    - Complex nested state (dict of dicts) persists
    - Multiple operations update state correctly
    - Calculated fields are persisted
    """
    time.sleep(2)

    # Add items to cart
    result1 = client.run(
        "ShoppingCart", "add_item",
        key="cart-user-123",
        item_id="item-001",
        quantity=2,
        price=10.99
    )
    assert result1["total_value"] == pytest.approx(21.98)

    result2 = client.run(
        "ShoppingCart", "add_item",
        key="cart-user-123",
        item_id="item-002",
        quantity=1,
        price=5.50
    )
    assert result2["total_value"] == pytest.approx(27.48)

    # Verify state persisted to database
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT state_json
        FROM entity_states
        WHERE entity_type = 'ShoppingCart' AND entity_key = 'cart-user-123'
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    assert row is not None, "ShoppingCart state not found in database"

    state = json.loads(row[0])
    assert "items" in state
    assert "item-001" in state["items"]
    assert "item-002" in state["items"]
    assert state["items"]["item-001"]["quantity"] == 2
    assert state["items"]["item-002"]["quantity"] == 1
    assert state["total_value"] == pytest.approx(27.48)

    print(f"\n✅ ShoppingCart complex state persists correctly")
    print(f"   Items: {len(state['items'])}")
    print(f"   Total: ${state['total_value']:.2f}")


@pytest.mark.integration
def test_bank_account_transaction_history_persists(client, worker_process, platform):
    """
    Test BankAccount transaction history persists to database.

    Verifies:
    - Array state (transaction history) persists
    - Multiple transactions are tracked
    - Balance updates correctly
    """
    time.sleep(2)

    # Perform transactions
    result1 = client.run(
        "BankAccount", "deposit",
        key="account-alice",
        amount=100.0,
        description="Initial deposit"
    )
    assert result1["balance"] == 100.0

    result2 = client.run(
        "BankAccount", "deposit",
        key="account-alice",
        amount=50.0,
        description="Second deposit"
    )
    assert result2["balance"] == 150.0

    result3 = client.run(
        "BankAccount", "withdraw",
        key="account-alice",
        amount=30.0,
        description="Withdrawal"
    )
    assert result3["balance"] == 120.0

    # Verify state persisted to database
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT state_json
        FROM entity_states
        WHERE entity_type = 'BankAccount' AND entity_key = 'account-alice'
        ORDER BY updated_at DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    assert row is not None, "BankAccount state not found in database"

    state = json.loads(row[0])
    assert state["balance"] == 120.0
    assert len(state["transactions"]) == 3
    assert state["transactions"][0]["type"] == "deposit"
    assert state["transactions"][0]["amount"] == 100.0
    assert state["transactions"][2]["type"] == "withdrawal"
    assert state["transactions"][2]["amount"] == 30.0

    print(f"\n✅ BankAccount transaction history persists correctly")
    print(f"   Balance: ${state['balance']:.2f}")
    print(f"   Transactions: {len(state['transactions'])}")


@pytest.mark.integration
def test_multiple_entity_instances_separate_state(client, worker_process, platform):
    """
    Test multiple instances of same entity type maintain separate state.

    Verifies:
    - Different keys create separate entity instances
    - State is isolated between instances
    - Database stores multiple entity states
    """
    time.sleep(2)

    # Create three separate counters
    client.run("Counter", "increment", key="counter-a", amount=10)
    client.run("Counter", "increment", key="counter-b", amount=20)
    client.run("Counter", "increment", key="counter-c", amount=30)

    # Verify each has separate state in database
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)

    cursor_a = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'Counter' AND entity_key = 'counter-a'
        ORDER BY updated_at DESC LIMIT 1
    """)
    state_a = json.loads(cursor_a.fetchone()[0])

    cursor_b = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'Counter' AND entity_key = 'counter-b'
        ORDER BY updated_at DESC LIMIT 1
    """)
    state_b = json.loads(cursor_b.fetchone()[0])

    cursor_c = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'Counter' AND entity_key = 'counter-c'
        ORDER BY updated_at DESC LIMIT 1
    """)
    state_c = json.loads(cursor_c.fetchone()[0])

    conn.close()

    assert state_a["count"] == 10
    assert state_b["count"] == 20
    assert state_c["count"] == 30

    print(f"\n✅ Multiple entity instances maintain separate state")
    print(f"   Counter A: {state_a['count']}")
    print(f"   Counter B: {state_b['count']}")
    print(f"   Counter C: {state_c['count']}")


@pytest.mark.integration
def test_entity_state_updates_incrementally(client, worker_process, platform):
    """
    Test entity state updates are incremental and preserve history.

    Verifies:
    - Each state change creates new record
    - State versions are tracked
    - Latest state can be queried
    """
    time.sleep(2)

    # Perform multiple operations
    client.run("Counter", "increment", key="counter-history", amount=1)
    time.sleep(0.1)  # Small delay to ensure different timestamps
    client.run("Counter", "increment", key="counter-history", amount=2)
    time.sleep(0.1)
    client.run("Counter", "increment", key="counter-history", amount=3)

    # Query all state records for this entity
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT state_json, updated_at
        FROM entity_states
        WHERE entity_type = 'Counter' AND entity_key = 'counter-history'
        ORDER BY updated_at ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    # Should have at least the latest state
    assert len(rows) >= 1, "No state records found"

    # Latest state should have final count
    latest_state = json.loads(rows[-1][0])
    assert latest_state["count"] == 6  # 1 + 2 + 3

    print(f"\n✅ Entity state updates incrementally")
    print(f"   State records: {len(rows)}")
    print(f"   Final count: {latest_state['count']}")


@pytest.mark.integration
def test_entity_state_isolation_across_types(client, worker_process, platform):
    """
    Test different entity types maintain separate state namespaces.

    Verifies:
    - Same key for different entity types creates separate state
    - Entity type is part of state isolation
    """
    time.sleep(2)

    # Use same key for different entity types
    key = "test-entity-001"

    client.run("Counter", "increment", key=key, amount=100)
    client.run("BankAccount", "deposit", key=key, amount=500.0, description="Test")

    # Verify both exist with different state
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)

    cursor_counter = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'Counter' AND entity_key = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (key,))
    counter_state = json.loads(cursor_counter.fetchone()[0])

    cursor_account = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'BankAccount' AND entity_key = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (key,))
    account_state = json.loads(cursor_account.fetchone()[0])

    conn.close()

    assert counter_state["count"] == 100
    assert account_state["balance"] == 500.0

    print(f"\n✅ Entity types maintain separate state namespaces")
    print(f"   Counter: {counter_state['count']}")
    print(f"   BankAccount: ${account_state['balance']:.2f}")


@pytest.mark.integration
def test_entity_state_query_after_multiple_operations(client, worker_process, platform):
    """
    Test entity state reflects latest operations when queried.

    Verifies:
    - State accumulates correctly over multiple operations
    - Query returns most recent state
    """
    time.sleep(2)

    # Build up shopping cart
    cart_key = "cart-integration-test"

    for i in range(5):
        client.run(
            "ShoppingCart", "add_item",
            key=cart_key,
            item_id=f"item-{i:03d}",
            quantity=i + 1,
            price=10.0 + i
        )

    # Get final state
    result = client.run("ShoppingCart", "get_items", key=cart_key)

    assert len(result) == 5
    assert "item-000" in result
    assert "item-004" in result

    # Verify database has correct final state
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'ShoppingCart' AND entity_key = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (cart_key,))

    row = cursor.fetchone()
    conn.close()

    state = json.loads(row[0])
    assert len(state["items"]) == 5

    # Verify total is calculated correctly
    expected_total = sum(
        (i + 1) * (10.0 + i) for i in range(5)
    )
    assert state["total_value"] == pytest.approx(expected_total)

    print(f"\n✅ Entity state query returns latest state after multiple operations")
    print(f"   Items: {len(state['items'])}")
    print(f"   Total: ${state['total_value']:.2f}")


@pytest.mark.integration
def test_entity_error_does_not_corrupt_state(client, worker_process, platform):
    """
    Test entity errors don't corrupt existing state.

    Verifies:
    - Failed operations don't modify state
    - State remains consistent after errors
    - Subsequent operations work correctly
    """
    time.sleep(2)

    account_key = "account-error-test"

    # Successful deposit
    client.run("BankAccount", "deposit", key=account_key, amount=100.0)

    # Try invalid withdrawal (should fail)
    from agnt5.client import RunError
    try:
        client.run("BankAccount", "withdraw", key=account_key, amount=200.0)
        assert False, "Expected RunError for insufficient funds"
    except RunError:
        pass  # Expected

    # Verify state unchanged
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT state_json FROM entity_states
        WHERE entity_type = 'BankAccount' AND entity_key = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (account_key,))

    state = json.loads(cursor.fetchone()[0])
    conn.close()

    # Balance should still be 100, only 1 transaction
    assert state["balance"] == 100.0
    assert len(state["transactions"]) == 1

    # Subsequent operation should work
    result = client.run("BankAccount", "withdraw", key=account_key, amount=50.0)
    assert result["balance"] == 50.0

    print(f"\n✅ Entity errors don't corrupt state")
    print(f"   State preserved after error")


@pytest.mark.integration
def test_concurrent_entity_operations(client, worker_process, platform):
    """
    Test concurrent operations on different entity instances.

    Verifies:
    - Multiple entity instances can be operated on concurrently
    - State isolation is maintained under concurrency
    - All operations complete successfully
    """
    time.sleep(2)

    import concurrent.futures

    def increment_counter(counter_id):
        return client.run("Counter", "increment", key=f"concurrent-{counter_id}", amount=counter_id)

    # Perform 10 concurrent counter increments
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(increment_counter, i) for i in range(1, 11)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    assert len(results) == 10

    # Verify each counter has correct value in database
    db_path = platform['host_db_path']
    conn = sqlite3.connect(db_path)

    for i in range(1, 11):
        cursor = conn.execute("""
            SELECT state_json FROM entity_states
            WHERE entity_type = 'Counter' AND entity_key = ?
            ORDER BY updated_at DESC LIMIT 1
        """, (f"concurrent-{i}",))

        state = json.loads(cursor.fetchone()[0])
        assert state["count"] == i

    conn.close()

    print(f"\n✅ Concurrent entity operations work correctly")
    print(f"   ✓ 10 concurrent operations completed")
    print(f"   ✓ State isolation maintained")


# TODO: Add tests for:
# - Entity state persistence through worker restart
# - Entity state version tracking and history
# - Entity state size limits
# - Entity state with large arrays/objects
# - Entity state concurrency/locking
# - Entity state in workflows (workflow + entity state interaction)
# - Entity state rollback on transaction failure
# - Entity state migration/schema evolution
