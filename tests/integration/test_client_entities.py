"""
Integration Tests: Entity Persistence

Tests entity state persistence and concurrency using the Client API.

These tests validate:
- Entity state persists to platform (not just in-memory)
- State survives worker restarts/crashes
- Concurrent updates don't cause lost writes (single-writer guarantee)
- Entity keys are properly isolated

Expected Results (Week 1 - RED Phase):
- test_entity_isolation: ✅ PASS (in-memory state works)
- test_entity_state_persists_across_restart: ❌ FAIL (TODO: platform persistence)
- test_entity_concurrent_updates_no_lost_writes: ❌ FAIL (TODO: distributed locking)
"""

import asyncio

import pytest

from conftest import restart_worker


@pytest.mark.integration
def test_entity_basic_operation(client, worker_process):
    """
    Test basic entity operation works.

    This should PASS - validates basic client → platform → worker flow.
    """
    # Add item to shopping cart
    result = client.entity("ShoppingCart", "user-123").add_item(
        item_id="widget-1",
        quantity=2,
        price=29.99
    )

    assert result["total_items"] == 1

    # Get total
    total = client.entity("ShoppingCart", "user-123").get_total()
    assert total == 59.98  # 2 * 29.99


@pytest.mark.integration
def test_entity_isolation_between_keys(client, worker_process):
    """
    Test different entity keys have isolated state.

    This should PASS - even in-memory state is properly isolated by key.
    """
    # Modify two different shopping carts
    result1 = client.entity("ShoppingCart", "alice").add_item(
        item_id="item-1",
        quantity=1,
        price=10.0
    )
    assert result1["total_items"] == 1

    result2 = client.entity("ShoppingCart", "bob").add_item(
        item_id="item-2",
        quantity=2,
        price=20.0
    )
    assert result2["total_items"] == 1

    # Verify isolation
    alice_total = client.entity("ShoppingCart", "alice").get_total()
    bob_total = client.entity("ShoppingCart", "bob").get_total()

    assert alice_total == 10.0
    assert bob_total == 40.0  # 2 * 20.0


@pytest.mark.integration
def test_entity_state_persists_across_restart(client, worker_process, platform):
    """
    Test entity state survives worker restart.

    ❌ Expected to FAIL - exposes TODO in entity.py:355-363

    Failure indicates:
    - Entity state is only stored in-memory in worker
    - EntityStateManager._rust_manager is None
    - load_state() and save_state() are not calling platform

    This test drives implementation of platform persistence in Week 2.
    """
    # 1. Add item to shopping cart
    result = client.entity("ShoppingCart", "user-123").add_item(
        item_id="widget-1",
        quantity=2,
        price=29.99
    )
    assert result["total_items"] == 1

    # 2. Restart worker (simulate crash)
    new_worker = restart_worker(worker_process, platform)

    # 3. Verify state was restored from platform
    # ❌ This will FAIL because state was not persisted
    total = client.entity("ShoppingCart", "user-123").get_total()

    # ❌ FAILS HERE - total will be 0.0 because state was lost
    assert total == 59.98, (
        f"Expected state to persist across restart, but got {total}. "
        "This indicates entity state is not being saved to platform. "
        "See entity.py:355-363 for TODOs."
    )

    # Cleanup
    new_worker.terminate()


@pytest.mark.integration
def test_entity_state_with_bank_account(client, worker_process, platform):
    """
    Test entity state persistence with BankAccount entity.

    ❌ Expected to FAIL - another example of missing platform persistence.

    Tests transaction history preservation across restarts.
    """
    # 1. Deposit money
    result = client.entity("BankAccount", "alice").deposit(amount=100.0)
    assert result["balance"] == 100.0

    # 2. Restart worker
    new_worker = restart_worker(worker_process, platform)

    # 3. Balance should be restored
    # ❌ This will FAIL
    balance = client.entity("BankAccount", "alice").get_balance()

    assert balance == 100.0, (
        f"Expected balance to persist, but got {balance}. "
        "Entity state was not persisted to platform."
    )

    # 4. Transaction history should also be preserved
    transactions = client.entity("BankAccount", "alice").get_transactions()

    assert len(transactions) == 1, (
        f"Expected 1 transaction, but got {len(transactions)}. "
        "Transaction history was not persisted."
    )

    # Cleanup
    new_worker.terminate()


@pytest.mark.integration
def test_entity_concurrent_updates_no_lost_writes(client, worker_process):
    """
    Test single-writer guarantee with concurrent updates.

    ❌ Expected to FAIL - exposes need for distributed locking.

    Without platform-coordinated locking, concurrent updates can cause:
    - Lost writes (last write doesn't see previous updates)
    - Race conditions
    - Inconsistent state

    This test drives implementation of distributed locking in Week 2-3.
    """
    # Reset counter
    client.entity("Counter", "shared").reset()

    # 50 concurrent increments
    async def run_increments():
        tasks = []
        for _ in range(50):
            # Each call is synchronous HTTP but we can parallelize
            task = asyncio.to_thread(
                lambda: client.entity("Counter", "shared").increment()
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    asyncio.run(run_increments())

    # Verify all updates applied (no lost writes)
    count = client.entity("Counter", "shared").get_count()

    # ❌ This will likely FAIL with count < 50
    # Without distributed locking, concurrent updates can be lost
    assert count == 50, (
        f"Expected 50 increments, but got {count}. "
        f"Lost {50 - count} writes due to lack of distributed locking. "
        "Entity methods need platform-coordinated single-writer guarantee."
    )


@pytest.mark.integration
def test_entity_multiple_method_calls(client, worker_process):
    """
    Test multiple method calls on same entity instance.

    This should PASS - validates state is maintained across method calls.
    """
    # Multiple operations on same cart
    client.entity("ShoppingCart", "user-456").add_item("item-1", 1, 10.0)
    client.entity("ShoppingCart", "user-456").add_item("item-2", 2, 20.0)
    client.entity("ShoppingCart", "user-456").add_item("item-3", 3, 30.0)

    # Check total
    total = client.entity("ShoppingCart", "user-456").get_total()

    # 1*10 + 2*20 + 3*30 = 10 + 40 + 90 = 140
    assert total == 140.0

    # Check items
    items = client.entity("ShoppingCart", "user-456").get_items()
    assert len(items) == 3


@pytest.mark.integration
@pytest.mark.skip(reason="Platform state verification not yet implemented")
def test_entity_state_in_platform_database(client, worker_process, platform):
    """
    Test entity state is actually in platform database.

    This test queries database directly to verify state persistence.
    Works with SQLite, PostgreSQL, and CockroachDB.

    ⏭️  SKIPPED - Requires platform verification utilities.

    Will be enabled in Week 2 after implementing utils.py
    """
    from tests.integration.utils import get_entity_state_from_platform

    # Add item
    client.entity("ShoppingCart", "user-789").add_item("widget", 1, 50.0)

    # Query platform database directly
    state = get_entity_state_from_platform(
        db_url=platform["db_url"],
        entity_type="ShoppingCart",
        key="user-789"
    )

    # Verify state exists in database
    assert state is not None
    assert "items" in state
    assert "widget" in state["items"]
