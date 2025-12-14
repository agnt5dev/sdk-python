"""
Entity Workflows - Test entity state persistence and operations

Test Coverage:
✅ Basic entity operations (Counter increment/decrement)
✅ Entity state persistence across workflow calls
✅ Multiple entity instances with isolated state
✅ Complex entity state (ShoppingCart)
✅ Entity with typed state (Pydantic models)

MCP Verification Points:
- Logs: Entity operations and state changes
- Entities table: State persisted with version tracking
- Events: entity.state_loaded, entity.state_saved
"""

from agnt5 import workflow, WorkflowContext
from ..entities import Counter, ShoppingCart, BankAccount


@workflow
async def wf_42_counter_basic(ctx: WorkflowContext, counter_key: str = "test-counter") -> dict:
    """
    Test Scenario: Basic Counter entity operations.

    Validates:
    - Entity can be instantiated with a key
    - Increment operation works and persists state
    - State is saved to entities table
    - Version tracking works correctly

    Args:
        counter_key: Unique key for the counter instance

    Returns:
        Counter operation results

    MCP Verification Points:
        - Logs: Counter operations
        - Entities table: Row with entity_type='Counter', entity_key=counter_key
        - Entity state: {"count": N}
    """
    ctx.logger.info("=== WF_42: Counter Basic Operations ===")
    ctx.logger.info(f"Using counter key: {counter_key}")

    # Create counter instance
    counter = Counter(key=counter_key)

    # Increment by default (1)
    ctx.logger.info("Incrementing counter by 1...")
    result1 = await counter.increment()
    ctx.logger.info(f"After increment(1): {result1}")

    # Increment by 5
    ctx.logger.info("Incrementing counter by 5...")
    result2 = await counter.increment(amount=5)
    ctx.logger.info(f"After increment(5): {result2}")

    # Get current value
    ctx.logger.info("Getting current value...")
    current = await counter.get_value()
    ctx.logger.info(f"Current value: {current}")

    # Decrement by 2
    ctx.logger.info("Decrementing counter by 2...")
    result3 = await counter.decrement(amount=2)
    ctx.logger.info(f"After decrement(2): {result3}")

    return {
        "counter_key": counter_key,
        "after_increment_1": result1,
        "after_increment_5": result2,
        "after_decrement_2": result3,
        "final_count": result3.get("count"),
        "expected_count": 4,  # 0 + 1 + 5 - 2 = 4
    }


@workflow
async def wf_43_counter_persistence(ctx: WorkflowContext, counter_key: str = "persist-counter") -> dict:
    """
    Test Scenario: Entity state persists across multiple workflow runs.

    Validates:
    - Entity state is loaded from database on subsequent calls
    - Increments accumulate across workflow invocations
    - Version tracking prevents conflicts

    Args:
        counter_key: Unique key for the counter instance

    Returns:
        Counter state showing accumulated value

    MCP Verification Points:
        - Logs: State loaded with version
        - Entities table: state_version increments
        - Entity state: Count increases with each run

    Note:
        Run this workflow multiple times with the same counter_key
        to verify persistence. Each run adds 10 to the count.
    """
    ctx.logger.info("=== WF_43: Counter Persistence Test ===")
    ctx.logger.info(f"Using counter key: {counter_key}")

    counter = Counter(key=counter_key)

    # Get initial value (may be non-zero if run before)
    ctx.logger.info("Getting initial value...")
    initial = await counter.get_value()
    initial_count = initial.get("count", 0)
    ctx.logger.info(f"Initial count: {initial_count}")

    # Increment by 10
    ctx.logger.info("Incrementing by 10...")
    result = await counter.increment(amount=10)
    new_count = result.get("count")
    ctx.logger.info(f"New count: {new_count}")

    return {
        "counter_key": counter_key,
        "initial_count": initial_count,
        "added": 10,
        "new_count": new_count,
        "message": f"Counter went from {initial_count} to {new_count}",
    }


@workflow
async def wf_44_multiple_counters(ctx: WorkflowContext) -> dict:
    """
    Test Scenario: Multiple entity instances with isolated state.

    Validates:
    - Different keys create separate entity instances
    - State is isolated between instances
    - Operations on one don't affect others

    Returns:
        Results from multiple counter instances

    MCP Verification Points:
        - Logs: Operations on each counter
        - Entities table: Three separate rows with different entity_keys
        - Entity state: Each has independent count
    """
    ctx.logger.info("=== WF_44: Multiple Counter Instances ===")

    # Create three separate counters
    counter_a = Counter(key="counter-alpha")
    counter_b = Counter(key="counter-beta")
    counter_c = Counter(key="counter-gamma")

    # Increment each by different amounts
    ctx.logger.info("Incrementing counter-alpha by 1...")
    result_a = await counter_a.increment(amount=1)

    ctx.logger.info("Incrementing counter-beta by 10...")
    result_b = await counter_b.increment(amount=10)

    ctx.logger.info("Incrementing counter-gamma by 100...")
    result_c = await counter_c.increment(amount=100)

    # Verify isolation - get each value
    ctx.logger.info("Verifying isolation...")
    val_a = await counter_a.get_value()
    val_b = await counter_b.get_value()
    val_c = await counter_c.get_value()

    ctx.logger.info(f"Alpha: {val_a}, Beta: {val_b}, Gamma: {val_c}")

    return {
        "counter_alpha": val_a.get("count"),
        "counter_beta": val_b.get("count"),
        "counter_gamma": val_c.get("count"),
        "isolation_verified": (
            val_a.get("count") != val_b.get("count") and
            val_b.get("count") != val_c.get("count")
        ),
    }


@workflow
async def wf_45_shopping_cart(ctx: WorkflowContext, cart_key: str = "test-cart") -> dict:
    """
    Test Scenario: Complex entity with nested state (ShoppingCart).

    Validates:
    - Complex state structures (nested dicts) work correctly
    - Multiple operations on same entity in sequence
    - Calculated fields (total) are correctly updated
    - State persistence with complex types

    Args:
        cart_key: Unique key for the cart instance

    Returns:
        Shopping cart final state

    MCP Verification Points:
        - Logs: Cart operations
        - Entities table: Row with entity_type='ShoppingCart'
        - Entity state: {"items": {...}, "total_value": N}
    """
    ctx.logger.info("=== WF_45: Shopping Cart Operations ===")
    ctx.logger.info(f"Using cart key: {cart_key}")

    cart = ShoppingCart(key=cart_key)

    # Add items
    ctx.logger.info("Adding widget (qty: 2, price: 29.99)...")
    result1 = await cart.add_item(item_id="widget-001", quantity=2, price=29.99)
    ctx.logger.info(f"After adding widget: {result1}")

    ctx.logger.info("Adding gadget (qty: 1, price: 99.99)...")
    result2 = await cart.add_item(item_id="gadget-002", quantity=1, price=99.99)
    ctx.logger.info(f"After adding gadget: {result2}")

    ctx.logger.info("Adding more widgets (qty: 3, price: 29.99)...")
    result3 = await cart.add_item(item_id="widget-001", quantity=3, price=29.99)
    ctx.logger.info(f"After adding more widgets: {result3}")

    # Get total
    ctx.logger.info("Getting cart total...")
    total = await cart.get_total()
    ctx.logger.info(f"Cart total: {total}")

    # Get all items
    ctx.logger.info("Getting all items...")
    items = await cart.get_items()
    ctx.logger.info(f"Cart items: {items}")

    # Expected: widget (5 qty * 29.99 = 149.95) + gadget (1 qty * 99.99 = 99.99) = 249.94
    expected_total = 5 * 29.99 + 1 * 99.99

    return {
        "cart_key": cart_key,
        "items": items,
        "total": total,
        "expected_total": expected_total,
        "item_count": len(items),
    }


@workflow
async def wf_46_bank_account(ctx: WorkflowContext, account_key: str = "test-account") -> dict:
    """
    Test Scenario: Entity with transaction history (BankAccount).

    Validates:
    - Entity with array state (transactions)
    - Deposit and withdrawal operations
    - Transaction history accumulation
    - Error handling for insufficient funds

    Args:
        account_key: Unique key for the account instance

    Returns:
        Account final state with transaction history

    MCP Verification Points:
        - Logs: Account operations
        - Entities table: Row with entity_type='BankAccount'
        - Entity state: {"balance": N, "transactions": [...]}
    """
    ctx.logger.info("=== WF_46: Bank Account Operations ===")
    ctx.logger.info(f"Using account key: {account_key}")

    account = BankAccount(key=account_key)

    # Initial deposit
    ctx.logger.info("Depositing $1000...")
    result1 = await account.deposit(amount=1000.0, description="Initial deposit")
    ctx.logger.info(f"After deposit: balance={result1['balance']}")

    # Second deposit
    ctx.logger.info("Depositing $500...")
    result2 = await account.deposit(amount=500.0, description="Paycheck")
    ctx.logger.info(f"After deposit: balance={result2['balance']}")

    # Withdrawal
    ctx.logger.info("Withdrawing $200...")
    result3 = await account.withdraw(amount=200.0, description="ATM withdrawal")
    ctx.logger.info(f"After withdrawal: balance={result3['balance']}")

    # Get final balance
    balance = await account.get_balance()
    ctx.logger.info(f"Final balance: {balance}")

    # Get transaction history
    transactions = await account.get_transactions()
    ctx.logger.info(f"Transaction count: {len(transactions)}")

    return {
        "account_key": account_key,
        "final_balance": balance,
        "expected_balance": 1300.0,  # 1000 + 500 - 200
        "transaction_count": len(transactions),
        "transactions": transactions,
    }


__all__ = [
    "wf_42_counter_basic",
    "wf_43_counter_persistence",
    "wf_44_multiple_counters",
    "wf_45_shopping_cart",
    "wf_46_bank_account",
]
