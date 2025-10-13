"""
Test Components for Integration Testing

Provides minimal functions, entities, and workflows for testing
platform integration end-to-end.

Components:
- Functions: greet, long_task, flaky_function, failing_function
- Entities: ShoppingCart, Counter, BankAccount
- Workflows: order_fulfillment, long_workflow
"""

import asyncio
from typing import Dict

from agnt5 import Context, Entity, function, workflow


# ==================== Functions ====================


@function
async def greet(ctx: Context, name: str) -> Dict:
    """Simple greeting function for basic execution tests."""
    ctx.logger.info(f"Greeting {name}")
    return {"message": f"Hello, {name}!"}


@function
async def long_task(ctx: Context, duration: int) -> Dict:
    """Simulates a long-running task."""
    ctx.logger.info(f"Starting long task ({duration}s)")
    await asyncio.sleep(duration)
    return {"status": "completed", "duration": duration}


@function(retries=5, retry_delay_ms=100)
async def flaky_function(ctx: Context, fail_count: int) -> Dict:
    """
    Fails `fail_count` times, then succeeds.

    Used to test retry logic.
    """
    attempt = ctx.attempt if hasattr(ctx, 'attempt') else 0

    ctx.logger.info(f"Flaky function attempt {attempt + 1}")

    if attempt < fail_count:
        raise Exception(f"Simulated failure (attempt {attempt + 1})")

    return {"succeeded": True, "attempts": attempt + 1}


@function
async def failing_function(ctx: Context, error: str) -> Dict:
    """Always fails with given error message."""
    ctx.logger.error(f"Failing with: {error}")
    raise Exception(error)


# ==================== Entities ====================


class ShoppingCart(Entity):
    """
    Shopping cart entity for testing state persistence.

    Tests:
    - State persists across worker restarts
    - State isolation between entity keys
    - Concurrent updates don't cause lost writes
    """

    async def add_item(self, item_id: str, quantity: int, price: float) -> Dict:
        """Add item to cart."""
        self.ctx.logger.info(f"Adding item {item_id} to cart")

        items = self.state.get("items", {})
        items[item_id] = {"quantity": quantity, "price": price}
        self.state.set("items", items)

        return {"total_items": len(items)}

    async def get_total(self) -> float:
        """Calculate cart total."""
        items = self.state.get("items", {})
        total = sum(
            item["quantity"] * item["price"]
            for item in items.values()
        )
        return total

    async def get_items(self) -> Dict:
        """Get all items in cart."""
        return self.state.get("items", {})

    async def clear(self) -> Dict:
        """Clear cart."""
        self.state.set("items", {})
        return {"status": "cleared"}


class Counter(Entity):
    """
    Counter entity for testing concurrency.

    Tests:
    - Concurrent increments don't cause lost updates
    - Single-writer guarantee works across multiple requests
    """

    async def increment(self) -> int:
        """Increment counter."""
        count = self.state.get("count", 0)

        # Simulate delay to expose race conditions
        await asyncio.sleep(0.01)

        new_count = count + 1
        self.state.set("count", new_count)

        return new_count

    async def get_count(self) -> int:
        """Get current count."""
        return self.state.get("count", 0)

    async def reset(self) -> Dict:
        """Reset counter to zero."""
        self.state.set("count", 0)
        return {"status": "reset", "count": 0}


class BankAccount(Entity):
    """
    Bank account entity for testing state durability.

    Tests:
    - Balance survives worker crashes
    - Transaction history is preserved
    """

    async def deposit(self, amount: float) -> Dict:
        """Deposit money into account."""
        balance = self.state.get("balance", 0.0)
        transactions = self.state.get("transactions", [])

        new_balance = balance + amount
        transactions.append({"type": "deposit", "amount": amount})

        self.state.set("balance", new_balance)
        self.state.set("transactions", transactions)

        return {"balance": new_balance}

    async def withdraw(self, amount: float) -> Dict:
        """Withdraw money from account."""
        balance = self.state.get("balance", 0.0)
        transactions = self.state.get("transactions", [])

        if balance < amount:
            raise ValueError(f"Insufficient funds: {balance} < {amount}")

        new_balance = balance - amount
        transactions.append({"type": "withdraw", "amount": amount})

        self.state.set("balance", new_balance)
        self.state.set("transactions", transactions)

        return {"balance": new_balance}

    async def get_balance(self) -> float:
        """Get current balance."""
        return self.state.get("balance", 0.0)

    async def get_transactions(self) -> list:
        """Get transaction history."""
        return self.state.get("transactions", [])


# ==================== Workflows ====================


@workflow
async def order_fulfillment(ctx: Context, order_id: str) -> Dict:
    """
    Multi-step order processing workflow.

    Tests:
    - Basic workflow execution
    - Sequential step execution
    - State management across steps
    """
    ctx.logger.info(f"Starting order fulfillment for {order_id}")

    # Step 1: Validate order
    ctx.state.set("status", "validating")
    ctx.logger.info("Validating order...")
    await asyncio.sleep(0.1)

    # Step 2: Process payment
    ctx.state.set("status", "processing_payment")
    ctx.logger.info("Processing payment...")
    await asyncio.sleep(0.1)
    payment_id = f"pay_{order_id}"

    # Step 3: Ship order
    ctx.state.set("status", "shipping")
    ctx.logger.info("Shipping order...")
    await asyncio.sleep(0.1)
    tracking_number = f"TRACK_{order_id}"

    # Complete
    ctx.state.set("status", "completed")

    return {
        "status": "completed",
        "order_id": order_id,
        "payment_id": payment_id,
        "tracking_number": tracking_number,
    }


@workflow
async def long_workflow(ctx: Context, steps: int) -> Dict:
    """
    Multi-step workflow for testing crash recovery.

    Tests:
    - Workflow resumes after worker crash
    - Steps are not re-executed (idempotency)
    - Checkpoint persistence works
    """
    ctx.logger.info(f"Starting long workflow with {steps} steps")

    ctx.state.set("completed_steps", [])

    for i in range(steps):
        ctx.logger.info(f"Executing step {i + 1}/{steps}")
        ctx.state.set("current_step", i)

        await asyncio.sleep(1)

        completed = ctx.state.get("completed_steps", [])
        completed.append(i)
        ctx.state.set("completed_steps", completed)

        ctx.logger.info(f"Step {i + 1} completed")

    ctx.state.set("status", "completed")

    return {
        "status": "completed",
        "steps_completed": steps,
        "completed_steps": ctx.state.get("completed_steps", [])
    }


@function
async def generate_text(ctx: Context, prompt: str) -> str:
    """Simulates streaming text generation."""
    ctx.logger.info(f"Generating text for prompt: {prompt}")

    # For streaming tests - future implementation
    # For now, return simple response
    return f"Generated response for: {prompt}"
