"""
Test Entities for Integration Testing

Provides stateful entities for testing state persistence and concurrency.
"""

from typing import Optional
from pydantic import BaseModel, Field
from agnt5 import Entity, EntityContext


# ============================================================================
# Entity State Schemas
# ============================================================================

class CounterState(BaseModel):
    """State schema for Counter entity."""
    count: int = Field(default=0, description="Current count value")


class ShoppingCartState(BaseModel):
    """State schema for ShoppingCart entity."""
    items: dict[str, dict] = Field(
        default_factory=dict,
        description="Items in cart {item_id: {quantity, price}}"
    )
    total_value: float = Field(default=0.0, description="Total cart value")


class BankAccountState(BaseModel):
    """State schema for BankAccount entity."""
    balance: float = Field(default=0.0, description="Current account balance")
    transactions: list[dict] = Field(
        default_factory=list,
        description="Transaction history"
    )


# ============================================================================
# Entity Definitions
# ============================================================================

class Counter(Entity, state_schema=CounterState):
    """Simple counter entity for testing basic state management."""

    async def increment(self, ctx: EntityContext, amount: int = 1) -> dict:
        """Increment the counter."""
        state = await ctx.get_state(CounterState)
        state.count += amount
        await ctx.set_state(state)
        return {"count": state.count}

    async def decrement(self, ctx: EntityContext, amount: int = 1) -> dict:
        """Decrement the counter."""
        state = await ctx.get_state(CounterState)
        state.count -= amount
        await ctx.set_state(state)
        return {"count": state.count}

    async def get_value(self, ctx: EntityContext) -> dict:
        """Get current counter value."""
        state = await ctx.get_state(CounterState)
        return {"count": state.count}

    async def reset(self, ctx: EntityContext) -> dict:
        """Reset counter to zero."""
        state = CounterState(count=0)
        await ctx.set_state(state)
        return {"count": 0}


class ShoppingCart(Entity, state_schema=ShoppingCartState):
    """Shopping cart entity for testing complex state operations."""

    async def add_item(
        self,
        ctx: EntityContext,
        item_id: str,
        quantity: int,
        price: float
    ) -> dict:
        """Add item to cart."""
        state = await ctx.get_state(ShoppingCartState)

        if item_id in state.items:
            # Update existing item
            state.items[item_id]["quantity"] += quantity
        else:
            # Add new item
            state.items[item_id] = {
                "quantity": quantity,
                "price": price
            }

        # Recalculate total
        state.total_value = sum(
            item["quantity"] * item["price"]
            for item in state.items.values()
        )

        await ctx.set_state(state)

        return {
            "item_id": item_id,
            "quantity": state.items[item_id]["quantity"],
            "total_items": len(state.items),
            "total_value": state.total_value
        }

    async def remove_item(self, ctx: EntityContext, item_id: str) -> dict:
        """Remove item from cart."""
        state = await ctx.get_state(ShoppingCartState)

        if item_id not in state.items:
            raise ValueError(f"Item {item_id} not in cart")

        del state.items[item_id]

        # Recalculate total
        state.total_value = sum(
            item["quantity"] * item["price"]
            for item in state.items.values()
        )

        await ctx.set_state(state)

        return {
            "item_id": item_id,
            "total_items": len(state.items),
            "total_value": state.total_value
        }

    async def get_total(self, ctx: EntityContext) -> float:
        """Get total cart value."""
        state = await ctx.get_state(ShoppingCartState)
        return state.total_value

    async def clear(self, ctx: EntityContext) -> dict:
        """Clear all items from cart."""
        state = ShoppingCartState()
        await ctx.set_state(state)
        return {"total_items": 0, "total_value": 0.0}


class BankAccount(Entity, state_schema=BankAccountState):
    """Bank account entity for testing transactional operations."""

    async def deposit(self, ctx: EntityContext, amount: float, description: str = "") -> dict:
        """Deposit money into account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        state = await ctx.get_state(BankAccountState)
        state.balance += amount
        state.transactions.append({
            "type": "deposit",
            "amount": amount,
            "description": description,
            "balance_after": state.balance
        })

        await ctx.set_state(state)

        return {
            "balance": state.balance,
            "transaction": state.transactions[-1]
        }

    async def withdraw(self, ctx: EntityContext, amount: float, description: str = "") -> dict:
        """Withdraw money from account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        state = await ctx.get_state(BankAccountState)

        if state.balance < amount:
            raise ValueError(f"Insufficient funds: balance={state.balance}, withdrawal={amount}")

        state.balance -= amount
        state.transactions.append({
            "type": "withdrawal",
            "amount": amount,
            "description": description,
            "balance_after": state.balance
        })

        await ctx.set_state(state)

        return {
            "balance": state.balance,
            "transaction": state.transactions[-1]
        }

    async def get_balance(self, ctx: EntityContext) -> float:
        """Get current balance."""
        state = await ctx.get_state(BankAccountState)
        return state.balance

    async def get_transactions(self, ctx: EntityContext) -> list[dict]:
        """Get transaction history."""
        state = await ctx.get_state(BankAccountState)
        return state.transactions


__all__ = [
    "Counter",
    "CounterState",
    "ShoppingCart",
    "ShoppingCartState",
    "BankAccount",
    "BankAccountState",
]
