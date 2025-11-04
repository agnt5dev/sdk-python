"""
Test Entities for Integration Testing

Provides stateful entities for testing state persistence and concurrency.
"""

from pydantic import BaseModel, Field
from agnt5 import Entity


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

class Counter(Entity):
    """Simple counter entity for testing basic state management."""

    _state_schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "Current count value"}
        }
    }

    async def increment(self, amount: int = 1) -> dict:
        """Increment the counter."""
        count = self.state.get("count", 0)
        count += amount
        self.state.set("count", count)
        return {"count": count}

    async def decrement(self, amount: int = 1) -> dict:
        """Decrement the counter."""
        count = self.state.get("count", 0)
        count -= amount
        self.state.set("count", count)
        return {"count": count}

    async def get_value(self) -> dict:
        """Get current counter value."""
        count = self.state.get("count", 0)
        return {"count": count}

    async def reset(self) -> dict:
        """Reset counter to zero."""
        self.state.set("count", 0)
        return {"count": 0}


class ShoppingCart(Entity):
    """Shopping cart entity for testing complex state operations."""

    _state_schema = {
        "type": "object",
        "properties": {
            "items": {"type": "object", "description": "Items in cart {item_id: {quantity, price}}"},
            "total_value": {"type": "number", "description": "Total cart value"}
        }
    }

    async def add_item(
        self,
        item_id: str,
        quantity: int,
        price: float
    ) -> dict:
        """Add item to cart."""
        items = self.state.get("items", {})

        if item_id in items:
            # Update existing item
            items[item_id]["quantity"] += quantity
        else:
            # Add new item
            items[item_id] = {
                "quantity": quantity,
                "price": price
            }

        self.state.set("items", items)

        # Calculate total
        total_value = sum(
            item["quantity"] * item["price"]
            for item in items.values()
        )
        self.state.set("total_value", total_value)

        return {
            "item_id": item_id,
            "quantity": items[item_id]["quantity"],
            "total_items": len(items),
            "total_value": total_value
        }

    async def remove_item(self, item_id: str) -> dict:
        """Remove item from cart."""
        items = self.state.get("items", {})

        if item_id not in items:
            raise ValueError(f"Item {item_id} not in cart")

        del items[item_id]
        self.state.set("items", items)

        # Recalculate total
        total_value = sum(
            item["quantity"] * item["price"]
            for item in items.values()
        )
        self.state.set("total_value", total_value)

        return {
            "item_id": item_id,
            "total_items": len(items),
            "total_value": total_value
        }

    async def get_total(self) -> float:
        """Get total cart value."""
        return self.state.get("total_value", 0.0)

    async def get_items(self) -> dict:
        """Get all items in cart."""
        return self.state.get("items", {})

    async def clear(self) -> dict:
        """Clear all items from cart."""
        self.state.set("items", {})
        self.state.set("total_value", 0.0)
        return {"total_items": 0, "total_value": 0.0}


class BankAccount(Entity):
    """Bank account entity for testing transactional operations."""

    _state_schema = {
        "type": "object",
        "properties": {
            "balance": {"type": "number", "description": "Current account balance"},
            "transactions": {"type": "array", "description": "Transaction history"}
        }
    }

    async def deposit(self, amount: float, description: str = "") -> dict:
        """Deposit money into account."""
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        balance = self.state.get("balance", 0.0)
        transactions = self.state.get("transactions", [])

        balance += amount
        transaction = {
            "type": "deposit",
            "amount": amount,
            "description": description,
            "balance_after": balance
        }
        transactions.append(transaction)

        self.state.set("balance", balance)
        self.state.set("transactions", transactions)

        return {
            "balance": balance,
            "transaction": transaction
        }

    async def withdraw(self, amount: float, description: str = "") -> dict:
        """Withdraw money from account."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        balance = self.state.get("balance", 0.0)
        transactions = self.state.get("transactions", [])

        if balance < amount:
            raise ValueError(f"Insufficient funds: balance={balance}, withdrawal={amount}")

        balance -= amount
        transaction = {
            "type": "withdrawal",
            "amount": amount,
            "description": description,
            "balance_after": balance
        }
        transactions.append(transaction)

        self.state.set("balance", balance)
        self.state.set("transactions", transactions)

        return {
            "balance": balance,
            "transaction": transaction
        }

    async def get_balance(self) -> float:
        """Get current balance."""
        return self.state.get("balance", 0.0)

    async def get_transactions(self) -> list[dict]:
        """Get transaction history."""
        return self.state.get("transactions", [])


__all__ = [
    "Counter",
    "CounterState",
    "ShoppingCart",
    "ShoppingCartState",
    "BankAccount",
    "BankAccountState",
]
