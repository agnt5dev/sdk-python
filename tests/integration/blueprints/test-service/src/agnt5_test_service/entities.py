"""
Test Entities for Integration Testing

Provides stateful entities for testing state persistence and concurrency.
"""

import asyncio
from agnt5 import Entity


class ShoppingCart(Entity):
    """
    Shopping cart entity for testing state persistence.

    Tests:
    - State persists across worker restarts
    - State isolation between entity keys
    - Concurrent updates don't cause lost writes
    """

    # Define state schema for platform registration and introspection
    _state_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "object",
                "description": "Map of item_id to item details",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "quantity": {"type": "integer", "description": "Number of items"},
                        "price": {"type": "number", "description": "Price per item"}
                    },
                    "required": ["quantity", "price"]
                }
            }
        },
        "description": "Shopping cart state with items and their quantities/prices"
    }

    async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
        """Add item to cart."""
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

    async def get_items(self) -> dict:
        """Get all items in cart."""
        return self.state.get("items", {})

    async def clear(self) -> dict:
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

    # Define state schema for platform registration and introspection
    _state_schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Current counter value",
                "default": 0
            }
        },
        "description": "Counter state with single integer value"
    }

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

    async def reset(self) -> dict:
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

    # Define state schema for platform registration and introspection
    _state_schema = {
        "type": "object",
        "properties": {
            "balance": {
                "type": "number",
                "description": "Current account balance",
                "default": 0.0
            },
            "transactions": {
                "type": "array",
                "description": "Transaction history",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["deposit", "withdraw"],
                            "description": "Transaction type"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Transaction amount"
                        }
                    },
                    "required": ["type", "amount"]
                },
                "default": []
            }
        },
        "description": "Bank account state with balance and transaction history"
    }

    async def deposit(self, amount: float) -> dict:
        """Deposit money into account."""
        balance = self.state.get("balance", 0.0)
        transactions = self.state.get("transactions", [])

        new_balance = balance + amount
        transactions.append({"type": "deposit", "amount": amount})

        self.state.set("balance", new_balance)
        self.state.set("transactions", transactions)

        return {"balance": new_balance}

    async def withdraw(self, amount: float) -> dict:
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


__all__ = ["ShoppingCart", "Counter", "BankAccount"]
