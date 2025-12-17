"""
Example: AGNT5 Entities

This example demonstrates stateful entity features:
- Creating entities with typed state (Pydantic models)
- Direct state attribute access (self.state.count vs self.state.get/set)
- Automatic mutation detection (reads don't trigger saves)
- @query decorator for high-frequency read methods
- Single-writer consistency (no race conditions)
- State isolation between entity instances

Entities are stateful objects with:
- Guaranteed single-writer semantics
- Automatic state persistence
- Hash-based mutation detection
- Each entity instance identified by a unique key

Each entity can be:
- Run standalone: python ex_02_entities.py
- Registered with platform: imported by app.py
- Invoked via client: client.entity("Counter", "user-123").increment(5)
"""

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from agnt5 import Entity
from agnt5.entity import query


# =============================================================================
# STATE MODELS (Pydantic)
# =============================================================================


class CounterState(BaseModel):
    """State model for Counter entity."""
    count: int = 0


class ConversationState(BaseModel):
    """State model for ConversationMemory entity."""
    messages: list[dict] = Field(default_factory=list)


class KeyValueState(BaseModel):
    """State model for KeyValueStore entity."""
    data: dict[str, Any] = Field(default_factory=dict)


class BankAccountState(BaseModel):
    """State model for BankAccount entity."""
    owner_name: str = ""
    balance: float = 0.0
    status: str = "inactive"
    transactions: list[dict] = Field(default_factory=list)
    created_at: str = ""


# =============================================================================
# COUNTER ENTITY
# =============================================================================


class Counter(Entity[CounterState]):
    """
    Counter entity with increment/decrement operations.

    Uses typed state with automatic mutation detection:
    - Write methods (increment, decrement, reset) auto-save on mutation
    - Read methods (@query) skip persistence for better performance

    State:
        count: int - The current counter value

    Example:
        counter = client.entity("Counter", "user-123")
        counter.increment(5)  # Returns: 5
        counter.get_count()   # Returns: 5
    """

    async def increment(self, amount: int = 1) -> int:
        """Increment the counter by amount."""
        self.state.count += amount
        return self.state.count

    async def decrement(self, amount: int = 1) -> int:
        """Decrement the counter by amount."""
        self.state.count -= amount
        return self.state.count

    @query
    async def get_count(self) -> int:
        """Get the current count (read-only, no state save)."""
        return self.state.count

    async def reset(self) -> int:
        """Reset counter to zero."""
        self.state.count = 0
        return 0


# =============================================================================
# CONVERSATION MEMORY ENTITY
# =============================================================================


class ConversationMemory(Entity[ConversationState]):
    """
    Conversation memory entity for AI agents and chatbots.

    Uses typed state with automatic mutation detection:
    - Write methods (add_message, set_system_message, clear) auto-save on mutation
    - Read methods (@query) skip persistence for better performance

    State:
        messages: list - List of messages [{role, content, timestamp}]

    Example:
        memory = client.entity("ConversationMemory", "session-abc")
        memory.add_message("user", "Hello!")
        memory.add_message("assistant", "Hi there!")
        memory.get_messages()  # Returns conversation history
    """

    async def add_message(self, role: str, content: str, metadata: dict = None) -> dict:
        """
        Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (tokens, model, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            message["metadata"] = metadata

        self.state.messages.append(message)

        return {
            "action": "added",
            "role": role,
            "message_count": len(self.state.messages),
        }

    @query
    async def get_messages(self, limit: int = None, roles: list = None) -> list:
        """
        Get messages from the conversation (read-only, no state save).

        Args:
            limit: Maximum number of messages to return (most recent)
            roles: Filter by roles (e.g., ["user", "assistant"])
        """
        messages = list(self.state.messages)  # Copy to avoid mutation

        if roles:
            messages = [m for m in messages if m["role"] in roles]

        if limit:
            messages = messages[-limit:]

        return messages

    @query
    async def get_last_message(self, role: str = None) -> dict:
        """Get the last message, optionally filtered by role (read-only)."""
        messages = list(self.state.messages)

        if role:
            messages = [m for m in messages if m["role"] == role]

        return messages[-1] if messages else None

    @query
    async def get_context_window(self, max_messages: int = 10) -> list:
        """
        Get recent messages for context window (read-only, no state save).

        Useful for passing to LLM as conversation history.
        Returns messages in the format expected by most LLM APIs.
        """
        messages = list(self.state.messages)
        recent = messages[-max_messages:] if max_messages else messages

        # Return in standard LLM format
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    @query
    async def summarize(self) -> dict:
        """Get conversation summary/statistics (read-only, no state save)."""
        messages = self.state.messages

        role_counts = {}
        for m in messages:
            role = m["role"]
            role_counts[role] = role_counts.get(role, 0) + 1

        total_chars = sum(len(m["content"]) for m in messages)

        return {
            "message_count": len(messages),
            "role_counts": role_counts,
            "total_characters": total_chars,
            "has_system_message": any(m["role"] == "system" for m in messages),
        }

    async def set_system_message(self, content: str) -> dict:
        """Set or update the system message (always first in conversation)."""
        # Remove existing system message if any
        self.state.messages = [m for m in self.state.messages if m["role"] != "system"]

        # Insert system message at the beginning
        system_msg = {
            "role": "system",
            "content": content,
            "timestamp": time.time(),
        }
        self.state.messages.insert(0, system_msg)

        return {"action": "system_message_set"}

    async def clear(self) -> dict:
        """Clear all messages from the conversation."""
        self.state.messages = []
        return {"action": "cleared"}


# =============================================================================
# KEY-VALUE STORE ENTITY
# =============================================================================


class KeyValueStore(Entity[KeyValueState]):
    """
    Simple key-value store entity.

    Uses typed state with automatic mutation detection:
    - Write methods (set_value, delete_key, clear) auto-save on mutation
    - Read methods (@query) skip persistence for better performance

    State:
        data: dict - The key-value pairs

    Example:
        store = client.entity("KeyValueStore", "config")
        store.set_value(item_key="theme", value="dark")
        store.get_value(item_key="theme")  # Returns: "dark"

    Note:
        Methods use 'item_key' instead of 'key' because 'key' is reserved
        for entity instance identification.
    """

    async def set_value(self, item_key: str, value: any) -> dict:
        """Set a key-value pair."""
        self.state.data[item_key] = value
        return {"key": item_key, "action": "set"}

    @query
    async def get_value(self, item_key: str, default: any = None) -> any:
        """Get a value by key (read-only, no state save)."""
        return self.state.data.get(item_key, default)

    async def delete_key(self, item_key: str) -> dict:
        """Delete a key."""
        if item_key in self.state.data:
            del self.state.data[item_key]
            return {"key": item_key, "action": "deleted"}
        return {"key": item_key, "action": "not_found"}

    @query
    async def list_keys(self) -> list:
        """Get all keys (read-only, no state save)."""
        return list(self.state.data.keys())

    async def clear(self) -> dict:
        """Clear all data."""
        self.state.data = {}
        return {"action": "cleared"}


# =============================================================================
# BANK ACCOUNT ENTITY (Advanced Example)
# =============================================================================


class BankAccount(Entity[BankAccountState]):
    """
    Bank account entity with transaction management.

    Demonstrates complex entity patterns:
    - Business logic validation (insufficient funds, account status)
    - Transaction history tracking
    - Account lifecycle management (activate, suspend)

    State:
        owner_name: str - Account owner's name
        balance: float - Current balance
        status: str - Account status (active, suspended, inactive)
        transactions: list - Transaction history
        created_at: str - Creation timestamp

    Example:
        account = client.entity("BankAccount", "alice-001")
        account.create_account(owner_name="Alice", initial_balance=1000.0)
        account.deposit(amount=500.0, description="Salary")
        account.withdraw(amount=200.0, description="ATM")
        account.get_balance()  # Returns: 1300.0
    """

    async def create_account(self, owner_name: str, initial_balance: float = 0.0) -> dict:
        """Initialize a new account."""
        from datetime import datetime

        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")

        self.state.owner_name = owner_name
        self.state.balance = initial_balance
        self.state.created_at = datetime.now().isoformat()
        self.state.transactions = []
        self.state.status = "active"

        return {
            "account_id": self.key,
            "owner_name": owner_name,
            "balance": initial_balance,
            "status": "active"
        }

    async def deposit(self, amount: float, description: str = "Deposit") -> dict:
        """Deposit money into the account."""
        from datetime import datetime

        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        if self.state.status != "active":
            raise ValueError(f"Cannot deposit to {self.state.status} account")

        self.state.balance += amount

        transaction = {
            "type": "deposit",
            "amount": amount,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "balance_after": self.state.balance
        }
        self.state.transactions.append(transaction)

        return {
            "success": True,
            "transaction": transaction,
            "new_balance": self.state.balance
        }

    async def withdraw(self, amount: float, description: str = "Withdrawal") -> dict:
        """Withdraw money from the account."""
        from datetime import datetime

        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        if self.state.status != "active":
            raise ValueError(f"Cannot withdraw from {self.state.status} account")

        if self.state.balance < amount:
            raise ValueError(f"Insufficient funds: ${self.state.balance} < ${amount}")

        self.state.balance -= amount

        transaction = {
            "type": "withdrawal",
            "amount": amount,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "balance_after": self.state.balance
        }
        self.state.transactions.append(transaction)

        return {
            "success": True,
            "transaction": transaction,
            "new_balance": self.state.balance
        }

    @query
    async def get_balance(self) -> float:
        """Get current account balance (read-only)."""
        return self.state.balance

    @query
    async def get_transaction_history(self, limit: int = None) -> list:
        """Get transaction history (read-only)."""
        transactions = list(self.state.transactions)
        if limit:
            return transactions[-limit:]
        return transactions

    @query
    async def get_account_info(self) -> dict:
        """Get complete account information (read-only)."""
        return {
            "account_id": self.key,
            "owner_name": self.state.owner_name,
            "balance": self.state.balance,
            "status": self.state.status,
            "created_at": self.state.created_at,
            "transaction_count": len(self.state.transactions)
        }

    async def suspend_account(self, reason: str) -> dict:
        """Suspend the account."""
        from datetime import datetime

        self.state.status = "suspended"
        return {
            "success": True,
            "status": "suspended",
            "reason": reason,
            "suspended_at": datetime.now().isoformat()
        }

    async def activate_account(self) -> dict:
        """Reactivate a suspended account."""
        if self.state.status == "active":
            return {"success": True, "message": "Account already active"}

        self.state.status = "active"
        return {"success": True, "status": "active"}


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    from agnt5.entity import with_entity_context

    @with_entity_context
    async def run_examples():
        print("=" * 60)
        print("AGNT5 Entities Example (Typed State API)")
        print("=" * 60)

        # Counter example
        print("\n--- Counter Entity (Entity[CounterState]) ---")
        counter = Counter(key="demo-counter")
        print(f"Initial count: {await counter.get_count()}")
        print(f"After increment(5): {await counter.increment(5)}")
        print(f"After increment(3): {await counter.increment(3)}")
        print(f"After decrement(2): {await counter.decrement(2)}")
        print(f"Final count: {await counter.get_count()}")

        # Conversation memory example
        print("\n--- ConversationMemory Entity (Entity[ConversationState]) ---")
        memory = ConversationMemory(key="demo-chat")
        await memory.set_system_message("You are a helpful assistant.")
        await memory.add_message("user", "Hello! How are you?")
        await memory.add_message("assistant", "I'm doing well, thanks for asking!")
        await memory.add_message("user", "Can you help me with Python?")
        print(f"Messages: {await memory.get_messages()}")
        print(f"Summary: {await memory.summarize()}")
        print(f"Context window: {await memory.get_context_window(max_messages=3)}")

        # Key-value store example
        print("\n--- KeyValueStore Entity (Entity[KeyValueState]) ---")
        store = KeyValueStore(key="demo-config")
        await store.set_value("theme", "dark")
        await store.set_value("language", "en")
        print(f"Theme: {await store.get_value('theme')}")
        print(f"Keys: {await store.list_keys()}")

        print("\n" + "=" * 60)
        print("Done!")

    await run_examples()


if __name__ == "__main__":
    asyncio.run(main())
