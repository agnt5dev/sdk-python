"""
Example: User Account Entity with Business Logic

This example demonstrates:
- Complex entity with business logic
- Error handling in entity methods
- Multiple methods operating on same state
- Real-world use case (bank account)
"""

import asyncio
from datetime import datetime
from typing import Dict, List

from agnt5 import Context, entity

# Create Account entity type
Account = entity("Account")


@Account.method
async def create_account(ctx: Context, owner_name: str, initial_balance: float = 0.0) -> Dict:
    """Initialize a new account."""
    if initial_balance < 0:
        raise ValueError("Initial balance cannot be negative")

    ctx.set("owner_name", owner_name)
    ctx.set("balance", initial_balance)
    ctx.set("created_at", datetime.now().isoformat())
    ctx.set("transactions", [])
    ctx.set("status", "active")

    ctx.logger.info(f"Account created for {owner_name} with balance ${initial_balance}")

    return {
        "account_id": ctx.object_id,
        "owner_name": owner_name,
        "balance": initial_balance,
        "status": "active"
    }


@Account.method
async def deposit(ctx: Context, amount: float, description: str = "Deposit") -> Dict:
    """Deposit money into the account."""
    if amount <= 0:
        raise ValueError("Deposit amount must be positive")

    # Check account status
    status = ctx.get("status", "inactive")
    if status != "active":
        raise ValueError(f"Cannot deposit to {status} account")

    # Update balance
    current_balance = ctx.get("balance", 0.0)
    new_balance = current_balance + amount
    ctx.set("balance", new_balance)

    # Record transaction
    transaction = {
        "type": "deposit",
        "amount": amount,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "balance_after": new_balance
    }

    transactions = ctx.get("transactions", [])
    transactions.append(transaction)
    ctx.set("transactions", transactions)

    ctx.logger.info(f"Deposited ${amount}: {description}")

    return {
        "success": True,
        "transaction": transaction,
        "new_balance": new_balance
    }


@Account.method
async def withdraw(ctx: Context, amount: float, description: str = "Withdrawal") -> Dict:
    """Withdraw money from the account."""
    if amount <= 0:
        raise ValueError("Withdrawal amount must be positive")

    # Check account status
    status = ctx.get("status", "inactive")
    if status != "active":
        raise ValueError(f"Cannot withdraw from {status} account")

    # Check sufficient funds
    current_balance = ctx.get("balance", 0.0)
    if current_balance < amount:
        raise ValueError(f"Insufficient funds: ${current_balance} < ${amount}")

    # Update balance
    new_balance = current_balance - amount
    ctx.set("balance", new_balance)

    # Record transaction
    transaction = {
        "type": "withdrawal",
        "amount": amount,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "balance_after": new_balance
    }

    transactions = ctx.get("transactions", [])
    transactions.append(transaction)
    ctx.set("transactions", transactions)

    ctx.logger.info(f"Withdrew ${amount}: {description}")

    return {
        "success": True,
        "transaction": transaction,
        "new_balance": new_balance
    }


@Account.method
async def transfer(ctx: Context, to_account_key: str, amount: float) -> Dict:
    """Transfer money to another account (simplified - just withdraws from this account)."""
    if amount <= 0:
        raise ValueError("Transfer amount must be positive")

    # In a real system, this would coordinate with the target account
    # For this example, we just withdraw from source account
    result = await withdraw(ctx, amount, description=f"Transfer to {to_account_key}")

    ctx.logger.info(f"Transferred ${amount} to account {to_account_key}")

    return {
        "success": True,
        "from_account": ctx.object_id,
        "to_account": to_account_key,
        "amount": amount,
        "new_balance": result["new_balance"]
    }


@Account.method
async def get_balance(ctx: Context) -> float:
    """Get current account balance."""
    return ctx.get("balance", 0.0)


@Account.method
async def get_transaction_history(ctx: Context, limit: int = None) -> List[Dict]:
    """Get transaction history."""
    transactions = ctx.get("transactions", [])

    if limit:
        return transactions[-limit:]
    return transactions


@Account.method
async def get_account_info(ctx: Context) -> Dict:
    """Get complete account information."""
    return {
        "account_id": ctx.object_id,
        "owner_name": ctx.get("owner_name"),
        "balance": ctx.get("balance", 0.0),
        "status": ctx.get("status", "inactive"),
        "created_at": ctx.get("created_at"),
        "transaction_count": len(ctx.get("transactions", []))
    }


@Account.method
async def suspend_account(ctx: Context, reason: str) -> Dict:
    """Suspend the account."""
    ctx.set("status", "suspended")
    ctx.set("suspended_reason", reason)
    ctx.set("suspended_at", datetime.now().isoformat())

    ctx.logger.warning(f"Account suspended: {reason}")

    return {
        "success": True,
        "status": "suspended",
        "reason": reason
    }


@Account.method
async def activate_account(ctx: Context) -> Dict:
    """Reactivate a suspended account."""
    current_status = ctx.get("status")

    if current_status == "active":
        return {"success": True, "message": "Account already active"}

    ctx.set("status", "active")
    ctx.delete("suspended_reason")
    ctx.delete("suspended_at")

    ctx.logger.info("Account reactivated")

    return {
        "success": True,
        "status": "active"
    }


async def main():
    print("=== User Account Entity Example ===\n")

    # Create two accounts
    alice_account = Account(key="alice-account-001")
    bob_account = Account(key="bob-account-002")

    # Initialize accounts
    print("1. Creating accounts...")
    await alice_account.create_account(owner_name="Alice", initial_balance=1000.0)
    await bob_account.create_account(owner_name="Bob", initial_balance=500.0)

    alice_info = await alice_account.get_account_info()
    bob_info = await bob_account.get_account_info()

    print(f"   Alice's account: ${alice_info['balance']}")
    print(f"   Bob's account: ${bob_info['balance']}\n")

    # Perform transactions
    print("2. Performing transactions...")

    print("   Alice deposits $500...")
    await alice_account.deposit(amount=500.0, description="Salary payment")
    balance = await alice_account.get_balance()
    print(f"   New balance: ${balance}\n")

    print("   Bob withdraws $200...")
    await bob_account.withdraw(amount=200.0, description="Cash withdrawal")
    balance = await bob_account.get_balance()
    print(f"   New balance: ${balance}\n")

    print("   Alice transfers $300 to another account...")
    await alice_account.transfer(to_account_key="charity-account", amount=300.0)
    balance = await alice_account.get_balance()
    print(f"   New balance: ${balance}\n")

    # Get transaction history
    print("3. Alice's Transaction History:")
    history = await alice_account.get_transaction_history()
    for i, txn in enumerate(history, 1):
        print(f"   {i}. {txn['type'].capitalize()}: ${txn['amount']} - {txn['description']}")
        print(f"      Balance after: ${txn['balance_after']}")

    print()

    # Test error handling
    print("4. Testing error handling...")

    print("   Trying to withdraw more than balance...")
    try:
        await bob_account.withdraw(amount=1000.0)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}\n")

    print("   Trying to deposit negative amount...")
    try:
        await alice_account.deposit(amount=-100.0)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}\n")

    # Test account suspension
    print("5. Testing account suspension...")
    print("   Suspending Bob's account...")
    await bob_account.suspend_account(reason="Suspicious activity")

    print("   Trying to withdraw from suspended account...")
    try:
        await bob_account.withdraw(amount=50.0)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}\n")

    print("   Reactivating account...")
    await bob_account.activate_account()

    print("   Withdrawing after reactivation...")
    result = await bob_account.withdraw(amount=50.0)
    print(f"   ✓ Withdrawal successful! New balance: ${result['new_balance']}\n")

    # Final account info
    print("6. Final Account Status:")
    alice_info = await alice_account.get_account_info()
    bob_info = await bob_account.get_account_info()

    print(f"   Alice: ${alice_info['balance']} ({alice_info['transaction_count']} transactions)")
    print(f"   Bob: ${bob_info['balance']} ({bob_info['transaction_count']} transactions)\n")


if __name__ == "__main__":
    asyncio.run(main())
