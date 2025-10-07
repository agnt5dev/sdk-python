"""
End-to-end test for DurableEntity: Client -> Gateway -> Worker

This test demonstrates the complete flow:
1. Worker registers Counter and ShoppingCart entities
2. Client calls entity methods via HTTP
3. Gateway routes to Worker Coordinator
4. Worker Coordinator dispatches to Worker
5. Worker executes entity methods with state isolation
6. Results flow back through the stack

Run this test with:
    python examples/18_entity_e2e_test.py
"""

import asyncio
import time

from agnt5 import Client, DurableEntity, MemoryEntity, WorkflowEntity


# Define entities (same as 17_durable_entity_basic.py)
class Counter(DurableEntity):
    """Simple counter with durable state."""

    async def increment(self, amount: int = 1) -> int:
        """Increment counter by amount."""
        count = await self.ctx.get("count", 0)
        count += amount
        await self.ctx.set("count", count)
        return count

    async def decrement(self, amount: int = 1) -> int:
        """Decrement counter by amount."""
        count = await self.ctx.get("count", 0)
        count -= amount
        await self.ctx.set("count", count)
        return count

    async def get_value(self) -> int:
        """Get current counter value."""
        return await self.ctx.get("count", 0)

    async def reset(self) -> dict:
        """Reset counter to zero."""
        await self.ctx.delete("count")
        return {"reset": True, "value": 0}


class ShoppingCart(DurableEntity):
    """Shopping cart with durable state."""

    async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
        """Add item to cart."""
        items = await self.ctx.get("items", {})

        if item_id in items:
            items[item_id]["quantity"] += quantity
        else:
            items[item_id] = {"quantity": quantity, "price": price}

        await self.ctx.set("items", items)
        total = await self._calculate_total()

        return {
            "item_id": item_id,
            "quantity": items[item_id]["quantity"],
            "cart_total": total,
        }

    async def remove_item(self, item_id: str) -> dict:
        """Remove item from cart."""
        items = await self.ctx.get("items", {})
        removed = items.pop(item_id, None)

        await self.ctx.set("items", items)
        total = await self._calculate_total()

        return {"removed": removed is not None, "cart_total": total}

    async def get_total(self) -> float:
        """Get cart total."""
        return await self._calculate_total()

    async def get_items(self) -> dict:
        """Get all items in cart."""
        return await self.ctx.get("items", {})

    async def checkout(self) -> dict:
        """Checkout and clear cart."""
        items = await self.ctx.get("items", {})
        total = await self._calculate_total()

        # Clear cart
        await self.ctx.clear_all()

        return {"items": items, "total": total, "checked_out": True}

    async def _calculate_total(self) -> float:
        """Private helper to calculate cart total."""
        items = await self.ctx.get("items", {})
        return sum(item["quantity"] * item["price"] for item in items.values())


class KnowledgeBase(MemoryEntity):
    """Knowledge base with semantic search capabilities."""
    max_memories: int = 50

    async def learn(self, topic: str, fact: str) -> dict:
        """Learn a new fact about a topic."""
        return await self.store(key=topic, content=fact, category="fact")

    async def search(self, query: str) -> list:
        """Search the knowledge base."""
        results = await self.recall(query, limit=3)
        return [
            {
                "topic": r["key"],
                "fact": r["content"],
                "score": r["score"]
            }
            for r in results
        ]


class OrderWorkflow(WorkflowEntity):
    """Durable order processing workflow with compensation."""
    max_retries: int = 2

    async def process_order(self, order_id: str, amount: float) -> dict:
        """
        Process an order through multiple steps.

        Steps:
        1. Validate order
        2. Charge payment
        3. Fulfill order
        """
        status = await self.get_status()

        # Step 1: Validate order
        if "validate" not in status["completed_steps"]:
            validation_result = {
                "order_id": order_id,
                "valid": True,
                "amount": amount
            }
            await self.mark_step_complete("validate", validation_result)

        # Step 2: Charge payment
        if "payment" not in status["completed_steps"]:
            payment_result = {
                "charged": True,
                "amount": amount,
                "transaction_id": f"txn_{order_id}"
            }
            await self.mark_step_complete("payment", payment_result)

        # Step 3: Fulfill order
        if "fulfill" not in status["completed_steps"]:
            fulfillment_result = {
                "shipped": True,
                "tracking_number": f"TRACK_{order_id}"
            }
            await self.mark_step_complete("fulfill", fulfillment_result)

        # Complete the workflow
        return await self.complete_workflow()


async def run_client_tests():
    """Run client-side entity tests (requires running dev server)."""
    print("\n=== End-to-End Entity Test ===\n")
    print("Prerequisites:")
    print("  - Dev server running on localhost:34181")
    print("  - Worker registered with Counter and ShoppingCart entities")
    print()

    # Wait a moment for worker to register
    print("Waiting 2 seconds for worker registration...")
    time.sleep(2)

    # Create client
    client = Client("http://localhost:34181")

    # Test Counter entity
    print("--- Testing Counter Entity ---\n")

    counter1 = client.entity("Counter", "user-123")
    counter2 = client.entity("Counter", "user-456")

    print("Counter 1 (user-123):")
    result = counter1.increment(amount=5)
    print(f"  increment(5) = {result}")

    result = counter1.increment(amount=3)
    print(f"  increment(3) = {result}")

    value = counter1.get_value()
    print(f"  get_value() = {value}")

    print("\nCounter 2 (user-456) - independent state:")
    result = counter2.increment(amount=10)
    print(f"  increment(10) = {result}")

    value = counter2.get_value()
    print(f"  get_value() = {value}")

    print("\nCounter 1 - verify still independent:")
    value = counter1.get_value()
    print(f"  get_value() = {value}")

    result = counter1.reset()
    print(f"  reset() = {result}")

    value = counter1.get_value()
    print(f"  get_value() after reset = {value}")

    # Test ShoppingCart entity
    print("\n--- Testing ShoppingCart Entity ---\n")

    cart = client.entity("ShoppingCart", "user-alice")

    print("Adding items:")
    result = cart.add_item(item_id="item-123", quantity=2, price=29.99)
    print(f"  add_item('item-123', qty=2, price=29.99) = {result}")

    result = cart.add_item(item_id="item-456", quantity=1, price=49.99)
    print(f"  add_item('item-456', qty=1, price=49.99) = {result}")

    result = cart.add_item(item_id="item-123", quantity=1, price=29.99)
    print(f"  add_item('item-123', qty=1, price=29.99) = {result}")

    print("\nCart state:")
    items = cart.get_items()
    print(f"  items = {items}")

    total = cart.get_total()
    print(f"  total = ${total:.2f}")

    print("\nRemoving item:")
    result = cart.remove_item(item_id="item-456")
    print(f"  remove_item('item-456') = {result}")

    print("\nCheckout:")
    result = cart.checkout()
    print(f"  checkout() = {result}")

    print("\nAfter checkout:")
    items = cart.get_items()
    print(f"  items = {items}")

    total = cart.get_total()
    print(f"  total = ${total:.2f}")

    print("\n✅ All client tests completed!")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("DurableEntity End-to-End Test")
    print("=" * 60)

    # Check if this script is being run as a worker or client
    import sys

    if "--client" in sys.argv:
        # Run client tests
        await run_client_tests()
    else:
        # Run as worker (default)
        print("\nRunning as Worker...")
        print("Entities registered: Counter, ShoppingCart")
        print("\nWaiting for client requests...")
        print("(Run with --client flag to test from client side)")
        print()

        # Worker runs continuously - entities are auto-registered
        # The Worker class will handle incoming requests
        # Note: service_name should match what Gateway expects (from deployment config)
        # In dev mode, this is typically the project/service name
        from agnt5 import Worker

        worker = Worker(service_name="default-service")

        print(f"Worker will register entities:")
        from agnt5.entity import EntityRegistry
        for entity_name in EntityRegistry.all().keys():
            print(f"  - {entity_name}")

        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
