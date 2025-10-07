"""
Basic example of DurableEntity (Cloudflare Durable Objects style).

This demonstrates the class-based entity API where:
- State is accessed via self.ctx
- Methods are regular async methods
- Single-writer consistency per key is automatic
"""

import asyncio
from agnt5 import DurableEntity


class Counter(DurableEntity):
    """
    Simple counter with durable state.
    Each counter is identified by a unique key.
    """

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
    """
    Shopping cart with durable state.
    Each cart is identified by user_id.
    """

    async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
        """Add item to cart."""
        items = await self.ctx.get("items", {})

        if item_id in items:
            items[item_id]["quantity"] += quantity
        else:
            items[item_id] = {
                "quantity": quantity,
                "price": price
            }

        await self.ctx.set("items", items)
        total = await self._calculate_total()

        return {
            "item_id": item_id,
            "quantity": items[item_id]["quantity"],
            "cart_total": total
        }

    async def remove_item(self, item_id: str) -> dict:
        """Remove item from cart."""
        items = await self.ctx.get("items", {})
        removed = items.pop(item_id, None)

        await self.ctx.set("items", items)
        total = await self._calculate_total()

        return {
            "removed": removed is not None,
            "cart_total": total
        }

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

        return {
            "items": items,
            "total": total,
            "checked_out": True
        }

    async def _calculate_total(self) -> float:
        """Private helper to calculate cart total."""
        items = await self.ctx.get("items", {})
        return sum(
            item["quantity"] * item["price"]
            for item in items.values()
        )


async def test_counter():
    """Test counter entity."""
    print("\n=== Testing Counter Entity ===\n")

    # Create counter instances
    counter1 = Counter(key="user-123")
    counter2 = Counter(key="user-456")

    # Test counter1
    print("Counter 1 (user-123):")
    result = await counter1.increment(5)
    print(f"  increment(5) = {result}")

    result = await counter1.increment(3)
    print(f"  increment(3) = {result}")

    value = await counter1.get_value()
    print(f"  get_value() = {value}")

    # Test counter2 (independent state)
    print("\nCounter 2 (user-456):")
    result = await counter2.increment(10)
    print(f"  increment(10) = {result}")

    value = await counter2.get_value()
    print(f"  get_value() = {value}")

    # Verify counter1 unchanged
    print("\nCounter 1 (verify independent):")
    value = await counter1.get_value()
    print(f"  get_value() = {value}")

    # Reset
    result = await counter1.reset()
    print(f"  reset() = {result}")

    value = await counter1.get_value()
    print(f"  get_value() after reset = {value}")


async def test_shopping_cart():
    """Test shopping cart entity."""
    print("\n=== Testing ShoppingCart Entity ===\n")

    # Create cart for user
    cart = ShoppingCart(key="user-alice")

    # Add items
    print("Adding items:")
    result = await cart.add_item("item-123", quantity=2, price=29.99)
    print(f"  add_item('item-123', qty=2, price=29.99) = {result}")

    result = await cart.add_item("item-456", quantity=1, price=49.99)
    print(f"  add_item('item-456', qty=1, price=49.99) = {result}")

    result = await cart.add_item("item-123", quantity=1, price=29.99)
    print(f"  add_item('item-123', qty=1, price=29.99) = {result}")

    # Get cart state
    print("\nCart state:")
    items = await cart.get_items()
    print(f"  items = {items}")

    total = await cart.get_total()
    print(f"  total = ${total:.2f}")

    # Remove item
    print("\nRemoving item:")
    result = await cart.remove_item("item-456")
    print(f"  remove_item('item-456') = {result}")

    # Checkout
    print("\nCheckout:")
    result = await cart.checkout()
    print(f"  checkout() = {result}")

    # Verify cart empty
    print("\nAfter checkout:")
    items = await cart.get_items()
    print(f"  items = {items}")

    total = await cart.get_total()
    print(f"  total = ${total:.2f}")


async def main():
    """Run all tests."""
    await test_counter()
    await test_shopping_cart()
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
