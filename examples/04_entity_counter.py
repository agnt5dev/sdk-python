"""
Example: Basic Entity Usage - Counter

This example demonstrates:
- Creating an entity by inheriting from Entity
- Synchronous state operations (self.state.get/set)
- Single-writer consistency (no race conditions)
- State isolation between entity instances
"""

import asyncio

from agnt5 import Entity


class Counter(Entity):
    """Counter entity with increment/decrement operations."""

    async def increment(self, amount: int = 1) -> int:
        """Increment the counter by amount."""
        current = self.state.get("count", 0)
        new_count = current + amount
        self.state.set("count", new_count)
        return new_count

    async def decrement(self, amount: int = 1) -> int:
        """Decrement the counter by amount."""
        current = self.state.get("count", 0)
        new_count = current - amount
        self.state.set("count", new_count)
        return new_count

    async def get_count(self) -> int:
        """Get the current count."""
        return self.state.get("count", 0)

    async def reset(self) -> int:
        """Reset counter to zero."""
        self.state.set("count", 0)
        return 0


async def main():
    print("=== Basic Entity Example: Counter ===\n")

    # Create counter instances
    counter1 = Counter(key="user-123")
    counter2 = Counter(key="user-456")

    print("1. Incrementing counter1...")
    result = await counter1.increment()
    print(f"   Counter1: {result}\n")

    result = await counter1.increment(amount=5)
    print(f"   Counter1: {result}\n")

    result = await counter1.increment(amount=3)
    print(f"   Counter1: {result}\n")

    print("2. Incrementing counter2 (isolated state)...")
    result = await counter2.increment(amount=10)
    print(f"   Counter2: {result}\n")

    print("3. Getting current counts...")
    count1 = await counter1.get_count()
    count2 = await counter2.get_count()
    print(f"   Counter1: {count1}")
    print(f"   Counter2: {count2}\n")

    print("4. Testing single-writer consistency...")
    print("   Running 100 concurrent increments on same counter...")

    # This demonstrates single-writer guarantee
    # All 100 increments will execute serially, preventing lost updates
    await counter1.reset()
    await asyncio.gather(*[counter1.increment() for _ in range(100)])

    final_count = await counter1.get_count()
    print(f"   Final count: {final_count} (should be 100, no lost updates!)\n")

    print("5. Parallel execution with different keys...")
    print("   Running increments on different counters in parallel...")

    counter3 = Counter(key="parallel-1")
    counter4 = Counter(key="parallel-2")

    # Different keys execute in parallel
    start = asyncio.get_event_loop().time()
    await asyncio.gather(
        *[counter3.increment() for _ in range(50)],
        *[counter4.increment() for _ in range(50)]
    )
    elapsed = asyncio.get_event_loop().time() - start

    count3 = await counter3.get_count()
    count4 = await counter4.get_count()

    print(f"   Counter3: {count3}")
    print(f"   Counter4: {count4}")
    print(f"   Executed in parallel! (Time: {elapsed:.3f}s)\n")


if __name__ == "__main__":
    asyncio.run(main())
