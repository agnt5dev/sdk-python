"""
Entity fixtures for integration tests.

This module provides reusable entity backends for testing various scenarios:
- Happy path: Basic entity operations
- Error scenarios: Entities that test failure modes
- Edge cases: Boundary conditions and corner cases
- Concurrency: Entities for testing concurrent access

These fixtures are separate from pedagogical examples, allowing tests to cover
complex scenarios including state corruption, validation failures, and race conditions.
"""

import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from agnt5 import Entity
from agnt5.entity import query


# =============================================================================
# STATE MODELS
# =============================================================================


class CounterState(BaseModel):
    """State for simple counter."""
    count: int = 0


class ComplexState(BaseModel):
    """Complex nested state for testing."""
    metadata: dict = Field(default_factory=dict)
    items: list = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    counter: int = 0


class LargeState(BaseModel):
    """Large state for size/performance testing."""
    data: str = ""
    chunks: list[str] = Field(default_factory=list)


class ValidationState(BaseModel):
    """State with validation rules."""
    value: int = 0
    status: str = "pending"


class ConcurrentState(BaseModel):
    """State for concurrency testing."""
    counter: int = 0
    last_modified_by: str = ""
    version: int = 0


# =============================================================================
# HAPPY PATH ENTITIES
# =============================================================================


class CounterEntity(Entity[CounterState]):
    """
    Simple counter entity for basic tests.

    Methods:
        increment(amount): Increase counter
        decrement(amount): Decrease counter
        get_count(): Get current count (read-only)
        reset(): Reset to zero
    """

    async def increment(self, amount: int = 1) -> int:
        """Increment the counter."""
        self.state.count += amount
        return self.state.count

    async def decrement(self, amount: int = 1) -> int:
        """Decrement the counter."""
        self.state.count -= amount
        return self.state.count

    @query
    async def get_count(self) -> int:
        """Get current count (read-only)."""
        return self.state.count

    async def reset(self) -> int:
        """Reset counter to zero."""
        self.state.count = 0
        return 0

    async def set_count(self, value: int) -> int:
        """Set counter to specific value."""
        self.state.count = value
        return self.state.count


class ComplexStateEntity(Entity[ComplexState]):
    """
    Entity with complex nested state for testing serialization and mutations.

    Methods:
        add_item(item): Add item to list
        set_metadata(key, value): Set metadata
        set_config(key, value): Set config
        increment_counter(): Increment counter
        get_state_summary(): Get state summary
    """

    async def add_item(self, item: Any) -> dict:
        """Add item to items list."""
        self.state.items.append(item)
        return {
            "item": item,
            "total_items": len(self.state.items),
        }

    async def set_metadata(self, key: str, value: Any) -> dict:
        """Set metadata field."""
        self.state.metadata[key] = value
        return {
            "key": key,
            "value": value,
        }

    async def set_config(self, key: str, value: Any) -> dict:
        """Set config field."""
        self.state.config[key] = value
        return {
            "key": key,
            "value": value,
        }

    async def increment_counter(self) -> int:
        """Increment internal counter."""
        self.state.counter += 1
        return self.state.counter

    @query
    async def get_state_summary(self) -> dict:
        """Get summary of state (read-only)."""
        return {
            "item_count": len(self.state.items),
            "metadata_keys": len(self.state.metadata),
            "config_keys": len(self.state.config),
            "counter": self.state.counter,
        }

    async def clear_all(self) -> dict:
        """Clear all state."""
        self.state.metadata = {}
        self.state.items = []
        self.state.config = {}
        self.state.counter = 0
        return {"cleared": True}


class LargeStateEntity(Entity[LargeState]):
    """
    Entity with large state for size/performance testing.

    Methods:
        set_data(size_kb): Set large data field
        add_chunk(size_kb): Add chunk to list
        get_size_info(): Get size information
        clear(): Clear all data
    """

    async def set_data(self, size_kb: int = 100) -> dict:
        """Set large data field."""
        self.state.data = "x" * (size_kb * 1024)
        return {
            "size_kb": size_kb,
            "length": len(self.state.data),
        }

    async def add_chunk(self, size_kb: int = 10) -> dict:
        """Add chunk to chunks list."""
        chunk = "y" * (size_kb * 1024)
        self.state.chunks.append(chunk)
        return {
            "chunk_size_kb": size_kb,
            "total_chunks": len(self.state.chunks),
        }

    @query
    async def get_size_info(self) -> dict:
        """Get size information (read-only)."""
        total_size = len(self.state.data) + sum(len(c) for c in self.state.chunks)
        return {
            "data_size_bytes": len(self.state.data),
            "chunk_count": len(self.state.chunks),
            "total_size_bytes": total_size,
            "total_size_kb": total_size // 1024,
        }

    async def clear(self) -> dict:
        """Clear all data."""
        self.state.data = ""
        self.state.chunks = []
        return {"cleared": True}


# =============================================================================
# ERROR SCENARIO ENTITIES
# =============================================================================


class ValidationEntity(Entity[ValidationState]):
    """
    Entity with strict validation for testing error handling.

    Methods:
        set_value(value): Set value (must be 0-1000)
        set_status(status): Set status (must be valid status)
        get_value(): Get value
        get_status(): Get status
    """

    async def set_value(self, value: int) -> dict:
        """Set value with validation."""
        if value < 0:
            raise ValueError(f"Value must be non-negative, got {value}")

        if value > 1000:
            raise ValueError(f"Value exceeds maximum (1000), got {value}")

        self.state.value = value
        return {
            "value": value,
            "valid": True,
        }

    async def set_status(self, status: str) -> dict:
        """Set status with validation."""
        valid_statuses = ["pending", "active", "completed", "failed"]

        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {valid_statuses}")

        self.state.status = status
        return {
            "status": status,
            "valid": True,
        }

    @query
    async def get_value(self) -> int:
        """Get value (read-only)."""
        return self.state.value

    @query
    async def get_status(self) -> str:
        """Get status (read-only)."""
        return self.state.status


class CorruptibleEntity(Entity[CounterState]):
    """
    Entity that can be corrupted for testing state corruption recovery.

    Methods:
        increment(): Normal increment
        corrupt_state(): Intentionally corrupt state
        get_count(): Get count
    """

    async def increment(self, amount: int = 1) -> int:
        """Normal increment."""
        self.state.count += amount
        return self.state.count

    async def corrupt_state(self) -> dict:
        """Intentionally corrupt state for testing."""
        # This would require direct DB access in real tests
        # For now, just set to invalid value
        self.state.count = -999999
        return {
            "corrupted": True,
            "message": "State intentionally corrupted for testing",
        }

    @query
    async def get_count(self) -> int:
        """Get count (read-only)."""
        return self.state.count


class ConcurrentEntity(Entity[ConcurrentState]):
    """
    Entity for testing concurrent access patterns.

    Methods:
        increment(worker_id): Increment with worker tracking
        slow_increment(worker_id, delay): Slow increment for race condition testing
        get_count(): Get count
        get_last_modified_by(): Get last modifier
        get_version(): Get version
    """

    async def increment(self, worker_id: str = "unknown") -> dict:
        """Increment counter with worker tracking."""
        self.state.counter += 1
        self.state.last_modified_by = worker_id
        self.state.version += 1
        return {
            "counter": self.state.counter,
            "modified_by": worker_id,
            "version": self.state.version,
        }

    async def slow_increment(self, worker_id: str = "unknown", delay: float = 0.5) -> dict:
        """Slow increment to induce race conditions."""
        import asyncio

        # Read current value
        current = self.state.counter

        # Artificial delay (simulates slow operation)
        await asyncio.sleep(delay)

        # Increment (potential race condition)
        self.state.counter = current + 1
        self.state.last_modified_by = worker_id
        self.state.version += 1

        return {
            "counter": self.state.counter,
            "modified_by": worker_id,
            "version": self.state.version,
        }

    @query
    async def get_count(self) -> int:
        """Get count (read-only)."""
        return self.state.counter

    @query
    async def get_last_modified_by(self) -> str:
        """Get last modifier (read-only)."""
        return self.state.last_modified_by

    @query
    async def get_version(self) -> int:
        """Get version (read-only)."""
        return self.state.version


class FailingEntity(Entity[CounterState]):
    """
    Entity that fails for testing error handling.

    Methods:
        always_fails(): Always raises error
        sometimes_fails(probability): Randomly fails
        fails_on_condition(condition): Fails if condition met
    """

    async def always_fails(self, error_type: str = "RuntimeError") -> dict:
        """Always raises error."""
        if error_type == "ValueError":
            raise ValueError("This entity always fails with ValueError")
        elif error_type == "TypeError":
            raise TypeError("This entity always fails with TypeError")
        else:
            raise RuntimeError("This entity always fails")

    async def sometimes_fails(self, fail_probability: float = 0.5) -> dict:
        """Randomly fails based on probability."""
        import random

        if random.random() < fail_probability:
            raise RuntimeError(f"Random failure (probability: {fail_probability})")

        self.state.count += 1
        return {
            "success": True,
            "count": self.state.count,
        }

    async def fails_on_condition(self, threshold: int = 5) -> dict:
        """Fails when count exceeds threshold."""
        self.state.count += 1

        if self.state.count > threshold:
            raise RuntimeError(f"Count exceeded threshold ({threshold})")

        return {
            "count": self.state.count,
            "threshold": threshold,
        }


# =============================================================================
# EDGE CASE ENTITIES
# =============================================================================


class EmptyStateEntity(Entity[CounterState]):
    """
    Entity for testing empty/default state handling.

    Methods:
        get_count(): Get count (should be 0 initially)
        increment(): Increment from default state
    """

    @query
    async def get_count(self) -> int:
        """Get count (read-only)."""
        return self.state.count

    async def increment(self, amount: int = 1) -> int:
        """Increment counter."""
        self.state.count += amount
        return self.state.count


class NegativeNumberEntity(Entity[CounterState]):
    """
    Entity for testing negative number handling.

    Methods:
        set_negative(value): Set to negative value
        decrement_below_zero(): Decrement below zero
        get_count(): Get count
    """

    async def set_negative(self, value: int) -> int:
        """Set to negative value."""
        if value >= 0:
            raise ValueError("Value must be negative for this test")

        self.state.count = value
        return self.state.count

    async def decrement_below_zero(self, amount: int = 10) -> int:
        """Decrement below zero."""
        self.state.count -= amount
        return self.state.count

    @query
    async def get_count(self) -> int:
        """Get count (read-only)."""
        return self.state.count


class UnicodeEntity(Entity[ComplexState]):
    """
    Entity for testing Unicode/internationalization.

    Methods:
        add_unicode_item(item): Add item with Unicode
        set_unicode_metadata(key, value): Set metadata with Unicode
        get_items(): Get items
    """

    async def add_unicode_item(self, item: str) -> dict:
        """Add item with Unicode characters."""
        self.state.items.append(item)
        return {
            "item": item,
            "length": len(item),
            "byte_length": len(item.encode("utf-8")),
            "total_items": len(self.state.items),
        }

    async def set_unicode_metadata(self, key: str, value: str) -> dict:
        """Set metadata with Unicode."""
        self.state.metadata[key] = value
        return {
            "key": key,
            "value": value,
            "byte_length": len(value.encode("utf-8")),
        }

    @query
    async def get_items(self) -> list:
        """Get items (read-only)."""
        return self.state.items


class TimeoutEntity(Entity[CounterState]):
    """
    Entity with slow operations for timeout testing.

    Methods:
        slow_operation(duration): Sleep for duration
        very_slow_operation(): Sleep for long time (should timeout)
        get_count(): Get count
    """

    async def slow_operation(self, duration: float = 1.0) -> dict:
        """Slow operation for timeout testing."""
        import asyncio

        await asyncio.sleep(duration)
        self.state.count += 1
        return {
            "duration": duration,
            "count": self.state.count,
        }

    async def very_slow_operation(self, duration: float = 30.0) -> dict:
        """Very slow operation (should timeout in tests)."""
        import asyncio

        await asyncio.sleep(duration)
        self.state.count += 1
        return {
            "duration": duration,
            "count": self.state.count,
        }

    @query
    async def get_count(self) -> int:
        """Get count (read-only)."""
        return self.state.count
