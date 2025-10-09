"""
Entity component for stateful operations with single-writer consistency.

Entities provide isolated state per unique key with automatic consistency guarantees.
In Phase 1, entities use in-memory state with asyncio locks for single-writer semantics.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Dict, Optional, Tuple

from .context import Context
from .exceptions import ConfigurationError, ExecutionError
from .function import _extract_function_schemas, _extract_function_metadata
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Global storage for in-memory entity state and locks
# Phase 2 will replace these with platform-backed durable storage
_entity_states: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (type, key) -> state
_entity_locks: Dict[Tuple[str, str], asyncio.Lock] = {}  # (type, key) -> lock

# Global entity registry
_ENTITY_REGISTRY: Dict[str, "EntityType"] = {}


class EntityRegistry:
    """Registry for entity types."""

    @staticmethod
    def register(entity_type: "EntityType") -> None:
        """Register an entity type."""
        if entity_type.name in _ENTITY_REGISTRY:
            logger.warning(f"Overwriting existing entity type '{entity_type.name}'")
        _ENTITY_REGISTRY[entity_type.name] = entity_type
        logger.debug(f"Registered entity type '{entity_type.name}'")

    @staticmethod
    def get(name: str) -> Optional["EntityType"]:
        """Get entity type by name."""
        return _ENTITY_REGISTRY.get(name)

    @staticmethod
    def all() -> Dict[str, "EntityType"]:
        """Get all registered entities."""
        return _ENTITY_REGISTRY.copy()

    @staticmethod
    def clear() -> None:
        """Clear all registered entities."""
        _ENTITY_REGISTRY.clear()
        logger.debug("Cleared entity registry")


class EntityType:
    """
    Metadata about an Entity class.

    Stores entity name, method schemas, and metadata for Worker auto-discovery
    and platform integration. Created automatically when Entity subclasses are defined.
    """

    def __init__(self, name: str, entity_class: type):
        """
        Initialize entity type metadata.

        Args:
            name: Entity type name (class name)
            entity_class: Reference to the Entity class
        """
        self.name = name
        self.entity_class = entity_class
        self._method_schemas: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}
        self._method_metadata: Dict[str, Dict[str, str]] = {}
        logger.debug(f"Created entity type: {name}")


# Utility functions for testing and debugging

def _clear_entity_state() -> None:
    """
    Clear all entity state and locks.

    Warning: Only use for testing. This will delete all entity state.
    """
    _entity_states.clear()
    _entity_locks.clear()
    logger.debug("Cleared all entity state and locks")


def _get_entity_state(entity_type: str, key: str) -> Optional[Dict[str, Any]]:
    """
    Get the current state of an entity instance.

    Args:
        entity_type: Entity type name
        key: Entity instance key

    Returns:
        State dict or None if entity has no state

    Note: For debugging and testing only.
    """
    state_key = (entity_type, key)
    return _entity_states.get(state_key)


def _get_all_entity_keys(entity_type: str) -> list[str]:
    """
    Get all keys for a given entity type.

    Args:
        entity_type: Entity type name

    Returns:
        List of keys that have state

    Note: For debugging and testing only.
    """
    return [
        key for (etype, key) in _entity_states.keys()
        if etype == entity_type
    ]


# ============================================================================
# New: Class-Based Entity API (Cloudflare Durable Objects style)
# ============================================================================

class EntityState:
    """
    Simple state interface for Entity instances.

    Provides a clean API for state management:
        self.state.get(key, default)
        self.state.set(key, value)
        self.state.delete(key)
        self.state.clear()

    State operations are synchronous and backed by the Context.
    """

    def __init__(self, context: Context):
        """Initialize state wrapper with a Context."""
        self._context = context

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self._context.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._context.set(key, value)

    def delete(self, key: str) -> None:
        """Delete key from state."""
        self._context.delete(key)

    def clear(self) -> None:
        """Clear all state."""
        # Clear all keys from the context state
        self._context._state.clear()


class Entity:
    """
    Base class for stateful entities with single-writer consistency.

    Entities provide a class-based API where:
    - State is accessed via self.state (clean, synchronous API)
    - Methods are regular async methods on the class
    - Each instance is bound to a unique key
    - Single-writer consistency per key is guaranteed automatically

    Example:
        ```python
        from agnt5 import Entity

        class ShoppingCart(Entity):
            async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
                items = self.state.get("items", {})
                items[item_id] = {"quantity": quantity, "price": price}
                self.state.set("items", items)
                return {"total_items": len(items)}

            async def get_total(self) -> float:
                items = self.state.get("items", {})
                return sum(item["quantity"] * item["price"] for item in items.values())

        # Usage
        cart = ShoppingCart(key="user-123")
        await cart.add_item("item-abc", quantity=2, price=29.99)
        total = await cart.get_total()
        ```

    Note:
        Methods are automatically wrapped to provide single-writer consistency per key.
        State operations are synchronous for simplicity.
    """

    def __init__(self, key: str):
        """
        Initialize an entity instance.

        Args:
            key: Unique identifier for this entity instance
        """
        self._key = key
        self._entity_type = self.__class__.__name__
        self._state_key = (self._entity_type, key)

        # State will be initialized during method execution
        self._state = None
        self._context = None

        logger.debug(f"Created Entity instance: {self._entity_type}:{key}")

    @property
    def state(self) -> EntityState:
        """
        Get the state interface for this entity.

        Available operations:
        - self.state.get(key, default)
        - self.state.set(key, value)
        - self.state.delete(key)
        - self.state.clear()

        Returns:
            EntityState for synchronous state operations
        """
        if self._state is None:
            # Create a context if not in method execution
            # This allows initialization and setup
            if self._context is None:
                self._context = Context(
                    run_id=f"{self._entity_type}:{self._key}:init",
                    component_type="entity",
                    object_id=self._key
                )
            self._state = EntityState(self._context)
        return self._state

    @property
    def key(self) -> str:
        """Get the entity instance key."""
        return self._key

    @property
    def entity_type(self) -> str:
        """Get the entity type name."""
        return self._entity_type

    def __getattribute__(self, name: str):
        """
        Intercept method calls to add single-writer consistency.

        This wraps all async methods (except private/magic methods) with:
        1. Lock acquisition (single-writer per key)
        2. Context setup with entity state
        3. Method execution
        4. State persistence
        """
        attr = object.__getattribute__(self, name)

        # Don't wrap private methods, properties, non-callables, or specific attributes
        if (name.startswith('_') or
            not callable(attr) or
            not asyncio.iscoroutinefunction(attr) or
            name in ('state', 'key', 'entity_type')):  # Skip properties
            return attr

        # Don't wrap if already wrapped
        if hasattr(attr, '_entity_wrapped'):
            return attr

        @functools.wraps(attr)
        async def entity_method_wrapper(*args, **kwargs):
            """
            Execute entity method with single-writer guarantee.

            This wrapper:
            1. Acquires lock for this entity instance (single-writer)
            2. Creates Context with entity state
            3. Executes method
            4. Updates state from Context
            """
            state_key = object.__getattribute__(self, '_state_key')
            entity_type = object.__getattribute__(self, '_entity_type')
            key = object.__getattribute__(self, '_key')

            # Get or create lock for this entity instance (single-writer guarantee)
            if state_key not in _entity_locks:
                _entity_locks[state_key] = asyncio.Lock()
            lock = _entity_locks[state_key]

            async with lock:
                # Get or create state for this entity instance
                if state_key not in _entity_states:
                    _entity_states[state_key] = {}
                state_dict = _entity_states[state_key]

                # Create Context with entity state
                ctx = Context(
                    run_id=f"{entity_type}:{key}:{name}",
                    component_type="entity",
                    object_id=key,
                    method_name=name
                )

                # Replace Context's internal state with entity state
                ctx._state = state_dict

                # Set context and state on instance for method access
                object.__setattr__(self, '_context', ctx)
                object.__setattr__(self, '_state', EntityState(ctx))

                try:
                    # Execute method
                    logger.debug(f"Executing {entity_type}:{key}.{name}")
                    result = await attr(*args, **kwargs)
                    logger.debug(f"Completed {entity_type}:{key}.{name}")
                    return result

                except Exception as e:
                    logger.error(
                        f"Error in {entity_type}:{key}.{name}: {e}",
                        exc_info=True
                    )
                    raise ExecutionError(
                        f"Entity method {name} failed: {e}"
                    ) from e
                finally:
                    # Clear context and state after execution
                    object.__setattr__(self, '_context', None)
                    object.__setattr__(self, '_state', None)

        # Mark as wrapped to avoid double-wrapping
        entity_method_wrapper._entity_wrapped = True
        return entity_method_wrapper


    def __init_subclass__(cls, **kwargs):
        """
        Auto-register Entity subclasses.

        This is called automatically when a class inherits from Entity.
        """
        super().__init_subclass__(**kwargs)

        # Don't register the base Entity class itself
        if cls.__name__ == 'Entity':
            return

        # Don't register SDK's built-in base classes (these are meant to be extended by users)
        if cls.__name__ in ('SessionEntity', 'MemoryEntity', 'WorkflowEntity'):
            return

        # Create an EntityType for this class, storing the class reference
        entity_type = EntityType(cls.__name__, entity_class=cls)

        # Register all public async methods
        for name, method in inspect.getmembers(cls, predicate=inspect.iscoroutinefunction):
            if not name.startswith('_'):
                # Extract schemas from the method
                input_schema, output_schema = _extract_function_schemas(method)
                method_metadata = _extract_function_metadata(method)

                # Store in entity type
                entity_type._method_schemas[name] = (input_schema, output_schema)
                entity_type._method_metadata[name] = method_metadata

                # Note: Actual method is not registered here
                # Execution happens via Entity.__getattribute__

        # Register the entity type
        EntityRegistry.register(entity_type)
        logger.debug(f"Auto-registered Entity subclass: {cls.__name__}")
