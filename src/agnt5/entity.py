"""
Entity component for stateful operations with single-writer consistency.

Entities provide isolated state per unique key with automatic consistency guarantees.
In Phase 1, entities use in-memory state with asyncio locks for single-writer semantics.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypeVar

from .context import Context
from .exceptions import ConfigurationError, ExecutionError

logger = logging.getLogger(__name__)

# Type for entity method handlers
T = TypeVar("T")
EntityMethod = Callable[..., Awaitable[T]]

# Global storage for in-memory entity state and locks
# Phase 2 will replace these with platform-backed durable storage
_entity_states: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (type, key) -> state
_entity_locks: Dict[Tuple[str, str], asyncio.Lock] = {}  # (type, key) -> lock


class EntityType:
    """
    Represents an entity type with registered methods.

    Entity types are created using the entity() function.
    Methods are registered using the @entity_type.method decorator.
    """

    def __init__(self, name: str):
        """
        Initialize an entity type.

        Args:
            name: Unique name for this entity type
        """
        self.name = name
        self._methods: Dict[str, EntityMethod] = {}
        logger.debug(f"Created entity type: {name}")

    def method(self, func: Optional[EntityMethod] = None) -> EntityMethod:
        """
        Decorator to register a method on this entity type.

        Methods receive a Context as the first parameter and can access
        entity state via ctx.get/set/delete.

        Args:
            func: The function to register as an entity method

        Returns:
            Decorated function

        Example:
            ```python
            Counter = entity("Counter")

            @Counter.method
            async def increment(ctx: Context, amount: int = 1) -> int:
                current = ctx.get("count", 0)
                new_count = current + amount
                ctx.set("count", new_count)
                return new_count
            ```
        """
        def decorator(f: EntityMethod) -> EntityMethod:
            # Validate function signature
            sig = inspect.signature(f)
            params = list(sig.parameters.values())

            if not params:
                raise ConfigurationError(
                    f"Entity method {f.__name__} must have at least one parameter (ctx: Context)"
                )

            first_param = params[0]
            # Check if first parameter is Context (can be class, string "Context", or empty)
            annotation = first_param.annotation
            is_context = (
                annotation == Context
                or annotation == "Context"
                or annotation is Context
                or annotation == inspect.Parameter.empty
                or (hasattr(annotation, "__name__") and annotation.__name__ == "Context")
            )
            if not is_context:
                raise ConfigurationError(
                    f"Entity method {f.__name__} first parameter must be 'ctx: Context', "
                    f"got '{annotation}' (type: {type(annotation).__name__})"
                )

            # Convert sync to async if needed
            if not asyncio.iscoroutinefunction(f):
                original_func = f

                @functools.wraps(original_func)
                async def async_wrapper(*args, **kwargs):
                    return original_func(*args, **kwargs)

                f = async_wrapper

            # Register method
            method_name = f.__name__
            if method_name in self._methods:
                logger.warning(
                    f"Overwriting existing method '{method_name}' on entity type '{self.name}'"
                )

            self._methods[method_name] = f
            logger.debug(f"Registered method '{method_name}' on entity type '{self.name}'")

            return f

        if func is None:
            return decorator
        return decorator(func)

    def __call__(self, key: str) -> "EntityInstance":
        """
        Create an instance of this entity type with a specific key.

        Args:
            key: Unique identifier for this entity instance

        Returns:
            EntityInstance that can invoke methods

        Example:
            ```python
            Counter = entity("Counter")

            counter1 = Counter(key="user-123")
            await counter1.increment(amount=5)
            ```
        """
        return EntityInstance(entity_type=self, key=key)


class EntityInstance:
    """
    An instance of an entity type bound to a specific key.

    Each instance has isolated state and guarantees single-writer consistency
    for operations on the same key.
    """

    def __init__(self, entity_type: EntityType, key: str):
        """
        Initialize an entity instance.

        Args:
            entity_type: The entity type
            key: Unique identifier for this instance
        """
        self._entity_type = entity_type
        self._key = key
        self._state_key = (entity_type.name, key)
        logger.debug(f"Created entity instance: {entity_type.name}:{key}")

    def __getattr__(self, method_name: str) -> Callable[..., Awaitable[Any]]:
        """
        Dynamically return a callable that invokes the entity method.

        Args:
            method_name: Name of the method to invoke

        Returns:
            Async callable that executes the method with single-writer guarantee

        Raises:
            AttributeError: If method doesn't exist on this entity type
        """
        if method_name.startswith("_"):
            # Don't intercept private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{method_name}'")

        if method_name not in self._entity_type._methods:
            available = ", ".join(self._entity_type._methods.keys())
            raise AttributeError(
                f"Entity type '{self._entity_type.name}' has no method '{method_name}'. "
                f"Available methods: {available or 'none'}"
            )

        method_func = self._entity_type._methods[method_name]

        @functools.wraps(method_func)
        async def method_wrapper(*args, **kwargs) -> Any:
            """
            Execute entity method with single-writer guarantee.

            This wrapper:
            1. Acquires lock for this entity instance (single-writer)
            2. Creates Context with entity state
            3. Executes method
            4. Updates state from Context
            """
            # Get or create lock for this entity instance (single-writer guarantee)
            if self._state_key not in _entity_locks:
                _entity_locks[self._state_key] = asyncio.Lock()
            lock = _entity_locks[self._state_key]

            async with lock:
                # Get or create state for this entity instance
                if self._state_key not in _entity_states:
                    _entity_states[self._state_key] = {}
                state_dict = _entity_states[self._state_key]

                # Create Context with entity state
                # Context state is a reference to the entity's state dict
                ctx = Context(
                    run_id=f"{self._entity_type.name}:{self._key}:{method_name}",
                    component_type="entity",
                    object_id=self._key,
                    method_name=method_name
                )

                # Replace Context's internal state with entity state
                # This allows ctx.get/set/delete to operate on entity state
                ctx._state = state_dict

                try:
                    # Execute method
                    logger.debug(
                        f"Executing {self._entity_type.name}:{self._key}.{method_name}"
                    )
                    result = await method_func(ctx, *args, **kwargs)
                    logger.debug(
                        f"Completed {self._entity_type.name}:{self._key}.{method_name}"
                    )
                    return result

                except Exception as e:
                    logger.error(
                        f"Error in {self._entity_type.name}:{self._key}.{method_name}: {e}",
                        exc_info=True
                    )
                    raise ExecutionError(
                        f"Entity method {method_name} failed: {e}"
                    ) from e

        return method_wrapper

    @property
    def entity_type(self) -> str:
        """Get the entity type name."""
        return self._entity_type.name

    @property
    def key(self) -> str:
        """Get the entity instance key."""
        return self._key


def entity(name: str) -> EntityType:
    """
    Create a new entity type.

    Entities provide stateful components with single-writer consistency.
    Each entity instance (identified by a unique key) has isolated state
    and guarantees that only one operation executes at a time per key.

    Args:
        name: Unique name for this entity type

    Returns:
        EntityType that can register methods and create instances

    Example:
        ```python
        from agnt5 import entity, Context

        # Define entity type
        Counter = entity("Counter")

        # Register methods
        @Counter.method
        async def increment(ctx: Context, amount: int = 1) -> int:
            current = ctx.get("count", 0)
            new_count = current + amount
            ctx.set("count", new_count)
            return new_count

        @Counter.method
        async def get_count(ctx: Context) -> int:
            return ctx.get("count", 0)

        # Create instances
        counter1 = Counter(key="user-123")
        counter2 = Counter(key="user-456")

        # Invoke methods (guaranteed single-writer per key)
        result = await counter1.increment(amount=5)  # Returns 5
        result = await counter1.increment(amount=3)  # Returns 8

        # Different keys execute in parallel
        await asyncio.gather(
            counter1.increment(amount=1),  # Parallel
            counter2.increment(amount=1)   # Parallel
        )
        ```

    Note:
        In Phase 1, entity state is in-memory and will be lost on process restart.
        Single-writer consistency uses asyncio.Lock (process-local).
        Phase 2 will add durable state and distributed locks via the platform.
    """
    return EntityType(name)


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
