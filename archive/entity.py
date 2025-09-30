"""Entity primitive for stateful components with single-writer consistency.

Entities provide durable state management with unique keys and method-level
access control (write vs shared/read-only).
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


class EntityType:
    """Represents an entity type with write and shared method decorators."""

    def __init__(self, name: str):
        self.name = name
        self._write_methods: Dict[str, Callable] = {}
        self._shared_methods: Dict[str, Callable] = {}

    def write(self, func: Callable) -> Callable:
        """Decorator for write methods (exclusive access per entity key)."""
        method_name = func.__name__
        self._write_methods[method_name] = func

        # Mark the function with entity metadata
        func._entity_type = self.name  # type: ignore[attr-defined]
        func._entity_access = "write"  # type: ignore[attr-defined]
        func._entity_method = method_name  # type: ignore[attr-defined]

        # Register with entity registry
        _ENTITY_REGISTRY.register_method(self.name, method_name, func, access="write")

        logger.debug(f"Registered write method '{method_name}' for entity '{self.name}'")
        return func

    def shared(self, func: Callable) -> Callable:
        """Decorator for shared/read-only methods (concurrent access allowed)."""
        method_name = func.__name__
        self._shared_methods[method_name] = func

        # Mark the function with entity metadata
        func._entity_type = self.name  # type: ignore[attr-defined]
        func._entity_access = "shared"  # type: ignore[attr-defined]
        func._entity_method = method_name  # type: ignore[attr-defined]

        # Register with entity registry
        _ENTITY_REGISTRY.register_method(self.name, method_name, func, access="shared")

        logger.debug(f"Registered shared method '{method_name}' for entity '{self.name}'")
        return func

    def get_method(self, method_name: str) -> Optional[Callable]:
        """Get a method by name."""
        if method_name in self._write_methods:
            return self._write_methods[method_name]
        if method_name in self._shared_methods:
            return self._shared_methods[method_name]
        return None


@dataclass
class EntityMethodDefinition:
    """Definition of an entity method."""

    entity_type: str
    method_name: str
    handler: Callable
    access: str  # "write" or "shared"
    module: str
    qualname: str


class EntityRegistry:
    """Thread-safe registry for entity types and methods."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entities: Dict[str, EntityType] = {}
        self._methods: Dict[
            tuple[str, str], EntityMethodDefinition
        ] = {}  # (entity_type, method_name) -> definition

    def register_entity(self, entity_type: EntityType) -> None:
        """Register an entity type."""
        with self._lock:
            if entity_type.name in self._entities:
                logger.warning(f"Entity type '{entity_type.name}' already registered")
            self._entities[entity_type.name] = entity_type

    def register_method(
        self, entity_type: str, method_name: str, handler: Callable, access: str
    ) -> None:
        """Register an entity method."""
        with self._lock:
            key = (entity_type, method_name)
            definition = EntityMethodDefinition(
                entity_type=entity_type,
                method_name=method_name,
                handler=handler,
                access=access,
                module=handler.__module__,
                qualname=handler.__qualname__,
            )
            self._methods[key] = definition
            logger.debug(f"Registered {access} method '{method_name}' for entity '{entity_type}'")

    def get_entity(self, entity_type: str) -> Optional[EntityType]:
        """Get an entity type by name."""
        with self._lock:
            return self._entities.get(entity_type)

    def get_method(self, entity_type: str, method_name: str) -> Optional[EntityMethodDefinition]:
        """Get an entity method definition."""
        with self._lock:
            return self._methods.get((entity_type, method_name))

    def list_entities(self) -> Dict[str, EntityType]:
        """List all registered entity types."""
        with self._lock:
            return dict(self._entities)

    def clear(self) -> None:
        """Clear all registrations (for testing)."""
        with self._lock:
            self._entities.clear()
            self._methods.clear()


# Global registry
_ENTITY_REGISTRY = EntityRegistry()


def entity(name: str) -> EntityType:
    """Create an entity type with the given name.

    Args:
        name: Unique entity type name (e.g., "ConversationAgent")

    Returns:
        EntityType instance with write and shared decorators

    Example:
        ```python
        agent = entity("ConversationAgent")

        @agent.write
        async def send_message(ctx, message: str) -> dict:
            history = await ctx.get("history", [])
            history.append({"role": "user", "content": message})
            ctx.set("history", history)
            return {"response": "processed"}

        @agent.shared
        async def get_history(ctx) -> list:
            return await ctx.get("history", [])
        ```
    """
    entity_type = EntityType(name)
    _ENTITY_REGISTRY.register_entity(entity_type)
    return entity_type


def get_entity_registry() -> EntityRegistry:
    """Get the global entity registry."""
    return _ENTITY_REGISTRY


class EntityProxy:
    """Proxy for invoking entity methods through context."""

    def __init__(self, entity_type: str, entity_key: str, context: Any):
        self._entity_type = entity_type
        self._entity_key = entity_key
        self._context = context

    async def __call__(self, method_name: str, *args, **kwargs) -> Any:
        """Invoke an entity method."""
        method_def = _ENTITY_REGISTRY.get_method(self._entity_type, method_name)
        if not method_def:
            raise AttributeError(
                f"Method '{method_name}' not found on entity type '{self._entity_type}'"
            )

        # TODO: Implement actual entity invocation through backend
        # For now, call the method directly with context
        handler = method_def.handler

        # Call the handler with context
        result = handler(self._context, *args, **kwargs)

        # Handle async results
        if inspect.iscoroutine(result) or inspect.isawaitable(result):
            return await result

        return result

    def __getattr__(self, method_name: str) -> Callable:
        """Create a callable for the entity method."""

        async def method_wrapper(*args, **kwargs):
            return await self(method_name, *args, **kwargs)

        return method_wrapper


__all__ = [
    "entity",
    "EntityType",
    "EntityProxy",
    "EntityRegistry",
    "EntityMethodDefinition",
    "get_entity_registry",
]
