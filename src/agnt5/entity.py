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
from .function import _extract_function_schemas, _extract_function_metadata
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Type for entity method handlers
T = TypeVar("T")
EntityMethod = Callable[..., Awaitable[T]]

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
    Represents an entity type with registered methods.

    Entity types are created using the entity() function.
    Methods are registered using the @entity_type.method decorator.
    """

    def __init__(self, name: str, entity_class: Optional[type] = None):
        """
        Initialize an entity type.

        Args:
            name: Unique name for this entity type
            entity_class: Optional reference to DurableEntity class (for class-based entities)
        """
        self.name = name
        self.entity_class = entity_class  # Store class reference for DurableEntity
        self._methods: Dict[str, EntityMethod] = {}
        self._method_schemas: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = {}
        self._method_metadata: Dict[str, Dict[str, str]] = {}
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

            # Extract schemas from type hints (use original func before async wrapping)
            original_func = original_func if 'original_func' in locals() else f
            input_schema, output_schema = _extract_function_schemas(original_func)

            # Extract metadata (description, etc.)
            method_metadata = _extract_function_metadata(original_func)

            # Register method
            method_name = f.__name__
            if method_name in self._methods:
                logger.warning(
                    f"Overwriting existing method '{method_name}' on entity type '{self.name}'"
                )

            self._methods[method_name] = f
            self._method_schemas[method_name] = (input_schema, output_schema)
            self._method_metadata[method_name] = method_metadata
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
    entity_type = EntityType(name)
    EntityRegistry.register(entity_type)
    return entity_type


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

class AsyncContextWrapper:
    """
    Wrapper that provides async API for Context state operations.

    This allows DurableEntity to use:
        await self.ctx.get(key, default)
        await self.ctx.set(key, value)
        await self.ctx.delete(key)

    While the underlying Context uses sync operations internally.
    """

    def __init__(self, context: Context):
        """Wrap a Context object with async API."""
        self._context = context

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from state (async API)."""
        return await self._context.get_async(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set value in state (async API)."""
        await self._context.set_async(key, value)

    async def delete(self, key: str) -> None:
        """Delete key from state (async API)."""
        await self._context.delete_async(key)

    async def clear_all(self) -> None:
        """Clear all state (async API)."""
        await self._context.clear_all()

    # Expose other context properties
    @property
    def run_id(self) -> str:
        return self._context.run_id

    @property
    def object_id(self) -> Optional[str]:
        return self._context.object_id

    @property
    def method_name(self) -> Optional[str]:
        return self._context.method_name


class DurableEntity:
    """
    Base class for class-based durable entities (Cloudflare Durable Objects style).

    Unlike the method-based entity() decorator, this provides a class-based API where:
    - State is accessed via self.ctx (like Cloudflare's this.ctx.storage)
    - Methods are regular async methods on the class
    - Each instance is bound to a unique key
    - Single-writer consistency per key is guaranteed

    Example:
        ```python
        from agnt5 import DurableEntity

        class ShoppingCart(DurableEntity):
            async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
                items = await self.ctx.get("items", {})
                items[item_id] = {"quantity": quantity, "price": price}
                await self.ctx.set("items", items)
                return {"total_items": len(items)}

            async def get_total(self) -> float:
                items = await self.ctx.get("items", {})
                return sum(item["quantity"] * item["price"] for item in items.values())

        # Usage
        cart = ShoppingCart(key="user-123")
        await cart.add_item("item-abc", quantity=2, price=29.99)
        total = await cart.get_total()
        ```

    Note:
        Methods are automatically wrapped to provide single-writer consistency per key.
        State operations (ctx.get/set/delete) are async in this API.
    """

    def __init__(self, key: str):
        """
        Initialize a durable entity instance.

        Args:
            key: Unique identifier for this entity instance
        """
        self._key = key
        self._entity_type = self.__class__.__name__
        self._state_key = (self._entity_type, key)

        # Create Context for state access (will be populated during method execution)
        self._ctx = None

        logger.debug(f"Created DurableEntity instance: {self._entity_type}:{key}")

    @property
    def ctx(self) -> "AsyncContextWrapper":
        """
        Get the context for state access.

        Available in method execution:
        - await self.ctx.get(key, default)
        - await self.ctx.set(key, value)
        - await self.ctx.delete(key)
        - await self.ctx.clear_all()

        Returns:
            AsyncContextWrapper for async state operations
        """
        if self._ctx is None:
            # Create a context if not in method execution
            # This allows initialization and setup
            self._ctx = Context(
                run_id=f"{self._entity_type}:{self._key}:init",
                component_type="entity",
                object_id=self._key
            )
        # Wrap in AsyncContextWrapper for async API
        return AsyncContextWrapper(self._ctx)

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
            name in ('ctx', 'key', 'entity_type')):  # Skip properties
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

                # Set context on instance for method access
                object.__setattr__(self, '_ctx', ctx)

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
                    # Clear context after execution
                    object.__setattr__(self, '_ctx', None)

        # Mark as wrapped to avoid double-wrapping
        entity_method_wrapper._entity_wrapped = True
        return entity_method_wrapper


    def __init_subclass__(cls, **kwargs):
        """
        Auto-register DurableEntity subclasses.

        This is called automatically when a class inherits from DurableEntity.
        """
        super().__init_subclass__(**kwargs)

        # Don't register the base DurableEntity class itself
        if cls.__name__ == 'DurableEntity':
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
                # Execution happens via DurableEntity.__getattribute__

        # Register the entity type
        EntityRegistry.register(entity_type)
        logger.debug(f"Auto-registered DurableEntity subclass: {cls.__name__}")


class SessionEntity(DurableEntity):
    """
    Session-based entity with built-in conversation history management.

    Inspired by Google ADK and OpenAI Agents SDK session patterns.
    Automatically manages message history with trimming and optional summarization.

    Configuration (class variables):
        max_turns: Maximum conversation turns to keep (default: 20)
        auto_summarize: Enable automatic summarization of old messages (default: False)
        history_key: State key for storing history (default: "_history")
        summary_key: State key for storing summary (default: "_summary")

    Built-in Methods:
        - add_message(role, content, **metadata): Add message to history
        - get_history(limit=None): Get conversation history
        - clear_history(): Clear all history
        - get_summary(): Get conversation summary (if auto_summarize enabled)

    Example:
        ```python
        from agnt5 import SessionEntity

        class Conversation(SessionEntity):
            max_turns: int = 20
            auto_summarize: bool = True

            async def chat(self, message: str) -> str:
                # Add user message (automatic)
                await self.add_message("user", message)

                # Get history (auto-trimmed)
                history = await self.get_history(limit=10)

                # Generate AI response
                response = await some_ai_call(history)

                # Add AI response (automatic)
                await self.add_message("assistant", response)

                return response

        # Usage
        conv = Conversation(key="user-123")
        response = await conv.chat("Hello!")  # History managed automatically
        ```
    """

    # Configuration (can be overridden in subclasses)
    max_turns: int = 20
    auto_summarize: bool = False
    history_key: str = "_history"
    summary_key: str = "_summary"

    async def add_message(
        self,
        role: str,
        content: str,
        **metadata
    ) -> dict:
        """
        Add a message to the conversation history.

        Args:
            role: Message role (e.g., "user", "assistant", "system")
            content: Message content
            **metadata: Additional metadata (name, timestamp, etc.)

        Returns:
            dict with message info and current history length
        """
        import time

        # Get current history
        history = await self.ctx.get(self.history_key, [])

        # Create message
        message = {
            "role": role,
            "content": content,
            "timestamp": metadata.get("timestamp", time.time()),
            **metadata
        }

        # Add to history
        history.append(message)

        # Trim if needed
        if len(history) > self.max_turns * 2:  # 2 messages per turn (user + assistant)
            if self.auto_summarize:
                # Summarize old messages before trimming
                await self._summarize_and_trim(history)
            else:
                # Just trim
                history = history[-(self.max_turns * 2):]

        # Save history
        await self.ctx.set(self.history_key, history)

        return {
            "role": role,
            "added": True,
            "history_length": len(history),
            "timestamp": message["timestamp"]
        }

    async def get_history(self, limit: Optional[int] = None) -> list:
        """
        Get conversation history.

        Args:
            limit: Maximum number of messages to return (None = all)

        Returns:
            List of message dicts
        """
        history = await self.ctx.get(self.history_key, [])

        if limit is not None:
            return history[-limit:]

        return history

    async def clear_history(self) -> dict:
        """
        Clear all conversation history.

        Returns:
            dict with status and cleared count
        """
        history = await self.ctx.get(self.history_key, [])
        count = len(history)

        await self.ctx.delete(self.history_key)

        if self.auto_summarize:
            await self.ctx.delete(self.summary_key)

        return {
            "cleared": True,
            "message_count": count
        }

    async def get_summary(self) -> Optional[str]:
        """
        Get conversation summary (if auto_summarize is enabled).

        Returns:
            Summary string or None if no summary exists
        """
        if not self.auto_summarize:
            return None

        return await self.ctx.get(self.summary_key)

    async def _summarize_and_trim(self, history: list) -> None:
        """
        Summarize old messages and trim history.

        This is a placeholder for future AI-powered summarization.
        For now, it just stores a simple summary and trims.

        Args:
            history: Current message history
        """
        # Messages to summarize (oldest half)
        to_summarize = history[:len(history) // 2]

        # Simple summary (in future, use AI to generate better summary)
        summary_text = f"Conversation summary: {len(to_summarize)} messages exchanged"

        # Get existing summary
        existing_summary = await self.ctx.get(self.summary_key)
        if existing_summary:
            summary_text = f"{existing_summary}\n{summary_text}"

        # Store summary
        await self.ctx.set(self.summary_key, summary_text)

        # Trim history (keep most recent messages)
        trimmed_history = history[len(history) // 2:]
        await self.ctx.set(self.history_key, trimmed_history)


class MemoryEntity(DurableEntity):
    """
    Memory entity for cross-session knowledge storage and retrieval.

    Provides semantic memory storage with search capabilities.
    In Phase 1: Simple keyword-based search (in-memory)
    Future: Vector embeddings with semantic search (Pinecone, Weaviate, etc.)

    Configuration (class variables):
        memory_key: State key for storing memories (default: "_memories")
        max_memories: Maximum memories to keep (default: 100)

    Built-in Methods:
        - store(key, content, **metadata): Store a memory
        - recall(query, limit=5): Search memories
        - forget(key): Delete a memory
        - list_memories(): List all stored memories

    Example:
        ```python
        from agnt5 import MemoryEntity

        class AgentMemory(MemoryEntity):
            max_memories: int = 50

            async def remember_fact(self, fact: str, category: str) -> dict:
                # Store with metadata
                return await self.store(
                    key=f"fact_{len(await self.list_memories())}",
                    content=fact,
                    category=category
                )

            async def find_facts(self, query: str) -> list:
                # Search memories
                results = await self.recall(query, limit=5)
                return [r["content"] for r in results]

        # Usage
        memory = AgentMemory(key="agent-123")
        await memory.remember_fact("Paris is the capital of France", category="geography")
        results = await memory.find_facts("capital France")
        ```
    """

    # Configuration
    memory_key: str = "_memories"
    max_memories: int = 100

    async def store(
        self,
        key: str,
        content: str,
        **metadata
    ) -> dict:
        """
        Store a memory with optional metadata.

        Args:
            key: Unique identifier for this memory
            content: The memory content to store
            **metadata: Additional metadata (tags, category, timestamp, etc.)

        Returns:
            dict with storage confirmation
        """
        import time

        # Get current memories
        memories = await self.ctx.get(self.memory_key, {})

        # Create memory entry
        memory = {
            "content": content,
            "timestamp": metadata.get("timestamp", time.time()),
            **metadata
        }

        # Store memory
        memories[key] = memory

        # Trim if needed
        if len(memories) > self.max_memories:
            # Remove oldest memories
            sorted_keys = sorted(
                memories.keys(),
                key=lambda k: memories[k].get("timestamp", 0)
            )
            for old_key in sorted_keys[:len(memories) - self.max_memories]:
                del memories[old_key]

        # Save memories
        await self.ctx.set(self.memory_key, memories)

        return {
            "stored": True,
            "key": key,
            "total_memories": len(memories)
        }

    async def recall(
        self,
        query: str,
        limit: int = 5
    ) -> list:
        """
        Search memories using keyword matching.

        Phase 1: Simple keyword search
        Future: Semantic search with embeddings

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching memories (sorted by relevance)
        """
        memories = await self.ctx.get(self.memory_key, {})

        if not memories:
            return []

        # Simple keyword matching (future: use embeddings)
        query_lower = query.lower()
        matches = []

        for key, memory in memories.items():
            content = memory.get("content", "").lower()

            # Calculate simple relevance score (number of matching words)
            query_words = set(query_lower.split())
            content_words = set(content.split())
            matching_words = query_words & content_words
            score = len(matching_words)

            if score > 0 or query_lower in content:
                matches.append({
                    "key": key,
                    "content": memory["content"],
                    "score": score if score > 0 else 0.5,  # Substring match gets 0.5
                    "timestamp": memory.get("timestamp"),
                    **{k: v for k, v in memory.items() if k not in ("content", "timestamp")}
                })

        # Sort by score (descending)
        matches.sort(key=lambda x: x["score"], reverse=True)

        return matches[:limit]

    async def forget(self, key: str) -> dict:
        """
        Delete a memory.

        Args:
            key: Memory key to delete

        Returns:
            dict with deletion status
        """
        memories = await self.ctx.get(self.memory_key, {})

        if key in memories:
            del memories[key]
            await self.ctx.set(self.memory_key, memories)
            return {"deleted": True, "key": key}

        return {"deleted": False, "key": key, "reason": "not_found"}

    async def list_memories(self) -> list:
        """
        List all stored memories.

        Returns:
            List of all memories with keys
        """
        memories = await self.ctx.get(self.memory_key, {})

        return [
            {"key": k, **v}
            for k, v in memories.items()
        ]

    async def clear_all_memories(self) -> dict:
        """
        Clear all memories.

        Returns:
            dict with status and count
        """
        memories = await self.ctx.get(self.memory_key, {})
        count = len(memories)

        await self.ctx.delete(self.memory_key)

        return {
            "cleared": True,
            "memory_count": count
        }


class WorkflowEntity(DurableEntity):
    """
    Workflow entity for durable multi-step processes.

    Provides orchestration for complex workflows with step tracking,
    compensation logic, and automatic state persistence.

    Similar to Temporal/Azure Durable Functions patterns.

    Configuration (class variables):
        workflow_key: State key for workflow state (default: "_workflow")
        max_retries: Maximum retries per step (default: 3)

    Built-in Methods:
        - get_status(): Get workflow execution status
        - mark_step_complete(step_name, result): Mark step as complete
        - mark_step_failed(step_name, error): Mark step as failed
        - rollback(to_step): Rollback to specific step
        - can_retry(step_name): Check if step can be retried

    Example:
        ```python
        from agnt5 import WorkflowEntity

        class OrderWorkflow(WorkflowEntity):
            async def process_order(self, order_id: str, items: list) -> dict:
                # Step 1: Validate order
                status = await self.get_status()
                if "validate" not in status["completed_steps"]:
                    try:
                        validation = await self._validate_order(order_id, items)
                        await self.mark_step_complete("validate", validation)
                    except Exception as e:
                        await self.mark_step_failed("validate", str(e))
                        raise

                # Step 2: Charge payment
                if "payment" not in status["completed_steps"]:
                    try:
                        charge = await self._charge_payment(order_id)
                        await self.mark_step_complete("payment", charge)
                    except Exception as e:
                        await self.mark_step_failed("payment", str(e))
                        # Rollback validation
                        await self.rollback("validate")
                        raise

                # Step 3: Ship order
                if "shipping" not in status["completed_steps"]:
                    try:
                        shipment = await self._ship_order(order_id)
                        await self.mark_step_complete("shipping", shipment)
                    except Exception as e:
                        await self.mark_step_failed("shipping", str(e))
                        raise

                return await self.get_status()

            async def _validate_order(self, order_id: str, items: list) -> dict:
                # Validation logic
                return {"valid": True, "order_id": order_id}

            async def _charge_payment(self, order_id: str) -> dict:
                # Payment logic
                return {"charged": True, "amount": 100}

            async def _ship_order(self, order_id: str) -> dict:
                # Shipping logic
                return {"shipped": True, "tracking": "TRACK123"}

        # Usage
        workflow = OrderWorkflow(key="order-456")
        result = await workflow.process_order("order-456", [{"sku": "ABC"}])
        status = await workflow.get_status()
        ```
    """

    # Configuration
    workflow_key: str = "_workflow"
    max_retries: int = 3

    async def get_status(self) -> dict:
        """
        Get current workflow execution status.

        Returns:
            dict: {
                "current_step": str or None,
                "completed_steps": list of step names,
                "failed_steps": dict mapping step names to error info,
                "started_at": timestamp or None,
                "completed_at": timestamp or None
            }
        """
        workflow_state = await self.ctx.get(
            self.workflow_key,
            {
                "current_step": None,
                "completed_steps": [],
                "failed_steps": {},
                "started_at": None,
                "completed_at": None,
            },
        )
        return workflow_state

    async def mark_step_complete(self, step_name: str, result: Any = None) -> dict:
        """
        Mark a workflow step as successfully completed.

        Args:
            step_name: Name of the step
            result: Optional result data from the step

        Returns:
            dict: Updated workflow status
        """
        workflow_state = await self.get_status()

        if workflow_state["started_at"] is None:
            workflow_state["started_at"] = time.time()

        # Add to completed steps if not already there
        if step_name not in workflow_state["completed_steps"]:
            workflow_state["completed_steps"].append(step_name)

        # Remove from failed steps if it was there
        if step_name in workflow_state["failed_steps"]:
            del workflow_state["failed_steps"][step_name]

        # Store step result
        if result is not None:
            step_results_key = f"{self.workflow_key}_results"
            step_results = await self.ctx.get(step_results_key, {})
            step_results[step_name] = {
                "result": result,
                "completed_at": time.time(),
            }
            await self.ctx.set(step_results_key, step_results)

        workflow_state["current_step"] = step_name

        await self.ctx.set(self.workflow_key, workflow_state)
        return workflow_state

    async def mark_step_failed(
        self, step_name: str, error: str, retry_count: int = 0
    ) -> dict:
        """
        Mark a workflow step as failed.

        Args:
            step_name: Name of the step
            error: Error message or description
            retry_count: Number of retries attempted

        Returns:
            dict: Updated workflow status
        """
        workflow_state = await self.get_status()

        if workflow_state["started_at"] is None:
            workflow_state["started_at"] = time.time()

        # Record failure
        workflow_state["failed_steps"][step_name] = {
            "error": error,
            "failed_at": time.time(),
            "retry_count": retry_count,
        }

        workflow_state["current_step"] = step_name

        await self.ctx.set(self.workflow_key, workflow_state)
        return workflow_state

    async def rollback(self, to_step: str) -> dict:
        """
        Rollback workflow to a specific step (for compensation logic).

        Args:
            to_step: Step name to rollback to

        Returns:
            dict: Updated workflow status
        """
        workflow_state = await self.get_status()

        # Find the index of the target step
        if to_step in workflow_state["completed_steps"]:
            target_index = workflow_state["completed_steps"].index(to_step)

            # Remove all steps after the target
            workflow_state["completed_steps"] = workflow_state["completed_steps"][
                : target_index + 1
            ]

            # Clear failed steps that are after the target
            workflow_state["failed_steps"] = {}

            workflow_state["current_step"] = to_step

            await self.ctx.set(self.workflow_key, workflow_state)

        return workflow_state

    async def can_retry(self, step_name: str) -> bool:
        """
        Check if a failed step can be retried based on max_retries.

        Args:
            step_name: Name of the step

        Returns:
            bool: True if step can be retried
        """
        workflow_state = await self.get_status()

        if step_name in workflow_state["failed_steps"]:
            retry_count = workflow_state["failed_steps"][step_name].get(
                "retry_count", 0
            )
            return retry_count < self.max_retries

        return True

    async def get_step_result(self, step_name: str) -> Any:
        """
        Get the result of a completed step.

        Args:
            step_name: Name of the step

        Returns:
            Any: Step result or None if not found
        """
        step_results_key = f"{self.workflow_key}_results"
        step_results = await self.ctx.get(step_results_key, {})

        if step_name in step_results:
            return step_results[step_name].get("result")

        return None

    async def complete_workflow(self) -> dict:
        """
        Mark the entire workflow as completed.

        Returns:
            dict: Final workflow status
        """
        workflow_state = await self.get_status()
        workflow_state["completed_at"] = time.time()
        workflow_state["current_step"] = None
        await self.ctx.set(self.workflow_key, workflow_state)
        return workflow_state

    async def reset_workflow(self) -> dict:
        """
        Reset workflow state (use with caution).

        Returns:
            dict: New empty workflow state
        """
        new_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": {},
            "started_at": None,
            "completed_at": None,
        }
        await self.ctx.set(self.workflow_key, new_state)

        # Clear step results
        step_results_key = f"{self.workflow_key}_results"
        await self.ctx.set(step_results_key, {})

        return new_state
