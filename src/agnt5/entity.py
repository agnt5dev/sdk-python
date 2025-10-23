"""
Entity component for stateful operations with single-writer consistency.
"""

import asyncio
import contextvars
import functools
import inspect
import json
from typing import Any, Dict, Optional, Tuple, get_type_hints

from ._schema_utils import extract_function_metadata, extract_function_schemas
from .exceptions import ExecutionError
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Context variable for worker-scoped state adapter
# This is set by Worker before entity execution and accessed by Entity instances
_entity_state_adapter_ctx: contextvars.ContextVar[Optional["EntityStateAdapter"]] = \
    contextvars.ContextVar('_entity_state_adapter', default=None)

# Global entity registry
_ENTITY_REGISTRY: Dict[str, "EntityType"] = {}


class EntityStateAdapter:
    """
    Thin Python adapter providing Pythonic interface to Rust EntityStateManager core.

    This adapter provides language-specific concerns only:
    - Worker-local asyncio.Lock for coarse-grained coordination
    - Type conversions between Python dict and JSON bytes
    - Pythonic async/await API over Rust core

    All business logic (caching, version tracking, retry logic, gRPC) lives in the Rust core.
    This keeps the Python layer simple (~150 LOC) and enables sharing business logic across SDKs.
    """

    def __init__(self, rust_core=None):
        """
        Initialize entity state adapter.

        Args:
            rust_core: Rust EntityStateManager instance (from _core module).
                      If None, operates in standalone/testing mode with in-memory state.
        """
        self._rust_core = rust_core
        # Worker-local locks for coarse-grained coordination within this worker
        self._local_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

        # Standalone mode: in-memory state storage when no Rust core
        # This enables testing without the full platform stack
        if rust_core is None:
            self._standalone_states: Dict[Tuple[str, str], Dict[str, Any]] = {}
            self._standalone_versions: Dict[Tuple[str, str], int] = {}
            logger.debug("Created EntityStateAdapter in standalone mode (in-memory state)")
        else:
            logger.debug("Created EntityStateAdapter with Rust core")

    def get_local_lock(self, state_key: Tuple[str, str]) -> asyncio.Lock:
        """
        Get worker-local asyncio.Lock for single-writer guarantee within this worker.

        This provides coarse-grained coordination for operations within the same worker.
        Cross-worker conflicts are handled by the Rust core via optimistic concurrency.

        Args:
            state_key: Tuple of (entity_type, entity_key)

        Returns:
            asyncio.Lock for this worker-local operation
        """
        if state_key not in self._local_locks:
            self._local_locks[state_key] = asyncio.Lock()
        return self._local_locks[state_key]

    async def load_state(self, entity_type: str, entity_key: str) -> Dict[str, Any]:
        """
        Load entity state (Rust handles cache-first logic and platform load).

        In standalone mode (no Rust core), uses in-memory state storage.

        Args:
            entity_type: Type of entity (e.g., "ShoppingCart", "Counter")
            entity_key: Unique key for entity instance

        Returns:
            State dictionary (empty dict if not found)
        """
        if not self._rust_core:
            # Standalone mode - return from in-memory storage
            state_key = (entity_type, entity_key)
            return self._standalone_states.get(state_key, {}).copy()

        try:
            # Rust checks cache first, loads from platform if needed
            state_json_bytes, version = await self._rust_core.py_get_cached_or_load(entity_type, entity_key)

            # Convert bytes to dict
            if state_json_bytes:
                state_json = state_json_bytes.decode('utf-8') if isinstance(state_json_bytes, bytes) else state_json_bytes
                return json.loads(state_json)
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to load state for {entity_type}:{entity_key}: {e}")
            return {}

    async def save_state(
        self,
        entity_type: str,
        entity_key: str,
        state: Dict[str, Any],
        expected_version: int
    ) -> int:
        """
        Save entity state (Rust handles version check and platform save).

        In standalone mode (no Rust core), stores in-memory with version tracking.

        Args:
            entity_type: Type of entity
            entity_key: Unique key for entity instance
            state: State dictionary to save
            expected_version: Expected current version (for optimistic locking)

        Returns:
            New version number after save

        Raises:
            RuntimeError: If version conflict or platform error
        """
        if not self._rust_core:
            # Standalone mode - store in memory with version tracking
            state_key = (entity_type, entity_key)
            current_version = self._standalone_versions.get(state_key, 0)

            # Optimistic locking check (even in standalone mode for consistency)
            if current_version != expected_version:
                raise RuntimeError(
                    f"Version conflict: expected {expected_version}, got {current_version}"
                )

            # Store state and increment version
            new_version = expected_version + 1
            self._standalone_states[state_key] = state.copy()
            self._standalone_versions[state_key] = new_version
            return new_version

        # Convert dict to JSON bytes
        state_json = json.dumps(state).encode('utf-8')

        # Rust handles optimistic locking and platform save
        new_version = await self._rust_core.py_save_state(
            entity_type,
            entity_key,
            state_json,
            expected_version
        )

        return new_version

    async def load_with_version(self, entity_type: str, entity_key: str) -> Tuple[Dict[str, Any], int]:
        """
        Load entity state with version (for update operations).

        In standalone mode (no Rust core), loads from in-memory storage with version.

        Args:
            entity_type: Type of entity
            entity_key: Unique key for entity instance

        Returns:
            Tuple of (state_dict, version)
        """
        if not self._rust_core:
            # Standalone mode - return from in-memory storage with version
            state_key = (entity_type, entity_key)
            state = self._standalone_states.get(state_key, {}).copy()
            version = self._standalone_versions.get(state_key, 0)
            return state, version

        try:
            state_json_bytes, version = await self._rust_core.py_get_cached_or_load(entity_type, entity_key)

            if state_json_bytes:
                state_json = state_json_bytes.decode('utf-8') if isinstance(state_json_bytes, bytes) else state_json_bytes
                state = json.loads(state_json)
            else:
                state = {}

            return state, version
        except Exception as e:
            logger.warning(f"Failed to load state with version for {entity_type}:{entity_key}: {e}")
            return {}, 0

    async def invalidate_cache(self, entity_type: str, entity_key: str) -> None:
        """
        Invalidate cache entry for specific entity.

        Args:
            entity_type: Type of entity
            entity_key: Unique key for entity instance
        """
        if self._rust_core:
            await self._rust_core.py_invalidate_cache(entity_type, entity_key)

    async def clear_cache(self) -> None:
        """Clear entire cache (useful for testing)."""
        if self._rust_core:
            await self._rust_core.py_clear_cache()

    def clear_all(self) -> None:
        """Clear all local locks (for testing)."""
        self._local_locks.clear()
        logger.debug("Cleared EntityStateAdapter local locks")

    async def get_state(self, entity_type: str, key: str) -> Optional[Dict[str, Any]]:
        """Get state for debugging/testing."""
        state, _ = await self.load_with_version(entity_type, key)
        return state if state else None

    def get_all_keys(self, entity_type: str) -> list[str]:
        """
        Get all keys for an entity type (testing/debugging only).

        Only works in standalone mode. Returns empty list in production mode.
        """
        if not hasattr(self, '_standalone_states'):
            return []

        keys = []
        for (etype, ekey) in self._standalone_states.keys():
            if etype == entity_type:
                keys.append(ekey)
        return keys


def _get_state_adapter() -> EntityStateAdapter:
    """
    Get the current entity state adapter from context.

    The state adapter must be set by Worker before entity execution.
    This ensures proper worker-scoped state isolation.

    Returns:
        EntityStateAdapter instance

    Raises:
        RuntimeError: If called outside of Worker context (state adapter not set)
    """
    adapter = _entity_state_adapter_ctx.get()
    if adapter is None:
        raise RuntimeError(
            "Entity requires state adapter context.\n\n"
            "In production:\n"
            "  Entities run automatically through Worker.\n\n"
            "In tests, use one of:\n"
            "  Option 1 - Decorator:\n"
            "    @with_entity_context\n"
            "    async def test_cart():\n"
            "        cart = ShoppingCart('key')\n"
            "        await cart.add_item(...)\n\n"
            "  Option 2 - Fixture:\n"
            "    async def test_cart(entity_context):\n"
            "        cart = ShoppingCart('key')\n"
            "        await cart.add_item(...)\n\n"
            "See: https://docs.agnt5.dev/sdk/entities#testing"
        )
    return adapter




# ============================================================================
# Testing Helpers
# ============================================================================

def with_entity_context(func):
    """
    Decorator that sets up entity state adapter for tests.

    Usage:
        @with_entity_context
        async def test_shopping_cart():
            cart = ShoppingCart(key="test")
            await cart.add_item("item", 1, 10.0)
            assert cart.state.get("items")
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        adapter = EntityStateAdapter()
        token = _entity_state_adapter_ctx.set(adapter)
        try:
            return await func(*args, **kwargs)
        finally:
            _entity_state_adapter_ctx.reset(token)
            adapter.clear_all()
    return wrapper


def create_entity_context():
    """
    Create an entity context for testing (can be used as pytest fixture).

    Usage in conftest.py or test file:
        import pytest
        from agnt5.entity import create_entity_context

        @pytest.fixture
        def entity_context():
            adapter, token = create_entity_context()
            yield adapter
            # Cleanup happens automatically

    Returns:
        Tuple of (EntityStateAdapter, context_token)
    """
    adapter = EntityStateAdapter()
    token = _entity_state_adapter_ctx.set(adapter)
    return adapter, token


def extract_state_schema(entity_class: type) -> Optional[Dict[str, Any]]:
    """
    Extract JSON schema from entity class for state structure documentation.

    The schema can be provided in multiple ways (in order of preference):
    1. Explicit _state_schema class attribute (most explicit)
    2. Docstring with state description
    3. Type annotations on __init__ method (least explicit, basic types only)

    Args:
        entity_class: The Entity subclass to extract schema from

    Returns:
        JSON schema dict or None if no schema could be extracted

    Examples:
        # Option 1: Explicit schema (recommended)
        class ShoppingCart(Entity):
            _state_schema = {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "description": "Cart items"},
                    "total": {"type": "number", "description": "Cart total"}
                },
                "description": "Shopping cart state"
            }

        # Option 2: Docstring
        class ShoppingCart(Entity):
            '''
            Shopping cart entity.

            State:
                items (list): List of cart items
                total (float): Total cart value
            '''

        # Option 3: Type hints (basic extraction)
        class ShoppingCart(Entity):
            def __init__(self, key: str):
                super().__init__(key)
                self.items: list = []
                self.total: float = 0.0
    """
    # Option 1: Check for explicit _state_schema attribute
    if hasattr(entity_class, '_state_schema'):
        schema = entity_class._state_schema
        logger.debug(f"Found explicit _state_schema for {entity_class.__name__}")
        return schema

    # Option 2: Extract from docstring (basic parsing)
    if entity_class.__doc__:
        doc = entity_class.__doc__.strip()
        if "State:" in doc or "state:" in doc.lower():
            # Found state documentation - create basic schema
            logger.debug(f"Found state documentation in docstring for {entity_class.__name__}")
            return {
                "type": "object",
                "description": f"State structure for {entity_class.__name__} (see docstring for details)"
            }

    # Option 3: Try to extract from __init__ type hints (very basic)
    try:
        init_method = entity_class.__init__
        type_hints = get_type_hints(init_method)
        # Remove 'key' and 'return' from hints
        state_hints = {k: v for k, v in type_hints.items() if k not in ('key', 'return')}

        if state_hints:
            logger.debug(f"Extracted type hints from __init__ for {entity_class.__name__}")
            properties = {}
            for name, type_hint in state_hints.items():
                # Basic type mapping
                if type_hint == str:
                    properties[name] = {"type": "string"}
                elif type_hint == int:
                    properties[name] = {"type": "integer"}
                elif type_hint == float:
                    properties[name] = {"type": "number"}
                elif type_hint == bool:
                    properties[name] = {"type": "boolean"}
                elif type_hint == list or str(type_hint).startswith('list'):
                    properties[name] = {"type": "array"}
                elif type_hint == dict or str(type_hint).startswith('dict'):
                    properties[name] = {"type": "object"}
                else:
                    properties[name] = {"type": "object", "description": str(type_hint)}

            if properties:
                return {
                    "type": "object",
                    "properties": properties,
                    "description": f"State structure inferred from type hints for {entity_class.__name__}"
                }
    except Exception as e:
        logger.debug(f"Could not extract type hints from {entity_class.__name__}: {e}")

    # No schema could be extracted
    logger.debug(f"No state schema found for {entity_class.__name__}")
    return None


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

    Stores entity name, method schemas, state schema, and metadata for Worker auto-discovery
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
        self._state_schema: Optional[Dict[str, Any]] = None
        logger.debug("Created entity type: %s", name)

    def set_state_schema(self, schema: Optional[Dict[str, Any]]) -> None:
        """
        Set the state schema for this entity type.

        Args:
            schema: JSON schema describing the entity's state structure
        """
        self._state_schema = schema
        if schema:
            logger.debug(f"Set state schema for {self.name}")

    def build_entity_definition(self) -> Dict[str, Any]:
        """
        Build complete entity definition for platform registration.

        Returns:
            Dictionary with entity name, state schema, and method schemas
        """
        # Build method schemas dict
        method_schemas = {}
        for method_name, (input_schema, output_schema) in self._method_schemas.items():
            method_metadata = self._method_metadata.get(method_name, {})
            method_schemas[method_name] = {
                "input_schema": input_schema,
                "output_schema": output_schema,
                "description": method_metadata.get("description", ""),
                "metadata": method_metadata
            }

        # Build complete definition
        definition = {
            "entity_name": self.name,
            "methods": method_schemas
        }

        # Add state schema if available
        if self._state_schema:
            definition["state_schema"] = self._state_schema

        return definition


# ============================================================================
# Class-Based Entity API (Cloudflare Durable Objects style)
# ============================================================================

class EntityState:
    """
    Simple state interface for Entity instances.

    Provides a clean API for state management:
        self.state.get(key, default)
        self.state.set(key, value)
        self.state.delete(key)
        self.state.clear()

    State operations are synchronous and backed by an internal dict.
    """

    def __init__(self, state_dict: Dict[str, Any]):
        """
        Initialize state wrapper with a state dict.

        Args:
            state_dict: Dictionary to use for state storage
        """
        self._state = state_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._state[key] = value

    def delete(self, key: str) -> None:
        """Delete key from state."""
        self._state.pop(key, None)

    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()


def _create_entity_method_wrapper(entity_type: str, method):
    """
    Create a wrapper for an entity method that provides single-writer consistency.

    This wrapper implements hybrid locking:
    1. Local lock (asyncio.Lock) for worker-scoped single-writer guarantee
    2. Optimistic concurrency (via Rust) for cross-worker conflicts
    3. Loads state via adapter (Rust handles cache + platform)
    4. Executes the method with clean EntityState interface
    5. Saves state via adapter (Rust handles version check + retry)

    Args:
        entity_type: Name of the entity type (class name)
        method: The async method to wrap

    Returns:
        Wrapped async method with single-writer consistency
    """
    @functools.wraps(method)
    async def entity_method_wrapper(self, *args, **kwargs):
        """Execute entity method with hybrid locking (local + optimistic)."""
        state_key = (entity_type, self._key)

        # Get state adapter
        adapter = _get_state_adapter()

        # Local lock for worker-scoped single-writer guarantee
        lock = adapter.get_local_lock(state_key)

        async with lock:
            # Load state with version (Rust handles cache-first + platform load)
            state_dict, current_version = await adapter.load_with_version(entity_type, self._key)

            logger.debug(
                "Loaded state for %s:%s (version %d)",
                entity_type, self._key, current_version
            )

            # Set up EntityState on instance for method access
            self._state = EntityState(state_dict)

            try:
                # Execute method
                logger.debug("Executing %s:%s.%s", entity_type, self._key, method.__name__)
                result = await method(self, *args, **kwargs)
                logger.debug("Completed %s:%s.%s", entity_type, self._key, method.__name__)

                # Save state after successful execution
                # Rust handles optimistic locking (version check)
                try:
                    new_version = await adapter.save_state(
                        entity_type,
                        self._key,
                        state_dict,
                        current_version
                    )
                    logger.info(
                        "Saved state for %s:%s (version %d -> %d)",
                        entity_type, self._key, current_version, new_version
                    )
                except Exception as e:
                    logger.error(
                        "Failed to save state for %s:%s: %s",
                        entity_type, self._key, e
                    )
                    # Don't fail the method execution just because persistence failed
                    # The state is still in the local dict for this execution

                return result

            except Exception as e:
                logger.error(
                    "Error in %s:%s.%s: %s",
                    entity_type, self._key, method.__name__, e,
                    exc_info=True
                )
                raise ExecutionError(
                    f"Entity method {method.__name__} failed: {e}"
                ) from e
            finally:
                # Clear state reference after execution
                self._state = None

    return entity_method_wrapper


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

        # State will be initialized during method execution by wrapper
        self._state = None

        logger.debug("Created Entity instance: %s:%s", self._entity_type, key)

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

        Raises:
            RuntimeError: If accessed outside of an entity method
        """
        if self._state is None:
            raise RuntimeError(
                f"Entity state can only be accessed within entity methods.\n\n"
                f"You tried to access state on {self._entity_type}(key='{self._key}') "
                f"outside of a method call.\n\n"
                f"❌ Wrong:\n"
                f"  cart = ShoppingCart(key='user-123')\n"
                f"  items = cart.state.get('items')  # Error!\n\n"
                f"✅ Correct:\n"
                f"  class ShoppingCart(Entity):\n"
                f"      async def get_items(self):\n"
                f"          return self.state.get('items', {{}})  # Works!\n\n"
                f"  cart = ShoppingCart(key='user-123')\n"
                f"  items = await cart.get_items()  # Call method instead"
            )

        # Type narrowing: after the raise, self._state is guaranteed to be not None
        assert self._state is not None
        return self._state

    @property
    def key(self) -> str:
        """Get the entity instance key."""
        return self._key

    @property
    def entity_type(self) -> str:
        """Get the entity type name."""
        return self._entity_type

    def __init_subclass__(cls, **kwargs):
        """
        Auto-register Entity subclasses and wrap methods.

        This is called automatically when a class inherits from Entity.
        It performs three tasks:
        1. Extracts state schema from the class
        2. Wraps all public async methods with single-writer consistency
        3. Registers the entity type with metadata for platform discovery
        """
        super().__init_subclass__(**kwargs)

        # Don't register the base Entity class itself
        if cls.__name__ == 'Entity':
            return

        # Don't register SDK's built-in base classes (these are meant to be extended by users)
        if cls.__name__ in ('SessionEntity', 'MemoryEntity'):
            return

        # Create an EntityType for this class, storing the class reference
        entity_type = EntityType(cls.__name__, entity_class=cls)

        # Extract and set state schema
        state_schema = extract_state_schema(cls)
        if state_schema:
            entity_type.set_state_schema(state_schema)
            logger.debug(f"Extracted state schema for {cls.__name__}")

        # Wrap all public async methods and register them
        for name, method in inspect.getmembers(cls, predicate=inspect.iscoroutinefunction):
            if not name.startswith('_'):
                # Extract schemas from the method
                input_schema, output_schema = extract_function_schemas(method)
                method_metadata = extract_function_metadata(method)

                # Store in entity type
                entity_type._method_schemas[name] = (input_schema, output_schema)
                entity_type._method_metadata[name] = method_metadata

                # Wrap the method with single-writer consistency
                # This happens once at class definition time (not per-call)
                wrapped_method = _create_entity_method_wrapper(cls.__name__, method)
                setattr(cls, name, wrapped_method)

        # Register the entity type
        EntityRegistry.register(entity_type)
        logger.debug(f"Auto-registered Entity subclass: {cls.__name__}")
