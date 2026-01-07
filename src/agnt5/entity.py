"""
Entity component for stateful operations with single-writer consistency.
"""

import asyncio
import contextvars
import functools
import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    from pydantic import BaseModel as PydanticBaseModel
    HAS_PYDANTIC = True
except ImportError:
    PydanticBaseModel = None  # type: ignore
    HAS_PYDANTIC = False

# TypeVar for generic Entity[StateType] support
# StateType can be a Pydantic model, TypedDict, dataclass, or plain dict
StateType = TypeVar("StateType")


# ============================================================================
# State Type Detection and Serialization Utilities
# ============================================================================

def _is_pydantic_model(obj_or_type) -> bool:
    """Check if object or type is a Pydantic model."""
    if not HAS_PYDANTIC:
        return False
    if isinstance(obj_or_type, type):
        return issubclass(obj_or_type, PydanticBaseModel)
    return isinstance(obj_or_type, PydanticBaseModel)


def _is_typed_dict(type_hint) -> bool:
    """Check if type hint is a TypedDict."""
    if type_hint is None:
        return False
    # TypedDict classes have __annotations__ and __total__
    return (
        isinstance(type_hint, type) and
        hasattr(type_hint, '__annotations__') and
        hasattr(type_hint, '__total__')
    )


def _get_state_type_kind(state_type: Optional[Type]) -> str:
    """
    Determine the kind of state type.

    Returns one of: 'pydantic', 'dataclass', 'typed_dict', 'untyped'
    """
    if state_type is None:
        return 'untyped'
    if _is_pydantic_model(state_type):
        return 'pydantic'
    if is_dataclass(state_type):
        return 'dataclass'
    if _is_typed_dict(state_type):
        return 'typed_dict'
    return 'untyped'


def _state_to_dict(state: Any, state_type: Optional[Type]) -> Dict[str, Any]:
    """
    Convert state object to dictionary for persistence.

    Handles Pydantic models, dataclasses, TypedDicts, and plain dicts.
    """
    if state is None:
        return {}

    kind = _get_state_type_kind(state_type)

    if kind == 'pydantic' and _is_pydantic_model(state):
        return state.model_dump()
    elif kind == 'dataclass' and is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    elif isinstance(state, dict):
        return state
    else:
        # Fallback: try to convert to dict
        return dict(state) if hasattr(state, '__iter__') else {}


def _dict_to_state(data: Dict[str, Any], state_type: Optional[Type]) -> Any:
    """
    Convert dictionary to typed state object.

    Creates Pydantic model, dataclass, or returns dict based on state_type.
    """
    if state_type is None:
        return data

    kind = _get_state_type_kind(state_type)

    if kind == 'pydantic':
        return state_type(**data)
    elif kind == 'dataclass':
        # Filter to only known fields for dataclass
        known_fields = {f.name for f in dataclass_fields(state_type)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return state_type(**filtered_data)
    elif kind == 'typed_dict':
        # TypedDict is just a dict with type hints
        return data
    else:
        return data


def _compute_state_hash(state: Any, state_type: Optional[Type]) -> str:
    """
    Compute a hash of the state for mutation detection.

    Uses JSON serialization with sorted keys for deterministic hashing.
    """
    kind = _get_state_type_kind(state_type)

    if kind == 'pydantic' and _is_pydantic_model(state):
        # Pydantic has optimized JSON serialization
        json_str = state.model_dump_json(exclude_none=False)
    elif kind == 'dataclass' and is_dataclass(state) and not isinstance(state, type):
        json_str = json.dumps(asdict(state), sort_keys=True, default=str)
    elif isinstance(state, dict):
        json_str = json.dumps(state, sort_keys=True, default=str)
    else:
        # Fallback
        json_str = json.dumps(_state_to_dict(state, state_type), sort_keys=True, default=str)

    return hashlib.md5(json_str.encode()).hexdigest()

from ._schema_utils import extract_function_metadata, extract_function_schemas
from ._telemetry import setup_module_logger
from .exceptions import ExecutionError

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

    async def load_state(
        self,
        entity_type: str,
        entity_key: str,
        scope: str = "global",
        scope_id: str = "",
    ) -> Dict[str, Any]:
        """
        Load entity state (Rust handles cache-first logic and platform load).

        In standalone mode (no Rust core), uses in-memory state storage.

        Args:
            entity_type: Type of entity (e.g., "ShoppingCart", "Counter")
            entity_key: Unique key for entity instance
            scope: Entity scope ("global", "session", "run", "user")
            scope_id: Scope identifier (session_id, run_id, user_id) - empty for global

        Returns:
            State dictionary (empty dict if not found)
        """
        if not self._rust_core:
            # Standalone mode - return from in-memory storage
            state_key = (entity_type, entity_key)
            return self._standalone_states.get(state_key, {}).copy()

        try:
            # Rust checks cache first, loads from platform if needed
            state_json_bytes, version = await self._rust_core.py_get_cached_or_load(
                entity_type, entity_key, scope, scope_id
            )

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
        expected_version: int,
        scope: str = "global",
        scope_id: str = "",
    ) -> int:
        """
        Save entity state (Rust handles version check and platform save).

        In standalone mode (no Rust core), stores in-memory with version tracking.

        Args:
            entity_type: Type of entity
            entity_key: Unique key for entity instance
            state: State dictionary to save
            expected_version: Expected current version (for optimistic locking)
            scope: Entity scope ("global", "session", "run", "user")
            scope_id: Scope identifier (session_id, run_id, user_id) - empty for global

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
            expected_version,
            scope,
            scope_id,
        )

        return new_version

    async def load_with_version(
        self,
        entity_type: str,
        entity_key: str,
        scope: str = "global",
        scope_id: str = "",
    ) -> Tuple[Dict[str, Any], int]:
        """
        Load entity state with version (for update operations).

        In standalone mode (no Rust core), loads from in-memory storage with version.

        Args:
            entity_type: Type of entity
            entity_key: Unique key for entity instance
            scope: Entity scope ("global", "session", "run", "user")
            scope_id: Scope identifier (session_id, run_id, user_id) - empty for global

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
            state_json_bytes, version = await self._rust_core.py_get_cached_or_load(
                entity_type, entity_key, scope, scope_id
            )

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


class TypedEntityState(Generic[StateType]):
    """
    Typed state wrapper for Entity instances.

    Provides direct attribute access for typed state (Pydantic, dataclass)
    while maintaining compatibility with the dict-based API.

    For Pydantic/dataclass:
        self.state.items  # Direct attribute access
        self.state.total = 100.0  # Direct attribute mutation

    For untyped (backward compat):
        self.state.get("items", [])
        self.state.set("items", [...])

    The underlying state object is accessible via ._typed_state for typed,
    or ._state for the dict representation.
    """

    def __init__(
        self,
        state_dict: Dict[str, Any],
        state_type: Optional[Type[StateType]] = None
    ):
        """
        Initialize typed state wrapper.

        Args:
            state_dict: Dictionary representation of state (for persistence)
            state_type: Optional type class (Pydantic model, dataclass, etc.)
        """
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, '_state', state_dict)
        object.__setattr__(self, '_state_type', state_type)

        # Create typed state object if type is provided
        if state_type is not None:
            typed_state = _dict_to_state(state_dict, state_type)
            object.__setattr__(self, '_typed_state', typed_state)
        else:
            object.__setattr__(self, '_typed_state', None)

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from typed state or fall back to dict access.

        For typed state (Pydantic/dataclass): returns attribute directly
        For untyped: raises AttributeError (use get() instead)
        """
        # Don't intercept private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        typed_state = object.__getattribute__(self, '_typed_state')
        if typed_state is not None:
            return getattr(typed_state, name)

        # For untyped state, provide helpful error
        raise AttributeError(
            f"Untyped state does not support attribute access. "
            f"Use self.state.get('{name}') instead, or define a typed state class."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set attribute on typed state and sync to dict.

        For typed state: sets attribute and syncs to _state dict
        For untyped: raises AttributeError (use set() instead)
        """
        # Don't intercept private attributes
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        typed_state = object.__getattribute__(self, '_typed_state')
        state_type = object.__getattribute__(self, '_state_type')
        state_dict = object.__getattribute__(self, '_state')

        if typed_state is not None:
            # Set on typed state object
            setattr(typed_state, name, value)
            # Sync back to dict for persistence
            state_dict.update(_state_to_dict(typed_state, state_type))
        else:
            raise AttributeError(
                f"Untyped state does not support attribute assignment. "
                f"Use self.state.set('{name}', value) instead."
            )

    # Dict-based API (backward compatible)
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state dict."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state dict and sync to typed state if present."""
        self._state[key] = value
        # Sync to typed state if present
        if self._typed_state is not None and self._state_type is not None:
            object.__setattr__(
                self,
                '_typed_state',
                _dict_to_state(self._state, self._state_type)
            )

    def delete(self, key: str) -> None:
        """Delete key from state."""
        self._state.pop(key, None)
        # Sync to typed state if present
        if self._typed_state is not None and self._state_type is not None:
            object.__setattr__(
                self,
                '_typed_state',
                _dict_to_state(self._state, self._state_type)
            )

    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        # Reset typed state
        if self._state_type is not None:
            object.__setattr__(
                self,
                '_typed_state',
                _dict_to_state({}, self._state_type)
            )

    def _get_dict(self) -> Dict[str, Any]:
        """Get the underlying dict representation (for persistence)."""
        if self._typed_state is not None:
            # Sync from typed state to ensure dict is up-to-date
            return _state_to_dict(self._typed_state, self._state_type)
        return self._state

    def _get_typed(self) -> Optional[StateType]:
        """Get the typed state object (Pydantic model, dataclass, etc.)."""
        return self._typed_state


# Decorator for marking read-only methods (optional optimization)
def query(func):
    """
    Mark an entity method as read-only (query).

    Query methods skip hash comparison and state persistence for maximum
    performance on high-frequency reads.

    Example:
        class Counter(Entity[CounterState]):
            @query
            async def get_count(self) -> int:
                return self.state.count

    Note:
        For most cases, you don't need this decorator - the Entity automatically
        detects whether state was mutated via hash comparison. Use @query only
        for high-frequency reads where even the hash computation overhead matters.
    """
    func._agnt5_method_type = 'query'
    return func


def _create_entity_method_wrapper(entity_type: str, method, state_type: Optional[Type] = None):
    """
    Create a wrapper for an entity method that provides single-writer consistency.

    This wrapper implements:
    1. Local lock (asyncio.Lock) for worker-scoped single-writer guarantee
    2. Optimistic concurrency (via Rust) for cross-worker conflicts
    3. Loads state via adapter (Rust handles cache + platform)
    4. Executes the method with TypedEntityState interface
    5. Hash-based mutation detection - only saves if state changed
    6. Support for @query decorator to skip mutation detection entirely

    Args:
        entity_type: Name of the entity type (class name)
        method: The async method to wrap
        state_type: Optional type class for typed state (Pydantic, dataclass, etc.)

    Returns:
        Wrapped async method with single-writer consistency and mutation detection
    """
    # Check if method is marked as @query (read-only)
    is_query = getattr(method, '_agnt5_method_type', None) == 'query'

    @functools.wraps(method)
    async def entity_method_wrapper(self, *args, **kwargs):
        """Execute entity method with hybrid locking and mutation detection."""
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

            # Set up TypedEntityState on instance for method access
            # Use the class-level state type if available
            effective_state_type = state_type or getattr(self.__class__, '_state_type', None)
            self._state = TypedEntityState(state_dict, effective_state_type)

            # Compute hash before method execution (skip for @query methods)
            if not is_query and effective_state_type is not None:
                # For typed state, compute hash of the typed object
                original_hash = _compute_state_hash(
                    self._state._get_typed() or state_dict,
                    effective_state_type
                )
            elif not is_query:
                # For untyped state, compute hash of the dict
                original_hash = _compute_state_hash(state_dict, None)
            else:
                original_hash = None  # Skip for @query

            try:
                # Execute method
                logger.debug("Executing %s:%s.%s", entity_type, self._key, method.__name__)
                result = await method(self, *args, **kwargs)
                logger.debug("Completed %s:%s.%s", entity_type, self._key, method.__name__)

                # For @query methods, skip persistence entirely
                if is_query:
                    logger.debug(
                        "Skipping state save for query method %s:%s.%s",
                        entity_type, self._key, method.__name__
                    )
                    return result

                # Get current state dict (sync from typed state if needed)
                current_state_dict = self._state._get_dict()

                # Compute hash after method execution
                if effective_state_type is not None:
                    new_hash = _compute_state_hash(
                        self._state._get_typed() or current_state_dict,
                        effective_state_type
                    )
                else:
                    new_hash = _compute_state_hash(current_state_dict, None)

                # Only save if state actually changed (hash-based mutation detection)
                if new_hash != original_hash:
                    try:
                        new_version = await adapter.save_state(
                            entity_type,
                            self._key,
                            current_state_dict,
                            current_version
                        )
                        logger.info(
                            "Saved state for %s:%s (version %d -> %d, hash changed)",
                            entity_type, self._key, current_version, new_version
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to save state for %s:%s: %s",
                            entity_type, self._key, e
                        )
                        # Don't fail the method execution just because persistence failed
                else:
                    logger.debug(
                        "Skipping state save for %s:%s.%s (no mutation detected)",
                        entity_type, self._key, method.__name__
                    )

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


class Entity(Generic[StateType]):
    """
    Base class for stateful entities with single-writer consistency.

    Entities provide a class-based API where:
    - State is accessed via self.state (clean, synchronous API)
    - Methods are regular async methods on the class
    - Each instance is bound to a unique key
    - Single-writer consistency per key is guaranteed automatically

    Supports typed state with Pydantic models for IDE autocomplete and validation:
        ```python
        from agnt5 import Entity
        from pydantic import BaseModel

        class CartState(BaseModel):
            items: dict[str, dict] = {}
            total: float = 0.0

        class ShoppingCart(Entity[CartState]):
            async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
                # IDE autocomplete works!
                self.state.items[item_id] = {"quantity": quantity, "price": price}
                self.state.total = sum(
                    item["quantity"] * item["price"]
                    for item in self.state.items.values()
                )
                return {"total_items": len(self.state.items)}

            async def get_total(self) -> float:
                return self.state.total  # No save triggered (auto-detected as read)
        ```

    Also supports dataclasses:
        ```python
        from dataclasses import dataclass, field

        @dataclass
        class CounterState:
            count: int = 0
            history: list = field(default_factory=list)

        class Counter(Entity[CounterState]):
            async def increment(self) -> int:
                self.state.count += 1
                return self.state.count
        ```

    And untyped state (backward compatible):
        ```python
        class Counter(Entity):
            async def increment(self) -> int:
                count = self.state.get("count", 0) + 1
                self.state.set("count", count)
                return count
        ```

    Note:
        Methods are automatically wrapped to provide single-writer consistency per key.
        State mutations are auto-detected via hash comparison - read-only methods
        don't trigger persistence.
    """

    # Class-level state type (set by __init_subclass__)
    _state_type: Optional[Type] = None

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
        It performs four tasks:
        1. Extracts state type from generic parameter (Entity[StateType])
        2. Extracts state schema from the class or state type
        3. Wraps all public async methods with single-writer consistency
        4. Registers the entity type with metadata for platform discovery
        """
        super().__init_subclass__(**kwargs)

        # Don't register the base Entity class itself
        if cls.__name__ == 'Entity':
            return

        # Don't register SDK's built-in base classes (these are meant to be extended by users)
        if cls.__name__ in ('SessionEntity', 'MemoryEntity'):
            return

        # Extract state type from generic parameter (Entity[CartState])
        state_type = None
        if hasattr(cls, '__orig_bases__'):
            for base in cls.__orig_bases__:
                origin = get_origin(base)
                if origin is Entity or (isinstance(origin, type) and issubclass(origin, Entity)):
                    args = get_args(base)
                    if args:
                        state_type = args[0]
                        break

        # Store state type on the class for later use
        cls._state_type = state_type

        if state_type is not None:
            kind = _get_state_type_kind(state_type)
            logger.debug(
                f"Extracted state type for {cls.__name__}: {state_type.__name__ if hasattr(state_type, '__name__') else state_type} ({kind})"
            )

        # Create an EntityType for this class, storing the class reference
        entity_type = EntityType(cls.__name__, entity_class=cls)

        # Extract state schema from Pydantic model if available
        if state_type is not None and _is_pydantic_model(state_type):
            try:
                # Pydantic v2 has model_json_schema()
                pydantic_schema = state_type.model_json_schema()
                entity_type.set_state_schema(pydantic_schema)
                logger.debug(f"Extracted Pydantic state schema for {cls.__name__}")
            except Exception as e:
                logger.debug(f"Could not extract Pydantic schema for {cls.__name__}: {e}")
                # Fall back to basic schema extraction
                state_schema = extract_state_schema(cls)
                if state_schema:
                    entity_type.set_state_schema(state_schema)
        else:
            # Fall back to basic schema extraction for non-Pydantic types
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

                # Wrap the method with single-writer consistency and typed state
                # Pass state_type so wrapper can use hash-based mutation detection
                wrapped_method = _create_entity_method_wrapper(cls.__name__, method, state_type)
                setattr(cls, name, wrapped_method)

        # Register the entity type
        EntityRegistry.register(entity_type)
        logger.debug(f"Auto-registered Entity subclass: {cls.__name__}")
