"""
Internal state adapter infrastructure for AGNT5 SDK.

This module provides the core state management infrastructure used by
workflows, agents, and other components. It is internal and not part
of the public API.
"""

import asyncio
import contextvars
import functools
import hashlib
import json
from dataclasses import asdict, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

try:
    from pydantic import BaseModel as PydanticBaseModel
    HAS_PYDANTIC = True
except ImportError:
    PydanticBaseModel = None  # type: ignore
    HAS_PYDANTIC = False

from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# TypeVar for generic state types
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


# ============================================================================
# Context Variable for State Adapter
# ============================================================================

# Context variable for worker-scoped state adapter
# This is set by Worker before execution and accessed by workflows/agents
_state_adapter_ctx: contextvars.ContextVar[Optional["StateAdapter"]] = \
    contextvars.ContextVar('_state_adapter', default=None)


class StateAdapter:
    """
    Thin Python adapter providing Pythonic interface to Rust EntityStateManager core.

    This adapter provides language-specific concerns only:
    - Worker-local asyncio.Lock for coarse-grained coordination
    - Type conversions between Python dict and JSON bytes
    - Pythonic async/await API over Rust core

    All business logic (caching, version tracking, retry logic, gRPC) lives in the Rust core.
    This keeps the Python layer simple and enables sharing business logic across SDKs.
    """

    def __init__(self, rust_core=None):
        """
        Initialize state adapter.

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
            self._standalone_states: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
            self._standalone_versions: Dict[Tuple[str, str, str, str], int] = {}
            logger.debug("Created StateAdapter in standalone mode (in-memory state)")
        else:
            logger.debug("Created StateAdapter with Rust core")

    @staticmethod
    def _scoped_state_key(
        entity_type: str,
        entity_key: str,
        scope: str = "global",
        scope_id: str = "",
    ) -> Tuple[str, str, str, str]:
        """Build the same logical identity used by platform entity state."""
        return (entity_type, entity_key, scope or "global", scope_id or "")

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
        Load state (Rust handles cache-first logic and platform load).

        In standalone mode (no Rust core), uses in-memory state storage.

        Args:
            entity_type: Type of entity (e.g., "WorkflowEntity", "Counter")
            entity_key: Unique key for entity instance
            scope: Entity scope ("global", "session", "run", "user")
            scope_id: Scope identifier (session_id, run_id, user_id) - empty for global

        Returns:
            State dictionary (empty dict if not found)
        """
        if not self._rust_core:
            # Standalone mode - return from in-memory storage
            state_key = self._scoped_state_key(entity_type, entity_key, scope, scope_id)
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
        Save state (Rust handles version check and platform save).

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
            state_key = self._scoped_state_key(entity_type, entity_key, scope, scope_id)
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
        Load state with version (for update operations).

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
            state_key = self._scoped_state_key(entity_type, entity_key, scope, scope_id)
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
            logger.error(f"Failed to load state with version for {entity_type}:{entity_key}: {e}")
            raise RuntimeError(
                f"Failed to load durable state for {entity_type}:{entity_key}: {e}"
            ) from e

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

    # -------------------------------------------------------------------
    # Session / Message operations
    # -------------------------------------------------------------------

    @property
    def has_native_messages(self) -> bool:
        """Whether native session/message operations are available."""
        return self._rust_core is not None

    async def send_message(
        self,
        correlation_id: str,
        message_type: str,
        payload: bytes,
    ) -> str:
        """Send a message via native runtime path.

        Args:
            correlation_id: Conversation thread ID (= session_id)
            message_type: "user", "assistant", "system", "tool"
            payload: JSON-encoded message content

        Returns:
            Message ID assigned by the runtime
        """
        if not self._rust_core:
            raise RuntimeError("Native message operations require Rust core (not in standalone mode)")
        return await self._rust_core.py_send_message(correlation_id, message_type, payload)

    async def list_messages(
        self,
        correlation_id: str,
        limit: int = 50,
    ) -> list:
        """List messages via native runtime path.

        Args:
            correlation_id: Conversation thread ID
            limit: Max messages to return

        Returns:
            List of proto-encoded message bytes
        """
        if not self._rust_core:
            raise RuntimeError("Native message operations require Rust core (not in standalone mode)")
        return await self._rust_core.py_list_messages(correlation_id, limit)

    async def create_session(
        self,
        session_id: str,
        component_name: str,
        session_type: str = "conversation",
    ) -> str:
        """Create a session via native runtime path.

        Args:
            session_id: Session identifier
            component_name: Agent or workflow name
            session_type: Session type

        Returns:
            Session ID
        """
        if not self._rust_core:
            raise RuntimeError("Native session operations require Rust core (not in standalone mode)")
        return await self._rust_core.py_create_session(session_id, component_name, session_type)

    def clear_all(self) -> None:
        """Clear all standalone state and local locks (for testing)."""
        self._local_locks.clear()
        if hasattr(self, '_standalone_states'):
            self._standalone_states.clear()
            self._standalone_versions.clear()
        logger.debug("Cleared StateAdapter local state and locks")

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
        for (etype, ekey, _scope, _scope_id) in self._standalone_states.keys():
            if etype == entity_type:
                keys.append(ekey)
        return keys


def _get_state_adapter() -> StateAdapter:
    """
    Get the current state adapter from context.

    The state adapter must be set by Worker before execution.
    This ensures proper worker-scoped state isolation.

    Returns:
        StateAdapter instance

    Raises:
        RuntimeError: If called outside of Worker context (state adapter not set)
    """
    adapter = _state_adapter_ctx.get()
    if adapter is None:
        raise RuntimeError(
            "State adapter context required.\n\n"
            "In production:\n"
            "  Workflows run automatically through Worker.\n\n"
            "In tests, use one of:\n"
            "  Option 1 - Decorator:\n"
            "    @with_state_context\n"
            "    async def test_workflow():\n"
            "        # Your test code\n\n"
            "  Option 2 - Fixture:\n"
            "    async def test_workflow(state_context):\n"
            "        # Your test code\n\n"
            "See: https://docs.agnt5.dev/sdk/workflows#testing"
        )
    return adapter


# ============================================================================
# Testing Helpers
# ============================================================================

def with_state_context(func):
    """
    Decorator that sets up state adapter for tests.

    Usage:
        @with_state_context
        async def test_workflow():
            # Test code that uses state
            pass
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        adapter = StateAdapter()
        token = _state_adapter_ctx.set(adapter)
        try:
            return await func(*args, **kwargs)
        finally:
            _state_adapter_ctx.reset(token)
            adapter.clear_all()
    return wrapper


def create_state_context():
    """
    Create a state context for testing (can be used as pytest fixture).

    Usage in conftest.py or test file:
        import pytest
        from agnt5._state_adapter import create_state_context

        @pytest.fixture
        def state_context():
            adapter, token = create_state_context()
            yield adapter
            # Cleanup happens automatically

    Returns:
        Tuple of (StateAdapter, context_token)
    """
    adapter = StateAdapter()
    token = _state_adapter_ctx.set(adapter)
    return adapter, token


# ============================================================================
# State Interface
# ============================================================================

class StateInterface:
    """
    Simple state interface for workflow/entity instances.

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

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying state dictionary."""
        return self._state


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# These aliases maintain backward compatibility for internal code
# that was using the old names from entity.py
EntityStateAdapter = StateAdapter
_entity_state_adapter_ctx = _state_adapter_ctx
EntityState = StateInterface
