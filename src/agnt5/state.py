"""
Simple state management API for coordination and control flow.

State API provides scoped key-value storage for:
- Run-scoped state (ephemeral, cleared when run completes)
- Session-scoped state (multi-turn conversation state)
- User-scoped state (long-term user preferences)

This is NOT for domain data - use Entity API or Memory API for that.
State is for coordination: phase tracking, flags, counters, simple config.

Example:
    # Run-scoped state (coordination within single workflow)
    ctx.state.set("phase", "research")
    phase = ctx.state.get("phase")

    # Session-scoped state (multi-turn conversation)
    ctx.session.state.set("context", {"topic": "travel"})

    # User-scoped state (long-term preferences)
    ctx.user.state.set("theme", "dark")
"""

import logging
from typing import Any, Dict, Optional

from .entity import EntityStateAdapter

logger = logging.getLogger(__name__)


class StateManager:
    """
    Simple key-value state manager for coordination and control flow.

    Uses the state_store table (formerly entities) under the hood with:
    - entity_type: "state"
    - entity_key: "kv" (single KV store per scope)
    - Stores all keys in one JSON object for simplicity

    This is intentionally simple - just get/set/delete operations.
    For complex state with business logic, use Entity API instead.
    """

    def __init__(
        self,
        state_adapter: EntityStateAdapter,
        scope: str,
        scope_id: str,
    ):
        """
        Initialize state manager.

        Args:
            state_adapter: Entity state adapter for platform communication
            scope: State scope ("run", "session", "user", "global")
            scope_id: Scope identifier (run_id, session_id, user_id)
        """
        self._state_adapter = state_adapter
        self._scope = scope
        self._scope_id = scope_id

        # In-memory cache of state (loaded lazily)
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_loaded = False

        logger.debug(
            f"Created StateManager(scope={scope}, scope_id={scope_id[:8]}...)"
        )

    async def _ensure_loaded(self) -> None:
        """Lazy-load state from platform on first access."""
        if self._cache_loaded:
            return

        # Load state from platform
        # Use entity_type="state", entity_key="kv" for all state
        state = await self._state_adapter.load_state(
            entity_type="state",
            entity_key="kv",
            scope=self._scope,
            scope_id=self._scope_id,
        )

        self._cache = state if state else {}
        self._cache_loaded = True

        logger.debug(
            f"Loaded state for {self._scope}:{self._scope_id[:8]}... "
            f"({len(self._cache)} keys)"
        )

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value by key.

        Args:
            key: State key
            default: Default value if key doesn't exist

        Returns:
            State value or default

        Example:
            phase = await ctx.state.get("phase", "init")
            count = await ctx.state.get("iteration_count", 0)
        """
        await self._ensure_loaded()
        return self._cache.get(key, default)  # type: ignore

    async def set(self, key: str, value: Any) -> None:
        """
        Set state value by key.

        Args:
            key: State key
            value: State value (must be JSON-serializable)

        Example:
            await ctx.state.set("phase", "research")
            await ctx.state.set("agents_completed", ["researcher", "writer"])
        """
        await self._ensure_loaded()

        # Update in-memory cache
        self._cache[key] = value  # type: ignore

        # Save to platform
        await self._save()

        logger.debug(f"Set state key '{key}' in {self._scope}:{self._scope_id[:8]}...")

    async def delete(self, key: str) -> None:
        """
        Delete state key.

        Args:
            key: State key to delete

        Example:
            await ctx.state.delete("temp_data")
        """
        await self._ensure_loaded()

        if key in self._cache:  # type: ignore
            del self._cache[key]  # type: ignore
            await self._save()
            logger.debug(
                f"Deleted state key '{key}' from {self._scope}:{self._scope_id[:8]}..."
            )

    async def clear(self) -> None:
        """
        Clear all state keys.

        Example:
            await ctx.state.clear()
        """
        await self._ensure_loaded()
        self._cache = {}  # type: ignore
        await self._save()
        logger.debug(f"Cleared all state for {self._scope}:{self._scope_id[:8]}...")

    async def keys(self) -> list:
        """
        Get list of all state keys.

        Returns:
            List of state keys

        Example:
            all_keys = await ctx.state.keys()
        """
        await self._ensure_loaded()
        return list(self._cache.keys())  # type: ignore

    async def _save(self) -> None:
        """Save state to platform with optimistic locking."""
        # Load current version
        current_state, current_version = await self._state_adapter.load_with_version(
            entity_type="state",
            entity_key="kv",
            scope=self._scope,
            scope_id=self._scope_id,
        )

        # Save with version check
        # Note: This may fail with version conflict if another worker modified state
        # In practice, state is typically worker-local so conflicts are rare
        try:
            await self._state_adapter.save_state(
                entity_type="state",
                entity_key="kv",
                state=self._cache,  # type: ignore
                expected_version=current_version,
                scope=self._scope,
                scope_id=self._scope_id,
            )
        except Exception as e:
            logger.error(
                f"Failed to save state for {self._scope}:{self._scope_id[:8]}...: {e}"
            )
            raise


class SessionContext:
    """
    Session context with state property.

    Provides access to session-scoped state that persists across
    multiple turns in a conversation.

    Example:
        await ctx.session.state.set("context", {"topic": "AI"})
        context = await ctx.session.state.get("context")
    """

    def __init__(self, state_adapter: EntityStateAdapter, session_id: str):
        """
        Initialize session context.

        Args:
            state_adapter: Entity state adapter
            session_id: Session identifier
        """
        self._state_adapter = state_adapter
        self._session_id = session_id
        self._state_manager: Optional[StateManager] = None

    @property
    def state(self) -> StateManager:
        """Get session-scoped state manager."""
        if not self._state_manager:
            self._state_manager = StateManager(
                state_adapter=self._state_adapter,
                scope="session",
                scope_id=self._session_id,
            )
        return self._state_manager


class UserContext:
    """
    User context with state property.

    Provides access to user-scoped state that persists across
    all sessions for a specific user.

    Example:
        await ctx.user.state.set("theme", "dark")
        theme = await ctx.user.state.get("theme", "light")
    """

    def __init__(self, state_adapter: EntityStateAdapter, user_id: str):
        """
        Initialize user context.

        Args:
            state_adapter: Entity state adapter
            user_id: User identifier
        """
        self._state_adapter = state_adapter
        self._user_id = user_id
        self._state_manager: Optional[StateManager] = None

    @property
    def state(self) -> StateManager:
        """Get user-scoped state manager."""
        if not self._state_manager:
            self._state_manager = StateManager(
                state_adapter=self._state_adapter,
                scope="user",
                scope_id=self._user_id,
            )
        return self._state_manager
