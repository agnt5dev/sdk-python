"""KV Memory — scoped key-value storage backed by CF_STATE.

Provides simple get/set/delete/keys/clear operations with automatic
scoping per session, user, run, or global.

Uses the existing StateAdapter -> EntityStateLoad/Save path,
which means zero runtime changes are needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .types import MemoryScope

logger = logging.getLogger(__name__)


class KVMemory:
    """Scoped key-value memory backed by the runtime's entity state (CF_STATE).

    Each KVMemory instance is bound to a scope (session, user, run, global)
    and stores all keys under a single entity state entry. Keys are
    JSON-serializable values.

    Example:
        ```python
        kv = KVMemory(state_adapter, scope=MemoryScope.SESSION, scope_id="sess-123")
        await kv.set("theme", "dark")
        theme = await kv.get("theme")  # "dark"
        await kv.delete("theme")
        ```
    """

    # Entity type used in CF_STATE for memory entries
    ENTITY_TYPE = "memory"
    # Entity key — all KV pairs stored in a single blob per scope
    ENTITY_KEY = "kv"

    def __init__(
        self,
        state_adapter,
        scope: MemoryScope,
        scope_id: str,
    ) -> None:
        """Initialize KV memory for a given scope.

        Args:
            state_adapter: StateAdapter instance for platform communication
            scope: Memory scope (SESSION, USER, RUN, GLOBAL)
            scope_id: Scope identifier (session_id, user_id, run_id, or "")
        """
        self._state_adapter = state_adapter
        self._scope = scope
        self._scope_id = scope_id

        # Lazy-loaded cache
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_loaded = False

    async def _ensure_loaded(self) -> None:
        """Lazy-load state from runtime on first access."""
        if self._cache_loaded:
            return

        state = await self._state_adapter.load_state(
            entity_type=self.ENTITY_TYPE,
            entity_key=self.ENTITY_KEY,
            scope=self._scope.value,
            scope_id=self._scope_id,
        )

        self._cache = state if state else {}
        self._cache_loaded = True

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key.

        Args:
            key: The key to look up
            default: Value to return if key is not found

        Returns:
            The stored value, or default if not found

        Example:
            theme = await ctx.memory.get("theme", "light")
        """
        await self._ensure_loaded()
        return self._cache.get(key, default)  # type: ignore[union-attr]

    async def set(self, key: str, value: Any) -> None:
        """Set a value by key.

        Args:
            key: The key to store under
            value: The value to store (must be JSON-serializable)

        Example:
            await ctx.memory.set("theme", "dark")
            await ctx.memory.set("preferences", {"lang": "en"})
        """
        await self._ensure_loaded()
        self._cache[key] = value  # type: ignore[index]
        await self._save()

    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: The key to delete

        Returns:
            True if the key existed, False otherwise

        Example:
            await ctx.memory.delete("temp_data")
        """
        await self._ensure_loaded()
        if key in self._cache:  # type: ignore[operator]
            del self._cache[key]  # type: ignore[index]
            await self._save()
            return True
        return False

    async def keys(self) -> List[str]:
        """Get all stored keys.

        Returns:
            List of all keys in this memory scope

        Example:
            all_keys = await ctx.memory.keys()
        """
        await self._ensure_loaded()
        return list(self._cache.keys())  # type: ignore[union-attr]

    async def clear(self) -> None:
        """Clear all keys in this memory scope.

        Example:
            await ctx.memory.clear()
        """
        await self._ensure_loaded()
        self._cache = {}
        await self._save()

    async def _save(self) -> None:
        """Persist current cache to runtime with optimistic locking."""
        _, current_version = await self._state_adapter.load_with_version(
            entity_type=self.ENTITY_TYPE,
            entity_key=self.ENTITY_KEY,
            scope=self._scope.value,
            scope_id=self._scope_id,
        )

        try:
            await self._state_adapter.save_state(
                entity_type=self.ENTITY_TYPE,
                entity_key=self.ENTITY_KEY,
                state=self._cache,
                expected_version=current_version,
                scope=self._scope.value,
                scope_id=self._scope_id,
            )
        except Exception as e:
            logger.error(
                f"Failed to save KV memory ({self._scope.value}:{self._scope_id[:8]}...): {e}"
            )
            raise
