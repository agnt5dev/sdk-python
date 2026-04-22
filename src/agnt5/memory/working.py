"""Working Memory — structured JSON scratchpad persisted across turns.

A single JSON document per scope that agents can read/write/merge.
Backed by CF_STATE with entity_type="working_memory".

Use cases:
- Track conversation context across turns (turn count, current task, etc.)
- Store extracted facts that the agent decides to remember
- Persist structured data that doesn't fit into simple KV pairs
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

from .types import MemoryScope

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Structured JSON scratchpad persisted across turns.

    Working memory stores a single JSON document per scope. It supports
    full replacement (set) and shallow merge (merge) operations.

    Example:
        ```python
        wm = ctx.memory.working

        # First turn — initialize
        await wm.set({"turn_count": 1, "topic": "travel"})

        # Next turn — merge updates
        await wm.merge({"turn_count": 2, "mood": "curious"})
        # Result: {"turn_count": 2, "topic": "travel", "mood": "curious"}

        # Read current state
        data = await wm.get()  # {"turn_count": 2, "topic": "travel", "mood": "curious"}

        # Clear
        await wm.clear()
        ```
    """

    ENTITY_TYPE = "working_memory"
    ENTITY_KEY = "data"

    def __init__(
        self,
        state_adapter,
        scope: MemoryScope,
        scope_id: str,
    ) -> None:
        """Initialize working memory for a given scope.

        Args:
            state_adapter: StateAdapter instance for platform communication
            scope: Memory scope (SESSION, USER, RUN, GLOBAL)
            scope_id: Scope identifier (session_id, user_id, run_id, or "")
        """
        self._state_adapter = state_adapter
        self._scope = scope
        self._scope_id = scope_id

        self._cache: Optional[Dict[str, Any]] = None
        self._cache_loaded = False

    async def _ensure_loaded(self) -> None:
        """Lazy-load from runtime on first access."""
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

    async def get(self) -> Dict[str, Any]:
        """Get the current working memory document.

        Returns:
            The working memory dict (empty dict if nothing stored yet)

        Example:
            data = await ctx.memory.working.get()
            turn = data.get("turn_count", 0)
        """
        await self._ensure_loaded()
        return copy.deepcopy(self._cache)  # type: ignore[arg-type]

    async def set(self, value: Dict[str, Any]) -> None:
        """Replace the entire working memory document.

        Args:
            value: New working memory content (must be a JSON-serializable dict)

        Example:
            await ctx.memory.working.set({"topic": "AI", "turn_count": 1})
        """
        self._cache = copy.deepcopy(value)
        self._cache_loaded = True
        await self._save()

    async def merge(self, partial: Dict[str, Any]) -> None:
        """Shallow-merge updates into working memory.

        Existing keys not in `partial` are preserved.
        Keys in `partial` with value `None` are deleted.

        Args:
            partial: Dict of keys to update/add/remove

        Example:
            await ctx.memory.working.merge({"turn_count": 2, "old_key": None})
        """
        await self._ensure_loaded()

        for key, value in partial.items():
            if value is None:
                self._cache.pop(key, None)  # type: ignore[union-attr]
            else:
                self._cache[key] = value  # type: ignore[index]

        await self._save()

    async def clear(self) -> None:
        """Clear all working memory.

        Example:
            await ctx.memory.working.clear()
        """
        self._cache = {}
        self._cache_loaded = True
        await self._save()

    async def _save(self) -> None:
        """Persist to runtime with optimistic locking."""
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
                f"Failed to save working memory ({self._scope.value}:{self._scope_id[:8]}...): {e}"
            )
            raise
