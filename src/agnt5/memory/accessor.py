"""Memory accessors — facades attached to ctx.memory and ctx.conversation.

MemoryAccessor provides unified access to all memory types:
- KV shortcuts: ctx.memory.get/set/delete (session-scoped by default)
- Scoped accessors: ctx.memory.user(), ctx.memory.run(), etc.
- Working memory: ctx.memory.working
- Semantic memory: ctx.memory.semantic (if configured)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from .kv import KVMemory
from .semantic import SemanticMemoryProvider
from .types import MemoryScope
from .working import WorkingMemory

logger = logging.getLogger(__name__)


class MemoryAccessor:
    """Unified memory accessor attached to ``ctx.memory``.

    Provides KV shortcuts that default to session scope, plus explicit
    scoped accessors for user/run/global memory.

    Example:
        ```python
        # Session-scoped KV (default)
        await ctx.memory.set("theme", "dark")
        theme = await ctx.memory.get("theme", "light")

        # User-scoped KV
        await ctx.memory.user().set("lang", "en")

        # Working memory
        wm = await ctx.memory.working.get()

        # Semantic (if configured)
        if ctx.memory.semantic:
            results = await ctx.memory.semantic.search("preferences")
        ```
    """

    def __init__(
        self,
        state_adapter,
        session_id: str,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        semantic_provider: Optional[SemanticMemoryProvider] = None,
    ) -> None:
        """Initialize memory accessor.

        Args:
            state_adapter: StateAdapter for platform state operations
            session_id: Session identifier (used for default scope)
            user_id: Optional user identifier (for user-scoped memory)
            run_id: Optional run identifier (for run-scoped memory)
            semantic_provider: Optional semantic memory provider
        """
        self._state_adapter = state_adapter
        self._session_id = session_id
        self._user_id = user_id
        self._run_id = run_id
        self._semantic_provider = semantic_provider

        # Lazily created scoped KV instances
        self._session_kv: Optional[KVMemory] = None
        self._user_kv: Optional[KVMemory] = None
        self._run_kv: Optional[KVMemory] = None
        self._global_kv: Optional[KVMemory] = None
        self._working: Optional[WorkingMemory] = None

    # === KV shortcuts (session-scoped by default) ===

    def _default_kv(self) -> KVMemory:
        """Get the default (session-scoped) KV memory."""
        if self._session_kv is None:
            self._session_kv = KVMemory(
                self._state_adapter,
                scope=MemoryScope.SESSION,
                scope_id=self._session_id,
            )
        return self._session_kv

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from session-scoped memory.

        Args:
            key: The key to look up
            default: Value to return if not found

        Returns:
            The stored value, or default
        """
        return await self._default_kv().get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """Set a value in session-scoped memory.

        Args:
            key: The key to store under
            value: The value to store (JSON-serializable)
        """
        await self._default_kv().set(key, value)

    async def delete(self, key: str) -> bool:
        """Delete a key from session-scoped memory.

        Args:
            key: The key to delete

        Returns:
            True if the key existed
        """
        return await self._default_kv().delete(key)

    async def keys(self) -> list:
        """Get all keys in session-scoped memory."""
        return await self._default_kv().keys()

    async def clear(self) -> None:
        """Clear all session-scoped memory."""
        await self._default_kv().clear()

    # === Scoped accessors ===

    def session(self) -> KVMemory:
        """Get session-scoped KV memory (same as default)."""
        return self._default_kv()

    def user(self) -> KVMemory:
        """Get user-scoped KV memory.

        Raises:
            RuntimeError: If no user_id was provided to the context
        """
        if not self._user_id:
            raise RuntimeError(
                "User-scoped memory requires a user_id. "
                "Pass user_id when creating the context or workflow."
            )
        if self._user_kv is None:
            self._user_kv = KVMemory(
                self._state_adapter,
                scope=MemoryScope.USER,
                scope_id=self._user_id,
            )
        return self._user_kv

    def run(self) -> KVMemory:
        """Get run-scoped KV memory (ephemeral, cleared after run)."""
        if not self._run_id:
            raise RuntimeError("Run-scoped memory requires a run_id.")
        if self._run_kv is None:
            self._run_kv = KVMemory(
                self._state_adapter,
                scope=MemoryScope.RUN,
                scope_id=self._run_id,
            )
        return self._run_kv

    def global_(self) -> KVMemory:
        """Get global (tenant-wide) KV memory."""
        if self._global_kv is None:
            self._global_kv = KVMemory(
                self._state_adapter,
                scope=MemoryScope.GLOBAL,
                scope_id="",
            )
        return self._global_kv

    # === Working memory ===

    @property
    def working(self) -> WorkingMemory:
        """Get working memory (structured scratchpad).

        Working memory is session-scoped by default.
        """
        if self._working is None:
            self._working = WorkingMemory(
                self._state_adapter,
                scope=MemoryScope.SESSION,
                scope_id=self._session_id,
            )
        return self._working

    # === Semantic memory (pluggable) ===

    @property
    def semantic(self) -> Optional[SemanticMemoryProvider]:
        """Get semantic memory provider, if configured.

        Returns:
            SemanticMemoryProvider instance, or None if not configured

        Example:
            if ctx.memory.semantic:
                results = await ctx.memory.semantic.search("color preferences")
        """
        return self._semantic_provider
