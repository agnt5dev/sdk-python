"""Memory accessors — facades attached to ctx.memory and ctx.conversation.

MemoryAccessor provides unified access to all memory types:
- KV shortcuts: ctx.memory.get/set/delete (session-scoped by default)
- Scoped accessors: ctx.memory.user(), ctx.memory.run(), etc.
- Working memory: ctx.memory.working
- Semantic memory: ctx.memory.semantic (if configured)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

from .kv import KVMemory
from .native import (
    MemoryService,
    NativeMemoryService,
    default_deployment_id,
    default_tenant_id,
)
from .semantic import SemanticMemoryProvider
from .types import MemoryRecord, MemoryScope, MemorySearchResult
from .working import WorkingMemory

logger = logging.getLogger(__name__)


class ScopedMemoryAccessor:
    """Hybrid scoped memory accessor.

    Calling the object returns the legacy scoped KV memory, preserving
    ``ctx.memory.user().set(...)``. Attribute access exposes semantic memory
    operations such as ``ctx.memory.user.save(...)``.
    """

    def __init__(
        self,
        *,
        parent: "MemoryAccessor",
        scope: MemoryScope,
        scope_id: Optional[str],
        kv_factory: Callable[[], KVMemory],
        missing_scope_message: Optional[str] = None,
    ) -> None:
        self._parent = parent
        self._scope = scope
        self._scope_id = scope_id
        self._kv_factory = kv_factory
        self._missing_scope_message = missing_scope_message

    def __call__(self) -> KVMemory:
        """Return the legacy scoped KV memory."""
        self._require_scope_id()
        return self._kv_factory()

    async def save(
        self,
        content: str,
        *,
        kind: str = "custom",
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        source_event_id: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        """Save content into semantic memory for this scope.

        Returns None when memory is disabled and the failure policy is
        ``best_effort``.
        """
        scope_id = self._require_scope_id()
        return await self._parent._memory_service.save(
            tenant_id=self._parent.tenant_id,
            deployment_id=self._parent.deployment_id,
            scope=self._semantic_scope,
            scope_id=scope_id,
            content=content,
            kind=kind,
            metadata=metadata,
            memory_id=memory_id,
            source_session_id=self._parent.session_id,
            source_run_id=self._parent.run_id,
            source_event_id=source_event_id,
        )

    async def search(
        self,
        query: str,
        *,
        kinds: Optional[Sequence[str]] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> list[MemorySearchResult]:
        """Search semantic memory for this scope."""
        scope_id = self._require_scope_id()
        return await self._parent._memory_service.search(
            tenant_id=self._parent.tenant_id,
            deployment_id=self._parent.deployment_id,
            query=query,
            scope_filters=[(self._semantic_scope, scope_id)],
            kinds=kinds,
            limit=limit,
            min_score=min_score,
        )

    async def forget(
        self,
        memory_id: Optional[str] = None,
        *,
        source_run_id: Optional[str] = None,
    ) -> int:
        """Forget semantic memory in this scope."""
        scope_id = self._require_scope_id()
        return await self._parent._memory_service.forget(
            tenant_id=self._parent.tenant_id,
            deployment_id=self._parent.deployment_id,
            memory_id=memory_id,
            scope=self._semantic_scope,
            scope_id=scope_id,
            source_run_id=source_run_id,
        )

    @property
    def scope(self) -> str:
        return self._semantic_scope

    @property
    def scope_id(self) -> Optional[str]:
        return self._scope_id

    @property
    def _semantic_scope(self) -> str:
        if self._scope == MemoryScope.GLOBAL:
            return "app"
        return self._scope.value

    def _require_scope_id(self) -> str:
        if self._scope_id is None:
            raise RuntimeError(self._missing_scope_message or "memory scope_id is required")
        return self._scope_id


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
        memory_service: Optional[MemoryService] = None,
        tenant_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ) -> None:
        """Initialize memory accessor.

        Args:
            state_adapter: StateAdapter for platform state operations
            session_id: Session identifier (used for default scope)
            user_id: Optional user identifier (for user-scoped memory)
            run_id: Optional run identifier (for run-scoped memory)
            semantic_provider: Optional semantic memory provider
            memory_service: Optional AGNT5-native memory service
            tenant_id: Tenant/project identity for semantic memory
            deployment_id: Deployment identity for semantic memory
        """
        self._state_adapter = state_adapter
        self._session_id = session_id
        self._user_id = user_id
        self._run_id = run_id
        self._semantic_provider = semantic_provider
        self._memory_service = memory_service or NativeMemoryService()
        self._tenant_id = tenant_id or default_tenant_id()
        self._deployment_id = deployment_id or default_deployment_id()

        # Lazily created scoped KV instances
        self._session_kv: Optional[KVMemory] = None
        self._user_kv: Optional[KVMemory] = None
        self._run_kv: Optional[KVMemory] = None
        self._global_kv: Optional[KVMemory] = None
        self._working: Optional[WorkingMemory] = None
        self._session_accessor: Optional[ScopedMemoryAccessor] = None
        self._user_accessor: Optional[ScopedMemoryAccessor] = None
        self._run_accessor: Optional[ScopedMemoryAccessor] = None
        self._app_accessor: Optional[ScopedMemoryAccessor] = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def user_id(self) -> Optional[str]:
        return self._user_id

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def deployment_id(self) -> str:
        return self._deployment_id

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

    @property
    def session(self) -> ScopedMemoryAccessor:
        """Get session-scoped memory.

        Use ``ctx.memory.session()`` for legacy KV memory or
        ``ctx.memory.session.save/search`` for semantic memory.
        """
        if self._session_accessor is None:
            self._session_accessor = ScopedMemoryAccessor(
                parent=self,
                scope=MemoryScope.SESSION,
                scope_id=self._session_id,
                kv_factory=self._default_kv,
            )
        return self._session_accessor

    @property
    def user(self) -> ScopedMemoryAccessor:
        """Get user-scoped memory.

        Use ``ctx.memory.user()`` for legacy KV memory or
        ``ctx.memory.user.save/search`` for semantic memory.
        """
        if self._user_accessor is None:
            self._user_accessor = ScopedMemoryAccessor(
                parent=self,
                scope=MemoryScope.USER,
                scope_id=self._user_id,
                kv_factory=self._user_kv_memory,
                missing_scope_message=(
                    "User-scoped memory requires a user_id. "
                    "Pass user_id when creating the context or workflow."
                ),
            )
        return self._user_accessor

    @property
    def run(self) -> ScopedMemoryAccessor:
        """Get run-scoped KV memory (ephemeral, cleared after run)."""
        if self._run_accessor is None:
            self._run_accessor = ScopedMemoryAccessor(
                parent=self,
                scope=MemoryScope.RUN,
                scope_id=self._run_id,
                kv_factory=self._run_kv_memory,
                missing_scope_message="Run-scoped memory requires a run_id.",
            )
        return self._run_accessor

    def _user_kv_memory(self) -> KVMemory:
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

    def _run_kv_memory(self) -> KVMemory:
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

    @property
    def app(self) -> ScopedMemoryAccessor:
        """Get app-scoped semantic memory.

        App memory is shared across the configured tenant/deployment. KV users
        should keep using ``ctx.memory.global_()``.
        """
        if self._app_accessor is None:
            self._app_accessor = ScopedMemoryAccessor(
                parent=self,
                scope=MemoryScope.GLOBAL,
                scope_id="",
                kv_factory=self.global_,
            )
        return self._app_accessor

    async def search(
        self,
        query: str,
        *,
        kinds: Optional[Sequence[str]] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> list[MemorySearchResult]:
        """Search session memory and user memory when a user_id is available."""
        scope_filters = [(MemoryScope.SESSION.value, self._session_id)]
        if self._user_id:
            scope_filters.append((MemoryScope.USER.value, self._user_id))
        return await self._memory_service.search(
            tenant_id=self._tenant_id,
            deployment_id=self._deployment_id,
            query=query,
            scope_filters=scope_filters,
            kinds=kinds,
            limit=limit,
            min_score=min_score,
        )

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
