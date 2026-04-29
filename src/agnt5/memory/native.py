"""AGNT5-native semantic memory service wrappers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional, Protocol, Sequence

from .types import MemoryRecord, MemorySearchResult

logger = logging.getLogger(__name__)


class MemoryService(Protocol):
    """Protocol implemented by AGNT5-native memory service clients."""

    async def save(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        scope: str,
        scope_id: str,
        content: str,
        kind: str = "custom",
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        source_run_id: Optional[str] = None,
        source_event_id: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        """Save a semantic memory record, or return None when disabled."""
        ...

    async def search(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        query: str,
        scope_filters: Sequence[tuple[str, str]],
        kinds: Optional[Sequence[str]] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> list[MemorySearchResult]:
        """Search semantic memory, returning an empty list when disabled."""
        ...

    async def forget(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        memory_id: Optional[str] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        source_run_id: Optional[str] = None,
    ) -> int:
        """Forget matching semantic memory records."""
        ...


class DisabledMemoryService:
    """Best-effort disabled memory service."""

    def __init__(self, reason: str = "memory service unavailable") -> None:
        self.reason = reason

    async def save(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        scope: str,
        scope_id: str,
        content: str,
        kind: str = "custom",
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        source_run_id: Optional[str] = None,
        source_event_id: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        return None

    async def search(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        query: str,
        scope_filters: Sequence[tuple[str, str]],
        kinds: Optional[Sequence[str]] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> list[MemorySearchResult]:
        return []

    async def forget(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        memory_id: Optional[str] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        source_run_id: Optional[str] = None,
    ) -> int:
        return 0


class NativeMemoryService:
    """Lazy wrapper around the Rust AGNT5 MemoryRuntime."""

    def __init__(self) -> None:
        self._runtime: Any = None
        self._disabled: Optional[DisabledMemoryService] = None
        self._lock = asyncio.Lock()

    async def _get_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self._disabled is not None:
            return None

        async with self._lock:
            if self._runtime is not None:
                return self._runtime
            if self._disabled is not None:
                return None

            try:
                from .. import _core
            except ImportError as exc:
                return self._disable_or_raise(f"Rust extension unavailable: {exc}")

            runtime_cls = getattr(_core, "Agnt5MemoryRuntime", None)
            if runtime_cls is None:
                return self._disable_or_raise("Rust extension does not expose Agnt5MemoryRuntime")

            try:
                runtime = await runtime_cls.from_env()
            except Exception as exc:
                return self._disable_or_raise(str(exc))

            self._runtime = runtime
            if not getattr(runtime, "is_available", False):
                reason = getattr(runtime, "unavailable_reason", None) or "memory disabled"
                logger.debug("AGNT5 memory runtime unavailable: %s", reason)
            return runtime

    def _disable_or_raise(self, reason: str) -> None:
        if _require_memory_or_fail():
            raise RuntimeError(reason)
        logger.debug("Disabling best-effort AGNT5 memory: %s", reason)
        self._disabled = DisabledMemoryService(reason)
        return None

    async def save(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        scope: str,
        scope_id: str,
        content: str,
        kind: str = "custom",
        metadata: Optional[dict[str, Any]] = None,
        memory_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        source_run_id: Optional[str] = None,
        source_event_id: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        runtime = await self._get_runtime()
        if runtime is None:
            return None

        native_record = await runtime.save(
            tenant_id,
            deployment_id,
            scope,
            scope_id,
            content,
            kind,
            json.dumps(metadata or {}),
            memory_id,
            source_session_id,
            source_run_id,
            source_event_id,
        )
        if native_record is None:
            return None
        return _record_from_native(native_record)

    async def search(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        query: str,
        scope_filters: Sequence[tuple[str, str]],
        kinds: Optional[Sequence[str]] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> list[MemorySearchResult]:
        runtime = await self._get_runtime()
        if runtime is None:
            return []

        native_results = await runtime.search(
            tenant_id,
            deployment_id,
            query,
            list(scope_filters),
            list(kinds or []),
            limit,
            min_score,
        )
        return [_search_result_from_native(result) for result in native_results]

    async def forget(
        self,
        *,
        tenant_id: str,
        deployment_id: str,
        memory_id: Optional[str] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
        source_run_id: Optional[str] = None,
    ) -> int:
        runtime = await self._get_runtime()
        if runtime is None:
            return 0

        return int(
            await runtime.forget(
                tenant_id,
                deployment_id,
                memory_id,
                scope,
                scope_id,
                source_run_id,
            )
        )


def default_tenant_id() -> str:
    """Resolve tenant/project identity for standalone SDK memory calls."""
    return os.getenv("AGNT5_TENANT_ID") or os.getenv("AGNT5_PROJECT_ID") or "default"


def default_deployment_id() -> str:
    """Resolve deployment identity for standalone SDK memory calls."""
    return os.getenv("AGNT5_DEPLOYMENT_ID") or "default"


def _require_memory_or_fail() -> bool:
    policy = os.getenv("AGNT5_MEMORY_FAILURE_POLICY", "best_effort").strip().lower()
    return policy in {"require_memory_or_fail", "require-memory-or-fail"}


def _record_from_native(native_record: Any) -> MemoryRecord:
    return MemoryRecord(
        id=native_record.id,
        tenant_id=native_record.tenant_id,
        deployment_id=native_record.deployment_id,
        scope=native_record.scope,
        scope_id=native_record.scope_id,
        kind=native_record.kind,
        content=native_record.content,
        metadata=json.loads(native_record.metadata_json or "{}"),
        embedding_model=native_record.embedding_model,
        embedding_dim=native_record.embedding_dim,
        source_session_id=native_record.source_session_id,
        source_run_id=native_record.source_run_id,
        source_event_id=native_record.source_event_id,
        created_at=native_record.created_at,
        updated_at=native_record.updated_at,
    )


def _search_result_from_native(native_result: Any) -> MemorySearchResult:
    return MemorySearchResult(
        record=_record_from_native(native_result.record),
        score=float(native_result.score),
        distance=float(native_result.distance),
    )
