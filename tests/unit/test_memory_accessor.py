"""Unit tests for MemoryAccessor."""

import pytest

from agnt5._state_adapter import StateAdapter
from agnt5.memory import (
    KVMemory,
    MemoryAccessor,
    MemoryRecord,
    MemorySearchResult,
    SemanticMemoryProvider,
    WorkingMemory,
)


class FakeMemoryService:
    def __init__(self):
        self.saved = []
        self.searches = []

    async def save(
        self,
        *,
        tenant_id,
        deployment_id,
        scope,
        scope_id,
        content,
        kind="custom",
        metadata=None,
        memory_id=None,
        source_session_id=None,
        source_run_id=None,
        source_event_id=None,
    ):
        self.saved.append(
            {
                "tenant_id": tenant_id,
                "deployment_id": deployment_id,
                "scope": scope,
                "scope_id": scope_id,
                "content": content,
                "kind": kind,
                "metadata": metadata or {},
                "source_session_id": source_session_id,
                "source_run_id": source_run_id,
                "source_event_id": source_event_id,
            }
        )
        return MemoryRecord(
            id=memory_id or "mem-1",
            tenant_id=tenant_id,
            deployment_id=deployment_id,
            scope=scope,
            scope_id=scope_id,
            kind=kind,
            content=content,
            metadata=metadata or {},
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            source_event_id=source_event_id,
        )

    async def search(
        self,
        *,
        tenant_id,
        deployment_id,
        query,
        scope_filters,
        kinds=None,
        limit=10,
        min_score=None,
    ):
        self.searches.append(
            {
                "tenant_id": tenant_id,
                "deployment_id": deployment_id,
                "query": query,
                "scope_filters": list(scope_filters),
                "kinds": list(kinds or []),
                "limit": limit,
                "min_score": min_score,
            }
        )
        return [
            MemorySearchResult(
                record=MemoryRecord(
                    id="mem-1",
                    tenant_id=tenant_id,
                    deployment_id=deployment_id,
                    scope=scope_filters[0][0],
                    scope_id=scope_filters[0][1],
                    kind="preference",
                    content="User prefers dark mode",
                ),
                score=0.9,
                distance=0.1,
            )
        ]

    async def forget(
        self,
        *,
        tenant_id,
        deployment_id,
        memory_id=None,
        scope=None,
        scope_id=None,
        source_run_id=None,
    ):
        return 1


@pytest.fixture
def adapter():
    return StateAdapter()


@pytest.fixture
def accessor(adapter):
    return MemoryAccessor(
        state_adapter=adapter,
        session_id="sess-001",
        user_id="user-123",
        run_id="run-abc",
        memory_service=FakeMemoryService(),
        tenant_id="tenant-a",
        deployment_id="dep-1",
    )


# --- KV shortcuts (session-scoped by default) ---


@pytest.mark.asyncio
async def test_default_set_get(accessor):
    await accessor.set("key", "value")
    assert await accessor.get("key") == "value"


@pytest.mark.asyncio
async def test_default_get_with_default(accessor):
    assert await accessor.get("missing", "fallback") == "fallback"


@pytest.mark.asyncio
async def test_default_delete(accessor):
    await accessor.set("k", "v")
    assert await accessor.delete("k") is True
    assert await accessor.get("k") is None


@pytest.mark.asyncio
async def test_default_keys_and_clear(accessor):
    await accessor.set("a", 1)
    await accessor.set("b", 2)
    assert sorted(await accessor.keys()) == ["a", "b"]
    await accessor.clear()
    assert await accessor.keys() == []


# --- Scoped accessors ---


@pytest.mark.asyncio
async def test_session_returns_kv(accessor):
    kv = accessor.session()
    assert isinstance(kv, KVMemory)
    await kv.set("x", 1)
    assert await kv.get("x") == 1


@pytest.mark.asyncio
async def test_session_semantic_save_uses_session_scope(accessor):
    record = await accessor.session.save(
        "User prefers dark mode",
        kind="preference",
        metadata={"source": "test"},
    )

    assert record is not None
    assert record.scope == "session"
    assert record.scope_id == "sess-001"
    assert record.tenant_id == "tenant-a"
    assert record.deployment_id == "dep-1"
    assert record.source_session_id == "sess-001"
    assert record.source_run_id == "run-abc"


@pytest.mark.asyncio
async def test_session_semantic_search_uses_session_filter(accessor):
    results = await accessor.session.search("theme", kinds=["preference"], limit=3, min_score=0.5)

    assert len(results) == 1
    assert accessor._memory_service.searches[-1]["scope_filters"] == [("session", "sess-001")]
    assert accessor._memory_service.searches[-1]["kinds"] == ["preference"]
    assert accessor._memory_service.searches[-1]["limit"] == 3
    assert accessor._memory_service.searches[-1]["min_score"] == 0.5


@pytest.mark.asyncio
async def test_user_returns_kv(accessor):
    kv = accessor.user()
    assert isinstance(kv, KVMemory)
    await kv.set("pref", "dark")
    assert await kv.get("pref") == "dark"


@pytest.mark.asyncio
async def test_user_semantic_save_uses_user_scope(accessor):
    record = await accessor.user.save("User likes compact answers")

    assert record is not None
    assert record.scope == "user"
    assert record.scope_id == "user-123"


@pytest.mark.asyncio
async def test_user_raises_without_user_id(adapter):
    accessor = MemoryAccessor(
        state_adapter=adapter,
        session_id="s1",
        user_id=None,
        run_id="r1",
    )
    with pytest.raises(RuntimeError, match="user_id"):
        accessor.user()


@pytest.mark.asyncio
async def test_user_semantic_save_raises_without_user_id(adapter):
    accessor = MemoryAccessor(
        state_adapter=adapter,
        session_id="s1",
        user_id=None,
        run_id="r1",
        memory_service=FakeMemoryService(),
    )
    with pytest.raises(RuntimeError, match="user_id"):
        await accessor.user.save("should fail")


@pytest.mark.asyncio
async def test_run_returns_kv(accessor):
    kv = accessor.run()
    assert isinstance(kv, KVMemory)


@pytest.mark.asyncio
async def test_global_returns_kv(accessor):
    kv = accessor.global_()
    assert isinstance(kv, KVMemory)


# --- Scope isolation via accessor ---


@pytest.mark.asyncio
async def test_scopes_are_isolated(accessor):
    await accessor.session().set("k", "session")
    await accessor.user().set("k", "user")
    await accessor.global_().set("k", "global")

    assert await accessor.session().get("k") == "session"
    assert await accessor.user().get("k") == "user"
    assert await accessor.global_().get("k") == "global"


@pytest.mark.asyncio
async def test_default_semantic_search_includes_session_and_user_filters(accessor):
    results = await accessor.search("preferences")

    assert len(results) == 1
    assert accessor._memory_service.searches[-1]["scope_filters"] == [
        ("session", "sess-001"),
        ("user", "user-123"),
    ]


# --- Working memory ---


@pytest.mark.asyncio
async def test_working_memory(accessor):
    wm = accessor.working
    assert isinstance(wm, WorkingMemory)
    await wm.set({"turn": 1})
    assert (await wm.get())["turn"] == 1


# --- Semantic memory ---


def test_semantic_is_none_by_default(accessor):
    assert accessor.semantic is None


def test_semantic_returns_provider_when_set(adapter):
    class FakeProvider(SemanticMemoryProvider):
        @property
        def provider_name(self):
            return "fake"

        async def store(self, content, metadata=None):
            return "id-1"

        async def search(self, query, limit=10, min_score=None):
            return []

        async def forget(self, memory_id):
            return True

    provider = FakeProvider()
    accessor = MemoryAccessor(
        state_adapter=adapter,
        session_id="s1",
        semantic_provider=provider,
    )
    assert accessor.semantic is provider
    assert accessor.semantic.provider_name == "fake"
