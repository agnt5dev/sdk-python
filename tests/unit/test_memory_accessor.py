"""Unit tests for MemoryAccessor."""

import pytest

from agnt5.memory import MemoryAccessor, KVMemory, WorkingMemory, SemanticMemoryProvider
from agnt5._state_adapter import StateAdapter


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
async def test_user_returns_kv(accessor):
    kv = accessor.user()
    assert isinstance(kv, KVMemory)
    await kv.set("pref", "dark")
    assert await kv.get("pref") == "dark"


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
