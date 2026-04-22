"""Unit tests for KVMemory."""

import pytest

from agnt5.memory import KVMemory, MemoryScope
from agnt5._state_adapter import StateAdapter


@pytest.fixture
def adapter():
    """Standalone in-memory state adapter."""
    return StateAdapter()


@pytest.fixture
def kv(adapter):
    """Session-scoped KV memory."""
    return KVMemory(adapter, scope=MemoryScope.SESSION, scope_id="sess-001")


@pytest.mark.asyncio
async def test_get_returns_default_when_empty(kv):
    assert await kv.get("missing") is None
    assert await kv.get("missing", "fallback") == "fallback"


@pytest.mark.asyncio
async def test_set_and_get(kv):
    await kv.set("theme", "dark")
    assert await kv.get("theme") == "dark"


@pytest.mark.asyncio
async def test_set_overwrites(kv):
    await kv.set("count", 1)
    await kv.set("count", 2)
    assert await kv.get("count") == 2


@pytest.mark.asyncio
async def test_delete_existing_key(kv):
    await kv.set("key", "val")
    assert await kv.delete("key") is True
    assert await kv.get("key") is None


@pytest.mark.asyncio
async def test_delete_missing_key(kv):
    assert await kv.delete("nope") is False


@pytest.mark.asyncio
async def test_keys(kv):
    await kv.set("a", 1)
    await kv.set("b", 2)
    result = await kv.keys()
    assert sorted(result) == ["a", "b"]


@pytest.mark.asyncio
async def test_clear(kv):
    await kv.set("x", 1)
    await kv.set("y", 2)
    await kv.clear()
    assert await kv.keys() == []
    assert await kv.get("x") is None


@pytest.mark.asyncio
async def test_scope_isolation(adapter):
    """Different scopes should not share data."""
    kv_a = KVMemory(adapter, scope=MemoryScope.SESSION, scope_id="sess-a")
    kv_b = KVMemory(adapter, scope=MemoryScope.SESSION, scope_id="sess-b")

    await kv_a.set("key", "from-a")
    await kv_b.set("key", "from-b")

    assert await kv_a.get("key") == "from-a"
    assert await kv_b.get("key") == "from-b"


@pytest.mark.asyncio
async def test_different_scope_types(adapter):
    """Session vs user vs global scopes are isolated."""
    kv_session = KVMemory(adapter, scope=MemoryScope.SESSION, scope_id="s1")
    kv_user = KVMemory(adapter, scope=MemoryScope.USER, scope_id="u1")
    kv_global = KVMemory(adapter, scope=MemoryScope.GLOBAL, scope_id="")

    await kv_session.set("k", "session")
    await kv_user.set("k", "user")
    await kv_global.set("k", "global")

    assert await kv_session.get("k") == "session"
    assert await kv_user.get("k") == "user"
    assert await kv_global.get("k") == "global"


@pytest.mark.asyncio
async def test_complex_values(kv):
    """JSON-serializable complex values should round-trip."""
    data = {"nested": {"list": [1, 2, 3], "flag": True}}
    await kv.set("complex", data)
    assert await kv.get("complex") == data
