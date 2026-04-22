"""Unit tests for WorkingMemory."""

import pytest

from agnt5.memory import WorkingMemory, MemoryScope
from agnt5._state_adapter import StateAdapter


@pytest.fixture
def adapter():
    """Standalone in-memory state adapter."""
    return StateAdapter()


@pytest.fixture
def wm(adapter):
    """Session-scoped working memory."""
    return WorkingMemory(adapter, scope=MemoryScope.SESSION, scope_id="sess-001")


@pytest.mark.asyncio
async def test_get_returns_empty_dict_initially(wm):
    data = await wm.get()
    assert data == {}


@pytest.mark.asyncio
async def test_set_and_get(wm):
    await wm.set({"topic": "travel", "turn": 1})
    data = await wm.get()
    assert data == {"topic": "travel", "turn": 1}


@pytest.mark.asyncio
async def test_set_replaces_entirely(wm):
    await wm.set({"a": 1, "b": 2})
    await wm.set({"c": 3})
    data = await wm.get()
    assert data == {"c": 3}


@pytest.mark.asyncio
async def test_merge_adds_keys(wm):
    await wm.set({"a": 1})
    await wm.merge({"b": 2})
    data = await wm.get()
    assert data == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_merge_overwrites_keys(wm):
    await wm.set({"count": 1, "name": "alice"})
    await wm.merge({"count": 2})
    data = await wm.get()
    assert data == {"count": 2, "name": "alice"}


@pytest.mark.asyncio
async def test_merge_deletes_with_none(wm):
    await wm.set({"a": 1, "b": 2, "c": 3})
    await wm.merge({"b": None})
    data = await wm.get()
    assert data == {"a": 1, "c": 3}


@pytest.mark.asyncio
async def test_clear(wm):
    await wm.set({"x": 1})
    await wm.clear()
    data = await wm.get()
    assert data == {}


@pytest.mark.asyncio
async def test_get_returns_deep_copy(wm):
    """Mutating the returned dict should not affect stored data."""
    await wm.set({"items": [1, 2, 3]})
    data = await wm.get()
    data["items"].append(4)

    fresh = await wm.get()
    assert fresh["items"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_scope_isolation(adapter):
    """Different scopes should not share working memory."""
    wm_a = WorkingMemory(adapter, scope=MemoryScope.SESSION, scope_id="sess-a")
    wm_b = WorkingMemory(adapter, scope=MemoryScope.SESSION, scope_id="sess-b")

    await wm_a.set({"owner": "a"})
    await wm_b.set({"owner": "b"})

    assert (await wm_a.get())["owner"] == "a"
    assert (await wm_b.get())["owner"] == "b"
