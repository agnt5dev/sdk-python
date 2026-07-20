"""Unit tests for ConversationAccessor."""

import pytest

from agnt5._state_adapter import StateAdapter
from agnt5.memory import ConversationAccessor, ConversationMessage


@pytest.fixture
def adapter():
    return StateAdapter()


@pytest.fixture
def conv(adapter):
    return ConversationAccessor(
        state_adapter=adapter,
        session_id="sess-001",
        component_name="test-agent",
    )


# --- Basic add/get ---

@pytest.mark.asyncio
async def test_empty_conversation(conv):
    messages = await conv.get_messages()
    assert messages == []


@pytest.mark.asyncio
async def test_add_and_get(conv):
    await conv.add("user", "Hello")
    await conv.add("assistant", "Hi there!")

    messages = await conv.get_messages()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there!"


@pytest.mark.asyncio
async def test_add_returns_id(conv):
    msg_id = await conv.add("user", "Test")
    assert msg_id == "0"

    msg_id2 = await conv.add("assistant", "Reply")
    assert msg_id2 == "1"


@pytest.mark.asyncio
async def test_add_with_metadata(conv):
    await conv.add("user", "Hello", metadata={"source": "web"})
    messages = await conv.get_messages()
    assert messages[0].metadata == {"source": "web"}


# --- Limit ---

@pytest.mark.asyncio
async def test_get_messages_limit(conv):
    for i in range(10):
        await conv.add("user", f"Message {i}")

    messages = await conv.get_messages(limit=3)
    assert len(messages) == 3
    # Should return most recent
    assert messages[0].content == "Message 7"
    assert messages[2].content == "Message 9"


# --- LM messages ---

@pytest.mark.asyncio
async def test_get_as_lm_messages(conv):
    await conv.add("user", "Hello")
    await conv.add("assistant", "Hi!")

    lm_messages = await conv.get_as_lm_messages()
    assert len(lm_messages) == 2
    assert lm_messages[0].role.value == "user"
    assert lm_messages[0].content == "Hello"
    assert lm_messages[1].role.value == "assistant"
    assert lm_messages[1].content == "Hi!"


# --- State ---

@pytest.mark.asyncio
async def test_get_state_empty(conv):
    state = await conv.get_state()
    assert state["session_id"] == "sess-001"
    assert state["message_count"] == 0
    assert state["created_at"] is None


@pytest.mark.asyncio
async def test_get_state_after_messages(conv):
    await conv.add("user", "Hello")
    await conv.add("assistant", "Hi!")

    state = await conv.get_state()
    assert state["message_count"] == 2
    assert state["created_at"] is not None
    assert state["last_message_at"] is not None


# --- Clear ---

@pytest.mark.asyncio
async def test_clear(conv):
    await conv.add("user", "Hello")
    await conv.add("assistant", "Hi!")
    await conv.clear()

    messages = await conv.get_messages()
    assert messages == []

    state = await conv.get_state()
    assert state["message_count"] == 0


# --- Session isolation ---

@pytest.mark.asyncio
async def test_session_isolation(adapter):
    conv_a = ConversationAccessor(adapter, session_id="sess-a")
    conv_b = ConversationAccessor(adapter, session_id="sess-b")

    await conv_a.add("user", "From A")
    await conv_b.add("user", "From B")

    msgs_a = await conv_a.get_messages()
    msgs_b = await conv_b.get_messages()

    assert len(msgs_a) == 1
    assert msgs_a[0].content == "From A"
    assert len(msgs_b) == 1
    assert msgs_b[0].content == "From B"


# --- Session ID property ---

def test_session_id_property(conv):
    assert conv.session_id == "sess-001"


# --- ConversationMessage ---

def test_conversation_message_to_dict():
    msg = ConversationMessage(role="user", content="Hello", timestamp=1000.0)
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "Hello"
    assert d["timestamp"] == 1000.0


def test_conversation_message_from_dict():
    d = {"role": "assistant", "content": "Hi!", "timestamp": 2000.0, "metadata": {"k": "v"}}
    msg = ConversationMessage.from_dict(d)
    assert msg.role == "assistant"
    assert msg.content == "Hi!"
    assert msg.metadata == {"k": "v"}


def test_conversation_message_to_lm_message():
    msg = ConversationMessage(role="user", content="Hello")
    lm_msg = msg.to_lm_message()
    assert lm_msg.role.value == "user"
    assert lm_msg.content == "Hello"
