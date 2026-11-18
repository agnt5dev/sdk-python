"""An agent declared ``per="user"`` keeps one instance per user of the app.

The worker tells the platform through the component metadata it registers,
and the agent reads the turn the platform starts: the items that waited in the
user's inbox.
"""

import pytest

from agnt5 import Agent
from agnt5.agent import AgentRegistry
from agnt5.worker._core import Worker
from agnt5.worker._executors import _agent_user_message


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.delenv("AGNT5_WORKER_MODE", raising=False)
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


def agent(name: str, **kwargs) -> Agent:
    return Agent(name=name, model="openai/gpt-test", instructions="Help", **kwargs)


def test_per_accepts_user_only():
    assert agent("support", per="user").per == "user"
    assert agent("digest").per is None
    with pytest.raises(ValueError, match="per must be"):
        agent("team", per="team")


def test_the_worker_registers_per_in_the_component_metadata():
    support = agent("support", per="user")
    digest = agent("digest")
    worker = Worker(service_name="assistants", agents=[support, digest])
    metadata = {
        component.name: dict(component.metadata)
        for component in worker._discover_components()
        if component.component_type == "agent"
    }
    assert metadata["support"] == {"per": "user"}
    assert metadata["digest"] == {}


def test_a_turn_reads_every_item_it_takes_oldest_first():
    turn = {
        "conversation_id": "api",
        "inbox_items": [
            {"item_id": "1", "channel": "api", "content": {"message": "Where is order 4412?"}},
            {"item_id": "2", "channel": "chat", "content": "It was a gift."},
            {"item_id": "3", "channel": "batch", "content": {"order": 4412}},
            {"item_id": "4", "channel": "api", "content": None},
        ],
    }
    assert _agent_user_message(turn) == (
        'Where is order 4412?\n\nIt was a gift.\n\n{"order": 4412}'
    )
    # A plain call is read as before.
    assert _agent_user_message({"message": "hi", "inbox_items": []}) == "hi"


class _Recorder:
    def __init__(self):
        self.events = []

    async def emit_now_async(self, event):
        self.events.append(event)


class _Ctx:
    def __init__(self):
        self.recorder = _Recorder()

    def _get_emitter(self):
        return self.recorder


def test_a_per_user_agent_gets_schedule_followup():
    assert "schedule_followup" in agent("support", per="user").tools
    assert "schedule_followup" not in agent("digest").tools


async def test_schedule_followup_asks_on_the_runs_own_journal():
    from agnt5.agent.instance_tools import (
        enter_instance_turn,
        followup_timer_id,
        schedule_followup_tool,
    )

    tool = schedule_followup_tool()
    ctx = _Ctx()
    # Outside a per-user turn there is no instance to follow up in.
    outside = await tool.handler(ctx, delay_seconds=60, message="check the PR")
    assert "error" in outside and not ctx.recorder.events

    token = enter_instance_turn("61676e35-0100-8000-8000-000000000001", "pr-42")
    try:
        result = await tool.handler(
            ctx, delay_seconds=60, message="check the PR", key="pr-42-review"
        )
        again = await tool.handler(ctx, delay_seconds=60, message="check the PR")
        same = await tool.handler(ctx, delay_seconds=60, message="check the PR")
        bad = await tool.handler(ctx, delay_seconds=60, message="x", key="a/b")
        late = await tool.handler(ctx, delay_seconds=10**9, message="x")
    finally:
        from agnt5.agent.instance_tools import current_instance_turn

        current_instance_turn.reset(token)

    assert result["timer_id"] == followup_timer_id(
        "61676e35-0100-8000-8000-000000000001", "pr-42-review"
    )
    assert again["timer_id"] == same["timer_id"], "the same call gets the same follow-up"
    assert "error" in bad and "error" in late
    event = ctx.recorder.events[0]
    assert event.event_type == "instance.followup_requested"
    data = event.to_dict()
    assert data["followup_key"] == "pr-42-review"
    assert data["conversation_id"] == "pr-42"
    assert data["content"] == {"message": "check the PR"}
    assert data["fire_at_ns"] > 0


async def test_send_messages_the_user_with_an_id_a_retry_keeps():
    from agnt5.agent.instance_tools import (
        current_instance_turn,
        enter_instance_turn,
        message_id,
        send_tool,
    )

    assert "send" in agent("support", per="user").tools
    assert "send" not in agent("digest").tools
    tool = send_tool()
    ctx = _Ctx()
    assert "error" in await tool.handler(ctx, text="hi")
    assert not ctx.recorder.events

    run_id = "61676e35-0100-8000-8000-000000000002"
    token = enter_instance_turn(run_id, "pr-42")
    try:
        first = await tool.handler(ctx, text="Your PR has a review")
        replay = await tool.handler(ctx, text="Your PR has a review")
        other = await tool.handler(ctx, text="Your PR was merged")
        keyed = await tool.handler(ctx, text="Your PR has a review", key="review")
    finally:
        current_instance_turn.reset(token)

    assert first["message_id"] == replay["message_id"], "a replay is the same message"
    assert other["message_id"] != first["message_id"]
    assert keyed["message_id"].startswith(message_id(run_id, "review."))
    event = ctx.recorder.events[0].to_dict()
    assert event["event_type"] == "instance.message_requested"
    assert event["conversation_id"] == "pr-42"
    assert event["content"] == {"text": "Your PR has a review"}
    assert first["message_id"] == message_id(run_id, event["message_key"])
