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
