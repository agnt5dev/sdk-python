import json

import pytest
from typing import Any, Dict

from agnt5.agent import (
    AgentContext,
    DurableAgent,
    clear_agents,
    get_registered_agents,
    stateful,
)
from agnt5.context import Context
from agnt5.function import execute_component


@pytest.fixture(autouse=True)
def _reset_agents():
    clear_agents()
    yield
    clear_agents()


def test_stateful_agent_registration():
    @stateful(name="support_agent", key=lambda tenant, user: f"{tenant}:{user}")
    class SupportAgent(DurableAgent):
        async def on_message(self, ctx: AgentContext, message):
            return {"echo": message}

    agents = get_registered_agents()
    assert "support_agent" in agents
    definition = agents["support_agent"]
    assert definition.agent_cls is SupportAgent
    assert definition.key_resolver("acme", "alice") == "acme:alice"


def test_duplicate_registration_raises():
    @stateful(name="dupe", key=lambda *_: "key")
    class FirstAgent(DurableAgent):
        async def on_message(self, ctx: AgentContext, message):
            return message

    with pytest.raises(ValueError):

        @stateful(name="dupe", key=lambda *_: "key")
        class SecondAgent(DurableAgent):
            async def on_message(self, ctx: AgentContext, message):
                return message


def test_requires_durable_agent_base():
    with pytest.raises(TypeError):

        @stateful(name="bad", key=lambda *_: "key")
        class NotAnAgent:  # type: ignore
            pass


def test_requires_key():
    with pytest.raises(ValueError):

        @stateful(name="missing")  # type: ignore
        class MissingKey(DurableAgent):
            async def on_message(self, ctx: AgentContext, message):
                return message


def test_agent_execute_component_produces_state_update():
    @stateful(name="counter_agent", key=lambda tid: tid)
    class CounterAgent(DurableAgent):
        async def on_message(self, ctx: AgentContext, payload: Dict[str, Any]):
            count = await ctx.memory.get("count", 0)
            await ctx.memory.set("count", count + 1)
            return {"count": count + 1}

    payload = json.dumps({"tenant": "acme"}).encode("utf-8")
    context = {
        "invocation_id": "inv-1",
        "service_name": "svc",
        "metadata": {"run_id": "run-1"},
        "step_checkpoints": [],
    }

    result = execute_component(
        handler_name="counter_agent",
        input_data=payload,
        context=context,
        component_type="agent",
        method_name="on_message",
        object_id="acme",
    )

    assert result.state_update is not None
    update = result.state_update
    snapshot = json.loads(update.new_state.decode("utf-8"))
    assert snapshot["count"] == 1
    assert json.loads(update.output_data.decode("utf-8")) == {"count": 1}
    assert update.transitions


@pytest.mark.asyncio
async def test_ctx_agent_call_runs_in_process():
    @stateful(name="chat_agent", key=lambda payload: payload["tenant"])
    class ChatAgent(DurableAgent):
        async def on_message(self, ctx: AgentContext, message: Dict[str, Any]):
            history = await ctx.memory.get("history", [])
            history.append(message["text"])
            await ctx.memory.set("history", history)
            return {"history": history}

    ctx = Context(
        run_id="run-1",
        step_id="step-1",
        attempt=1,
        invocation_id="inv-root",
        service_name="svc",
        component_name="caller",
        metadata={"headers": {"x-trace": "abc"}},
    )

    first = await ctx.agent.call("chat_agent", {"tenant": "acme", "text": "hi"})
    assert first.object_id == "acme"
    assert first.output == {"history": ["hi"]}
    assert first.state["history"] == ["hi"]
    assert first.transitions

    second = await ctx.agent.call("chat_agent", {"tenant": "acme", "text": "there"})
    assert second.output == {"history": ["hi", "there"]}
    assert second.state["history"] == ["hi", "there"]
    assert len(second.transitions) == 1

    cached_state = ctx.agent.get_state("chat_agent", "acme")
    assert cached_state == {"history": ["hi", "there"]}

    ctx.agent.reset("chat_agent", object_id="acme")
    assert ctx.agent.get_state("chat_agent", "acme") is None
