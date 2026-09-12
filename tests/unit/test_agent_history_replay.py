"""AGNT5-1108: hosted replay must reuse the originally selected session history."""

import asyncio
import base64
import copy
from types import SimpleNamespace

import pytest
from test_worker_agent_streaming import _DummyExecutor, _RecordingWorker
from test_workflow_activation import WorkflowActivationTransport

from agnt5._serialization import serialize
from agnt5.activation import (
    ActivationClient,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationKind,
    activation_id,
    current_activation,
)
from agnt5.agent import Agent, AgentContext
from agnt5.exceptions import ActivationError, ActivationErrorCode
from agnt5.lm import GenerateResponse, LMClient, Message
from agnt5.tool import tool


class ReplayTransport(WorkflowActivationTransport):
    """Retain accepted receipts across replacement workers; reject changed inputs."""

    def __init__(self):
        super().__init__()
        self.semantics = {}
        self.outputs = {}
        self.history_ids = set()
        self.crash_at = None

    async def begin(self, request):
        aid = activation_id(
            request.project_id,
            request.run_id,
            request.parent_activation_id,
            request.kind,
            request.stable_key,
        )
        semantics = (request.input_digest, request.definition_digest, request.recovery_policy)
        if aid in self.semantics and self.semantics[aid] != semantics:
            raise ActivationError(
                ActivationErrorCode.NON_DETERMINISTIC_REPLAY,
                "stable activation key was reused with different durable semantics",
            )
        self.semantics[aid] = semantics
        if request.stable_key.startswith("agent_history:"):
            self.history_ids.add(aid)
        if aid in self.outputs:
            self.begin_requests.append(request)
            return ActivationDecision(
                kind=ActivationDecisionKind.REPLAY,
                activation_id=aid,
                attempt=1,
                accepted_journal_offset=12,
                replay_output=self.outputs[aid],
            )
        return await super().begin(request)

    async def complete(self, **request):
        is_history = request["activation_id"] in self.history_ids
        if is_history and self.crash_at == "before_accept":
            self.crash_at = None
            raise asyncio.CancelledError("worker died before snapshot acceptance")
        self.outputs[request["activation_id"]] = request["output"]
        receipt = await super().complete(**request)
        if is_history and self.crash_at == "after_accept":
            self.crash_at = None
            raise asyncio.CancelledError("worker died after snapshot acceptance")
        return receipt


class HistoryState:
    def __init__(self):
        self.data = {}
        self.version = 0

    async def load_state(self, *args, **kwargs):
        return copy.deepcopy(self.data)

    async def load_with_version(self, *args, **kwargs):
        return copy.deepcopy(self.data), self.version

    async def save_state(self, entity_type, key, value, expected_version, **kwargs):
        assert expected_version == self.version
        self.data = copy.deepcopy(value)
        self.version += 1
        return self.version


def metadata(attempt=0):
    return {
        "dispatch_mode": "pull",
        "durable_activation_v1": "true",
        "execution_mode": "eval_worker",
        "project_id": "project-fixture",
        "worker_session_id": f"fixture-session-{attempt}",
        "lease_id": f"fixture-lease-{attempt}",
        "component_name": "docs_chat_assistant",
        "activation_artifact_sha256": base64.b64encode(b"a" * 32).decode(),
        "activation_definition_version": "v1",
        "activation_definition_config": '["object",[]]',
    }


@pytest.fixture
def hosted(monkeypatch):
    transport = ReplayTransport()
    state = HistoryState()
    counts = {"model": 0, "tool": 0}
    model_inputs = []

    async def generate(**kwargs):
        model_inputs.append(copy.deepcopy(kwargs["prompt"]))
        counts["model"] += 1
        if counts["model"] % 2:
            return GenerateResponse(
                text="",
                tool_calls=[
                    {
                        "id": "call-docs-1",
                        "name": "docs_shell",
                        "arguments": '{"command":"read-docs"}',
                    }
                ],
            )
        return GenerateResponse(text=f"fixture answer {counts['model'] // 2}")

    def init(self, *args, **kwargs):
        self._provider = "openai"
        self._default_model = None
        self._rust_lm = SimpleNamespace(generate=generate)

    async def no_runs_history(self):
        return []

    monkeypatch.setattr(LMClient, "__init__", init)
    monkeypatch.setattr(LMClient, "_convert_response", lambda self, response: response)
    monkeypatch.setattr(AgentContext, "_load_from_runs_api", no_runs_history)

    @tool
    async def docs_shell(ctx, command: str) -> str:
        """Read a deterministic documentation excerpt."""
        counts["tool"] += 1
        return "fixture docs"

    agent = Agent(
        name="docs_chat_assistant",
        model="openai/gpt-4o-mini",
        tools=[docs_shell],
        instructions="Answer using the fixture docs",
        before_model_callback=lambda ctx, request: None,
    )

    async def execute(attempt=0, run_id="run-replay-fixture", message="question", managed=False):
        request = SimpleNamespace(
            invocation_id=run_id,
            input_data=serialize({"message": message, "session_history_managed": managed}),
            runtime_context=None,
            session_id="shared-conversation",
            user_id="",
            attempt=attempt,
            is_streaming=False,
            component_name=agent.name,
            component_type="agent",
            metadata=metadata(attempt),
        )
        executor = _DummyExecutor(_RecordingWorker())
        executor._entity_state_adapter = state
        executor._activation_client_for_metadata = lambda _: ActivationClient(transport)
        return await executor._execute_agent(agent, request.input_data, request)

    return SimpleNamespace(
        execute=execute,
        transport=transport,
        state=state,
        counts=counts,
        model_inputs=model_inputs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_hosted_replay_reuses_history_model_and_tool_receipts(hosted, managed):
    first = await hosted.execute(managed=managed)
    assert first.success, first.error_message
    second = await hosted.execute(attempt=1, managed=managed)
    assert second.success, second.error_message
    assert second.output_data == first.output_data
    assert hosted.counts == {"model": 2, "tool": 1}
    model_requests = [r for r in hosted.transport.begin_requests if r.kind is ActivationKind.MODEL]
    assert [r.stable_key for r in model_requests] == [
        "model:openai/gpt-4o-mini:0",
        "model:openai/gpt-4o-mini:1",
        "model:openai/gpt-4o-mini:0",
        "model:openai/gpt-4o-mini:1",
    ]
    assert all(r.parent_activation_id == "" for r in model_requests)
    assert len(hosted.transport.history_ids) == (0 if managed else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before_accept", "after_accept"])
async def test_snapshot_acceptance_precedes_session_writes_and_model_execution(hosted, boundary):
    hosted.state.data = {"messages": [{"role": "user", "content": "original history"}]}
    hosted.transport.crash_at = boundary
    with pytest.raises(asyncio.CancelledError):
        await hosted.execute()
    assert hosted.counts == {"model": 0, "tool": 0}
    assert hosted.state.version == 0
    hosted.state.data = {"messages": [{"role": "user", "content": "changed while offline"}]}
    recovered = await hosted.execute(attempt=1)
    assert recovered.success, recovered.error_message
    expected = "original history" if boundary == "after_accept" else "changed while offline"
    assert [m["content"] for m in hosted.model_inputs[0]] == [expected, "question"]
    assert hosted.counts == {"model": 2, "tool": 1}


@pytest.mark.asyncio
async def test_new_run_keeps_completed_prior_turns_in_the_same_session(hosted):
    first = await hosted.execute(run_id="first-turn", message="first question")
    assert first.success, first.error_message
    second = await hosted.execute(run_id="second-turn", message="follow-up")
    assert second.success, second.error_message
    second_input = [m["content"] for m in hosted.model_inputs[2]]
    assert "first question" in second_input
    assert "fixture answer 1" in second_input
    assert second_input[-1] == "follow-up"
    replayed = await hosted.execute(attempt=1, run_id="second-turn", message="follow-up")
    assert replayed.success, replayed.error_message
    assert replayed.output_data == second.output_data
    assert hosted.counts == {"model": 4, "tool": 2}
    assert len(hosted.transport.history_ids) == 2


def context_for(transport, *, durable=True):
    authority = metadata()
    if not durable:
        authority.pop("durable_activation_v1")
    context = AgentContext(
        run_id="history-run",
        agent_name="docs_chat_assistant",
        session_id="history-session",
        state_manager=HistoryState(),
        trace_metadata=authority,
    )
    context._activation_client = ActivationClient(transport)
    return context


@pytest.mark.asyncio
async def test_history_snapshot_preserves_message_fields_and_sequential_invocations():
    transport = ReplayTransport()
    context = context_for(transport)
    initial = [
        Message.assistant("calling", tool_calls=[{"id": "tool-1", "name": "lookup"}]),
        Message.tool_result("tool-1", "result"),
        Message.system("rules"),
    ]
    selected = initial

    async def load():
        return copy.deepcopy(selected)

    context.get_conversation_history = load
    assert current_activation() is None
    assert await context._get_initial_conversation_history() == initial
    selected = [Message.user("next invocation")]
    assert await context._get_initial_conversation_history() == selected
    replay = context_for(transport)

    async def no_live_read():
        pytest.fail("a completed snapshot must not reload mutable history")

    replay.get_conversation_history = no_live_read
    restored = await replay._get_initial_conversation_history()
    assert restored == initial
    restored[0].tool_calls[0]["name"] = "mutated by caller"
    assert await replay._get_initial_conversation_history() == selected
    assert current_activation() is None
    assert [r.stable_key for r in transport.begin_requests] == [
        "agent_history:docs_chat_assistant:0",
        "agent_history:docs_chat_assistant:1",
        "agent_history:docs_chat_assistant:0",
        "agent_history:docs_chat_assistant:1",
    ]


@pytest.mark.asyncio
async def test_non_durable_history_remains_live_and_missing_durable_client_fails_closed():
    transport = ReplayTransport()
    context = context_for(transport, durable=False)
    selected = [Message.user("first")]

    async def load():
        return copy.deepcopy(selected)

    context.get_conversation_history = load
    assert await context._get_initial_conversation_history() == selected
    selected = [Message.user("latest")]
    assert await context._get_initial_conversation_history() == selected
    assert transport.begin_requests == []
    context._trace_metadata["durable_activation_v1"] = "true"
    context._activation_client = None
    with pytest.raises(ActivationError, match="no activation client"):
        await context._get_initial_conversation_history()
