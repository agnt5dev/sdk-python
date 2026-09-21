"""Durable agent work is shown under the iteration that issued it (AGNT5-1101).

Agent iterations are journal events, not admitted activations, so a model or
tool activation's durable parent is the enclosing step. The activation carries
the iteration as a reader-only display parent instead; ownership is untouched.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agnt5 import Agent, Context, tool
from agnt5.activation import (
    ActivationClient,
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationKind,
    ActivationRecoveryPolicy,
    _reset_current_activation,
    _set_current_activation,
    activation_id,
    activation_request_from_context,
    current_display_parent_correlation_id,
    display_parent_scope,
)
from agnt5.agent import AgentContext, handoff
from agnt5.context import set_current_context
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, Message, TokenUsage
from agnt5.lm.client import LMClient
from agnt5.lm.events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)


class MockLanguageModel(LanguageModel):
    """Scripted model: one response per call, with optional tool calls."""

    def __init__(self, responses=None, tool_calls=None):
        self.responses = responses or ["Mock response"]
        self.tool_calls_list = tool_calls or []
        self.call_count = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        tool_calls = None
        if self.call_count < len(self.tool_calls_list):
            tool_calls = self.tool_calls_list[self.call_count]
        self.call_count += 1
        return GenerateResponse(
            text=response_text,
            tool_calls=tool_calls,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def stream(self, request: GenerateRequest):
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        common = {
            "name": "mock-model",
            "correlation_id": "mock-corr",
            "parent_correlation_id": "mock-parent",
        }
        yield LMContentBlockStarted(block_type="text", index=0, **common)
        yield LMContentBlockDelta(content=response_text, block_type="text", index=0, **common)
        yield LMContentBlockCompleted(block_type="text", index=0, **common)
        yield LMCompleted(
            output_data={"text": response_text, "tool_calls": []},
            input_tokens=10,
            output_tokens=20,
            **common,
        )


def _durable_trace_metadata(component_name: str) -> dict:
    return {
        "durable_activation_v1": "true",
        "project_id": "project-1",
        "component_name": component_name,
        "worker_session_id": "worker-1",
        "run_authority": "run-authority",
        "lease_authority": "lease-authority",
        "activation_definition_version": "v1",
        "activation_artifact_sha256": "00" * 32,
        "activation_definition_config": '["object",[]]',
    }


class _RecordingActivationClient:
    def __init__(self):
        self.requests = []

    async def run(self, request, execute, **options):
        self.requests.append(request)
        decision = ActivationDecision(
            kind=ActivationDecisionKind.EXECUTE,
            activation_id=activation_id(
                request.project_id,
                request.run_id,
                request.parent_activation_id,
                request.kind,
                request.stable_key,
            ),
            attempt=1,
            accepted_journal_offset=len(self.requests),
            fence_token=b"fence",
        )
        if callback := options.get("on_admitted"):
            callback(decision)
        return await execute(), decision


class _ModelActivationTransport:
    def __init__(self):
        self.begin_requests = []

    async def begin(self, request):
        self.begin_requests.append(request)
        return ActivationDecision(
            kind=ActivationDecisionKind.EXECUTE,
            activation_id=activation_id(
                request.project_id,
                request.run_id,
                request.parent_activation_id,
                request.kind,
                request.stable_key,
            ),
            attempt=1,
            accepted_journal_offset=7,
            fence_token=b"fence",
        )

    async def complete(self, **request):
        return ActivationCompletionReceipt(
            activation_id=request["activation_id"],
            attempt=request["attempt"],
            accepted_journal_offset=8,
        )

    async def fail(self, **request):  # pragma: no cover - failure path unused
        raise AssertionError("unexpected failure")


def _rust_response(text: str):
    """Shape of the native provider response the LM client normalizes."""

    response = MagicMock()
    response.id = "response-1"
    response.content = text
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    response.usage.cached_tokens = 0
    response.usage.cache_creation_tokens = 0
    response.object = None
    response.tool_calls = None
    return response


def _iteration_ids(emitted) -> list[str]:
    return [e.correlation_id for e in emitted if e.event_type == "agent.iteration.started"]


def _durable_context() -> Context:
    return Context(
        run_id="run-1",
        correlation_id="corr-1",
        parent_correlation_id="",
        trace_metadata=_durable_trace_metadata("router"),
    )


def test_request_carries_the_scoped_display_parent_and_nothing_else():
    ctx = _durable_context()

    def build():
        return activation_request_from_context(
            ctx,
            kind=ActivationKind.TOOL,
            stable_key="tool:1",
            input_value={},
            recovery_policy=ActivationRecoveryPolicy.FAIL,
        )

    assert build().display_parent_correlation_id == ""
    with display_parent_scope("iteration-1"):
        request = build()
        assert request.display_parent_correlation_id == "iteration-1"
        assert request.parent_activation_id == ""
        # Nested work belongs to the admitted activation, not to the iteration
        # that described it; the scope comes back once the activation ends.
        token = _set_current_activation(
            ActivationDecision(
                kind=ActivationDecisionKind.EXECUTE,
                activation_id="actv1_tool",
                attempt=1,
                accepted_journal_offset=1,
            )
        )
        try:
            assert current_display_parent_correlation_id() == ""
            nested = build()
            assert nested.display_parent_correlation_id == ""
            assert nested.parent_activation_id == "actv1_tool"
            with display_parent_scope("inner-iteration"):
                assert build().display_parent_correlation_id == "inner-iteration"
        finally:
            _reset_current_activation(token)
        assert current_display_parent_correlation_id() == "iteration-1"
    assert build().display_parent_correlation_id == ""


@pytest.mark.asyncio
async def test_durable_tool_activation_points_at_its_iteration():
    @tool(recovery_policy="idempotent_retry")
    async def charge(ctx: Context, amount: int) -> dict:
        return {"amount": amount}

    mock_lm = MockLanguageModel(
        responses=["charging", "charged"],
        tool_calls=[
            [{"id": "call_1", "name": "charge", "arguments": '{"amount": 42}'}],
            None,
        ],
    )
    agent = Agent(name="biller", model=mock_lm, instructions="Bill", tools=[charge])
    ctx = AgentContext(
        run_id="run-1",
        agent_name="biller",
        trace_metadata=_durable_trace_metadata("biller"),
    )
    client = _RecordingActivationClient()
    ctx._activation_client = client
    emitted = []
    ctx.emit = emitted.append

    [event async for event in agent.stream("bill me", context=ctx)]

    iterations = _iteration_ids(emitted)
    assert len(iterations) == 2
    by_kind = {request.kind: request for request in client.requests}
    assert by_kind[ActivationKind.TOOL].display_parent_correlation_id == iterations[0]
    # The history step is the loop's own bookkeeping, outside any iteration.
    assert by_kind[ActivationKind.STEP].display_parent_correlation_id == ""
    assert current_display_parent_correlation_id() == ""


@pytest.mark.asyncio
async def test_later_iterations_point_at_themselves():
    @tool(recovery_policy="idempotent_retry")
    async def lookup(ctx: Context, key: str) -> dict:
        return {"key": key}

    mock_lm = MockLanguageModel(
        responses=["first", "second", "done"],
        tool_calls=[
            [{"id": "call_1", "name": "lookup", "arguments": '{"key": "a"}'}],
            [{"id": "call_2", "name": "lookup", "arguments": '{"key": "b"}'}],
            None,
        ],
    )
    agent = Agent(name="finder", model=mock_lm, instructions="Find", tools=[lookup])
    ctx = AgentContext(
        run_id="run-1",
        agent_name="finder",
        trace_metadata=_durable_trace_metadata("finder"),
    )
    client = _RecordingActivationClient()
    ctx._activation_client = client
    emitted = []
    ctx.emit = emitted.append

    [event async for event in agent.stream("find", context=ctx)]

    iterations = _iteration_ids(emitted)
    tool_requests = [r for r in client.requests if r.kind is ActivationKind.TOOL]
    assert len(iterations) == 3
    assert [r.display_parent_correlation_id for r in tool_requests] == iterations[:2]
    assert len(set(r.parent_activation_id for r in tool_requests)) == 1


@pytest.mark.asyncio
async def test_handoff_child_activation_points_at_the_calling_iteration():
    target = Agent(
        name="target",
        model=MockLanguageModel(responses=["handled"]),
        instructions="Handle delegated work",
    )
    source = Agent(
        name="source",
        model=MockLanguageModel(
            responses=["delegating", "relayed"],
            tool_calls=[
                [{"id": "call_1", "name": "transfer_to_target", "arguments": '{"message": "go"}'}],
                None,
            ],
        ),
        instructions="Delegate work",
        handoffs=[handoff(target)],
    )
    ctx = AgentContext(
        run_id="run-1",
        agent_name="source",
        trace_metadata=_durable_trace_metadata("source"),
    )
    client = _RecordingActivationClient()
    ctx._activation_client = client
    emitted = []
    ctx.emit = emitted.append

    [event async for event in source.stream("start", context=ctx)]

    child = next(r for r in client.requests if r.kind is ActivationKind.CHILD)
    source_iterations = [
        e.correlation_id
        for e in emitted
        if e.event_type == "agent.iteration.started" and e.metadata.get("name") == "source"
    ]
    assert child.display_parent_correlation_id == source_iterations[0]
    # The delegated agent's own bookkeeping runs inside the child activation and
    # is owned by it, so it carries no borrowed display parent.
    inner_steps = [
        r for r in client.requests if r.kind is ActivationKind.STEP and r.parent_activation_id != ""
    ]
    assert inner_steps and all(r.display_parent_correlation_id == "" for r in inner_steps)


@pytest.mark.asyncio
async def test_durable_model_activation_points_at_its_iteration():
    transport = _ModelActivationTransport()
    ctx = AgentContext(
        run_id="run-1",
        agent_name="writer",
        trace_metadata=_durable_trace_metadata("writer"),
    )
    ctx._activation_client = ActivationClient(transport)
    emitted = []
    ctx.emit = emitted.append
    response = _rust_response("done")

    token = set_current_context(ctx)
    try:
        with patch("agnt5.lm.client.RustLanguageModel") as rust:
            instance = MagicMock()
            instance.generate = AsyncMock(return_value=response)
            rust.return_value = instance
            # A model callback routes the loop through generate(); the streaming
            # route is covered by the tool tests and by the stream helper.
            agent = Agent(
                name="writer",
                model="openai/gpt-4o-mini",
                instructions="Write",
                after_model_callback=lambda ctx, request, response: None,
            )
            result = await agent.run("hello", context=ctx)
    finally:
        token.var.reset(token)

    assert result.output == "done"
    model_requests = [r for r in transport.begin_requests if r.kind is ActivationKind.MODEL]
    assert len(model_requests) == 1
    assert model_requests[0].display_parent_correlation_id == _iteration_ids(emitted)[0]
    assert json.loads(model_requests[0].input_data)["model"] == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_direct_model_call_outside_an_agent_has_no_display_parent():
    transport = _ModelActivationTransport()
    ctx = _durable_context()
    ctx._activation_client = ActivationClient(transport)
    response = _rust_response("plain")

    token = set_current_context(ctx)
    try:
        with patch("agnt5.lm.client.RustLanguageModel") as rust:
            instance = MagicMock()
            instance.generate = AsyncMock(return_value=response)
            rust.return_value = instance
            await LMClient(provider="openai").generate(
                GenerateRequest(model="openai/gpt-4o-mini", messages=[Message.user("hi")])
            )
    finally:
        token.var.reset(token)

    assert transport.begin_requests[0].display_parent_correlation_id == ""
