"""A tool that must pause for a person, held by an agent where the run cannot pause.

An agent hands a tool's ordinary errors back to the model. A tool that could
not pause is different: if the model saw that as a tool result it would carry
on as if the person had been asked, so the agent fails the run instead. It
fails before the first model call, because where the agent runs decides
whether it can pause, not which tool the model happens to pick.
"""

from types import SimpleNamespace

import pytest

from agnt5 import Agent, Context, ToolCannotPauseError, tool
from agnt5._serialization import serialize
from agnt5.agent import AgentRegistry
from agnt5.exceptions import ConfigurationError, WaitingForUserInputException
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, Message, TokenUsage
from agnt5.tool import AskUserTool, RequestApprovalTool, ToolRegistry
from agnt5.worker._executors import ExecutorMixin
from agnt5.workflow import WorkflowContext, WorkflowEntity


@pytest.fixture(autouse=True)
def clear_registries():
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    ToolRegistry.clear()
    AgentRegistry.clear()


class ToolCallingModel(LanguageModel):
    """Calls one tool, then answers "done" once it has been asked again."""

    def __init__(self, tool_name: str, arguments: str = "{}"):
        self.tool_name = tool_name
        self.arguments = arguments
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        if len(self.requests) == 1:
            return GenerateResponse(
                text="",
                tool_calls=[{"id": "call-1", "name": self.tool_name, "arguments": self.arguments}],
                usage=usage,
            )
        return GenerateResponse(text="done", tool_calls=None, usage=usage)

    async def stream(self, request: GenerateRequest):
        raise NotImplementedError
        yield  # pragma: no cover

    def shown_to_model(self) -> str:
        """Everything the model was sent after its tool call."""
        return repr([request.messages for request in self.requests[1:]])


def _workflow_ctx() -> WorkflowContext:
    return WorkflowContext(workflow_entity=WorkflowEntity(run_id="pause-run"), run_id="pause-run")


def _record_publication():
    published = []

    @tool(confirmation=True)
    async def record_publication(ctx: Context, title: str) -> str:
        """Record a publication."""
        published.append(title)
        return f"recorded {title}"

    return ToolRegistry.get("record_publication"), published


def _publisher(model, tools):
    return Agent(name="publisher", model=model, instructions="Publish.", tools=tools)


# An agent inside a step


@pytest.mark.asyncio
async def test_agent_in_a_step_fails_when_its_confirmation_tool_cannot_pause():
    probe, published = _record_publication()
    model = ToolCallingModel("record_publication", '{"title": "Q3 report"}')
    agent = _publisher(model, [probe])
    ctx = _workflow_ctx()

    async def publish():
        return (await agent.run("Publish it", context=ctx)).output

    with pytest.raises(ToolCannotPauseError, match="record_publication") as raised:
        await ctx.step("publish", publish)

    assert isinstance(raised.value, ConfigurationError)
    assert raised.value.tool_name == "record_publication"
    assert "from the workflow body" in str(raised.value)
    assert published == []
    # Refused before the model was asked anything.
    assert model.requests == []


@pytest.mark.asyncio
async def test_streaming_agent_in_a_step_fails_when_its_confirmation_tool_cannot_pause():
    probe, published = _record_publication()
    model = ToolCallingModel("record_publication", '{"title": "Q3 report"}')
    agent = _publisher(model, [probe])
    ctx = _workflow_ctx()

    async def publish():
        return [event async for event in agent.stream("Publish it", context=ctx)]

    with pytest.raises(ToolCannotPauseError, match="record_publication"):
        await ctx.step("publish", publish)

    assert published == []
    assert model.requests == []


@pytest.mark.asyncio
async def test_agent_in_a_step_fails_even_when_the_model_would_not_call_the_tool():
    """Otherwise the mistake passes every run in which the model skips the tool."""
    probe, _published = _record_publication()

    @tool
    async def lookup(ctx: Context) -> str:
        """Look something up."""
        return "found"

    model = ToolCallingModel("lookup")
    agent = _publisher(model, [lookup, probe])
    ctx = _workflow_ctx()

    async def research():
        return (await agent.run("Look it up", context=ctx)).output

    with pytest.raises(ToolCannotPauseError, match="record_publication"):
        await ctx.step("research", research)

    assert model.requests == []


@pytest.mark.asyncio
async def test_agent_without_a_pause_tool_still_runs_in_a_step():
    @tool
    async def lookup(ctx: Context) -> str:
        """Look something up."""
        return "found"

    model = ToolCallingModel("lookup")
    agent = _publisher(model, [lookup])
    ctx = _workflow_ctx()

    async def research():
        return (await agent.run("Look it up", context=ctx)).output

    assert await ctx.step("research", research) == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,tool_name,arguments",
    [
        (AskUserTool, "ask_user", '{"question": "Which focus?"}'),
        (RequestApprovalTool, "request_approval", '{"action": "Close the ticket"}'),
    ],
)
async def test_agent_in_a_step_fails_when_a_built_in_pause_tool_cannot_pause(
    tool_class, tool_name, arguments
):
    ctx = _workflow_ctx()
    model = ToolCallingModel(tool_name, arguments)
    agent = _publisher(model, [tool_class(ctx)])

    async def intake():
        return (await agent.run("Start", context=ctx)).output

    with pytest.raises(ToolCannotPauseError, match=tool_name) as raised:
        await ctx.step("intake", intake)

    assert raised.value.tool_name == tool_name
    assert "from the workflow body" in str(raised.value)
    assert model.requests == []


@pytest.mark.asyncio
async def test_agent_outside_a_workflow_fails_when_its_confirmation_tool_cannot_pause():
    probe, published = _record_publication()
    model = ToolCallingModel("record_publication", '{"title": "Q3 report"}')
    agent = _publisher(model, [probe])
    plain_ctx = Context(run_id="plain-run", correlation_id="c", parent_correlation_id="p")

    with pytest.raises(ToolCannotPauseError, match="outside a workflow"):
        await agent.run("Publish it", context=plain_ctx)

    assert published == []
    assert model.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_fails_when_a_tool_finds_it_cannot_pause_only_once_called(streaming):
    """The check at the start can't see every case, so the loop still refuses."""

    @tool
    async def escalate(ctx: Context) -> str:
        """Escalate to a person."""
        raise ToolCannotPauseError("escalate cannot pause here", tool_name="escalate")

    model = ToolCallingModel("escalate")
    agent = _publisher(model, [escalate])

    with pytest.raises(ToolCannotPauseError, match="escalate cannot pause here"):
        if streaming:
            [event async for event in agent.stream("Escalate", context=_workflow_ctx())]
        else:
            await agent.run("Escalate", context=_workflow_ctx())

    # The model was asked once and never shown the failure as a tool result.
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_agent_continued_after_a_pause_fails_when_its_tool_cannot_pause():
    """The loop a resumed agent continues in refuses the same way."""
    probe, published = _record_publication()
    model = ToolCallingModel("record_publication", '{"title": "Q3 report"}')
    agent = _publisher(model, [probe])
    ctx = _workflow_ctx()

    async def publish():
        return await agent._continue_execution_from_iteration(
            context=ctx, messages=[Message.user("Publish it")], iteration=0, all_tool_calls=[]
        )

    with pytest.raises(ToolCannotPauseError, match="record_publication"):
        await ctx.step("publish", publish)

    assert published == []
    assert len(model.requests) == 1


# Ordinary tool errors still go back to the model


@pytest.mark.asyncio
async def test_agent_in_a_step_still_hands_an_ordinary_tool_error_to_the_model():
    @tool
    async def lookup(ctx: Context) -> str:
        """Look something up."""
        raise ConfigurationError("lookup is misconfigured")

    model = ToolCallingModel("lookup")
    agent = _publisher(model, [lookup])
    ctx = _workflow_ctx()

    async def research():
        return (await agent.run("Look it up", context=ctx)).output

    assert await ctx.step("research", research) == "done"
    assert len(model.requests) == 2
    assert "lookup is misconfigured" in model.shown_to_model()


# In the workflow body the same tools still pause


@pytest.mark.asyncio
async def test_agent_in_the_workflow_body_still_pauses_for_confirmation():
    probe, published = _record_publication()
    model = ToolCallingModel("record_publication", '{"title": "Q3 report"}')
    agent = _publisher(model, [probe])

    with pytest.raises(WaitingForUserInputException) as raised:
        await agent.run("Publish it", context=_workflow_ctx())

    assert raised.value.input_type == "approval"
    assert published == []


@pytest.mark.asyncio
async def test_agent_in_the_workflow_body_still_pauses_for_ask_user():
    ctx = _workflow_ctx()
    model = ToolCallingModel("ask_user", '{"question": "Which focus?"}')
    agent = _publisher(model, [AskUserTool(ctx)])

    with pytest.raises(WaitingForUserInputException) as raised:
        await agent.run("Start", context=ctx)

    assert raised.value.question == "Which focus?"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,tool_name,arguments,input_type",
    [
        (AskUserTool, "ask_user", '{"question": "Which focus?"}', "text"),
        (RequestApprovalTool, "request_approval", '{"action": "Close the ticket"}', "approval"),
    ],
)
async def test_built_in_pause_tool_made_without_a_context_pauses_in_the_workflow_body(
    tool_class, tool_name, arguments, input_type
):
    """It finds the workflow running the agent, as a confirmation tool does."""
    model = ToolCallingModel(tool_name, arguments)
    agent = _publisher(model, [tool_class()])

    with pytest.raises(WaitingForUserInputException) as raised:
        await agent.run("Start", context=_workflow_ctx())

    assert raised.value.input_type == input_type


# A confirmation tool run directly


class _Executor(ExecutorMixin):
    def __init__(self) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = None
        self.service_name = "test"


def _tool_request(payload, metadata=None):
    return SimpleNamespace(
        invocation_id="run-direct-tool",
        input_data=serialize(payload),
        runtime_context=None,
        metadata=dict(metadata or {}),
        session_id="",
        user_id="",
        attempt=0,
        is_streaming=False,
        component_name="record_publication",
    )


@pytest.fixture
def emitted(monkeypatch):
    """Event types the run emitted, in order, instead of sending them anywhere."""
    events = []

    async def capture(self, event):
        events.append(event.event_type)

    async def capture_batch(self, batch):
        events.extend(event.event_type for event in batch)

    monkeypatch.setattr(Context, "emit_async", capture)
    monkeypatch.setattr(Context, "emit_batch_async", capture_batch)
    return events


async def _run_tool(probe, payload, metadata=None):
    request = _tool_request(payload, metadata)
    return await _Executor()._execute_tool(probe, request.input_data, request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatch_metadata,paused_event_type",
    [({}, "run.paused"), ({"dispatch_mode": "pull", "lease_id": "lease-1"}, "workflow.paused")],
)
async def test_confirmation_tool_run_directly_pauses_then_runs_once_approved(
    emitted, dispatch_metadata, paused_event_type
):
    probe, published = _record_publication()

    paused = await _run_tool(probe, {"title": "Q3 report"}, dispatch_metadata)

    assert paused is not None
    assert paused.success is True
    assert paused.event_type == paused_event_type
    assert paused.metadata["pause_reason"] == "user_input_required"
    assert paused.metadata["input_type"] == "approval"
    assert paused.metadata["component_type"] == "tool"
    assert "record_publication" in paused.metadata["question"]
    assert published == []
    assert "run.failed" not in emitted and "tool.failed" not in emitted
    if not dispatch_metadata:
        # Push runs journal the pause themselves; the platform resumes from it.
        assert "workflow.paused" in emitted
        assert emitted[:2] == ["run.started", "tool.started"]

    # The platform dispatches the run again with the pause's metadata and the answer.
    emitted.clear()
    resume_metadata = {**dispatch_metadata, **paused.metadata, "user_response": "approve"}
    completed = await _run_tool(probe, {"title": "Q3 report"}, resume_metadata)

    assert published == ["Q3 report"]
    assert "workflow.paused" not in emitted
    assert "run.started" not in emitted
    assert "tool.completed" in emitted
    if dispatch_metadata:
        assert completed.success is True
        assert completed.event_type == "run.completed"
    else:
        assert completed is None
        assert emitted[-1] == "run.completed"


@pytest.mark.asyncio
async def test_confirmation_tool_run_directly_does_not_run_when_rejected(emitted):
    probe, published = _record_publication()

    paused = await _run_tool(probe, {"title": "Q3 report"})
    await _run_tool(probe, {"title": "Q3 report"}, {**paused.metadata, "user_response": "reject"})

    assert published == []
    assert emitted[-1] == "run.completed"


@pytest.mark.asyncio
async def test_tool_without_confirmation_run_directly_keeps_its_plain_context(emitted):
    seen = []

    @tool
    async def lookup(ctx: Context, key: str) -> str:
        """Look something up."""
        seen.append(type(ctx))
        return key

    await _run_tool(ToolRegistry.get("lookup"), {"key": "a"})

    assert seen == [Context]
    assert emitted == ["run.started", "tool.started", "tool.completed", "run.completed"]
