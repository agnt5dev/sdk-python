"""Required checkpoints must yield the asyncio loop across SDK execution paths."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agnt5 import Agent, Context
from agnt5.agent import AgentContext, AgentRegistry
from agnt5.context import _current_context, get_current_context, set_current_context
from agnt5.events import ComponentType, Started, is_checkpoint_event
from agnt5.function import FunctionContext, FunctionRegistry, function
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, LMClient, Message
from agnt5.lm.events import LMCompleted
from agnt5.sandbox import ExecuteCodeResult, ReadFileResult, WriteFileResult
from agnt5.sandbox_tools import (
    _make_execute_code_handler,
    _make_read_file_handler,
    _make_write_file_handler,
    _SandboxToolState,
)
from agnt5.tool import Tool, ToolRegistry
from agnt5.workflow import WorkflowContext, WorkflowEntity

CASES = [
    ("function", "function.started"),
    ("function", "function.completed"),
    ("function", "function.failed"),
    ("agent", "agent.started"),
    ("agent", "agent.completed"),
    ("agent", "agent.failed"),
    ("agent", "agent.iteration.started"),
    ("agent", "agent.iteration.completed"),
    ("agent_tool", "tool_call.started"),
    ("agent_tool", "tool_call.completed"),
    ("agent_tool", "tool_call.failed"),
    ("agent_legacy", "agent.started"),
    ("agent_legacy", "agent.completed"),
    ("agent_legacy", "agent.failed"),
    ("agent_legacy_tool", "tool_call.started"),
    ("agent_legacy_tool", "tool_call.completed"),
    ("agent_legacy_tool", "tool_call.failed"),
    ("agent_sandbox", "sandbox.execute.started"),
    ("agent_sandbox", "sandbox.execute.completed"),
    ("agent_sandbox", "sandbox.execute.failed"),
    ("model_generate", "lm.started"),
    ("model_generate", "lm.completed"),
    ("model_generate", "lm.failed"),
    ("model_stream", "lm.started"),
    ("model_stream", "lm.completed"),
    ("model_stream", "lm.failed"),
    ("sandbox_execute", "sandbox.execute.started"),
    ("sandbox_execute", "sandbox.execute.completed"),
    ("sandbox_execute", "sandbox.execute.failed"),
    ("sandbox_write", "sandbox.file.written"),
    ("sandbox_read", "sandbox.file.read"),
    ("workflow_stream", "workflow.started"),
]


@pytest.fixture(autouse=True)
def clear_registries():
    FunctionRegistry.clear()
    AgentRegistry.clear()
    ToolRegistry.clear()
    yield
    FunctionRegistry.clear()
    AgentRegistry.clear()
    ToolRegistry.clear()


@pytest.fixture(params=["ordinary", "pull", "fire_and_forget"])
def checkpoint_metadata(request, monkeypatch):
    import agnt5.events as events_module

    monkeypatch.setattr(
        events_module, "_FIRE_AND_FORGET_NONTERMINAL", request.param == "fire_and_forget"
    )
    if request.param == "pull":
        return {"dispatch_mode": "pull", "pull_completion_lifecycle_v1": "true"}
    return {}


class PendingCheckpointWorker:
    def __init__(self, target):
        self.target = target
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.events = []
        self.acks = []
        self.reject = False
        self.failure = RuntimeError("checkpoint rejected")

    def emit_event_sync(self, **event):
        raise AssertionError("async SDK code used a blocking checkpoint wait")

    def queue_event(self, **event):
        assert not is_checkpoint_event(event["event_type"]), "required ACK was deferred"

    async def emit_event_async(self, **event):
        self.events.append(event)
        if event["event_type"] == self.target and event["run_id"] == "blocked-run":
            self.entered.set()
            await self.release.wait()
            if self.reject:
                raise self.failure
        self.acks.append(event)


def operation_for(path, worker, metadata, *, sandbox=None):
    state = {"executed": False, "tool_executed": False, "sandbox_executed": False,
             "success_observed": False}
    fail = worker.target.endswith(".failed")
    options = dict(run_id="blocked-run", correlation_id="root",
                   parent_correlation_id="parent", worker=worker, trace_metadata=metadata)
    context = Context(**options)

    async def backend(*args, **kwargs):
        state["executed"] = True
        if fail:
            raise RuntimeError("backend failed")
        return GenerateResponse(text="answer")

    if path == "function":
        context = FunctionContext(**options)

        @function
        async def dependency(ctx):
            return await backend()

        def invoke():
            return dependency(context)
    elif path.startswith("agent"):
        class Model(LanguageModel):
            calls = 0

            async def generate(self, request):
                state["executed"] = True
                self.calls += 1
                if path in ("agent", "agent_legacy") and fail:
                    raise RuntimeError("backend failed")
                calls = ([{"id": "call-1", "name": "dependency", "arguments": "{}"}]
                         if path in ("agent_tool", "agent_legacy_tool", "agent_sandbox") and self.calls == 1 else None)
                return GenerateResponse(text="answer", tool_calls=calls)

            async def stream(self, request):
                response = await self.generate(request)
                yield LMCompleted(name="model", correlation_id="model-1",
                                  parent_correlation_id="iteration-1", model="model",
                                  provider="test", output_data={"text": response.text})

        async def handler(ctx):
            state["tool_executed"] = True
            if path == "agent_sandbox":
                async def execute(*args, **kwargs):
                    state["sandbox_executed"] = True
                    if fail:
                        raise RuntimeError("backend failed")
                    return ExecuteCodeResult(stdout="answer", stderr="", exit_code=0,
                                             execution_time_ms=1)

                sandbox = _SandboxToolState(SimpleNamespace(execute_code=execute))
                return await _make_execute_code_handler(sandbox)(ctx, "print(1)")
            if fail:
                raise RuntimeError("tool failed")
            return {"ok": True}

        tools = ([Tool(name="dependency", description="test", handler=handler)]
                 if path in ("agent_tool", "agent_legacy_tool", "agent_sandbox") else [])
        agent = Agent(name="probe", model=Model(), instructions="Answer briefly", tools=tools,
                      sandbox=sandbox)
        context = AgentContext(run_id=options["run_id"], agent_name=agent.name,
                               worker=worker, trace_metadata=metadata)
        context._get_initial_conversation_history = AsyncMock(return_value=[])
        context.save_conversation_history = AsyncMock()
        def invoke():
            if path in ("agent_legacy", "agent_legacy_tool"):
                return agent._run_impl("hello", context=context)
            return agent.run("hello", context=context)
    elif path.startswith("model"):
        client = object.__new__(LMClient)
        client._provider = "openai"
        client._default_model = "test"
        client._convert_response = lambda response: response

        async def chunks(**kwargs):
            response = await backend()
            yield SimpleNamespace(chunk_type="completed", block_type=None, index=0,
                                  text=response.text, model="test", usage=None,
                                  finish_reason="stop", tool_calls=None)

        client._rust_lm = SimpleNamespace(generate=backend, stream_iter=chunks)
        request = GenerateRequest(model="openai/test", messages=[Message.user("hello")])

        async def invoke():
            if path == "model_generate":
                return await client.generate(request)
            async for event in client.stream(request):
                if event.event_type == "lm.completed":
                    state["success_observed"] = True

    elif path.startswith("sandbox"):
        async def execute(*args, **kwargs):
            await backend()
            return ExecuteCodeResult(stdout="answer", stderr="", exit_code=0, execution_time_ms=1)

        async def write(*args, **kwargs):
            state["executed"] = True
            return WriteFileResult(success=True, path="file", size=1)

        async def read(*args, **kwargs):
            state["executed"] = True
            return ReadFileResult(path="file", content=b"x", size=1, mode=0, is_dir=False)

        sandbox = _SandboxToolState(SimpleNamespace(execute_code=execute, write_file=write,
                                                   read_file=read))
        if path == "sandbox_execute":
            def invoke():
                return _make_execute_code_handler(sandbox)(context, "print(1)")
        elif path == "sandbox_write":
            def invoke():
                return _make_write_file_handler(sandbox)(context, "file", "x")
        else:
            def invoke():
                return _make_read_file_handler(sandbox)(context, "file")
    else:
        context = WorkflowContext(workflow_entity=WorkflowEntity(run_id="blocked-run"), **options)

        async def events():
            yield Started(name="streamed", correlation_id="streamed-1",
                          parent_correlation_id="root", component_type=ComponentType.WORKFLOW)
            state["executed"] = True

        def invoke():
            return context._consume_streaming_result(events(), "streamed")

    outer = Context(run_id="outer", correlation_id="outer", parent_correlation_id="")

    async def run():
        token = set_current_context(context if path != "function" else outer)
        try:
            return await invoke()
        finally:
            state["context_after"] = get_current_context()
            _current_context.reset(token)

    state["expected_context"] = context if path != "function" else outer
    original_parent = context.parent_correlation_id
    state["raises"] = fail and path not in ("agent_tool", "agent_legacy_tool", "agent_sandbox")
    state["context"] = context
    state["original_parent"] = original_parent
    return run(), state


async def wait_for_checkpoint(task, worker):
    entered = asyncio.create_task(worker.entered.wait())
    try:
        done, _ = await asyncio.wait({task, entered}, timeout=2,
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
            pytest.fail("operation finished without awaiting its required checkpoint")
        assert entered in done, "operation did not reach the checkpoint"
    finally:
        entered.cancel()
        await asyncio.gather(entered, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path,boundary", CASES)
async def test_async_paths_yield_while_preserving_required_ack(checkpoint_metadata, path, boundary):
    worker = PendingCheckpointWorker(boundary)
    operation, state = operation_for(path, worker, checkpoint_metadata)
    task = asyncio.create_task(operation)
    try:
        await wait_for_checkpoint(task, worker)
        progress = asyncio.Event()

        @function
        async def independent(ctx):
            progress.set()
            return "independent"

        other = FunctionContext(run_id="other-run", correlation_id="other",
                                parent_correlation_id="parent", worker=worker,
                                trace_metadata=checkpoint_metadata)
        assert await asyncio.wait_for(independent(other), timeout=2) == "independent"
        assert progress.is_set()
        assert not task.done()
        assert not state["success_observed"]
        assert all(event["event_type"] != boundary or event["run_id"] != "blocked-run"
                   for event in worker.acks)
        if boundary.endswith(".started"):
            if path in ("agent_tool", "agent_legacy_tool"):
                assert not state["tool_executed"]
            elif path == "agent_sandbox":
                assert not state["sandbox_executed"]
            else:
                assert not state["executed"]
        worker.release.set()
        if state["raises"]:
            with pytest.raises(RuntimeError, match="backend failed"):
                await task
        else:
            await task
        assert any(event["event_type"] == boundary and event["run_id"] == "blocked-run"
                   for event in worker.acks)
        assert state["context_after"] is state["expected_context"]
        for event in worker.events:
            payload = json.loads(event["event_data"])
            assert event["metadata"]["correlation_id"] == payload["correlation_id"]
            assert event["metadata"].get("parent_correlation_id", "") == payload.get("parent_correlation_id", "")
    finally:
        worker.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path,boundary", CASES)
async def test_async_checkpoint_rejection_is_observable(path, boundary):
    worker = PendingCheckpointWorker(boundary)
    worker.reject = True
    worker.release.set()
    operation, _ = operation_for(path, worker, {})
    with pytest.raises(RuntimeError, match="checkpoint rejected") as caught:
        await operation
    assert caught.value is worker.failure


@pytest.mark.asyncio
@pytest.mark.parametrize("path,boundary", CASES)
async def test_async_checkpoint_cancellation_does_not_report_success(path, boundary):
    worker = PendingCheckpointWorker(boundary)
    operation, state = operation_for(path, worker, {})
    task = asyncio.create_task(operation)
    try:
        await wait_for_checkpoint(task, worker)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not state["success_observed"]
        assert state["context_after"] is state["expected_context"]
        context = state["context"]
        assert context.parent_correlation_id == state["original_parent"]
        assert all(event["event_type"] != boundary for event in worker.acks)
    finally:
        worker.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["agent", "agent_legacy"])
@pytest.mark.parametrize("outcome", ["cancel", "reject", "success"])
async def test_agent_start_checkpoint_always_closes_sandbox(checkpoint_metadata, path, outcome):
    worker = PendingCheckpointWorker("agent.started")
    sandbox = SimpleNamespace(close=AsyncMock(), execute_code=AsyncMock(),
                              write_file=AsyncMock(), read_file=AsyncMock(), list_files=AsyncMock())
    operation, state = operation_for(path, worker, checkpoint_metadata, sandbox=sandbox)
    task = asyncio.create_task(operation)
    try:
        await wait_for_checkpoint(task, worker)
        assert not state["executed"]
        assert not task.done()
        sandbox.close.assert_not_awaited()
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "reject":
            worker.reject = True
            worker.release.set()
            with pytest.raises(RuntimeError, match="checkpoint rejected") as caught:
                await task
            assert caught.value is worker.failure
        else:
            worker.release.set()
            await task

        sandbox.close.assert_awaited_once()
        assert state["executed"] == (outcome == "success")
        assert state["context_after"] is state["expected_context"]
        assert state["context"].parent_correlation_id == state["original_parent"]
        if outcome != "success":
            assert not any(event["event_type"] in ("agent.completed", "agent.failed")
                           for event in worker.events)
    finally:
        worker.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
