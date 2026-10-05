"""ctx.caller: who called a run through a hosted MCP server."""

from types import SimpleNamespace

import pytest

from agnt5 import Caller
from agnt5._serialization import serialize
from agnt5.context import caller_from_metadata
from agnt5.function import FunctionContext
from agnt5.worker._executors import ExecutorMixin
from agnt5.workflow import WorkflowContext, WorkflowEntity

MCP_METADATA = {
    "trigger_type": "mcp",
    "mcp.server": "support",
    "mcp.tool": "lookup_order",
    "mcp.subject": "user_123",
    "mcp.auth_method": "oauth",
    "mcp.client": "claude-desktop",
    "project_id": "proj_1",
}

EXPECTED = Caller(
    server="support",
    tool="lookup_order",
    subject="user_123",
    auth_method="oauth",
    client="claude-desktop",
)


def test_caller_from_mcp_metadata():
    assert caller_from_metadata(MCP_METADATA) == EXPECTED


def test_api_key_caller():
    caller = caller_from_metadata(
        {**MCP_METADATA, "mcp.subject": "service_key:key_9", "mcp.auth_method": "api_key", "mcp.client": "curl/8.0"}
    )
    assert caller.subject == "service_key:key_9"
    assert caller.auth_method == "api_key"
    assert caller.client == "curl/8.0"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"project_id": "proj_1"},
        {"trigger_type": "cron", "mcp.server": "support"},
        # trigger_type alone is not reserved; mcp.* is, so it must be there.
        {"trigger_type": "mcp"},
        {**MCP_METADATA, "trigger_type": "api"},
    ],
)
def test_no_caller_when_the_run_was_not_started_by_mcp(metadata):
    assert caller_from_metadata(metadata) is None


def test_caller_is_frozen_and_carries_only_the_caller_fields():
    with pytest.raises(AttributeError):
        EXPECTED.subject = "someone_else"  # type: ignore[misc]
    assert set(vars(EXPECTED)) == {"server", "tool", "subject", "auth_method", "client"}


def test_function_and_workflow_contexts_expose_caller():
    fn_ctx = FunctionContext(
        run_id="run_1", correlation_id="c", parent_correlation_id="p", trace_metadata=MCP_METADATA
    )
    wf_ctx = WorkflowContext(WorkflowEntity("run_1"), run_id="run_1", trace_metadata=MCP_METADATA)
    assert fn_ctx.caller == EXPECTED
    assert wf_ctx.caller == EXPECTED


def test_contexts_without_mcp_metadata_have_no_caller():
    fn_ctx = FunctionContext(run_id="run_1", correlation_id="c", parent_correlation_id="p")
    wf_ctx = WorkflowContext(WorkflowEntity("run_1"), run_id="run_1", trace_metadata={"trigger_type": "cron"})
    assert fn_ctx.caller is None
    assert wf_ctx.caller is None


class _Executor(ExecutorMixin):
    def __init__(self) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = None
        self.service_name = "test"


def _request(metadata):
    return SimpleNamespace(
        invocation_id="run-mcp",
        input_data=serialize({}),
        runtime_context=None,
        metadata=dict(metadata),
        session_id="",
        user_id="",
        attempt=0,
        is_streaming=False,
        component_name="component",
    )


@pytest.fixture
def quiet_emit(monkeypatch):
    async def emit_async(self, event):
        return None

    async def emit_batch_async(self, events):
        return None

    for cls in (FunctionContext, WorkflowContext):
        monkeypatch.setattr(cls, "emit_async", emit_async)
        monkeypatch.setattr(cls, "emit_batch_async", emit_batch_async)


@pytest.mark.asyncio
async def test_a_dispatched_workflow_sees_the_caller_from_dispatch_metadata(quiet_emit):
    seen = []

    async def handler(ctx):
        seen.append(ctx.caller)
        return {"ok": True}

    request = _request(MCP_METADATA)
    await _Executor()._execute_workflow(
        SimpleNamespace(name="triage", handler=handler), request.input_data, request
    )
    assert seen == [EXPECTED]


@pytest.mark.asyncio
async def test_a_dispatched_function_sees_the_caller_from_dispatch_metadata(quiet_emit):
    seen = []

    async def handler(ctx):
        seen.append(ctx.caller)
        return {"ok": True}

    request = _request(MCP_METADATA)
    await _Executor()._execute_function(
        SimpleNamespace(name="lookup_order", handler=handler, retries=None, timeout_ms=None),
        request.input_data,
        request,
    )
    assert seen == [EXPECTED]
