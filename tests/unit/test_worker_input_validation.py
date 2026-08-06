"""Worker executor input validation tests."""

import base64
from types import SimpleNamespace

import pytest

from agnt5._serialization import deserialize, serialize
from agnt5.activation import ActivationDecision, ActivationDecisionKind, activation_id
from agnt5.worker._executors import ExecutorMixin, _agent_missing_message_error


class _DummyExecutor(ExecutorMixin):
    def __init__(self) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = None
        self.service_name = "test"


class _DurableExecutor(_DummyExecutor):
    def _activation_client_for_metadata(self, _metadata):
        return self

    async def begin(self, request):
        return ActivationDecision(
            kind=ActivationDecisionKind.EXECUTE,
            activation_id=activation_id(
                request.project_id,
                request.run_id,
                request.parent_activation_id,
                request.kind,
                request.stable_key,
            ),
            attempt=2,
            accepted_journal_offset=11,
            fence_token=b"fence-2",
        )


def _request(payload):
    return SimpleNamespace(
        invocation_id="run-invalid-input",
        input_data=serialize(payload),
        runtime_context=None,
        metadata={},
        session_id="",
        user_id="",
        attempt=0,
        is_streaming=False,
        component_name="component",
    )


def _assert_clear_non_object_failure(response, payload):
    assert response is not None
    assert response.success is False
    assert "Component input must be a JSON object" in response.error_message
    assert type(payload).__name__ in response.error_message


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["Alabama", ["a", "b"], 42])
@pytest.mark.parametrize("component_type", ["function", "tool", "agent"])
async def test_non_object_inputs_fail_before_generic_executor_handlers(component_type, payload):
    executor = _DummyExecutor()
    handler_called = False

    async def handler(ctx, **kwargs):
        nonlocal handler_called
        handler_called = True
        return {"ok": True}

    if component_type == "function":
        response = await executor._execute_function(
            SimpleNamespace(
                name="invalid_input_function",
                handler=handler,
                retries=None,
                timeout_ms=None,
            ),
            b"",
            _request(payload),
        )
    elif component_type == "tool":

        class Tool:
            name = "invalid_input_tool"

            async def invoke(self, ctx, **kwargs):
                nonlocal handler_called
                handler_called = True
                return {"ok": True}

        response = await executor._execute_tool(Tool(), b"", _request(payload))
    else:
        response = await executor._execute_agent(
            SimpleNamespace(name="invalid_input_agent"),
            b"",
            _request(payload),
        )

    _assert_clear_non_object_failure(response, payload)
    assert handler_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["Alabama", ["a", "b"], 42])
async def test_non_object_inputs_fail_before_workflow_handler(payload):
    executor = _DummyExecutor()
    handler_called = False

    async def handler(ctx, **kwargs):
        nonlocal handler_called
        handler_called = True
        return {"ok": True}

    request = _request(payload)
    response = await executor._execute_workflow(
        SimpleNamespace(name="invalid_input_workflow", handler=handler),
        request.input_data,
        request,
    )

    _assert_clear_non_object_failure(response, payload)
    assert handler_called is False


@pytest.mark.asyncio
async def test_pull_workflow_pause_returns_terminal_without_queueing_workflow_terminal():
    executor = _DummyExecutor()
    queued_events = []

    async def handler(ctx):
        ctx.emit = queued_events.append
        await ctx.wait_for_user(
            "Approve deployment?",
            input_type="approval",
            options=[
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"},
            ],
        )

    request = _request({})
    request.metadata = {
        "dispatch_mode": "pull",
        "lease_id": "lease-hitl",
    }
    request.component_type = "workflow"
    response = await executor._execute_workflow(
        SimpleNamespace(name="approval_workflow", handler=handler),
        request.input_data,
        request,
    )

    assert response is not None
    assert response.success is True
    assert response.event_type == "workflow.paused"
    assert response.attempt == 0
    assert deserialize(response.output_data) == {
        "_paused": True,
        "question": "Approve deployment?",
        "pause_index": 0,
    }
    queued_types = [event.event_type for event in queued_events]
    assert "workflow.paused" not in queued_types
    assert "approval.requested" in queued_types
    assert "workflow.step.paused" in queued_types
    assert response.metadata["question"] == "Approve deployment?"
    assert response.metadata["step_name"] == "wait_for_user_0"
    assert response.metadata["step_correlation_id"]
    assert response.metadata["workflow_correlation_id"]


@pytest.mark.asyncio
async def test_negotiated_activation_without_native_client_fails_before_workflow_code():
    executor = _DummyExecutor()
    handler_called = False

    async def handler(ctx):
        nonlocal handler_called
        handler_called = True

    request = _request({})
    request.metadata = {"durable_activation_v1": "true"}
    response = await executor._execute_workflow(
        SimpleNamespace(name="durable_workflow", handler=handler),
        request.input_data,
        request,
    )

    assert response is not None
    assert response.success is False
    assert "executor has no activation client" in response.error_message
    assert handler_called is False


@pytest.mark.asyncio
async def test_durable_workflow_sleep_returns_typed_worker_suspension():
    executor = _DurableExecutor()

    async def handler(ctx):
        await ctx.sleep(2.5, name="backoff")

    request = _request({})
    request.metadata = {
        "durable_activation_v1": "true",
        "durable_suspension_v1": "true",
        "project_id": "project-1",
        "worker_session_id": "session-1",
        "lease_id": "lease-1",
        "activation_artifact_sha256": base64.b64encode(b"a" * 32).decode(),
        "activation_definition_version": "v1",
        "activation_definition_config": '["object",[]]',
        "component_name": "durable_workflow",
    }
    response = await executor._execute_workflow(
        SimpleNamespace(name="durable_workflow", handler=handler),
        request.input_data,
        request,
    )

    assert response is not None
    assert response.success is True
    assert response.event_type == "workflow.paused"
    assert response.worker_suspension is not None
    assert response.worker_suspension.timer_key == "sleep:backoff"
    assert response.worker_suspension.delay_ms == 2500
    assert response.worker_suspension.attempt == 2
    assert bytes(response.worker_suspension.fence_token) == b"fence-2"


def test_agent_missing_message_error_lists_received_keys():
    message = _agent_missing_message_error({"state": "Alabama"})

    assert "requires a 'message' key" in message
    assert "Received keys: ['state']" in message
    assert "dataset input matches the component's expected schema" in message
