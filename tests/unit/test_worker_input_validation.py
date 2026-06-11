"""Worker executor input validation tests."""

from types import SimpleNamespace

import pytest

from agnt5._serialization import serialize
from agnt5.worker._executors import ExecutorMixin, _agent_missing_message_error


class _DummyExecutor(ExecutorMixin):
    def __init__(self) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = None
        self.service_name = "test"


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
async def test_non_object_inputs_fail_before_generic_executor_handlers(
    component_type, payload
):
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


def test_agent_missing_message_error_lists_received_keys():
    message = _agent_missing_message_error({"state": "Alabama"})

    assert "requires a 'message' key" in message
    assert "Received keys: ['state']" in message
    assert "dataset input matches the component's expected schema" in message
