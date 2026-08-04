"""Worker executor contextvar lifecycle tests."""

from types import SimpleNamespace

import pytest

from agnt5._state_adapter import _entity_state_adapter_ctx
from agnt5.tracing import _current_span
from agnt5.worker._executors import ExecutorMixin


class _DummyExecutor(ExecutorMixin):
    def __init__(self) -> None:
        self._entity_state_adapter = object()
        self._checkpoint_client = None
        self._rust_worker = None
        self.service_name = "test"


@pytest.mark.asyncio
async def test_execute_with_context_resets_span_and_state_adapter_contextvars():
    executor_mixin = _DummyExecutor()
    request = SimpleNamespace(
        input_data=b"{}",
        runtime_context=SimpleNamespace(trace_id="trace-123", span_id="span-123"),
    )
    initial_span = _current_span.get()
    initial_adapter = _entity_state_adapter_ctx.get()

    def create_context(input_dict, req):
        assert _entity_state_adapter_ctx.get() is executor_mixin._entity_state_adapter
        return SimpleNamespace(input=input_dict, request=req)

    async def execute(ctx, input_dict, req):
        current_span = _current_span.get()
        assert current_span is not None
        assert current_span.trace_id == "trace-123"
        assert current_span.span_id == "span-123"
        assert _entity_state_adapter_ctx.get() is executor_mixin._entity_state_adapter
        return None

    await executor_mixin._execute_with_context(
        request,
        create_context,
        execute,
        "Test",
    )

    assert _current_span.get() is initial_span
    assert _entity_state_adapter_ctx.get() is initial_adapter
