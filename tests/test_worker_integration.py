import json

import pytest

from agnt5 import Context, durable
from agnt5.function import clear_registry
from agnt5.worker import Worker
from agnt5._compat import _rust_available

try:
    from agnt5._core import PyExecuteComponentRequest, PyStepCheckpoint
except Exception:  # pragma: no cover - extension module not available
    PyExecuteComponentRequest = None  # type: ignore


@pytest.mark.skipif(not _rust_available, reason="Rust core not available")
@pytest.mark.skipif(PyExecuteComponentRequest is None, reason="PyO3 bindings unavailable")
def test_worker_replays_step_checkpoints(tmp_path):
    if not hasattr(PyExecuteComponentRequest, "step_checkpoints"):
        pytest.skip("PyExecuteComponentRequest missing step_checkpoints attribute")

    clear_registry()

    @durable.function(name="cached_handler")
    async def cached_handler(ctx: Context) -> int:
        return await ctx.step("cached", lambda: 99)

    worker = Worker(service_name="svc-test", service_version="1.0.0")
    worker._register_components()

    request = PyExecuteComponentRequest(
        invocation_id="inv-1",
        service_name="svc-test",
        component_name="cached_handler",
        input_data=list(b"{}"),
        metadata={"run_id": "run-1", "attempt": "1"},
        step_checkpoints=[
            PyStepCheckpoint(
                name="cached",
                key=None,
                status="SUCCEEDED",
                attempt=1,
                updated_at="2024-11-30T12:34:56+00:00",
                result=json.dumps(41).encode("utf-8"),
            )
        ],
    )

    response = worker._handle_message(request)
    assert response.success is True
    assert json.loads(bytes(response.output_data).decode("utf-8")) == 41

    clear_registry()
