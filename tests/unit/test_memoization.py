import json

import pytest
from agnt5.context import Context


class FakeCheckpointClient:
    def __init__(self, payload):
        self.payload = payload

    async def get_memoized_step(self, tenant_id, run_id, step_key):
        return json.dumps(self.payload).encode()


@pytest.mark.asyncio
async def test_lm_cache_hash_mismatch_is_not_replayed():
    ctx = Context(
        run_id="run-1",
        correlation_id="corr",
        parent_correlation_id="",
        enable_memoization=True,
        trace_metadata={"project_id": "project-1"},
    )
    assert ctx._memo is not None
    ctx._memo._checkpoint_client = FakeCheckpointClient(
        {"input_hash": "stored-hash", "output_data": {"text": "stale output"}}
    )
    ctx._memo._connected = True

    cached = await ctx._memo.get_cached_lm_result("lm.0", "current-hash")

    assert cached is None


@pytest.mark.asyncio
async def test_tool_cache_hash_mismatch_is_not_replayed():
    ctx = Context(
        run_id="run-1",
        correlation_id="corr",
        parent_correlation_id="",
        enable_memoization=True,
        trace_metadata={"project_id": "project-1"},
    )
    assert ctx._memo is not None
    ctx._memo._checkpoint_client = FakeCheckpointClient(
        {"input_hash": "stored-hash", "output_data": {"value": "stale"}}
    )
    ctx._memo._connected = True

    found, cached = await ctx._memo.get_cached_tool_result("tool.search.0", "current-hash")

    assert found is False
    assert cached is None
