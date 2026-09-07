"""Durable state writes must acknowledge without occupying the asyncio loop."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from agnt5 import WorkflowContext
from agnt5.workflow import WorkflowEntity


def context(worker):
    ctx = WorkflowContext(run_id="state-run", workflow_entity=WorkflowEntity("state-run"))
    ctx._worker = worker
    ctx._trace_metadata = {"pull_completion_lifecycle_v1": "true"}
    return ctx


@pytest.mark.asyncio
async def test_async_state_write_waits_for_ack_without_blocking_other_runs():
    entered, ack = asyncio.Event(), asyncio.Event()

    async def checkpoint(**kwargs):
        assert kwargs["event_type"] == "workflow.state.changed"
        entered.set()
        await ack.wait()

    worker = Mock(emit_event_async=AsyncMock(side_effect=checkpoint))
    ctx = context(worker)
    writing = asyncio.create_task(ctx.state.set_async("stage", "paid"))
    try:
        await asyncio.wait_for(entered.wait(), 0.5)
        await asyncio.sleep(0)  # another run can be scheduled while the RPC waits
        assert not writing.done()
        assert ctx.state.get("stage") is None
        worker.emit_event_sync.assert_not_called()
        worker.queue_event.assert_not_called()
        ack.set()
        await writing
        assert ctx.state.get("stage") == "paid"
        assert len(ctx._workflow_entity._state_changes) == 1
    finally:
        ack.set()
        await writing


@pytest.mark.asyncio
async def test_failed_async_state_write_does_not_publish_local_success():
    worker = Mock(emit_event_async=AsyncMock(side_effect=RuntimeError("append failed")))
    ctx = context(worker)
    with pytest.raises(RuntimeError, match="append failed"):
        await ctx.state.set_async("stage", "paid")
    assert ctx.state.get("stage") is None
    assert not ctx._workflow_entity._state_changes


@pytest.mark.asyncio
async def test_async_delete_and_sync_compatibility():
    worker = Mock(emit_event_async=AsyncMock())
    ctx = context(worker)
    ctx.state.set("stage", "paid")
    worker.emit_event_sync.assert_called_once()
    await ctx.state.delete_async("stage")
    assert ctx.state.get("stage") is None
    assert ctx._workflow_entity._state_changes[-1]["deleted"] is True
    worker.emit_event_async.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_writes_preserve_order_per_workflow():
    entered, ack = asyncio.Event(), asyncio.Event()
    values = []

    async def checkpoint(**kwargs):
        import json

        values.append(json.loads(kwargs["event_data"])["value"])
        if len(values) == 1:
            entered.set()
            await ack.wait()

    ctx = context(Mock(emit_event_async=AsyncMock(side_effect=checkpoint)))
    first = asyncio.create_task(ctx.state.set_async("stage", "first"))
    await asyncio.wait_for(entered.wait(), 0.5)
    second = asyncio.create_task(ctx.state.set_async("stage", "second"))
    await asyncio.sleep(0)
    assert values == ["first"]
    ack.set()
    await asyncio.gather(first, second)
    assert values == ["first", "second"]
    assert ctx.state.get("stage") == "second"


@pytest.mark.asyncio
async def test_state_batch_also_requires_ack_when_lifecycle_is_deferred():
    from agnt5.events import StateChanged

    worker = Mock(emit_event_batch_async=AsyncMock())
    ctx = context(worker)
    await ctx.emit_batch_async([StateChanged(
        name="workflow", correlation_id="state-change", parent_correlation_id="",
        key="stage", value="paid",
    )])
    worker.queue_event.assert_not_called()
    worker.emit_event_batch_async.assert_awaited_once()
