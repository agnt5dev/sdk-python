"""Durability behavior for lifecycle event emission."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import agnt5.events as events_module
from agnt5.events import Completed, ComponentType, EventEmitter, OutputDelta, Started


def checkpoint_started() -> Started:
    return Started(
        name="durable-step",
        correlation_id="step-1",
        parent_correlation_id="workflow-1",
        component_type=ComponentType.WORKFLOW,
    )


def terminal_checkpoint() -> Completed:
    return Completed(
        name="durable-workflow",
        correlation_id="workflow-1",
        parent_correlation_id="run-1",
        component_type=ComponentType.WORKFLOW,
    )


def transient_delta() -> OutputDelta:
    return OutputDelta(
        name="output",
        correlation_id="output-1",
        parent_correlation_id="workflow-1",
        content="chunk",
    )


def test_sync_checkpoint_failure_propagates_without_queue_fallback() -> None:
    worker = MagicMock()
    worker.emit_event_sync.side_effect = RuntimeError("journal unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    with pytest.raises(RuntimeError, match="journal unavailable"):
        emitter.emit(checkpoint_started())

    worker.queue_event.assert_not_called()


def test_sync_transient_queue_failure_remains_best_effort() -> None:
    worker = MagicMock()
    worker.queue_event.side_effect = RuntimeError("stream queue unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    envelope = emitter.emit(transient_delta())

    assert envelope.event_type == "output.delta"


@pytest.mark.asyncio
async def test_async_checkpoint_failure_propagates() -> None:
    worker = MagicMock()
    worker.emit_event_async = AsyncMock(side_effect=RuntimeError("journal unavailable"))
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    with pytest.raises(RuntimeError, match="journal unavailable"):
        await emitter.emit_async(terminal_checkpoint())


@pytest.mark.asyncio
async def test_async_nonterminal_checkpoint_enqueue_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(events_module, "_FIRE_AND_FORGET_NONTERMINAL", True)
    worker = MagicMock()
    worker.queue_event.side_effect = RuntimeError("journal queue unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    with pytest.raises(RuntimeError, match="journal queue unavailable"):
        await emitter.emit_async(checkpoint_started())


@pytest.mark.asyncio
async def test_async_checkpoint_batch_enqueue_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(events_module, "_FIRE_AND_FORGET_NONTERMINAL", True)
    worker = MagicMock()
    worker.queue_event.side_effect = RuntimeError("journal queue unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    with pytest.raises(RuntimeError, match="journal queue unavailable"):
        await emitter.emit_batch_async([checkpoint_started()])


@pytest.mark.asyncio
async def test_async_transient_queue_failure_remains_best_effort() -> None:
    worker = MagicMock()
    worker.queue_event.side_effect = RuntimeError("stream queue unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    envelope = await emitter.emit_async(transient_delta())

    assert envelope.event_type == "output.delta"


@pytest.mark.asyncio
async def test_async_transient_batch_queue_failure_remains_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(events_module, "_FIRE_AND_FORGET_NONTERMINAL", True)
    worker = MagicMock()
    worker.queue_event.side_effect = RuntimeError("stream queue unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    await emitter.emit_batch_async([transient_delta()])
