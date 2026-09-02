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


def test_event_metadata_cannot_override_execution_authority() -> None:
    emitter = EventEmitter(
        run_id="run-1",
        base_metadata={
            "dispatch_mode": "pull",
            "worker_id": "worker-1",
            "worker_session_id": "session-1",
            "lease_id": "lease-1",
            "lease_attempt": "1",
            "assignment_commit_offset": "42",
        },
    )

    metadata = emitter._event_metadata(
        {
            "lease_id": "forged-lease",
            "worker_id": "forged-worker",
            "assignment_commit_offset": "999",
            "custom": "preserved",
        }
    )

    assert metadata["lease_id"] == "lease-1"
    assert metadata["worker_id"] == "worker-1"
    assert metadata["assignment_commit_offset"] == "42"
    assert metadata["custom"] == "preserved"


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


def test_observed_lifecycle_event_queues_without_checkpoint_rpc() -> None:
    worker = MagicMock()
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    envelope = emitter.emit_observed(checkpoint_started())

    assert envelope.event_type == "workflow.started"
    worker.emit_event_sync.assert_not_called()
    worker.emit_event_async.assert_not_called()
    worker.queue_event.assert_called_once()
    assert worker.queue_event.call_args.kwargs["is_streaming"] is False


def test_observed_lifecycle_queue_failure_is_best_effort() -> None:
    worker = MagicMock()
    worker.queue_event.side_effect = RuntimeError("journal queue unavailable")
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    envelope = emitter.emit_observed(checkpoint_started())

    assert envelope.event_type == "workflow.started"


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


def lifecycle_batch() -> list:
    return [
        Started(
            name="run",
            correlation_id="run-1",
            parent_correlation_id="",
            component_type=ComponentType.RUN,
        ),
        Started(
            name="wf",
            correlation_id="workflow-1",
            parent_correlation_id="run-1",
            component_type=ComponentType.WORKFLOW,
        ),
    ]


@pytest.mark.asyncio
async def test_deferred_lifecycle_queues_nonterminal_checkpoints_for_complete_job() -> None:
    worker = MagicMock()
    worker.emit_event_async = AsyncMock()
    worker.emit_event_batch_async = AsyncMock()
    emitter = EventEmitter(run_id="run-1", defer_lifecycle=True)
    emitter.set_worker(worker)

    await emitter.emit_batch_async(lifecycle_batch())
    await emitter.emit_async(checkpoint_started())

    # sdk-core holds queued events for the run and carries them in CompleteJob.
    assert worker.queue_event.call_count == 3
    queued_types = [call.kwargs["event_type"] for call in worker.queue_event.call_args_list]
    assert queued_types == ["run.started", "workflow.started", "workflow.started"]
    assert all(call.kwargs["is_streaming"] is False for call in worker.queue_event.call_args_list)
    worker.emit_event_batch_async.assert_not_awaited()
    worker.emit_event_async.assert_not_awaited()

    # Terminal events still await so the core can pre-flush and fence them.
    await emitter.emit_async(terminal_checkpoint())
    worker.emit_event_async.assert_awaited_once()
    assert worker.emit_event_async.await_args.kwargs["event_type"] == "workflow.completed"


@pytest.mark.asyncio
async def test_lifecycle_awaits_the_batch_rpc_when_not_deferred(monkeypatch) -> None:
    monkeypatch.setattr(events_module, "_FIRE_AND_FORGET_NONTERMINAL", False)
    worker = MagicMock()
    worker.emit_event_batch_async = AsyncMock()
    emitter = EventEmitter(run_id="run-1")
    emitter.set_worker(worker)

    await emitter.emit_batch_async(lifecycle_batch())

    worker.queue_event.assert_not_called()
    worker.emit_event_batch_async.assert_awaited_once()
    assert not emitter.defer_lifecycle


def test_context_defers_lifecycle_only_for_negotiated_non_streaming_pull_runs() -> None:
    from agnt5.context import Context

    negotiated = {"dispatch_mode": "pull", "pull_completion_lifecycle_v1": "true"}
    deferred = Context(
        run_id="run-1",
        correlation_id="c",
        parent_correlation_id="p",
        trace_metadata=negotiated,
    )
    assert deferred._get_emitter().defer_lifecycle is True

    streaming = Context(
        run_id="run-2",
        correlation_id="c",
        parent_correlation_id="p",
        is_streaming=True,
        trace_metadata=negotiated,
    )
    assert streaming._get_emitter().defer_lifecycle is False

    legacy = Context(
        run_id="run-3",
        correlation_id="c",
        parent_correlation_id="p",
        trace_metadata={"dispatch_mode": "pull"},
    )
    assert legacy._get_emitter().defer_lifecycle is False
