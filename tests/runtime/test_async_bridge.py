"""The Python/Rust bridge, checked against a real agnt5-runtime instead of fakes.

The runtime, gateway, transport and worker process are all real. The only
scripted piece is the agent's model, so a stream needs no provider.
"""

from __future__ import annotations

import re
import statistics
import uuid

import pytest

pytestmark = pytest.mark.runtime

APP = "bridge_app.py"


def _run_overlapping(runtime, workers, count: int) -> dict[str, dict]:
    """Submit `count` runs that hold a context value for a second, so they
    overlap on one worker, and return each run's output by its key."""
    workers.start(APP, durable_steps=True)
    client = runtime.client()
    keys = [uuid.uuid4().hex[:8] for _ in range(count)]
    with runtime.known_sqlite_lock():
        submitted = {
            key: client.submit(
                "hold_a_context_value", {"key": key, "hold_s": 1.0}, component_type="workflow"
            )
            for key in keys
        }
        outputs = {}
        for key, run in submitted.items():
            result = client.wait_for_result(run.run_id, timeout=60)
            assert result.is_success, result.error
            outputs[key] = {**result.output, "submitted_run_id": run.run_id}
    # Runs really overlapped on the worker, so a leak between them would show.
    assert _peak_concurrency(workers.recorded()) >= 2, "no two runs overlapped"
    return outputs


def test_concurrent_runs_each_keep_their_own_context(isolated_runtime, isolated_workers):
    """Runs that overlap on one worker never see each other's context values."""
    outputs = _run_overlapping(isolated_runtime, isolated_workers, count=3)

    for key, output in outputs.items():
        assert output["seen_at_start"] == "unset", f"run {key} started with another run's value"
        assert output["seen_at_end"] == key
        # A sync step runs in a thread, and still sees its own run.
        assert output["step_run_id"] == output["submitted_run_id"]


def test_each_run_sees_its_own_trace_context(agnt5_runtime, workers):
    """The worker hands each run its execution span's trace headers, which LLM
    calls use to parent their spans. Two runs get two different contexts."""
    workers.start(APP, durable_steps=True)
    client = agnt5_runtime.client()

    traceparents = []
    for _ in range(2):
        run = client.submit(
            "hold_a_context_value",
            {"key": uuid.uuid4().hex[:8], "hold_s": 0.0},
            component_type="workflow",
        )
        result = client.wait_for_result(run.run_id, timeout=60)
        assert result.is_success, result.error
        traceparents.append(result.output["traceparent"])

    for traceparent in traceparents:
        assert traceparent and re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]", traceparent), (
            f"run started without a trace context: {traceparent!r}"
        )
    assert traceparents[0] != traceparents[1], "two runs shared one trace context"


def test_a_sync_step_sees_its_runs_context(agnt5_runtime, workers):
    """A sync step runs in an executor thread; the run's context goes with it."""
    workers.start(APP, durable_steps=True)
    client = agnt5_runtime.client()

    run = client.submit(
        "hold_a_context_value",
        {"key": uuid.uuid4().hex[:8], "hold_s": 0.0},
        component_type="workflow",
    )
    result = client.wait_for_result(run.run_id, timeout=60)

    assert result.is_success, result.error
    assert result.output["step_run_id"] == run.run_id


def _peak_concurrency(recorded: list[dict]) -> int:
    """The most runs that were between their start and end at the same moment."""
    edges = sorted(
        (r["at"], 1 if r["event"] == "start" else -1)
        for r in recorded
        if r["event"] in ("start", "end")
    )
    running = peak = 0
    for _, step in edges:
        running += step
        peak = max(peak, running)
    return peak


def _stream_agent(agnt5_runtime, key: str):
    return list(
        agnt5_runtime.client().stream_events(
            "streaming_reporter", {"message": key}, component_type="agent", wait_timeout=60
        )
    )


def test_a_streamed_agent_does_not_stall_at_each_block_boundary(agnt5_runtime, workers):
    """At each block boundary the worker waits for the run's stream events to
    leave its queue before asking the model for more. The wait ends when they
    leave, not after a fixed delay, so a six-block answer streams promptly."""
    worker = workers.start(APP, durable_steps=True)
    key = uuid.uuid4().hex[:8]

    events = _stream_agent(agnt5_runtime, key)

    completed = [event for event in events if event.event_type == "run.completed"]
    assert completed, f"{[event.event_type for event in events]}\n{worker.log_tail()}"
    assert "block-0 block-1 block-2 block-3 block-4 block-5" in str(completed[-1].data)

    starts = sorted(
        r["at"] for r in workers.recorded() if r["event"] == "block_start" and r["key"] == key
    )
    assert len(starts) == 6, worker.log_tail()
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:])]
    # The old fixed wait was 200 ms per boundary. The sender drains every
    # 50 ms, so a prompt boundary stays well under that.
    assert statistics.median(gaps) < 0.15, f"gaps between blocks: {[round(g, 3) for g in gaps]}"


def test_a_streamed_agents_tokens_reach_the_client(agnt5_runtime, workers):
    """Each block's text reaches a streaming client as it is produced."""
    workers.start(APP, durable_steps=True)
    key = uuid.uuid4().hex[:8]

    events = _stream_agent(agnt5_runtime, key)

    deltas = [str(event.data) for event in events if event.event_type == "lm.message.delta"]
    assert [index for index in range(6) if any(f"block-{index}" in d for d in deltas)] == list(
        range(6)
    ), [event.event_type for event in events]
