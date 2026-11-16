"""Durable workflow behavior, checked against a real agnt5-runtime instead of fakes."""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

pytestmark = pytest.mark.runtime


def test_each_step_runs_once_and_the_run_returns_its_result(agnt5_runtime, workers):
    workers.start()
    order_id = uuid.uuid4().hex[:8]

    result = agnt5_runtime.client().run(
        "refund_order", {"order_id": order_id}, component_type="workflow"
    )

    assert result.is_success, result.error
    assert result.output == {
        "steps": [f"lookup:{order_id}", f"refund:{order_id}", f"notify:{order_id}"]
    }
    assert workers.effects() == [f"lookup:{order_id}", f"refund:{order_id}", f"notify:{order_id}"]


def _sleep_through_a_worker_crash(agnt5_runtime, workers, sleep_s: float):
    """Start a run that sleeps, kill its worker once the runtime holds the
    sleep's timer, start a fresh worker, and wait for the result."""
    first = workers.start()
    client = agnt5_runtime.client()
    run_key = uuid.uuid4().hex[:8]

    started = time.monotonic()
    submitted = client.submit(
        "sleep_between_steps", {"run_key": run_key, "seconds": sleep_s}, component_type="workflow"
    )
    # Kill the worker only once the runtime holds the sleep's timer. Killing it
    # earlier, before the suspension reaches the runtime, is a different case:
    # the run then waits for the worker's lease to expire.
    agnt5_runtime.wait_until(
        lambda: "timer.scheduled" in agnt5_runtime.event_types(submitted.run_id),
        timeout=30,
        what="the runtime to schedule the sleep's timer",
        on_timeout=first.log_tail,
    )
    first.kill()
    second = workers.start()

    result = client.wait_for_result(submitted.run_id, timeout=120)
    elapsed = time.monotonic() - started
    assert result.is_success, f"{result.error}\n{second.log_tail()}"
    return run_key, submitted.run_id, elapsed


@pytest.mark.xfail(
    strict=False,
    reason=(
        "needs an agnt5-sdk-core release that derives the activation artifact "
        "identity when none is configured (agnt5dev/sdk-core#48); until then a "
        "self-hosted worker falls back to an in-process sleep. Remove this mark "
        "when Cargo.lock moves to that release."
    ),
)
def test_durable_sleep_survives_a_worker_crash(agnt5_runtime, workers):
    """The sleep's timer lives in the runtime, so killing the worker mid-sleep
    neither restarts the sleep nor repeats the step before it."""
    run_key, run_id, elapsed = _sleep_through_a_worker_crash(agnt5_runtime, workers, sleep_s=6.0)

    assert workers.effects().count(f"before:{run_key}") == 1
    assert workers.effects().count(f"after:{run_key}") == 1
    assert agnt5_runtime.event_types(run_id).count("timer.scheduled") == 1
    assert elapsed >= 6.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known runtime bug: a run resumed from a durable timer stays leased to the "
        "worker that suspended it, so after that worker crashes the run waits for "
        "the old lease (60 s) before a healthy worker can take it"
    ),
)
def test_a_run_resumed_after_a_worker_crash_moves_to_a_healthy_worker_promptly(
    agnt5_runtime, workers
):
    _, _, elapsed = _sleep_through_a_worker_crash(agnt5_runtime, workers, sleep_s=6.0)
    assert elapsed < 6.0 + 30, f"run took {elapsed:.1f}s"


def test_a_paused_run_continues_with_the_answer(agnt5_runtime, workers):
    worker = workers.start()
    client = agnt5_runtime.client()
    run_key = uuid.uuid4().hex[:8]

    submitted = client.submit("ask_for_approval", {"run_key": run_key}, component_type="workflow")
    agnt5_runtime.wait_until(
        lambda: agnt5_runtime.status(submitted.run_id) == "paused",
        timeout=30,
        what="the run to pause",
        on_timeout=worker.log_tail,
    )

    resumed = httpx.post(
        f"{agnt5_runtime.http_url}/v1/runs/{submitted.run_id}/resume",
        json={"user_response": "approved"},
        timeout=10,
    )
    assert resumed.status_code == 200, (
        f"{resumed.text}\nevents: {agnt5_runtime.event_types(submitted.run_id)}"
    )

    result = client.wait_for_result(submitted.run_id, timeout=60)
    assert result.is_success, f"{result.error}\n{worker.log_tail()}"
    assert result.output == {"answer": "approved"}
    assert workers.effects().count(f"before_pause:{run_key}") == 1
    assert workers.effects().count(f"after_pause:{run_key}") == 1
