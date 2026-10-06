"""ctx.progress: report how far a run has got."""

import asyncio
import json
import math

import pytest

from agnt5 import progress as progress_module
from agnt5.events import Completed, ComponentType
from agnt5.function import FunctionContext
from agnt5.progress import ProgressReporter, progress_payload
from agnt5.workflow import WorkflowContext, WorkflowEntity


class FakeWorker:
    """Records what reaches the journal through the native worker."""

    def __init__(self, fail: bool = False) -> None:
        self.appended: list[dict] = []
        self.queued: list[dict] = []
        self.fail = fail

    async def emit_event_async(self, **kwargs):
        if self.fail:
            raise RuntimeError("engine unavailable")
        self.appended.append(kwargs)

    def queue_event(self, **kwargs):
        self.queued.append(kwargs)

    def progress(self) -> list[dict]:
        return [
            json.loads(call["event_data"])
            for call in self.appended
            if call["event_type"] == "progress.update"
        ]


@pytest.fixture(autouse=True)
def fast_interval(monkeypatch):
    monkeypatch.setattr(progress_module, "MIN_INTERVAL_SECONDS", 0.05)


def function_ctx(worker=None) -> FunctionContext:
    ctx = FunctionContext(
        run_id="run_1",
        correlation_id="fn_cid",
        parent_correlation_id="run_cid",
        worker=worker,
        trace_metadata={"lease_id": "lease_1", "project_id": "proj_1"},
    )
    ctx._component_name = "embed_docs"
    return ctx


async def settle(seconds: float = 0.0) -> None:
    await asyncio.sleep(seconds)
    for _ in range(5):
        await asyncio.sleep(0)


def figures(worker: FakeWorker) -> list:
    return [(p.get("progress"), p.get("total"), p.get("message")) for p in worker.progress()]


# --- what a report carries --------------------------------------------------


def test_a_report_carries_progress_total_and_message():
    assert progress_payload(3, 10, "Embedding") == {"progress": 3.0, "total": 10.0, "message": "Embedding"}
    assert progress_payload(0.5) == {"progress": 0.5}
    # An empty message says nothing; a long one is cut.
    assert progress_payload(1, message="") == {"progress": 1.0}
    assert len(progress_payload(1, message="x" * 5000)["message"]) == progress_module.MAX_MESSAGE_CHARS


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (("3",), TypeError),
        ((True,), TypeError),
        ((None,), TypeError),
        ((1, "10"), TypeError),
        ((1, None, 42), TypeError),
        ((math.nan,), ValueError),
        ((math.inf,), ValueError),
        ((1, math.inf), ValueError),
        ((1, 0), ValueError),
        ((1, -5), ValueError),
    ],
)
def test_bad_reports_raise(args, error):
    with pytest.raises(error):
        progress_payload(*args)


def test_locally_a_report_goes_nowhere_but_is_still_checked():
    ctx = function_ctx(worker=None)
    ctx.progress(1, total=2, message="fine")
    with pytest.raises(TypeError):
        ctx.progress("1")


# --- sending ----------------------------------------------------------------


async def test_a_report_is_appended_to_the_journal_at_once():
    worker = FakeWorker()
    ctx = function_ctx(worker)
    ctx.progress(1, total=4, message="Embedded a.md")
    await settle()

    assert figures(worker) == [(1.0, 4.0, "Embedded a.md")]
    call = worker.appended[0]
    # A direct append, not the queue a pull run holds until it completes.
    assert worker.queued == []
    assert call["run_id"] == "run_1"
    assert call["metadata"]["correlation_id"] == "fn_cid"
    assert call["metadata"]["parent_correlation_id"] == "run_cid"
    assert call["metadata"]["lease_id"] == "lease_1", "carries the run's authority"
    data = worker.progress()[0]
    assert data["event_type"] == "progress.update"
    assert data["name"] == "embed_docs"


async def test_reports_are_coalesced_to_the_latest():
    worker = FakeWorker()
    ctx = function_ctx(worker)
    ctx.progress(1, total=50)
    await settle()
    assert figures(worker) == [(1.0, 50.0, None)], "the first goes out at once"
    for i in range(2, 51):
        ctx.progress(i, total=50)
        await asyncio.sleep(0)
    assert len(figures(worker)) == 1, "the rest wait out the interval"
    await settle(0.08)
    assert figures(worker) == [(1.0, 50.0, None), (50.0, 50.0, None)], "then only the latest"


async def test_progress_never_goes_backwards():
    worker = FakeWorker()
    ctx = function_ctx(worker)
    ctx.progress(5, total=10, message="five")
    await settle()
    ctx.progress(2, total=10, message="two")  # backwards: dropped
    ctx.progress(5, total=10, message="five")  # a repeat: dropped
    await settle(0.08)
    assert figures(worker) == [(5.0, 10.0, "five")]

    ctx.progress(5, total=10, message="checking")  # same figure, new message
    await settle(0.08)
    assert figures(worker)[-1] == (5.0, 10.0, "checking")


async def test_a_sync_handler_reports_from_its_worker_thread():
    worker = FakeWorker()
    ctx = function_ctx(worker)

    def handler():
        ctx.progress(1, total=2, message="from a thread")

    await asyncio.get_running_loop().run_in_executor(None, handler)
    await settle(0.01)
    assert figures(worker) == [(1.0, 2.0, "from a thread")]


async def test_a_finished_run_sends_nothing_more():
    worker = FakeWorker()
    ctx = function_ctx(worker)
    ctx.progress(1, total=3)
    await settle()
    ctx.progress(2, total=3)  # waiting out the interval
    await ctx.emit_async(
        Completed(
            name="embed_docs",
            correlation_id="run_cid",
            parent_correlation_id="",
            component_type=ComponentType.RUN,
        )
    )
    ctx.progress(3, total=3)
    await settle(0.08)
    assert figures(worker) == [(1.0, 3.0, None)]
    assert worker.appended[-1]["event_type"] == "run.completed"


async def test_a_report_that_cannot_be_sent_never_fails_the_run():
    worker = FakeWorker(fail=True)
    ctx = function_ctx(worker)
    ctx.progress(1)
    await settle()
    ctx.progress(2)
    await settle(0.08)
    worker.fail = False
    ctx.progress(3)
    await settle(0.08)
    assert figures(worker) == [(3.0, None, None)]


async def test_workflows_report_the_same_way():
    worker = FakeWorker()
    ctx = WorkflowContext(WorkflowEntity("run_1"), run_id="run_1", worker=worker)
    ctx.progress(2, total=5, message="Drafting")
    await settle()
    assert figures(worker) == [(2.0, 5.0, "Drafting")]


async def test_a_reporter_without_a_loop_keeps_the_latest_for_later():
    sent = []

    async def send(payload):
        sent.append(payload)

    reporter = ProgressReporter(send, loop=None, interval=0)
    # No running loop in this thread: nothing can be sent yet.
    await asyncio.get_running_loop().run_in_executor(None, reporter.report, {"progress": 1.0})
    assert sent == []
    reporter.report({"progress": 2.0})
    await settle()
    assert sent == [{"progress": 2.0}]
