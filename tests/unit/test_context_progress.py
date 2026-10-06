"""ctx.progress: report how far a run has got."""

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

from agnt5 import progress as progress_module
from agnt5.events import Completed, ComponentType
from agnt5.function import FunctionContext
from agnt5.progress import ProgressReporter, progress_payload
from agnt5.worker._core import Worker
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
    # Reporters live per run; tests reuse run ids.
    monkeypatch.setattr(progress_module, "_reporters", {})


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


async def dispatch(coro):
    """Run a handler coroutine the way the worker runs a dispatch."""
    return await Worker._track_invocation(SimpleNamespace(_inflight={}), "run_1", coro)


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


async def test_the_last_report_is_written_before_the_run_finishes():
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
    ctx.progress(3, total=3)  # after the end: dropped
    await settle(0.08)
    assert figures(worker) == [(1.0, 3.0, None), (2.0, 3.0, None)]
    assert [c["event_type"] for c in worker.appended] == [
        "progress.update",
        "progress.update",
        "run.completed",
    ]


async def test_a_report_made_just_before_returning_is_written():
    # A pull run returns its terminal in the response: the worker drains the
    # run's reporter before handing it back, with no chance to yield first.
    worker = FakeWorker()

    async def handler():
        function_ctx(worker).progress(4, total=4, message="Done")
        return "ok"

    assert await dispatch(handler()) == "ok"
    assert figures(worker) == [(4.0, 4.0, "Done")]
    assert progress_module._reporters == {}, "dispatch contexts use their execution"


async def test_a_throttled_report_is_written_when_the_execution_ends():
    worker = FakeWorker()

    async def handler():
        ctx = function_ctx(worker)
        ctx.progress(1, total=2)
        await settle()
        ctx.progress(2, total=2)
        return "ok"

    await dispatch(handler())
    assert figures(worker) == [(1.0, 2.0, None), (2.0, 2.0, None)]


async def test_a_report_after_the_execution_ended_is_dropped():
    # Code the handler left running (a stray task) reports after the end.
    worker = FakeWorker()
    leftover: list[asyncio.Task] = []

    async def handler():
        ctx = function_ctx(worker)
        ctx.progress(1, total=3)

        async def stray():
            await asyncio.sleep(0.02)
            ctx.progress(2, total=3)  # a context that reported before
            function_ctx(worker).progress(3, total=3)  # and a fresh one

        leftover.append(asyncio.create_task(stray()))
        return "ok"

    await dispatch(handler())
    await leftover[0]
    await settle(0.08)
    assert figures(worker) == [(1.0, 3.0, None)]


async def test_a_retry_of_the_run_starts_clean():
    worker = FakeWorker()

    async def attempt(figure: float):
        function_ctx(worker).progress(figure, total=10)
        return "ok"

    await dispatch(attempt(8))
    # The same run again, starting lower: not held back by the last attempt
    # (the runtime keeps the run's own figure from going backwards).
    await dispatch(attempt(2))
    assert figures(worker) == [(8.0, 10.0, None), (2.0, 10.0, None)]


async def test_a_report_carries_the_context_that_made_it():
    worker = FakeWorker()
    workflow_ctx = function_ctx(worker)
    step_ctx = function_ctx(worker)
    step_ctx._correlation_id = "step_cid"
    step_ctx._component_name = "embed_one"

    workflow_ctx.progress(5, total=10)
    await settle()
    step_ctx.progress(7, total=10)  # accepted, waiting out the interval
    workflow_ctx.progress(3, total=10)  # backwards: dropped, changes nothing
    await settle(0.08)
    sent = worker.progress()
    assert [(p["progress"], p["correlation_id"], p["name"]) for p in sent] == [
        (5.0, "fn_cid", "embed_docs"),
        (7.0, "step_cid", "embed_one"),
    ]


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

    async def send(payload, _source):
        sent.append(payload)

    reporter = ProgressReporter(send, loop=None, interval=0)
    # No running loop in this thread: nothing can be sent yet.
    await asyncio.get_running_loop().run_in_executor(None, reporter.report, {"progress": 1.0})
    assert sent == []
    reporter.report({"progress": 2.0})
    await settle()
    assert sent == [{"progress": 2.0}]


async def test_a_run_shares_one_reporter_across_its_contexts():
    # A workflow and its ctx.task children report for the same run: one
    # interval and one never-backwards rule between them.
    worker = FakeWorker()
    parent = WorkflowContext(WorkflowEntity("run_1"), run_id="run_1", worker=worker)
    children = [function_ctx(worker) for _ in range(3)]
    parent.progress(4, total=10)
    await settle()
    for i, child in enumerate(children):
        child.progress(5 + i, total=10)
    children[0].progress(2, total=10)  # below the run's last figure: dropped
    await settle()
    assert figures(worker) == [(4.0, 10.0, None)], "one record, not one per context"
    await settle(0.08)
    assert figures(worker) == [(4.0, 10.0, None), (7.0, 10.0, None)]


async def test_a_cancelled_execution_writes_nothing_more():
    worker = FakeWorker()
    reported = asyncio.Event()

    async def handler():
        ctx = function_ctx(worker)
        ctx.progress(1, total=3)
        await settle()
        ctx.progress(2, total=3)  # waiting out the interval
        reported.set()
        await asyncio.sleep(10)

    task = asyncio.ensure_future(dispatch(handler()))
    await reported.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await settle(0.08)
    assert figures(worker) == [(1.0, 3.0, None)]


async def test_a_failed_execution_still_writes_its_last_report():
    # Executors turn a handler's exception into a failed terminal response;
    # the last report goes ahead of it.
    worker = FakeWorker()

    async def handler():
        ctx = function_ctx(worker)
        ctx.progress(1, total=3)
        ctx.progress(2, total=3)
        return "run.failed response"

    assert await dispatch(handler()) == "run.failed response"
    # Both came before the reporter's first turn: only the latest is written.
    assert figures(worker) == [(2.0, 3.0, None)]


async def test_an_agent_names_itself_on_its_reports():
    from agnt5.agent.context import AgentContext

    worker = FakeWorker()
    ctx = AgentContext(run_id="run_1", agent_name="researcher", worker=worker)
    ctx.progress(1, total=2, message="Searching")
    await settle()
    assert worker.progress()[0]["name"] == "researcher"


async def test_a_task_function_names_itself_on_its_reports():
    from agnt5.function import function

    worker = FakeWorker()

    @function(name="embed_one")
    async def embed_one(ctx: FunctionContext, doc: str) -> str:
        ctx.progress(1, total=1, message=f"Embedded {doc}")
        return doc

    workflow_ctx = WorkflowContext(WorkflowEntity("run_1"), run_id="run_1", worker=worker)
    assert await workflow_ctx.step(embed_one, "a.md") == "a.md"
    await settle()
    assert [p["name"] for p in worker.progress()] == ["embed_one"]


async def test_a_function_cancelled_mid_run_writes_nothing_more(monkeypatch):
    # Through the worker's real function executor, whose cancellation path
    # returns normally (the gateway already wrote run.cancelled).
    from agnt5.worker._executors import ExecutorMixin

    async def quiet(self, event):  # lifecycle records aren't under test
        return None

    monkeypatch.setattr(FunctionContext, "emit_async", quiet)
    monkeypatch.setattr(FunctionContext, "emit_batch_async", quiet)

    class Executor(ExecutorMixin):
        def __init__(self, worker) -> None:
            self._entity_state_adapter = object()
            self._checkpoint_client = None
            self._rust_worker = worker
            self.service_name = "test"

    worker = FakeWorker()

    async def handler(ctx):
        ctx.progress(1, total=3)
        await settle()
        ctx.progress(2, total=3)  # waiting out the interval
        raise asyncio.CancelledError()  # CancelExecution → task.cancel()

    request = SimpleNamespace(
        invocation_id="run_1",
        input_data=b"{}",
        runtime_context=None,
        metadata={},
        session_id="",
        user_id="",
        attempt=0,
        is_streaming=False,
        component_name="embed_docs",
    )
    config = SimpleNamespace(name="embed_docs", handler=handler, retries=None, timeout_ms=None)
    result = await dispatch(Executor(worker)._execute_function(config, request.input_data, request))
    assert result is None
    await settle(0.08)
    assert figures(worker) == [(1.0, 3.0, None)]
