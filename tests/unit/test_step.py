"""Durable steps exercise the same activation client as hosted workflow calls."""

import asyncio
import base64
from collections import Counter

import pytest

from agnt5 import StepContext, step
from agnt5.activation import (
    ActivationClient,
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationFailureReceipt,
    ActivationKind,
    activation_id,
)
from agnt5.context import get_current_context
from agnt5.exceptions import ActivationError, ActivationErrorCode
from agnt5.function import FunctionContext, FunctionRegistry, function
from agnt5.workflow import WorkflowContext, WorkflowEntity


class CheckpointProbe:
    """Any additional durable event RPC would turn two ACKs back into four."""

    def __init__(self):
        self.checkpoints = []

    async def emit_event_async(self, **event):
        self.checkpoints.append(event)

    def queue_event(self, **event):
        assert event["event_type"].startswith("log.")


class JournalTransport:
    """A committed result survives contexts, including when its ACK is lost."""

    def __init__(self):
        self.calls = []
        self.committed = {}
        self.inputs = {}
        self.attempts = Counter()
        self.block = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.error = None
        self.lose_completion_ack = False

    async def wait(self, phase, run_id):
        if self.block == (phase, run_id):
            self.entered.set()
            await self.release.wait()
        if self.error and self.error[0] == phase:
            raise self.error[1]

    async def begin(self, request):
        self.calls.append(("begin", request))
        await self.wait("begin", request.run_id)
        identity = activation_id(
            request.project_id,
            request.run_id,
            request.parent_activation_id,
            request.kind,
            request.stable_key,
        )
        previous = self.inputs.setdefault(identity, request.input_digest)
        if previous != request.input_digest:
            raise ActivationError(ActivationErrorCode.NON_DETERMINISTIC_REPLAY, "input changed")
        replay = identity in self.committed
        if not replay:
            self.attempts[identity] += 1
        return ActivationDecision(
            kind=ActivationDecisionKind.REPLAY if replay else ActivationDecisionKind.EXECUTE,
            activation_id=identity,
            attempt=self.attempts[identity],
            accepted_journal_offset=1,
            fence_token=b"fence" if not replay else b"",
            replay_output=self.committed.get(identity),
        )

    async def complete(self, **request):
        self.calls.append(("complete", request))
        await self.wait("complete", request["run_id"])
        self.committed[request["activation_id"]] = request["output"]
        if self.lose_completion_ack:
            raise ActivationError(ActivationErrorCode.UNKNOWN_OUTCOME, "completion ACK lost")
        return ActivationCompletionReceipt(
            activation_id=request["activation_id"],
            attempt=request["attempt"],
            accepted_journal_offset=2,
        )

    async def fail(self, **request):
        self.calls.append(("fail", request))
        await self.wait("fail", request["run_id"])
        return ActivationFailureReceipt(
            activation_id=request["activation_id"],
            attempt=request["attempt"],
            accepted_journal_offset=2,
            status="RETRY_READY" if request["retryable"] else "FAILED",
        )


def workflow_context(transport, mode="pull", run_id="run-1", client=None):
    metadata = {
        "project_id": "project-1",
        "worker_session_id": "worker-session",
        "lease_id": "lease-1",
        "durable_activation_v1": "true",
        "activation_artifact_sha256": base64.b64encode(b"a" * 32).decode(),
        "activation_definition_version": "v1",
        "component_name": "workflow",
    }
    if mode == "pull":
        metadata.update(dispatch_mode="pull", pull_completion_lifecycle_v1="true")
    probe = CheckpointProbe()
    entity = WorkflowEntity(run_id=run_id, component_name="workflow")
    context = WorkflowContext(
        workflow_entity=entity,
        run_id=run_id,
        worker=probe,
        activation_client=client or ActivationClient(transport),
        trace_metadata=metadata,
    )
    return context, entity, probe


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pull", "push"])
async def test_step_has_only_begin_and_complete_acks(mode):
    transport = JournalTransport()
    context, entity, probe = workflow_context(transport, mode)

    @step
    async def double(ctx: StepContext, value: int) -> dict:
        assert isinstance(ctx, StepContext)
        assert get_current_context() is ctx
        assert ctx.activation.idempotency_key.startswith("agnt5:actv1_")
        assert ctx.run_id == context.run_id
        assert ctx.correlation_id == ctx.activation.activation_id
        assert ctx.metadata["lease_id"] == "lease-1"
        return {"value": value * 2}

    assert await context.step(double, 3, key="item-3") == {"value": 6}
    assert [phase for phase, _ in transport.calls] == ["begin", "complete"]
    request = transport.calls[0][1]
    assert request.kind is ActivationKind.STEP
    assert request.stable_key == "step:double:item-3"
    assert entity.get_completed_step(request.stable_key) == {"value": 6}
    assert not probe.checkpoints
    assert FunctionRegistry.get("double") is None
    assert context.activation is None
    assert context._step_event_stack == []


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["begin", "complete"])
@pytest.mark.parametrize("entrypoint", ["step", "run"])
async def test_step_obeys_ack_and_dependency_boundaries(boundary, entrypoint):
    transport = JournalTransport()
    transport.block = (boundary, "run-1")
    context, entity, probe = workflow_context(transport)
    executed = []

    @step
    async def first():
        executed.append("first")
        return 7

    @step
    async def second(value):
        executed.append("second")
        return value + 1

    async def workflow_body():
        async def invoke(handler, *args, key):
            if entrypoint == "run":
                with pytest.warns(DeprecationWarning, match=r"ctx\.run\(\).*ctx\.step\(\)"):
                    return await context.run(handler, *args, key=key)
            return await context.step(handler, *args, key=key)

        value = await invoke(first, key="first")
        assert entity.has_completed_step("step:first:first")
        return await invoke(second, value, key="second")

    task = asyncio.create_task(workflow_body())
    try:
        await asyncio.wait_for(transport.entered.wait(), 1)
        await asyncio.sleep(0)
        assert not task.done()
        assert not entity.has_completed_step("step:first:first")
        assert executed == ([] if boundary == "begin" else ["first"])
        assert len(transport.calls) == (1 if boundary == "begin" else 2)
        transport.block = None
        transport.release.set()
        assert await asyncio.wait_for(task, 1) == 8
        assert executed == ["first", "second"]
        assert not probe.checkpoints
    finally:
        transport.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_step_replays_after_a_fresh_context():
    transport = JournalTransport()
    executed = []

    @step
    async def load():
        executed.append(True)
        return {"saved": 42}

    context, _, _ = workflow_context(transport)
    assert await context.step(load, key="load") == {"saved": 42}
    recovered, entity, probe = workflow_context(transport)
    assert await recovered.step(load, key="load") == {"saved": 42}
    assert executed == [True]
    assert [phase for phase, _ in transport.calls] == ["begin", "complete", "begin"]
    assert entity.has_completed_step("step:load:load")
    assert not probe.checkpoints


@pytest.mark.asyncio
async def test_lost_completion_ack_blocks_dependency_and_replays_committed_result():
    transport = JournalTransport()
    transport.lose_completion_ack = True
    executed = []

    @step
    async def write():
        executed.append("write")
        return "committed"

    context, entity, probe = workflow_context(transport)
    with pytest.raises(ActivationError) as caught:
        await context.step(write, key="write")
        executed.append("dependent")
    assert caught.value.code is ActivationErrorCode.UNKNOWN_OUTCOME
    assert executed == ["write"]
    assert not entity.has_completed_step("step:write:write")
    assert not probe.checkpoints
    assert not any(phase == "fail" for phase, _ in transport.calls)
    transport.lose_completion_ack = False
    recovered, _, _ = workflow_context(transport)
    assert await recovered.step(write, key="write") == "committed"
    assert executed == ["write"]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["begin", "complete"])
async def test_rejected_activation_never_memoizes_or_starts_dependent_work(boundary):
    transport = JournalTransport()
    transport.error = (boundary, ActivationError(ActivationErrorCode.STALE_AUTHORITY, "lease lost"))
    context, entity, _ = workflow_context(transport)
    executed = []

    @step
    async def write():
        executed.append("handler")
        return 1

    with pytest.raises(ActivationError):
        await context.step(write, key="write")
        executed.append("dependent")
    assert executed == ([] if boundary == "begin" else ["handler"])
    assert not entity.has_completed_step("step:write:write")
    assert context.activation is None
    assert context._step_event_stack == []


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["begin", "complete"])
async def test_cancellation_does_not_complete_or_memoize_step(boundary):
    transport = JournalTransport()
    transport.block = (boundary, "run-1")
    context, entity, _ = workflow_context(transport)

    @step
    async def work():
        return 1

    task = asyncio.create_task(context.step(work, key="work"))
    try:
        await asyncio.wait_for(transport.entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not entity.has_completed_step("step:work:work")
        assert not transport.committed
        assert not any(phase == "fail" for phase, _ in transport.calls)
        assert context._step_event_stack == []
    finally:
        transport.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["begin", "complete"])
async def test_other_runs_progress_while_one_run_awaits_ack(boundary):
    transport = JournalTransport()
    transport.block = (boundary, "run-1")
    client = ActivationClient(transport)
    blocked, _, _ = workflow_context(transport, client=client)
    sibling, entity, probe = workflow_context(transport, run_id="run-2", client=client)

    @step
    async def work(ctx: StepContext):
        return ctx.run_id

    task = asyncio.create_task(blocked.step(work, key="same-key"))
    try:
        await asyncio.wait_for(transport.entered.wait(), 1)
        assert await asyncio.wait_for(sibling.step(work, key="same-key"), 1) == "run-2"
        assert entity.has_completed_step("step:work:same-key")
        assert not task.done()
        assert not probe.checkpoints
        transport.release.set()
        assert await asyncio.wait_for(task, 1) == "run-1"
    finally:
        transport.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_steps_have_distinct_keys_and_task_local_context():
    transport = JournalTransport()
    context, _, probe = workflow_context(transport)
    seen = []

    @step
    async def item(ctx: StepContext, value: int):
        active = ctx.activation.activation_id
        await asyncio.sleep(0)
        assert get_current_context() is ctx
        assert ctx.activation.activation_id == active
        seen.append(active)
        return value * 2

    assert await asyncio.gather(*(context.step(item, i, key=str(i)) for i in range(20))) == list(
        range(0, 40, 2)
    )
    assert len(set(seen)) == 20
    assert Counter(phase for phase, _ in transport.calls) == {"begin": 20, "complete": 20}
    assert not probe.checkpoints
    assert context._step_event_stack == []


@pytest.mark.asyncio
async def test_step_retries_through_acknowledged_failure_with_same_identity():
    transport = JournalTransport()
    context, _, probe = workflow_context(transport)
    attempts = []

    @step(retries={"max_attempts": 3, "initial_interval_ms": 0})
    async def flaky(ctx: StepContext):
        attempts.append(ctx.attempt)
        if len(attempts) < 3:
            raise ValueError("retry")
        return 42

    assert await context.step(flaky, key="retry") == 42
    assert attempts == [0, 1, 2]
    assert [phase for phase, _ in transport.calls] == [
        "begin",
        "fail",
        "begin",
        "fail",
        "begin",
        "complete",
    ]
    assert {req.stable_key for phase, req in transport.calls if phase == "begin"} == {
        "step:flaky:retry"
    }
    assert not probe.checkpoints


@pytest.mark.asyncio
async def test_step_waits_for_failure_ack_before_raising():
    transport = JournalTransport()
    transport.block = ("fail", "run-1")
    context, entity, _ = workflow_context(transport)

    @step
    async def broken():
        raise ValueError("boom")

    task = asyncio.create_task(context.step(broken, key="broken"))
    try:
        await asyncio.wait_for(transport.entered.wait(), 1)
        assert not task.done()
        assert not entity.has_completed_step("step:broken:broken")
        transport.release.set()
        with pytest.raises(ValueError, match="boom"):
            await asyncio.wait_for(task, 1)
    finally:
        transport.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_step_arguments_sync_context_and_run_alias():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)

    @step(name="sync_value")
    def work(ctx: StepContext, value, *, suffix):
        assert get_current_context() is ctx
        assert ctx.activation is not None
        return {"value": value, "suffix": suffix}

    with pytest.warns(DeprecationWarning, match=r"ctx\.run\(\).*ctx\.step\(\)"):
        assert await context.run(work, None, suffix="ok", key="sync") == {
            "value": None,
            "suffix": "ok",
        }


@pytest.mark.asyncio
async def test_step_timeout_is_recorded_as_failure():
    transport = JournalTransport()
    context, entity, _ = workflow_context(transport)

    @step(timeout_ms=1)
    async def slow():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await context.step(slow, key="slow")
    assert [phase for phase, _ in transport.calls] == ["begin", "fail"]
    assert not entity.has_completed_step("step:slow:slow")


@pytest.mark.asyncio
async def test_step_requires_explicit_durable_execution():
    calls = []

    @step
    async def work():
        calls.append(True)
        return 1

    with pytest.raises(RuntimeError, match=r"ctx\.step"):
        await work()
    context, _, _ = workflow_context(JournalTransport())
    context._activation_client = None
    with pytest.raises(ActivationError) as caught:
        await context.step(work)
    assert caught.value.code is ActivationErrorCode.DURABILITY_UNAVAILABLE
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["step", "run"])
async def test_legacy_function_retains_its_function_checkpoints(entrypoint):
    transport = JournalTransport()
    context, _, probe = workflow_context(transport)

    @function(name="legacy_step_compatibility")
    async def work(ctx: FunctionContext, value: int):
        assert isinstance(ctx, FunctionContext)
        return value

    try:
        if entrypoint == "run":
            with pytest.warns(DeprecationWarning, match=r"ctx\.run\(\).*ctx\.step\(\)"):
                result = await context.run(work, 7, key="legacy")
        else:
            result = await context.step(work, 7, key="legacy")
        assert result == 7
        assert [phase for phase, _ in transport.calls] == ["begin", "complete"]
        assert [e["event_type"] for e in probe.checkpoints] == [
            "function.started",
            "function.completed",
        ]
    finally:
        FunctionRegistry.discard("legacy_step_compatibility")


@pytest.mark.asyncio
@pytest.mark.parametrize("run_count", [10, 100])
async def test_concurrent_runs_keep_receipts_and_results_isolated(run_count):
    transport = JournalTransport()
    client = ActivationClient(transport)
    entered = asyncio.Queue()
    release = asyncio.Event()
    contexts = [
        workflow_context(transport, run_id=f"run-{i}", client=client) for i in range(run_count)
    ]

    @step
    async def work(ctx: StepContext):
        entered.put_nowait(ctx.run_id)
        await release.wait()
        assert get_current_context() is ctx
        return ctx.run_id

    tasks = [asyncio.create_task(ctx.step(work, key="same")) for ctx, _, _ in contexts]
    try:
        started = {await asyncio.wait_for(entered.get(), 2) for _ in tasks}
        assert len(started) == run_count
        assert not transport.committed
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*tasks), 2) == [
            f"run-{i}" for i in range(run_count)
        ]
        assert len(transport.committed) == run_count
        for i, (ctx, entity, probe) in enumerate(contexts):
            assert entity.get_completed_step("step:work:same") == f"run-{i}"
            assert ctx.activation is None
            assert ctx._step_event_stack == []
            assert not probe.checkpoints
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_nested_steps_keep_the_admitted_parent_in_their_identity():
    transport = JournalTransport()
    context, _, probe = workflow_context(transport)

    @step
    async def child(value):
        return value

    @step
    async def parent(value):
        return await context.step(child, value, key="child")

    assert await context.step(parent, 1, key="first") == 1
    assert await context.step(parent, 2, key="second") == 2
    begins = [r for phase, r in transport.calls if phase == "begin"]
    assert begins[1].parent_activation_id == activation_id(
        begins[0].project_id, begins[0].run_id, "", ActivationKind.STEP, begins[0].stable_key
    )
    assert begins[3].parent_activation_id != begins[1].parent_activation_id
    assert not probe.checkpoints


@pytest.mark.asyncio
async def test_failure_ack_error_stops_retries():
    transport = JournalTransport()
    transport.error = (
        "fail",
        ActivationError(ActivationErrorCode.UNKNOWN_OUTCOME, "failure ACK lost"),
    )
    context, entity, _ = workflow_context(transport)
    attempts = []

    @step(retries={"max_attempts": 3, "initial_interval_ms": 0})
    async def broken():
        attempts.append(True)
        raise ValueError("boom")

    with pytest.raises(ActivationError) as caught:
        await context.step(broken, key="broken")
    assert caught.value.code is ActivationErrorCode.UNKNOWN_OUTCOME
    assert attempts == [True]
    assert [phase for phase, _ in transport.calls] == ["begin", "fail"]
    assert not entity.has_completed_step("step:broken:broken")


@pytest.mark.asyncio
async def test_step_retry_budget_exhaustion_records_the_final_failure():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)

    @step(retries={"max_attempts": 2, "initial_interval_ms": 0})
    async def broken():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await context.step(broken, key="broken")
    assert [phase for phase, _ in transport.calls] == ["begin", "fail", "begin", "fail"]
    assert [r["retryable"] for phase, r in transport.calls if phase == "fail"] == [True, False]


def test_step_carries_schema_and_policy_without_registering_a_component():
    @step(name="typed_step", retries=2, timeout_ms=1000)
    async def work(ctx: StepContext, value: int) -> str:
        return str(value)

    config = work._agnt5_step_config
    assert config.name == "typed_step"
    assert config.input_schema["properties"] == {"value": {"type": "integer"}}
    assert config.output_schema == {"type": "string"}
    assert config.retries.max_attempts == 2
    assert config.timeout_ms == 1000
    assert FunctionRegistry.get("typed_step") is None


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
def test_step_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_ms"):
        step(timeout_ms=timeout)


@pytest.mark.asyncio
async def test_step_retry_executes_the_same_input_snapshot():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    attempts = []
    values = [1, 2]

    @step(retries={"max_attempts": 2, "initial_interval_ms": 0})
    async def mutate(items):
        attempts.append(items.copy())
        result = items.pop()
        if len(attempts) == 1:
            raise ValueError("retry after mutation")
        return result

    assert await context.step(mutate, values, key="snapshot") == 2
    assert attempts == [[1, 2], [1, 2]]
    assert values == [1, 2]
    begins = [r for phase, r in transport.calls if phase == "begin"]
    assert begins[0].input_digest == begins[1].input_digest


@pytest.mark.asyncio
async def test_step_propagates_activation_to_nested_model_calls():
    from agnt5.lm import GenerateRequest, GenerateResponse, LMClient

    transport = JournalTransport()
    context, _, probe = workflow_context(transport)

    @step
    async def research(ctx: StepContext):
        client = LMClient(provider="openai")

        async def generate(request):
            return GenerateResponse(text="research result")

        client.generate = generate
        return await client._generate_durable(GenerateRequest(model="openai/gpt-4o-mini"), ctx)

    assert (await context.step(research, key="research")).text == "research result"
    begins = [r for phase, r in transport.calls if phase == "begin"]
    assert [r.kind for r in begins] == [ActivationKind.STEP, ActivationKind.MODEL]
    assert begins[1].parent_activation_id == activation_id(
        begins[0].project_id,
        begins[0].run_id,
        begins[0].parent_activation_id,
        begins[0].kind,
        begins[0].stable_key,
    )
    assert not probe.checkpoints


@pytest.mark.asyncio
async def test_run_deprecation_points_to_the_caller_and_preserves_replay():
    import inspect

    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    executed = []

    @step
    async def work(value):
        executed.append(value)
        return {"value": value}

    with pytest.warns(DeprecationWarning, match=r"ctx\.run\(\).*ctx\.step\(\)") as caught:
        call_line = inspect.currentframe().f_lineno + 1
        result = await context.run(work, value=7, key="same")
    assert result == {"value": 7}
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == call_line
    recovered, _, _ = workflow_context(transport)
    with pytest.warns(DeprecationWarning, match=r"ctx\.run\(\).*ctx\.step\(\)"):
        assert await recovered.run(work, value=7, key="same") == {"value": 7}
    assert executed == [7]
    assert [phase for phase, _ in transport.calls] == ["begin", "complete", "begin"]
    assert {r.stable_key for phase, r in transport.calls if phase == "begin"} == {"step:work:same"}


@pytest.mark.asyncio
async def test_run_deprecation_preserves_the_original_handler_error():
    transport = JournalTransport()
    context, entity, _ = workflow_context(transport)
    error = ValueError("handler failed")

    @step
    async def broken():
        raise error

    with pytest.warns(DeprecationWarning, match=r"ctx\.run\(\).*ctx\.step\(\)"):
        with pytest.raises(ValueError) as caught:
            await context.run(broken, key="broken")
    assert caught.value is error
    assert [phase for phase, _ in transport.calls] == ["begin", "fail"]
    assert not entity.has_completed_step("step:broken:broken")


@pytest.mark.asyncio
async def test_step_does_not_emit_run_deprecation():
    import warnings

    context, _, _ = workflow_context(JournalTransport())

    @step
    async def work():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        assert await context.step(work) == 1
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.asyncio
@pytest.mark.parametrize('child_key', [None, 'explicit'])
@pytest.mark.parametrize('finish_kind', ['declared', 'callback', 'function'])
async def test_nested_replay_preserves_following_root_step_identity(child_key, finish_kind):
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    effects = []

    @step
    async def child():
        return 'child'

    @step
    async def parent():
        return await context.step(child, key=child_key)

    async def finish_body():
        effects.append('finish')
        return 'done'

    declared_finish = step(finish_body)
    async def legacy_body(ctx):
        return await finish_body()

    legacy_finish = function(name=f'nested_replay_finish_{finish_kind}_{child_key}')(legacy_body)

    async def run():
        assert await context.step(parent) == 'child'
        if finish_kind == 'declared':
            return await context.step(declared_finish)
        if finish_kind == 'callback':
            return await context.step('finish', finish_body)
        return await context.step(legacy_finish)

    assert await run() == 'done'
    original = [r.stable_key for phase, r in transport.calls if phase == 'begin' and not r.parent_activation_id]
    context, _, _ = workflow_context(transport)
    transport.calls.clear()
    assert await run() == 'done'
    replay = [r.stable_key for phase, r in transport.calls if phase == 'begin']
    assert replay == original
    assert effects == ['finish']


@pytest.mark.asyncio
async def test_parent_retry_replays_child_and_resets_captured_context_allocations():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    child_effects = []
    allocations = []

    @step
    async def child():
        child_effects.append(True)
        return 7

    @step(retries={'max_attempts': 2, 'initial_interval_ms': 0})
    async def parent(ctx):
        allocations.append(context.allocate_activation_key('model', 'research'))
        result = await context.step(child)
        if ctx.attempt == 0:
            raise ValueError('retry parent after child committed')
        return result

    assert await context.step(parent) == 7
    assert child_effects == [True]
    assert allocations == ['model:research:0', 'model:research:0']
    children = [r for phase, r in transport.calls if phase == 'begin' and r.parent_activation_id]
    assert children[0].stable_key == children[1].stable_key
    assert children[0].parent_activation_id == children[1].parent_activation_id


@pytest.mark.asyncio
async def test_parallel_parents_keep_legacy_children_under_their_own_activation():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)

    @step
    async def parent(value):
        await asyncio.sleep(0)
        return await context.step('child', lambda: value)

    assert await asyncio.gather(context.step(parent, 1), context.step(parent, 2)) == [1, 2]
    children = [r for phase, r in transport.calls if phase == 'begin' and r.parent_activation_id]
    assert len(children) == 2
    assert children[0].stable_key == children[1].stable_key == 'step:child:0'
    assert children[0].parent_activation_id != children[1].parent_activation_id


@pytest.mark.asyncio
@pytest.mark.parametrize('wait', ['sleep', 'user'])
async def test_declared_steps_reject_workflow_waits_before_mutating_positions(wait):
    from agnt5.exceptions import ConfigurationError
    transport = JournalTransport()
    context, entity, _ = workflow_context(transport)

    @step
    async def parent():
        if wait == 'sleep':
            return await context.sleep(1)
        return await context.wait_for_user('Approve?')

    with pytest.raises(ConfigurationError, match='between workflow steps'):
        await context.step(parent, key='parent')
    assert entity._pause_index == 0
    assert context._step_counter == 1


@pytest.mark.asyncio
async def test_sync_handler_timeout_waits_for_thread_before_failure_and_retry():
    import threading
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    entered = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0
    calls = 0

    @step(timeout_ms=10, retries={'max_attempts': 2, 'initial_interval_ms': 0})
    def slow():
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        entered.set()
        release.wait(2)
        active -= 1
        return 'late'

    task = asyncio.create_task(context.step(slow))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0.03)
        assert not task.done()
        assert calls == 1
        assert [phase for phase, _ in transport.calls] == ['begin']
        release.set()
        # The second attempt can now succeed; the timed-out first result is discarded.
        assert await task == 'late'
        assert calls == 2 and peak == 1 and active == 0
        assert [phase for phase, _ in transport.calls] == ['begin', 'fail', 'begin', 'complete']
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['timeout', 'cancel'])
async def test_async_cleanup_cannot_turn_timeout_or_cancellation_into_success(reason):
    transport = JournalTransport()
    context, entity, _ = workflow_context(transport)
    entered = asyncio.Event()

    @step(timeout_ms=10 if reason == 'timeout' else None)
    async def suppress_cancel():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return 'late'

    task = asyncio.create_task(context.step(suppress_cancel, key='guard'))
    await entered.wait()
    if reason == 'cancel':
        task.cancel()
    with pytest.raises(TimeoutError if reason == 'timeout' else asyncio.CancelledError):
        await task
    assert not entity.has_completed_step('step:suppress_cancel:guard')
    assert [phase for phase, _ in transport.calls] == (['begin', 'fail'] if reason == 'timeout' else ['begin'])


@pytest.mark.asyncio
async def test_cancellation_during_sync_timeout_drain_prevents_retry():
    import threading
    transport = JournalTransport()
    context, entity, _ = workflow_context(transport)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    @step(timeout_ms=5, retries={'max_attempts': 2, 'initial_interval_ms': 0})
    def blocked():
        calls.append(True)
        entered.set()
        release.wait(2)
        return 7

    task = asyncio.create_task(context.step(blocked, key='guard'))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == [True]
        assert [phase for phase, _ in transport.calls] == ['begin']
        assert not entity.has_completed_step('step:blocked:guard')
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_legacy_only_nested_steps_preserve_their_existing_wire_identity():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)

    @function(name='legacy_ordinal_parent_compat')
    async def parent(ctx):
        return await context.step('legacy_child', lambda: 'child')

    assert await context.step(parent) == 'child'
    begins = [r for phase, r in transport.calls if phase == 'begin']
    assert [(r.parent_activation_id, r.stable_key) for r in begins] == [
        ('', 'step:legacy_ordinal_parent_compat:0'), ('', 'step:legacy_child:1')]


@pytest.mark.asyncio
async def test_declared_child_of_legacy_parent_does_not_shift_following_root_identity():
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    effects = []

    @step
    async def child():
        return 7

    @function(name='legacy_parent_declared_child_compat')
    async def parent(ctx):
        return await context.step(child)

    @step
    async def finish():
        effects.append(True)
        return 8

    async def run():
        assert await context.step(parent) == 7
        return await context.step(finish)

    assert await run() == 8
    begins = [r for phase, r in transport.calls if phase == 'begin']
    assert begins[1].parent_activation_id == activation_id(
        begins[0].project_id, begins[0].run_id, '', ActivationKind.STEP, begins[0].stable_key)
    assert begins[2].stable_key == 'step:finish:1'
    context, _, _ = workflow_context(transport)
    assert await run() == 8
    assert effects == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['timeout', 'cancel'])
async def test_sync_factory_returned_awaitable_can_unwind_after_thread_exit(reason):
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def pending():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    @step(timeout_ms=10 if reason == 'timeout' else None)
    def factory():
        return pending()

    task = asyncio.create_task(context.step(factory))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if reason == 'cancel':
            task.cancel()
        with pytest.raises(TimeoutError if reason == 'timeout' else asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 1)
        assert cleaned.is_set()
        assert [phase for phase, _ in transport.calls] == (['begin', 'fail'] if reason == 'timeout' else ['begin'])
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_legacy_parents_keep_unnamed_sleep_keys_workflow_wide():
    transport = JournalTransport()
    context, entity, probe = workflow_context(transport)
    probe.queue_event = lambda **event: probe.checkpoints.append(event)

    @function(name='legacy_unnamed_sleep_compat')
    async def parent(ctx):
        await context.sleep(0.001)
        return True

    assert await context.step(parent, key='one')
    assert await context.step(parent, key='two')
    assert entity.has_completed_step('sleep:sleep_1')
    assert entity.has_completed_step('sleep:sleep_3')
    assert context._step_counter == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['timeout', 'cancel'])
async def test_late_sync_factory_never_starts_returned_coroutine(reason):
    import threading
    transport = JournalTransport()
    context, _, _ = workflow_context(transport)
    entered = threading.Event()
    release = threading.Event()
    effects = []
    coroutines = []
    calls = 0

    @step(timeout_ms=10, retries={'max_attempts': 2, 'initial_interval_ms': 0})
    def factory():
        nonlocal calls
        calls += 1
        attempt = calls
        entered.set()
        release.wait(2)

        async def continuation():
            effects.append(attempt)
            await asyncio.sleep(0)
            return attempt

        result = continuation()
        coroutines.append(result)
        return result

    task = asyncio.create_task(context.step(factory))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0.03)
        if reason == 'cancel':
            task.cancel()
            await asyncio.sleep(0)
        release.set()
        if reason == 'cancel':
            with pytest.raises(asyncio.CancelledError):
                await task
            assert effects == [] and calls == 1
        else:
            assert await task == 2
            assert effects == [2] and calls == 2
        assert coroutines[0].cr_frame is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
