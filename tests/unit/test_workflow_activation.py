import base64

import pytest

from agnt5.activation import (
    ActivationClient,
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationFailureReceipt,
    activation_id,
)
from agnt5.events import Completed, Failed, Started
from agnt5.exceptions import ActivationError, ActivationErrorCode
from agnt5.function import FunctionContext, FunctionRegistry, function
from agnt5.workflow import WorkflowContext, WorkflowEntity


@pytest.fixture(autouse=True)
def clear_function_registry():
    FunctionRegistry.clear()
    yield
    FunctionRegistry.clear()


class WorkflowActivationTransport:
    def __init__(self):
        self.begin_requests = []
        self.complete_requests = []
        self.fail_requests = []
        self.replay_output = None
        self.complete_error = None

    async def begin(self, request):
        self.begin_requests.append(request)
        return ActivationDecision(
            kind=(
                ActivationDecisionKind.REPLAY
                if self.replay_output is not None
                else ActivationDecisionKind.EXECUTE
            ),
            activation_id=activation_id(
                request.project_id,
                request.run_id,
                request.parent_activation_id,
                request.kind,
                request.stable_key,
            ),
            attempt=1,
            accepted_journal_offset=11,
            fence_token=b"fence-1" if self.replay_output is None else b"",
            replay_output=self.replay_output,
        )

    async def complete(self, **request):
        self.complete_requests.append(request)
        if self.complete_error is not None:
            raise self.complete_error
        return ActivationCompletionReceipt(
            activation_id=request["activation_id"],
            attempt=request["attempt"],
            accepted_journal_offset=12,
        )

    async def fail(self, **request):
        self.fail_requests.append(request)
        return ActivationFailureReceipt(
            activation_id=request["activation_id"],
            attempt=request["attempt"],
            accepted_journal_offset=12,
            status="FAILED",
        )


def activation_context(transport):
    entity = WorkflowEntity(run_id="run-1", component_name="workflow")
    context = WorkflowContext(
        workflow_entity=entity,
        run_id="run-1",
        activation_client=ActivationClient(transport),
        trace_metadata={
            "project_id": "project-1",
            "worker_session_id": "session-1",
            "lease_id": "lease-1",
            "activation_artifact_sha256": base64.b64encode(b"a" * 32).decode(),
            "activation_definition_version": "v1",
            "activation_definition_config": '["object",[]]',
            "component_name": "workflow",
        },
    )
    events = []
    context.emit = events.append
    return context, entity, events


@pytest.mark.asyncio
async def test_checkpoint_form_uses_activation_and_memoizes_only_after_ack():
    transport = WorkflowActivationTransport()
    context, entity, events = activation_context(transport)
    executed = False

    async def load():
        nonlocal executed
        executed = True
        assert len(transport.begin_requests) == 1
        assert not entity.has_completed_step("step:load:0")
        return {"value": 42}

    result = await context.step("load", load)

    assert executed
    assert result == {"value": 42}
    assert transport.begin_requests[0].stable_key == "step:load:0"
    assert len(transport.complete_requests) == 1
    assert entity.get_completed_step("step:load:0") == {"value": 42}
    assert [type(event) for event in events] == [Started, Completed]
    assert events[1].metadata["activation_id"].startswith("actv1_")
    assert events[1].metadata["accepted_journal_offset"] == "12"


@pytest.mark.asyncio
async def test_checkpoint_form_replays_without_executing_user_code():
    transport = WorkflowActivationTransport()
    transport.replay_output = b'{"cached":true}'
    context, entity, events = activation_context(transport)

    async def must_not_run():
        raise AssertionError("user code ran on REPLAY")

    result = await context.step("load", must_not_run)

    assert result == {"cached": True}
    assert transport.complete_requests == []
    assert entity.get_completed_step("step:load:0") == {"cached": True}
    assert [type(event) for event in events] == [Started, Completed]
    assert events[1].metadata["cache_hit"] == "true"


@pytest.mark.asyncio
async def test_checkpoint_form_does_not_return_or_memoize_when_completion_ack_is_lost():
    transport = WorkflowActivationTransport()
    transport.complete_error = ActivationError(
        ActivationErrorCode.UNKNOWN_OUTCOME,
        "completion acknowledgement was lost",
    )
    context, entity, events = activation_context(transport)

    with pytest.raises(ActivationError) as caught:
        await context.step("load", lambda: "value")

    assert caught.value.code is ActivationErrorCode.UNKNOWN_OUTCOME
    assert not entity.has_completed_step("step:load:0")
    assert [type(event) for event in events] == [Started]
    assert context._step_event_stack == []


@pytest.mark.asyncio
async def test_checkpoint_form_waits_for_failure_receipt_before_raising_user_error():
    transport = WorkflowActivationTransport()
    context, entity, events = activation_context(transport)

    async def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await context.step("load", fail)

    assert len(transport.fail_requests) == 1
    assert transport.fail_requests[0]["external_outcome_certainty"] == "UNKNOWN"
    assert not entity.has_completed_step("step:load:0")
    assert [type(event) for event in events] == [Started, Failed]


@pytest.mark.asyncio
async def test_function_form_uses_same_activation_boundary_and_explicit_key():
    transport = WorkflowActivationTransport()
    context, entity, events = activation_context(transport)
    executed = False

    @function
    async def load(ctx: FunctionContext, item: str):
        nonlocal executed
        executed = True
        assert ctx._trace_metadata["project_id"] == "project-1"
        assert not entity.has_completed_step("step:load:item-42")
        return {"item": item}

    result = await context.step(load, "record", key="item-42")

    assert executed
    assert result == {"item": "record"}
    assert transport.begin_requests[0].stable_key == "step:load:item-42"
    assert entity.get_completed_step("step:load:item-42") == {"item": "record"}
    assert [type(event) for event in events] == [Started, Started, Completed, Completed]
    assert events[0].component_type.value == "workflow"
    assert events[1].component_type.value == "function"


@pytest.mark.asyncio
async def test_function_form_replay_skips_registered_function():
    transport = WorkflowActivationTransport()
    transport.replay_output = b'{"cached":true}'
    context, entity, events = activation_context(transport)

    @function
    async def load(_ctx: FunctionContext):
        raise AssertionError("registered function ran on REPLAY")

    result = await context.step(load)

    assert result == {"cached": True}
    assert entity.get_completed_step("step:load:0") == {"cached": True}
    assert [type(event) for event in events] == [Started, Completed]


@pytest.mark.asyncio
async def test_function_form_does_not_memoize_when_completion_ack_is_lost():
    transport = WorkflowActivationTransport()
    transport.complete_error = ActivationError(
        ActivationErrorCode.UNKNOWN_OUTCOME,
        "completion acknowledgement was lost",
    )
    context, entity, events = activation_context(transport)

    @function
    async def load(_ctx: FunctionContext):
        return "value"

    with pytest.raises(ActivationError) as caught:
        await context.step(load)

    assert caught.value.code is ActivationErrorCode.UNKNOWN_OUTCOME
    assert not entity.has_completed_step("step:load:0")
    assert [type(event) for event in events] == [Started, Started, Completed]
