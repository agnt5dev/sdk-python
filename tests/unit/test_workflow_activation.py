import base64
import json

import pytest

from agnt5.activation import (
    ActivationClient,
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationFailureReceipt,
    ActivationKind,
    activation_id,
)
from agnt5.events import Completed, Started
from agnt5.exceptions import (
    ActivationError,
    ActivationErrorCode,
    DurableSleepSuspension,
)
from agnt5.function import FunctionContext, FunctionRegistry, function
from agnt5.lm import GenerateRequest, GenerateResponse, LMClient
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


def durable_sleep_context(transport):
    context, entity, events = activation_context(transport)
    context._trace_metadata["durable_suspension_v1"] = "true"
    return context, entity, events


@pytest.mark.asyncio
async def test_durable_sleep_yields_typed_timer_authority_without_local_wait():
    transport = WorkflowActivationTransport()
    context, entity, _events = durable_sleep_context(transport)

    with pytest.raises(DurableSleepSuspension) as caught:
        await context.sleep(2.5, name="backoff")

    suspension = caught.value
    assert suspension.timer_key == "sleep:backoff"
    assert suspension.delay_ms == 2500
    assert suspension.attempt == 1
    assert suspension.fence_token == b"fence-1"
    assert len(suspension.input_digest) == 32
    assert len(suspension.definition_digest) == 32
    assert not entity.has_completed_step("sleep:backoff")
    assert transport.begin_requests[0].kind is ActivationKind.TIMER
    assert transport.begin_requests[0].display_name == "sleep:backoff"
    assert json.loads(transport.begin_requests[0].input_data) == {
        "delay_ms": 2500,
        "timer_key": "sleep:backoff",
    }
    assert transport.complete_requests == []


@pytest.mark.asyncio
async def test_durable_sleep_resume_completes_only_matching_timer_activation():
    transport = WorkflowActivationTransport()
    context, entity, _events = durable_sleep_context(transport)
    context._trace_metadata.update(
        {
            "timer_key": "sleep:backoff",
            "activation_id": activation_id(
                "project-1",
                "run-1",
                "",
                ActivationKind.TIMER,
                "sleep:backoff",
            ),
        }
    )

    await context.sleep(2.5, name="backoff")

    assert entity.has_completed_step("sleep:backoff")
    assert transport.begin_requests == []


@pytest.mark.asyncio
async def test_durable_sleep_replay_skips_an_earlier_completed_timer():
    transport = WorkflowActivationTransport()
    context, entity, _events = durable_sleep_context(transport)
    entity.record_step_completion(
        "sleep:first",
        "sleep",
        {"delay_ms": 1000, "timer_key": "sleep:first"},
        None,
    )
    context._trace_metadata.update(
        {
            "timer_key": "sleep:second",
            "activation_id": activation_id(
                "project-1",
                "run-1",
                "",
                ActivationKind.TIMER,
                "sleep:second",
            ),
        }
    )

    await context.sleep(1, name="first")
    await context.sleep(1, name="second")

    assert entity.has_completed_step("sleep:first")
    assert entity.has_completed_step("sleep:second")
    assert transport.begin_requests == []


@pytest.mark.asyncio
async def test_sleep_zero_is_noop_and_negative_is_rejected():
    transport = WorkflowActivationTransport()
    context, _entity, _events = durable_sleep_context(transport)

    await context.sleep(0, name="noop")
    with pytest.raises(ValueError, match="negative"):
        await context.sleep(-1, name="invalid")

    assert transport.begin_requests == []


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
    # The activation is the step boundary record: nothing decorative is emitted.
    assert events == []
    assert transport.begin_requests[0].display_name == "load"
    assert json.loads(transport.begin_requests[0].input_data) == {
        "step_name": "load",
        "handler_name": "checkpoint",
        "input": {"args": [], "kwargs": {}},
    }
    assert transport.complete_requests[0]["usage"].latency_ms >= 0
    assert context._step_event_stack == []
    assert context.activation is None


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
    assert events == []
    assert transport.begin_requests[0].display_name == "load"
    assert context._step_event_stack == []


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
    assert events == []
    assert context._step_event_stack == []
    assert context.activation is None


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
    assert transport.fail_requests[0]["latency_ms"] >= 0
    assert not entity.has_completed_step("step:load:0")
    assert events == []
    assert context._step_event_stack == []


@pytest.mark.asyncio
async def test_function_form_uses_same_activation_boundary_and_explicit_key():
    transport = WorkflowActivationTransport()
    context, entity, events = activation_context(transport)
    executed = False

    observed = {}

    @function
    async def load(ctx: FunctionContext, item: str):
        nonlocal executed
        executed = True
        assert ctx._trace_metadata["project_id"] == "project-1"
        assert not entity.has_completed_step("step:load:item-42")
        observed["activation"] = context.activation
        observed["stack"] = list(context._step_event_stack)
        return {"item": item}

    result = await context.step(load, "record", key="item-42")

    assert executed
    assert result == {"item": "record"}
    assert transport.begin_requests[0].stable_key == "step:load:item-42"
    assert transport.begin_requests[0].display_name == "load_0"
    assert json.loads(transport.begin_requests[0].input_data) == {
        "step_name": "load_0",
        "handler_name": "load",
        "input": {"args": ["record"], "kwargs": {}},
    }
    assert entity.get_completed_step("step:load:item-42") == {"item": "record"}
    # Only the nested function.* pair is emitted; it parents to the activation.
    step_activation_id = activation_id(
        "project-1", "run-1", "", ActivationKind.STEP, "step:load:item-42"
    )
    assert [type(event) for event in events] == [Started, Completed]
    assert all(event.component_type.value == "function" for event in events)
    assert all(event.parent_correlation_id == step_activation_id for event in events)
    assert events[0].correlation_id == events[1].correlation_id
    assert observed["activation"] is not None
    assert observed["activation"].activation_id == step_activation_id
    assert observed["stack"] == [step_activation_id]
    assert context._step_event_stack == []
    assert context.activation is None


@pytest.mark.asyncio
async def test_function_form_propagates_activation_client_to_nested_model_call():
    transport = WorkflowActivationTransport()
    context, _entity, _events = activation_context(transport)

    @function
    async def load(ctx: FunctionContext):
        client = LMClient(provider="openai")

        async def generate(request: GenerateRequest) -> GenerateResponse:
            del request
            return GenerateResponse(text="nested durable result")

        client.generate = generate
        return await client._generate_durable(
            GenerateRequest(model="openai/gpt-4o-mini"),
            ctx,
        )

    result = await context.step(load)

    assert result.text == "nested durable result"
    assert [request.kind for request in transport.begin_requests] == [
        ActivationKind.STEP,
        ActivationKind.MODEL,
    ]
    step_request, model_request = transport.begin_requests
    assert model_request.parent_activation_id == activation_id(
        step_request.project_id,
        step_request.run_id,
        step_request.parent_activation_id,
        step_request.kind,
        step_request.stable_key,
    )
    assert model_request.display_name == "openai/gpt-4o-mini"


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
    assert events == []


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
    assert [type(event) for event in events] == [Started, Completed]
    assert all(event.component_type.value == "function" for event in events)
    assert context._step_event_stack == []
