import base64
import hashlib
import math
import time

import pytest

from agnt5.activation import (
    ActivationClient,
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationDefinition,
    ActivationKind,
    ActivationRecoveryPolicy,
    BeginActivationRequest,
    UInt64,
    activation_id,
    canonical_activation_value,
    stable_step_key,
)
from agnt5.exceptions import ActivationError, ActivationErrorCode


def test_canonical_activation_values_match_frozen_vectors():
    values = [
        (None, b'["null"]'),
        (True, b'["bool",true]'),
        (-42, b'["i64","-42"]'),
        (UInt64(42), b'["u64","42"]'),
        (1.0, b'["f64","3ff0000000000000"]'),
        (-0.0, b'["f64","0000000000000000"]'),
        ("caf\u00e9/", '["string","caf\u00e9/"]'.encode()),
        (b"\x00\xff", b'["bytes","AP8"]'),
        ([None, False, "x"], b'["array",[["null"],["bool",false],["string","x"]]]'),
        (
            {"name": "alpha", "count": 2},
            b'["object",[["count",["i64","2"]],["name",["string","alpha"]]]]',
        ),
    ]
    for value, expected in values:
        encoded = canonical_activation_value(value)
        assert encoded == expected


def test_definition_and_identity_match_frozen_vectors():
    definition = ActivationDefinition(
        artifact_sha256=base64.b64decode("0lJSBAIElTtKmSY0S/XeONW7020B5x6yW0xopTX5kkg="),
        component_name="workflow",
        definition_version="v1",
        canonical_config=b'["object",[]]',
    )
    assert base64.b64encode(definition.digest).decode() == (
        "iTziD0lZ9kXRtq7RUj58/nzuTDQQtdgYp+MDNrAGVmw="
    )
    assert (
        activation_id("project-1", "run-1", "parent-1", ActivationKind.STEP, "step/load")
        == "actv1_9LU0V32sQX2U3CaQSCW37t-WWSvBAe04qTWqTD6mN-w"
    )


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, 2**63, {1: "bad"}, "\ud800", object()],
)
def test_canonical_activation_values_reject_unsafe_inputs(value):
    with pytest.raises(ActivationError) as caught:
        canonical_activation_value(value)
    assert caught.value.code is ActivationErrorCode.INVALID_ARGUMENT


def test_stable_step_key_has_explicit_and_compatibility_forms():
    assert stable_step_key("load", 0) == "step:load:0"
    assert stable_step_key("load", 0, "item-42") == "step:load:item-42"


class RecordingTransport:
    def __init__(self, decision):
        self.decision = decision
        self.complete_calls = []
        self.fail_calls = []

    async def begin(self, request):
        self.request = request
        return self.decision

    async def complete(self, **request):
        self.complete_calls.append(request)
        return ActivationCompletionReceipt(
            activation_id=request["activation_id"],
            attempt=request["attempt"],
            accepted_journal_offset=12,
        )

    async def fail(self, **request):
        self.fail_calls.append(request)
        raise AssertionError("not expected")


def activation_request():
    return BeginActivationRequest(
        project_id="project-1",
        run_id="run-1",
        parent_activation_id="",
        kind=ActivationKind.STEP,
        stable_key="step:load:0",
        input_digest=hashlib.sha256(b'["null"]').digest(),
        definition_digest=b"d" * 32,
        recovery_policy=ActivationRecoveryPolicy.DURABLE_STEPS,
        worker_session_id="session-1",
        run_authority=b"run-authority",
        lease_authority=b"lease-authority",
    )


@pytest.mark.asyncio
async def test_activation_client_executes_only_after_admission_and_completion_ack():
    request = activation_request()
    expected_id = activation_id(
        request.project_id,
        request.run_id,
        request.parent_activation_id,
        request.kind,
        request.stable_key,
    )
    transport = RecordingTransport(
        ActivationDecision(
            kind=ActivationDecisionKind.EXECUTE,
            activation_id=expected_id,
            attempt=1,
            fence_token=b"fence-1",
            accepted_journal_offset=11,
        )
    )
    client = ActivationClient(transport)
    called = False
    started = time.monotonic()

    async def execute():
        nonlocal called
        called = True
        assert transport.request == request
        return {"value": 42}

    result, receipt = await client.run(
        request,
        execute,
        encode_output=lambda value: b'{"value":42}',
        decode_output=lambda value: value,
        latency_ms=lambda: int((time.monotonic() - started) * 1000),
    )

    assert called
    assert result == {"value": 42}
    assert receipt.accepted_journal_offset == 12
    assert transport.complete_calls[0]["output_digest"] == hashlib.sha256(b'{"value":42}').digest()


@pytest.mark.asyncio
async def test_activation_client_replays_without_running_user_code():
    request = activation_request()
    expected_id = activation_id(
        request.project_id,
        request.run_id,
        request.parent_activation_id,
        request.kind,
        request.stable_key,
    )
    transport = RecordingTransport(
        ActivationDecision(
            kind=ActivationDecisionKind.REPLAY,
            activation_id=expected_id,
            attempt=1,
            accepted_journal_offset=12,
            replay_output=b'{"cached":true}',
        )
    )
    client = ActivationClient(transport)

    async def must_not_run():
        raise AssertionError("user code ran on REPLAY")

    result, receipt = await client.run(
        request,
        must_not_run,
        encode_output=lambda value: value,
        decode_output=lambda value: value.decode(),
        latency_ms=lambda: 0,
    )

    assert result == '{"cached":true}'
    assert receipt.kind is ActivationDecisionKind.REPLAY
    assert transport.complete_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "code"),
    [
        (ActivationDecisionKind.WAIT, ActivationErrorCode.CONTENDED),
        (ActivationDecisionKind.CONFLICT, ActivationErrorCode.NON_DETERMINISTIC_REPLAY),
        (ActivationDecisionKind.CANCELLED, ActivationErrorCode.CANCELLED),
        (ActivationDecisionKind.UNKNOWN_OUTCOME, ActivationErrorCode.UNKNOWN_OUTCOME),
    ],
)
async def test_activation_client_refuses_non_execution_decisions(kind, code):
    request = activation_request()
    transport = RecordingTransport(
        ActivationDecision(
            kind=kind,
            activation_id=activation_id(
                request.project_id,
                request.run_id,
                request.parent_activation_id,
                request.kind,
                request.stable_key,
            ),
            attempt=1,
            accepted_journal_offset=11,
        )
    )
    client = ActivationClient(transport)

    with pytest.raises(ActivationError) as caught:
        await client.run(
            request,
            lambda: pytest.fail("user code ran"),
            encode_output=lambda value: value,
            decode_output=lambda value: value,
            latency_ms=lambda: 0,
        )
    assert caught.value.code is code
