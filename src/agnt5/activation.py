"""Durable activation V1 values, identities, and typed adapter contracts."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from .exceptions import ActivationError, ActivationErrorCode

DURABLE_ACTIVATION_V1 = "durable_activation_v1"
_IDENTITY_DOMAIN = b"agnt5.activation.identity.v1\0"
_DEFINITION_DOMAIN = b"agnt5.activation.definition.v1\0"
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_U64_MAX = 2**64 - 1

T = TypeVar("T")


class UInt64(int):
    """An explicitly unsigned integer for cross-language canonical values."""

    def __new__(cls, value: int) -> "UInt64":
        if isinstance(value, bool) or value < 0 or value > _U64_MAX:
            raise ValueError("UInt64 must be between 0 and 2^64 - 1")
        return int.__new__(cls, value)


class ActivationKind(IntEnum):
    """Stable V1 activation kinds."""

    STEP = 1
    MODEL = 2
    AGENT = 3
    TOOL = 4
    CHILD = 5
    APPROVAL = 6
    TIMER = 7
    EVAL = 8


class ActivationRecoveryPolicy(IntEnum):
    """Stable V1 activation recovery policies."""

    IDEMPOTENT_RETRY = 1
    DURABLE_STEPS = 2
    UNKNOWN_OUTCOME = 3
    COMPENSATE = 4
    FAIL = 5


class ActivationDecisionKind(str, Enum):
    """Complete V1 begin-decision surface."""

    EXECUTE = "EXECUTE"
    REPLAY = "REPLAY"
    WAIT = "WAIT"
    CONFLICT = "CONFLICT"
    CANCELLED = "CANCELLED"
    UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"


@dataclass(frozen=True)
class BeginActivationRequest:
    project_id: str
    run_id: str
    parent_activation_id: str
    kind: ActivationKind
    stable_key: str
    input_digest: bytes
    definition_digest: bytes
    recovery_policy: ActivationRecoveryPolicy
    worker_session_id: str
    run_authority: bytes
    lease_authority: bytes


@dataclass(frozen=True)
class ActivationDecision:
    kind: ActivationDecisionKind
    activation_id: str
    attempt: int
    accepted_journal_offset: int
    fence_token: bytes = b""
    replay_output: bytes | None = None
    message: str = ""


@dataclass(frozen=True)
class ActivationCompletionReceipt:
    activation_id: str
    attempt: int
    accepted_journal_offset: int
    replayed: bool = False


@dataclass(frozen=True)
class ActivationFailureReceipt:
    activation_id: str
    attempt: int
    accepted_journal_offset: int
    status: str
    replayed: bool = False


class ActivationTransport(Protocol):
    """Coarse native boundary used by language-local workflow code."""

    async def begin(self, request: BeginActivationRequest) -> ActivationDecision: ...

    async def complete(
        self,
        *,
        project_id: str,
        run_id: str,
        activation_id: str,
        attempt: int,
        fence_token: bytes,
        output: bytes,
        output_digest: bytes,
        latency_ms: int,
    ) -> ActivationCompletionReceipt: ...

    async def fail(
        self,
        *,
        project_id: str,
        run_id: str,
        activation_id: str,
        attempt: int,
        fence_token: bytes,
        error_code: str,
        error_data: bytes,
        retryable: bool,
        external_outcome_certainty: str,
    ) -> ActivationFailureReceipt: ...


@dataclass(frozen=True)
class ActivationDefinition:
    artifact_sha256: bytes
    component_name: str
    definition_version: str
    canonical_config: bytes

    @property
    def digest(self) -> bytes:
        return activation_definition_digest(
            self.artifact_sha256,
            self.component_name,
            self.definition_version,
            self.canonical_config,
        )


class ActivationClient:
    """Fail-closed activation operations over an injected native transport."""

    def __init__(self, transport: ActivationTransport) -> None:
        self._transport = transport

    async def run(
        self,
        request: BeginActivationRequest,
        execute: Callable[[], Awaitable[T]],
        *,
        encode_output: Callable[[T], bytes],
        decode_output: Callable[[bytes], T],
        latency_ms: Callable[[], int],
        on_admitted: Callable[[ActivationDecision], None] | None = None,
        on_completed: Callable[
            [ActivationDecision, ActivationDecision | ActivationCompletionReceipt], None
        ]
        | None = None,
        on_failed: Callable[[ActivationDecision, ActivationFailureReceipt, Exception], None]
        | None = None,
    ) -> tuple[T, ActivationDecision | ActivationCompletionReceipt]:
        """Execute or replay one activation, returning only after durable acceptance."""

        expected_id = activation_id(
            request.project_id,
            request.run_id,
            request.parent_activation_id,
            request.kind,
            request.stable_key,
        )
        decision = await self._transport.begin(request)
        if decision.activation_id != expected_id:
            raise ActivationError(
                ActivationErrorCode.UNKNOWN_OUTCOME,
                f"runtime returned activation ID {decision.activation_id!r}, expected {expected_id!r}",
                activation_id=decision.activation_id,
                attempt=decision.attempt,
            )
        if decision.kind is ActivationDecisionKind.REPLAY:
            if decision.replay_output is None:
                raise ActivationError(
                    ActivationErrorCode.UNKNOWN_OUTCOME,
                    "REPLAY receipt is missing its canonical output",
                    activation_id=decision.activation_id,
                    attempt=decision.attempt,
                )
            if on_admitted is not None:
                on_admitted(decision)
            result = decode_output(decision.replay_output)
            if on_completed is not None:
                on_completed(decision, decision)
            return result, decision
        if decision.kind is not ActivationDecisionKind.EXECUTE:
            raise _decision_error(decision)
        if decision.attempt <= 0 or not decision.fence_token:
            raise ActivationError(
                ActivationErrorCode.UNKNOWN_OUTCOME,
                "EXECUTE receipt is missing fenced authority",
                activation_id=decision.activation_id,
                attempt=decision.attempt,
            )
        if on_admitted is not None:
            on_admitted(decision)

        try:
            result = await execute()
        except Exception as user_error:
            error_data = json.dumps(
                {"message": str(user_error), "type": type(user_error).__name__},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            receipt = await self._transport.fail(
                project_id=request.project_id,
                run_id=request.run_id,
                activation_id=decision.activation_id,
                attempt=decision.attempt,
                fence_token=decision.fence_token,
                error_code="STEP_FAILED",
                error_data=error_data,
                retryable=False,
                external_outcome_certainty="UNKNOWN",
            )
            if (
                receipt.activation_id != decision.activation_id
                or receipt.attempt != decision.attempt
            ):
                raise ActivationError(
                    ActivationErrorCode.UNKNOWN_OUTCOME,
                    "runtime returned a failure receipt for different activation authority",
                    activation_id=decision.activation_id,
                    attempt=decision.attempt,
                ) from user_error
            if on_failed is not None:
                on_failed(decision, receipt, user_error)
            raise

        output = encode_output(result)
        receipt = await self._transport.complete(
            project_id=request.project_id,
            run_id=request.run_id,
            activation_id=decision.activation_id,
            attempt=decision.attempt,
            fence_token=decision.fence_token,
            output=output,
            output_digest=hashlib.sha256(output).digest(),
            latency_ms=latency_ms(),
        )
        if receipt.activation_id != decision.activation_id or receipt.attempt != decision.attempt:
            raise ActivationError(
                ActivationErrorCode.UNKNOWN_OUTCOME,
                "runtime returned a completion receipt for different activation authority",
                activation_id=decision.activation_id,
                attempt=decision.attempt,
            )
        if on_completed is not None:
            on_completed(decision, receipt)
        return result, receipt


def canonical_activation_value(value: Any) -> bytes:
    """Encode one value using the frozen tagged canonical V1 representation."""

    encoded = _canonical_value(value)
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def activation_definition_digest(
    artifact_sha256: bytes,
    component_name: str,
    definition_version: str,
    canonical_config: bytes,
) -> bytes:
    """Compute the frozen definition digest for one deployed component."""

    if len(artifact_sha256) != 32:
        raise ActivationError(
            ActivationErrorCode.INVALID_ARGUMENT,
            "activation artifact SHA-256 must contain exactly 32 bytes",
        )
    try:
        config = json.loads(canonical_config)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivationError(
            ActivationErrorCode.INVALID_ARGUMENT,
            "activation definition config is not valid canonical JSON",
        ) from error
    if json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode() != canonical_config:
        raise ActivationError(
            ActivationErrorCode.INVALID_ARGUMENT,
            "activation definition config is not canonically encoded",
        )
    framed = b"".join(
        _frame(part)
        for part in (
            artifact_sha256,
            component_name.encode("utf-8"),
            definition_version.encode("utf-8"),
            DURABLE_ACTIVATION_V1.encode("ascii"),
            canonical_config,
        )
    )
    return hashlib.sha256(_DEFINITION_DOMAIN + framed).digest()


def activation_id(
    project_id: str,
    run_id: str,
    parent_activation_id: str,
    kind: ActivationKind,
    stable_key: str,
) -> str:
    """Compute the stable V1 identity for one logical activation."""

    encoded = _IDENTITY_DOMAIN
    for value in (project_id, run_id, parent_activation_id):
        encoded += _frame(value.encode("utf-8"))
    encoded += struct.pack(">I", int(kind))
    encoded += _frame(stable_key.encode("utf-8"))
    digest = base64.urlsafe_b64encode(hashlib.sha256(encoded).digest()).rstrip(b"=").decode()
    return f"actv1_{digest}"


def stable_step_key(name: str, ordinal: int, explicit_key: str | None = None) -> str:
    """Return the additive explicit key or sequential compatibility fallback."""

    if not name:
        raise ActivationError(ActivationErrorCode.INVALID_ARGUMENT, "step name cannot be empty")
    if explicit_key is not None:
        if not explicit_key:
            raise ActivationError(
                ActivationErrorCode.INVALID_ARGUMENT,
                "explicit step key cannot be empty",
            )
        return f"step:{name}:{explicit_key}"
    if ordinal < 0:
        raise ActivationError(
            ActivationErrorCode.INVALID_ARGUMENT,
            "sequential step ordinal cannot be negative",
        )
    return f"step:{name}:{ordinal}"


def decode_sha256(value: str) -> bytes:
    """Decode a hexadecimal or padded/unpadded Base64 SHA-256 value."""

    if not value:
        raise ActivationError(
            ActivationErrorCode.DURABILITY_UNAVAILABLE,
            "activation artifact SHA-256 is unavailable",
        )
    decoders = (
        lambda candidate: bytes.fromhex(candidate),
        lambda candidate: base64.b64decode(candidate, validate=True),
        lambda candidate: base64.urlsafe_b64decode(candidate + "=" * (-len(candidate) % 4)),
    )
    for decoder in decoders:
        try:
            decoded = decoder(value)
        except (ValueError, base64.binascii.Error):
            continue
        if len(decoded) == 32:
            return decoded
    raise ActivationError(
        ActivationErrorCode.INVALID_ARGUMENT,
        "activation artifact SHA-256 must encode exactly 32 bytes",
    )


def _canonical_value(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, UInt64):
        return ["u64", str(value)]
    if isinstance(value, int):
        if value < _I64_MIN or value > _I64_MAX:
            raise ActivationError(
                ActivationErrorCode.INVALID_ARGUMENT,
                "canonical activation integer exceeds signed 64-bit range",
            )
        return ["i64", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ActivationError(
                ActivationErrorCode.INVALID_ARGUMENT,
                "canonical activation values reject NaN and infinity",
            )
        bits = 0 if value == 0 else struct.unpack(">Q", struct.pack(">d", value))[0]
        return ["f64", f"{bits:016x}"]
    if isinstance(value, str):
        _validate_unicode(value)
        return ["string", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.urlsafe_b64encode(bytes(value)).rstrip(b"=").decode("ascii")
        return ["bytes", encoded]
    if isinstance(value, (list, tuple)):
        return ["array", [_canonical_value(item) for item in value]]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ActivationError(
                ActivationErrorCode.INVALID_ARGUMENT,
                "canonical activation objects require string keys",
            )
        for key in value:
            _validate_unicode(key)
        return ["object", [[key, _canonical_value(value[key])] for key in sorted(value)]]
    raise ActivationError(
        ActivationErrorCode.INVALID_ARGUMENT,
        f"unsupported canonical activation value type {type(value).__name__}",
    )


def _validate_unicode(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ActivationError(
            ActivationErrorCode.INVALID_ARGUMENT,
            "canonical activation strings must contain valid UTF-8",
        ) from error


def _frame(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _decision_error(decision: ActivationDecision) -> ActivationError:
    codes = {
        ActivationDecisionKind.WAIT: ActivationErrorCode.CONTENDED,
        ActivationDecisionKind.CONFLICT: ActivationErrorCode.NON_DETERMINISTIC_REPLAY,
        ActivationDecisionKind.CANCELLED: ActivationErrorCode.CANCELLED,
        ActivationDecisionKind.UNKNOWN_OUTCOME: ActivationErrorCode.UNKNOWN_OUTCOME,
    }
    messages = {
        ActivationDecisionKind.WAIT: "activation is already executing",
        ActivationDecisionKind.CONFLICT: "stable step key was reused with different input or definition",
        ActivationDecisionKind.CANCELLED: "activation was cancelled",
        ActivationDecisionKind.UNKNOWN_OUTCOME: "activation has an unknown external outcome",
    }
    return ActivationError(
        codes.get(decision.kind, ActivationErrorCode.UNKNOWN_OUTCOME),
        decision.message or messages.get(decision.kind, "runtime returned an invalid decision"),
        activation_id=decision.activation_id,
        attempt=decision.attempt,
    )
