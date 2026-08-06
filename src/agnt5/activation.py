"""Durable activation V1 values, identities, and typed adapter contracts."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from .exceptions import ActivationError, ActivationErrorCode

DURABLE_ACTIVATION_V1 = "durable_activation_v1"
_IDENTITY_DOMAIN = b"agnt5.activation.identity.v1\0"
_DEFINITION_DOMAIN = b"agnt5.activation.definition.v1\0"
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_U64_MAX = 2**64 - 1
_NATIVE_ACTIVATION_ERROR_PREFIX = "AGNT5_ACTIVATION_ERROR:"

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
    FUNCTION = 2
    AGENT = FUNCTION
    MODEL = 3
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


@dataclass(frozen=True)
class ActivationExecution:
    """Authority exposed to user code while one activation is executing."""

    activation_id: str
    attempt: int
    idempotency_key: str


@dataclass(frozen=True)
class ActivationUsage:
    """Bounded accounting committed with an accepted activation."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class ActivationEvidence:
    """Bounded immutable evidence committed with a terminal activation."""

    evidence_type: str
    payload: bytes
    sha256: bytes

    @classmethod
    def inline(cls, evidence_type: str, payload: bytes) -> "ActivationEvidence":
        return cls(
            evidence_type=evidence_type,
            payload=bytes(payload),
            sha256=hashlib.sha256(payload).digest(),
        )


_current_activation: ContextVar[ActivationExecution | None] = ContextVar(
    "agnt5_current_activation",
    default=None,
)


def current_activation() -> ActivationExecution | None:
    """Return the active durable unit for downstream idempotency propagation."""

    return _current_activation.get()


def _set_current_activation(decision: ActivationDecision) -> Token:
    return _current_activation.set(
        ActivationExecution(
            activation_id=decision.activation_id,
            attempt=decision.attempt,
            idempotency_key=f"agnt5:{decision.activation_id}",
        )
    )


def _reset_current_activation(token: Token) -> None:
    _current_activation.reset(token)


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
        usage: ActivationUsage,
        evidence: tuple[ActivationEvidence, ...],
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
        evidence: tuple[ActivationEvidence, ...],
    ) -> ActivationFailureReceipt: ...


def _native_activation_error(error: Exception) -> ActivationError:
    if isinstance(error, ActivationError):
        return error
    message = str(error)
    marker = message.find(_NATIVE_ACTIVATION_ERROR_PREFIX)
    if marker >= 0:
        try:
            detail = json.loads(message[marker + len(_NATIVE_ACTIVATION_ERROR_PREFIX) :])
            try:
                code = ActivationErrorCode(detail.get("code", ""))
            except ValueError:
                code = ActivationErrorCode.UNKNOWN_OUTCOME
            return ActivationError(
                code,
                detail.get("message") or message,
                activation_id=detail.get("activationId") or "",
                attempt=int(detail.get("attempt") or 0),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return ActivationError(ActivationErrorCode.UNKNOWN_OUTCOME, message)


class NativeActivationTransport:
    """Typed Python adapter over the feature-gated PyO3 activation client."""

    def __init__(self, native_client: Any) -> None:
        self._native_client = native_client

    async def begin(self, request: BeginActivationRequest) -> ActivationDecision:
        try:
            response = await self._native_client.begin_activation(
                request.project_id,
                request.run_id,
                request.parent_activation_id,
                int(request.kind),
                request.stable_key,
                list(request.input_digest),
                list(request.definition_digest),
                int(request.recovery_policy),
                request.worker_session_id,
                list(request.run_authority),
                list(request.lease_authority),
            )
        except Exception as error:
            raise _native_activation_error(error) from error
        return ActivationDecision(
            kind=ActivationDecisionKind(response.kind),
            activation_id=response.activation_id,
            attempt=int(response.attempt),
            accepted_journal_offset=int(response.accepted_journal_offset),
            fence_token=bytes(response.fence_token),
            replay_output=(
                bytes(response.replay_output) if response.replay_output is not None else None
            ),
            message=response.message,
        )

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
        usage: ActivationUsage,
        evidence: tuple[ActivationEvidence, ...],
    ) -> ActivationCompletionReceipt:
        try:
            response = await self._native_client.complete_activation(
                project_id,
                run_id,
                activation_id,
                attempt,
                list(fence_token),
                list(output),
                list(output_digest),
                usage.tokens_in,
                usage.tokens_out,
                usage.cost_usd,
                usage.latency_ms,
                usage.provider,
                usage.model,
                [
                    (item.evidence_type, list(item.payload), list(item.sha256))
                    for item in evidence
                ],
            )
        except Exception as error:
            raise _native_activation_error(error) from error
        return ActivationCompletionReceipt(
            activation_id=response.activation_id,
            attempt=int(response.attempt),
            accepted_journal_offset=int(response.accepted_journal_offset),
            replayed=bool(response.replayed),
        )

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
        evidence: tuple[ActivationEvidence, ...],
    ) -> ActivationFailureReceipt:
        try:
            response = await self._native_client.fail_activation(
                project_id,
                run_id,
                activation_id,
                attempt,
                list(fence_token),
                error_code,
                list(error_data),
                retryable,
                external_outcome_certainty,
                [
                    (item.evidence_type, list(item.payload), list(item.sha256))
                    for item in evidence
                ],
            )
        except Exception as error:
            raise _native_activation_error(error) from error
        return ActivationFailureReceipt(
            activation_id=response.activation_id,
            attempt=int(response.attempt),
            accepted_journal_offset=int(response.accepted_journal_offset),
            status=response.status,
            replayed=bool(response.replayed),
        )


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

    async def begin(self, request: BeginActivationRequest) -> ActivationDecision:
        """Admit one activation and validate the runtime-owned identity."""

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
        if decision.kind is ActivationDecisionKind.EXECUTE and (
            decision.attempt <= 0 or not decision.fence_token
        ):
            raise ActivationError(
                ActivationErrorCode.UNKNOWN_OUTCOME,
                "EXECUTE receipt is missing fenced authority",
                activation_id=decision.activation_id,
                attempt=decision.attempt,
            )
        return decision

    async def complete(
        self,
        request: BeginActivationRequest,
        decision: ActivationDecision,
        *,
        output: bytes,
        usage: ActivationUsage,
        evidence: tuple[ActivationEvidence, ...] = (),
    ) -> ActivationCompletionReceipt:
        """Commit one fenced completion and validate the returned authority."""

        receipt = await self._transport.complete(
            project_id=request.project_id,
            run_id=request.run_id,
            activation_id=decision.activation_id,
            attempt=decision.attempt,
            fence_token=decision.fence_token,
            output=output,
            output_digest=hashlib.sha256(output).digest(),
            usage=usage,
            evidence=evidence,
        )
        if receipt.activation_id != decision.activation_id or receipt.attempt != decision.attempt:
            raise ActivationError(
                ActivationErrorCode.UNKNOWN_OUTCOME,
                "runtime returned a completion receipt for different activation authority",
                activation_id=decision.activation_id,
                attempt=decision.attempt,
            )
        return receipt

    async def fail(
        self,
        request: BeginActivationRequest,
        decision: ActivationDecision,
        *,
        error_code: str,
        error_data: bytes,
        retryable: bool,
        external_outcome_certainty: str = "UNKNOWN",
        evidence: tuple[ActivationEvidence, ...] = (),
    ) -> ActivationFailureReceipt:
        """Commit one fenced failure and validate the returned authority."""

        receipt = await self._transport.fail(
            project_id=request.project_id,
            run_id=request.run_id,
            activation_id=decision.activation_id,
            attempt=decision.attempt,
            fence_token=decision.fence_token,
            error_code=error_code,
            error_data=error_data,
            retryable=retryable,
            external_outcome_certainty=external_outcome_certainty,
            evidence=evidence,
        )
        if receipt.activation_id != decision.activation_id or receipt.attempt != decision.attempt:
            raise ActivationError(
                ActivationErrorCode.UNKNOWN_OUTCOME,
                "runtime returned a failure receipt for different activation authority",
                activation_id=decision.activation_id,
                attempt=decision.attempt,
            )
        return receipt

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
        failure_error_code: str = "STEP_FAILED",
        failure_retryable: bool = False,
        failure_external_outcome_certainty: str = "UNKNOWN",
        completion_usage: Callable[[T], ActivationUsage] | None = None,
        completion_evidence: Callable[[T], tuple[ActivationEvidence, ...]] | None = None,
        failure_evidence: Callable[[Exception], tuple[ActivationEvidence, ...]] | None = None,
    ) -> tuple[T, ActivationDecision | ActivationCompletionReceipt]:
        """Execute or replay one activation, returning only after durable acceptance."""

        decision = await self.begin(request)
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
            receipt = await self.fail(
                request,
                decision,
                error_code=failure_error_code,
                error_data=error_data,
                retryable=failure_retryable,
                external_outcome_certainty=failure_external_outcome_certainty,
                evidence=failure_evidence(user_error) if failure_evidence is not None else (),
            )
            if on_failed is not None:
                on_failed(decision, receipt, user_error)
            raise

        output = encode_output(result)
        usage = completion_usage(result) if completion_usage is not None else ActivationUsage()
        usage = replace(usage, latency_ms=latency_ms())
        receipt = await self.complete(
            request,
            decision,
            output=output,
            usage=usage,
            evidence=completion_evidence(result) if completion_evidence is not None else (),
        )
        if on_completed is not None:
            on_completed(decision, receipt)
        return result, receipt


def activation_request_from_context(
    context: Any,
    *,
    kind: ActivationKind,
    stable_key: str,
    input_value: Any,
    recovery_policy: ActivationRecoveryPolicy,
) -> BeginActivationRequest:
    """Build one journal-bound request from negotiated worker context authority."""

    metadata = dict(getattr(context, "_trace_metadata", None) or {})
    project_id = metadata.get("project_id") or metadata.get("tenant_id") or ""
    run_id = getattr(context, "run_id", "")
    worker_session_id = metadata.get("worker_session_id") or metadata.get("worker_id") or ""
    run_authority = metadata.get("run_authority") or run_id
    lease_authority = metadata.get("lease_authority") or metadata.get("lease_id") or ""
    definition_version = metadata.get("activation_definition_version") or ""
    component_name = (
        metadata.get("component_name")
        or getattr(context, "component_name", None)
        or getattr(context, "_agent_name", None)
        or ""
    )
    if not all(
        (
            project_id,
            run_id,
            worker_session_id,
            run_authority,
            lease_authority,
            component_name,
            definition_version,
        )
    ):
        raise ActivationError(
            ActivationErrorCode.DURABILITY_UNAVAILABLE,
            "durable activation requires project, run, worker-session, run, lease, component, and definition authority",
        )
    canonical_config = metadata.get("activation_definition_config", '["object",[]]').encode(
        "utf-8"
    )
    return BeginActivationRequest(
        project_id=project_id,
        run_id=run_id,
        parent_activation_id=metadata.get("parent_activation_id", ""),
        kind=kind,
        stable_key=stable_key,
        input_digest=hashlib.sha256(canonical_activation_value(input_value)).digest(),
        definition_digest=activation_definition_digest(
            decode_sha256(metadata.get("activation_artifact_sha256", "")),
            component_name,
            definition_version,
            canonical_config,
        ),
        recovery_policy=recovery_policy,
        worker_session_id=worker_session_id,
        run_authority=run_authority.encode("utf-8"),
        lease_authority=lease_authority.encode("utf-8"),
    )


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
