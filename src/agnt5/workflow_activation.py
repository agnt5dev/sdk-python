"""Internal durable-activation execution shared by Python workflow step forms."""

from __future__ import annotations

import hashlib
import inspect
import time
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from ._ids import generate_cid
from ._serialization import deserialize, serialize
from .activation import (
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationDefinition,
    ActivationFailureReceipt,
    ActivationKind,
    ActivationRecoveryPolicy,
    BeginActivationRequest,
    canonical_activation_value,
    decode_sha256,
)
from .events import Completed, ComponentType, Failed, OperationType, Started
from .exceptions import ActivationError, ActivationErrorCode

if TYPE_CHECKING:
    from .workflow import WorkflowContext

T = TypeVar("T")


async def execute_checkpoint_callable(
    context: WorkflowContext,
    name: str,
    step_key: str,
    func_or_awaitable: Callable[..., Awaitable[T]] | Awaitable[T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> T:
    """Execute only the user-code portion of an arbitrary checkpoint step."""

    with context.memo_child_scope("step", step_key):
        if inspect.isasyncgen(func_or_awaitable):
            return await context._consume_streaming_result(func_or_awaitable, name)
        if inspect.iscoroutine(func_or_awaitable) or inspect.isawaitable(func_or_awaitable):
            return await func_or_awaitable
        if callable(func_or_awaitable):
            call_result = func_or_awaitable(*args, **kwargs)
            if inspect.isasyncgen(call_result):
                return await context._consume_streaming_result(call_result, name)
            if inspect.iscoroutine(call_result) or inspect.isawaitable(call_result):
                return await call_result
            return call_result
        raise ValueError(
            f"step() second argument must be awaitable or callable, got {type(func_or_awaitable)}"
        )


async def run_durable_step(
    context: WorkflowContext,
    *,
    name: str,
    step_key: str,
    handler_name: str,
    input_value: Any,
    execute: Callable[[], Awaitable[T]],
) -> T:
    """Run one step through the journal-authoritative activation protocol."""

    metadata = context._trace_metadata or {}
    project_id = metadata.get("project_id", "") or metadata.get("tenant_id", "")
    worker_session_id = metadata.get("worker_session_id", "") or metadata.get("worker_id", "")
    run_authority = metadata.get("run_authority", "") or context.run_id
    lease_authority = metadata.get("lease_authority", "") or metadata.get("lease_id", "")
    component_name = (
        context._workflow_name or metadata.get("component_name", "") or context._workflow_entity.key
    )
    definition_version = metadata.get("activation_definition_version", "")
    if not all(
        (
            project_id,
            context.run_id,
            worker_session_id,
            run_authority,
            lease_authority,
            component_name,
            definition_version,
        )
    ):
        raise ActivationError(
            ActivationErrorCode.DURABILITY_UNAVAILABLE,
            "durable activation requires project, run, worker-session, run, lease, and definition authority",
        )
    canonical_config = metadata.get("activation_definition_config", '["object",[]]').encode("utf-8")
    definition = ActivationDefinition(
        artifact_sha256=decode_sha256(metadata.get("activation_artifact_sha256", "")),
        component_name=component_name,
        definition_version=definition_version,
        canonical_config=canonical_config,
    )
    request = BeginActivationRequest(
        project_id=project_id,
        run_id=context.run_id,
        parent_activation_id=metadata.get("parent_activation_id", ""),
        kind=ActivationKind.STEP,
        stable_key=step_key,
        input_digest=hashlib.sha256(canonical_activation_value(input_value)).digest(),
        definition_digest=definition.digest,
        recovery_policy=ActivationRecoveryPolicy.DURABLE_STEPS,
        worker_session_id=worker_session_id,
        run_authority=run_authority.encode("utf-8"),
        lease_authority=lease_authority.encode("utf-8"),
    )
    step_event_id = str(uuid.uuid4())
    step_correlation_id = generate_cid()
    started_at = time.monotonic()

    def parent_correlation_id() -> str:
        return (
            context._step_event_stack[-1] if context._step_event_stack else context._correlation_id
        )

    def on_admitted(decision: ActivationDecision) -> None:
        context.emit(
            Started(
                name=name,
                correlation_id=step_correlation_id,
                parent_correlation_id=parent_correlation_id(),
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                input_data={"step_name": name, "handler_name": handler_name},
                metadata={
                    "name": name,
                    "step_key": step_key,
                    "activation_id": decision.activation_id,
                    "activation_attempt": str(decision.attempt),
                    "accepted_journal_offset": str(decision.accepted_journal_offset),
                },
            )
        )
        context._step_event_stack.append(step_event_id)

    def pop_step() -> None:
        if not context._step_event_stack:
            return
        popped_id = context._step_event_stack.pop()
        if popped_id != step_event_id:
            context._logger.warning(
                f"Step event stack mismatch in durable step: expected {step_event_id}, got {popped_id}"
            )

    def on_completed(
        decision: ActivationDecision,
        receipt: ActivationDecision | ActivationCompletionReceipt,
    ) -> None:
        pop_step()
        context.emit(
            Completed(
                name=name,
                correlation_id=step_correlation_id,
                parent_correlation_id=parent_correlation_id(),
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                output_data={"step_name": name, "handler_name": handler_name},
                duration_ms=int((time.monotonic() - started_at) * 1000),
                metadata={
                    "name": name,
                    "step_key": step_key,
                    "cache_hit": str(decision.kind is ActivationDecisionKind.REPLAY).lower(),
                    "activation_id": decision.activation_id,
                    "activation_attempt": str(decision.attempt),
                    "accepted_journal_offset": str(receipt.accepted_journal_offset),
                },
            )
        )

    def on_failed(
        decision: ActivationDecision,
        receipt: ActivationFailureReceipt,
        error: Exception,
    ) -> None:
        pop_step()
        context.emit(
            Failed(
                name=name,
                correlation_id=step_correlation_id,
                parent_correlation_id=parent_correlation_id(),
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                error_code=type(error).__name__,
                error_message=str(error),
                metadata={
                    "name": name,
                    "step_key": step_key,
                    "activation_id": decision.activation_id,
                    "activation_attempt": str(decision.attempt),
                    "accepted_journal_offset": str(receipt.accepted_journal_offset),
                },
            )
        )

    try:
        result, _receipt = await context._activation_client.run(
            request,
            execute,
            encode_output=serialize,
            decode_output=deserialize,
            latency_ms=lambda: int((time.monotonic() - started_at) * 1000),
            on_admitted=on_admitted,
            on_completed=on_completed,
            on_failed=on_failed,
        )
    except Exception:
        if context._step_event_stack and context._step_event_stack[-1] == step_event_id:
            context._step_event_stack.pop()
        raise
    context._workflow_entity.record_step_completion(step_key, handler_name, input_value, result)
    return result
