"""Internal durable-activation execution shared by Python workflow step forms."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from ._ids import generate_cid
from ._serialization import deserialize, serialize
from ._telemetry import truncate_span_attribute_value
from .activation import (
    ActivationCompletionReceipt,
    ActivationDecision,
    ActivationDecisionKind,
    ActivationDefinition,
    ActivationFailureReceipt,
    ActivationKind,
    ActivationRecoveryPolicy,
    BeginActivationRequest,
    activation_id,
    canonical_activation_value,
    decode_sha256,
)
from .events import Completed, ComponentType, Failed, OperationType, Started
from .exceptions import ActivationError, ActivationErrorCode, DurableSleepSuspension
from .function import FunctionContext, FunctionRegistry

if TYPE_CHECKING:
    from .workflow import WorkflowContext

T = TypeVar("T")


def _activation_definition(
    context: WorkflowContext,
) -> tuple[dict[str, str], str, ActivationDefinition]:
    metadata = context._trace_metadata or {}
    project_id = metadata.get("project_id", "") or metadata.get("tenant_id", "")
    component_name = (
        context._workflow_name or metadata.get("component_name", "") or context._workflow_entity.key
    )
    definition_version = metadata.get("activation_definition_version", "")
    if not all((project_id, component_name, definition_version)):
        raise ActivationError(
            ActivationErrorCode.DURABILITY_UNAVAILABLE,
            "durable activation requires project and deployed definition identity",
        )
    definition = ActivationDefinition(
        artifact_sha256=decode_sha256(metadata.get("activation_artifact_sha256", "")),
        component_name=component_name,
        definition_version=definition_version,
        canonical_config=metadata.get("activation_definition_config", '["object",[]]').encode(
            "utf-8"
        ),
    )
    return metadata, project_id, definition


async def run_durable_sleep(
    context: WorkflowContext,
    *,
    timer_key: str,
    delay_ms: int,
) -> None:
    """Admit a timer activation and return control to the runtime as suspension."""

    if context._workflow_entity.has_completed_step(timer_key):
        return

    metadata, project_id, definition = _activation_definition(context)
    worker_session_id = metadata.get("worker_session_id", "") or metadata.get("worker_id", "")
    run_authority = metadata.get("run_authority", "") or context.run_id
    lease_authority = metadata.get("lease_authority", "") or metadata.get("lease_id", "")
    if not all((worker_session_id, run_authority, lease_authority)):
        raise ActivationError(
            ActivationErrorCode.DURABILITY_UNAVAILABLE,
            "durable sleep requires worker-session, run, and lease authority",
        )

    input_value = {"delay_ms": delay_ms, "timer_key": timer_key}
    input_digest = hashlib.sha256(canonical_activation_value(input_value)).digest()
    request = BeginActivationRequest(
        project_id=project_id,
        run_id=context.run_id,
        parent_activation_id=metadata.get("parent_activation_id", ""),
        kind=ActivationKind.TIMER,
        stable_key=timer_key,
        input_digest=input_digest,
        definition_digest=definition.digest,
        recovery_policy=ActivationRecoveryPolicy.DURABLE_STEPS,
        worker_session_id=worker_session_id,
        run_authority=run_authority.encode("utf-8"),
        lease_authority=lease_authority.encode("utf-8"),
    )
    expected_id = activation_id(
        request.project_id,
        request.run_id,
        request.parent_activation_id,
        request.kind,
        request.stable_key,
    )
    resumed_timer_key = metadata.get("timer_key", "")
    resumed_activation_id = metadata.get("activation_id", "")
    if resumed_timer_key == timer_key:
        if resumed_activation_id != expected_id:
            raise ActivationError(
                ActivationErrorCode.NON_DETERMINISTIC_REPLAY,
                "timer resume authority does not match the deterministic sleep activation",
                activation_id=resumed_activation_id,
            )
        context._workflow_entity.record_step_completion(timer_key, "sleep", input_value, None)
        return

    decision = await context._activation_client.begin(request)
    if decision.kind is not ActivationDecisionKind.EXECUTE:
        from .activation import _decision_error

        raise _decision_error(decision)

    continuation: dict[str, Any] = {
        "completed_steps": context._workflow_entity._completed_steps,
        "step_events": context._workflow_entity._step_events,
        "workflow_correlation_id": context._correlation_id,
    }
    state = context._workflow_entity._state
    if state is not None:
        continuation["workflow_state"] = state.get_state_snapshot()

    raise DurableSleepSuspension(
        activation_id=decision.activation_id,
        attempt=decision.attempt,
        fence_token=decision.fence_token,
        timer_key=timer_key,
        input_digest=input_digest,
        definition_digest=definition.digest,
        continuation=json.dumps(continuation, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ),
        delay_ms=delay_ms,
    )


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


async def execute_function_callable(
    context: WorkflowContext,
    handler_name: str,
    step_name: str,
    step_key: str,
    step_correlation_id: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Execute the language-local @function body and its nested lifecycle events."""

    from ._serialization import serialize_to_str
    from .context import _current_context, set_current_context
    from .tracing import create_span

    func_config = FunctionRegistry.get(handler_name)
    if func_config is None:
        raise ValueError(f"Function '{handler_name}' not found in registry")
    input_repr = (
        truncate_span_attribute_value(serialize_to_str({"args": args, "kwargs": kwargs}))
        if args or kwargs
        else "{}"
    )
    with create_span(
        f"workflow.task.{handler_name}",
        "function",
        context._runtime_context,
        {
            "step_name": step_name,
            "handler_name": handler_name,
            "run_id": context.run_id,
            "input.data": input_repr,
        },
    ) as span:
        func_correlation_id = generate_cid()
        func_context = FunctionContext(
            run_id=context.run_id,
            correlation_id=func_correlation_id,
            parent_correlation_id=step_correlation_id,
            runtime_context=context._runtime_context,
            worker=context._worker,
            trace_metadata=context._trace_metadata,
            memo_namespace=context.allocate_memo_child_scope("step", step_key),
        )
        if len(args) == 1 and isinstance(args[0], dict):
            function_input = args[0]
        elif kwargs.get("input") and isinstance(kwargs.get("input"), dict):
            function_input = kwargs["input"]
        elif kwargs:
            function_input = dict(kwargs)
        else:
            function_input = {"args": list(args)} if args else {}
        context.emit(
            Started(
                name=handler_name,
                correlation_id=func_correlation_id,
                parent_correlation_id=step_correlation_id,
                component_type=ComponentType.FUNCTION,
                input_data=function_input,
                attempt=0,
            )
        )
        function_started_at = time.time_ns()
        context_token = set_current_context(func_context)
        call_kwargs = dict(kwargs)
        try:
            try:
                if not args and "input" in call_kwargs:
                    input_data = call_kwargs.pop("input")
                    handler_result = func_config.handler(func_context, input_data, **call_kwargs)
                else:
                    handler_result = func_config.handler(func_context, *args, **call_kwargs)
                if inspect.isasyncgen(handler_result):
                    result = await context._consume_streaming_result(handler_result, step_name)
                elif inspect.iscoroutine(handler_result):
                    result = await handler_result
                else:
                    result = handler_result
            finally:
                _current_context.reset(context_token)

            try:
                span.set_attribute(
                    "output.data",
                    truncate_span_attribute_value(serialize_to_str(result)),
                )
            except (TypeError, ValueError):
                span.set_attribute("output.data", truncate_span_attribute_value(repr(result)))
            context.emit(
                Completed(
                    name=handler_name,
                    correlation_id=func_correlation_id,
                    parent_correlation_id=step_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    output_data=result if isinstance(result, dict) else {"result": result},
                    duration_ms=(time.time_ns() - function_started_at) // 1_000_000,
                )
            )
            return result
        except Exception as error:
            context.emit(
                Failed(
                    name=handler_name,
                    correlation_id=func_correlation_id,
                    parent_correlation_id=step_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    error_code=type(error).__name__,
                    error_message=str(error),
                    duration_ms=(time.time_ns() - function_started_at) // 1_000_000,
                )
            )
            span.set_attribute("error", "true")
            span.set_attribute("error.message", str(error))
            span.set_attribute("error.type", type(error).__name__)
            raise


async def run_durable_step(
    context: WorkflowContext,
    *,
    name: str,
    step_key: str,
    handler_name: str,
    input_value: Any,
    execute: Callable[[str], Awaitable[T]],
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
            lambda: execute(step_correlation_id),
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
