"""Compatibility layer exposing decorators and execution helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .agent import get_agent_registry
from .context import Context
from .durable import DurableFunctionDefinition, function as durable_function
from .durable import get_registry

logger = logging.getLogger(__name__)


@dataclass
class StateTransitionPayload:
    operation: str
    key: str
    old_value: Optional[bytes]
    new_value: Optional[bytes]
    timestamp_ms: int


@dataclass
class StateUpdatePayload:
    new_state: bytes
    transitions: list[StateTransitionPayload]
    output_data: bytes = b""


@dataclass
class ComponentExecutionResult:
    output_data: bytes = b""
    metadata: Dict[str, str] = field(default_factory=dict)
    state_update: Optional[StateUpdatePayload] = None


def function(*args, **kwargs):  # type: ignore[no-redef]
    """Backward compatible alias for ``durable.function``."""

    return durable_function(*args, **kwargs)


handler = function


def get_registered_functions() -> Dict[str, Any]:
    return {name: definition.handler for name, definition in get_registry().functions().items()}


def get_function_metadata(func: Any) -> Optional[Dict[str, Any]]:
    definition: Optional[DurableFunctionDefinition]
    definition = getattr(func, "_agnt5_durable", None)
    if not definition:
        registry = get_registry().functions()
        for candidate in registry.values():
            if candidate.handler is func:
                definition = candidate
                break
    if not definition:
        return None

    signature = inspect.signature(definition.handler)
    parameters = []
    for index, param in enumerate(signature.parameters.values()):
        if index == 0 and param.name == "ctx":
            continue
        param_entry = {"name": param.name, "kind": str(param.kind).split(".")[-1]}
        if param.annotation is not inspect.Parameter.empty:
            param_entry["annotation"] = getattr(param.annotation, "__name__", str(param.annotation))
        if param.default is not inspect.Parameter.empty:
            param_entry["default"] = param.default
        parameters.append(param_entry)

    metadata = definition.to_metadata()
    metadata["parameters"] = parameters
    return metadata


def clear_registry() -> None:
    get_registry().clear()


def execute_component(
    handler_name: str,
    input_data: bytes,
    context: Dict[str, Any],
    *,
    component_type: str = "function",
    method_name: Optional[str] = None,
    object_id: Optional[str] = None,
    state_snapshot: Optional[bytes] = None,
    journal_position: Optional[int] = None,
) -> ComponentExecutionResult:
    component_key = (component_type or "function").lower()
    if component_key == "agent":
        return _execute_agent(
            handler_name=handler_name,
            input_data=input_data,
            context=context,
            method_name=method_name,
            state_snapshot=state_snapshot,
            journal_position=journal_position,
            object_id=object_id,
        )
    if component_key in {"function", "unspecified"}:
        return _execute_function(
            handler_name=handler_name,
            input_data=input_data,
            context=context,
            state_snapshot=state_snapshot,
            journal_position=journal_position,
            component_type=component_key,
        )
    if component_key in {"spawn", "task"}:
        # Spawned child invocations reuse durable function handling.
        return _execute_function(
            handler_name=handler_name,
            input_data=input_data,
            context=context,
            state_snapshot=state_snapshot,
            journal_position=journal_position,
            component_type="function",
        )
    raise ValueError(f"Unsupported component type '{component_type}' for handler '{handler_name}'")


def _execute_function(
    *,
    handler_name: str,
    input_data: bytes,
    context: Dict[str, Any],
    state_snapshot: Optional[bytes],
    journal_position: Optional[int],
    component_type: str,
) -> ComponentExecutionResult:
    registry = get_registry()
    definition = registry.get(handler_name)
    if not definition:
        raise ValueError(f"Handler '{handler_name}' not found in registry")

    ctx, outbound_metadata = _build_context(
        handler_name=handler_name,
        context=context,
        state_snapshot=state_snapshot,
        journal_position=journal_position,
        component_type=component_type,
    )

    decoded_input = _decode_input(input_data)

    try:
        result = _invoke_handler(definition.handler, ctx, decoded_input)
    except Exception as exc:
        logger.exception("Durable function '%s' failed", handler_name)
        raise RuntimeError(f"Function execution failed: {exc}") from exc

    output_bytes = _encode_output(result)

    checkpoints = ctx.export_new_checkpoints()
    if checkpoints:
        outbound_metadata["step_checkpoints"] = json.dumps(checkpoints)
    else:
        outbound_metadata.pop("step_checkpoints", None)

    return ComponentExecutionResult(output_data=output_bytes, metadata=outbound_metadata)


def _execute_agent(
    *,
    handler_name: str,
    input_data: bytes,
    context: Dict[str, Any],
    method_name: Optional[str],
    state_snapshot: Optional[bytes],
    journal_position: Optional[int],
    object_id: Optional[str],
) -> ComponentExecutionResult:
    registry = get_agent_registry()
    definition = registry.get(handler_name)
    if not definition:
        raise ValueError(f"Stateful agent '{handler_name}' is not registered")

    ctx, outbound_metadata = _build_context(
        handler_name=handler_name,
        context=context,
        state_snapshot=state_snapshot,
        journal_position=journal_position,
        component_type="agent",
        object_id=object_id,
        method_name=method_name,
    )

    agent_instance = definition.agent_cls()
    target_method = method_name or "on_message"
    if not hasattr(agent_instance, target_method):
        raise AttributeError(
            f"Agent '{handler_name}' does not implement method '{target_method}'"
        )

    decoded_input = _decode_input(input_data)
    method = getattr(agent_instance, target_method)
    result = method(ctx, decoded_input)
    if inspect.isawaitable(result):
        result = _run_awaitable(result)

    output_bytes = _encode_output(result)

    checkpoints = ctx.export_new_checkpoints()
    if checkpoints:
        outbound_metadata["step_checkpoints"] = json.dumps(checkpoints)

    state_update = ctx.memory.export_state_update()
    if state_update:
        state_update.output_data = output_bytes
        transitions = [
            StateTransitionPayload(
                operation=transition.operation,
                key=transition.key,
                old_value=transition.old_value,
                new_value=transition.new_value,
                timestamp_ms=transition.timestamp_ms,
            )
            for transition in state_update.transitions
        ]
        normalized_update = StateUpdatePayload(
            new_state=state_update.new_state,
            transitions=transitions,
            output_data=state_update.output_data,
        )
        return ComponentExecutionResult(
            output_data=b"",
            metadata=outbound_metadata,
            state_update=normalized_update,
        )

    return ComponentExecutionResult(output_data=output_bytes, metadata=outbound_metadata)


def _build_context(
    *,
    handler_name: str,
    context: Dict[str, Any],
    state_snapshot: Optional[bytes],
    journal_position: Optional[int],
    component_type: str,
    object_id: Optional[str] = None,
    method_name: Optional[str] = None,
) -> tuple[Context, Dict[str, str]]:
    invocation_metadata = dict(context.get("metadata") or {})
    run_id = invocation_metadata.get("run_id") or context.get("invocation_id")
    step_id = invocation_metadata.get("step_id") or handler_name
    attempt_raw = invocation_metadata.get("attempt", 1)
    try:
        attempt = int(attempt_raw)
    except (TypeError, ValueError):
        attempt = 1

    step_checkpoint_payload = context.get("step_checkpoints") if context else None
    if not step_checkpoint_payload:
        step_checkpoint_payload = invocation_metadata.get("step_checkpoints")
    parsed_checkpoints = _parse_step_checkpoints(step_checkpoint_payload)

    ctx = Context(
        run_id=str(run_id) if run_id is not None else str(context.get("invocation_id")),
        step_id=str(step_id),
        attempt=attempt,
        invocation_id=str(context.get("invocation_id") or ""),
        service_name=str(context.get("service_name") or ""),
        component_name=handler_name,
        metadata=invocation_metadata,
        step_checkpoints=parsed_checkpoints,
        parent_run_id=invocation_metadata.get("parent_run_id"),
        component_type=component_type,
        object_id=object_id,
        method_name=method_name,
        state_snapshot=state_snapshot,
        journal_position=journal_position,
    )

    outbound_metadata: Dict[str, str] = dict(invocation_metadata)
    outbound_metadata["component_type"] = component_type
    if object_id:
        outbound_metadata["object_id"] = object_id
    if method_name:
        outbound_metadata["method_name"] = method_name
    if journal_position is not None:
        outbound_metadata["journal_position"] = str(journal_position)

    return ctx, outbound_metadata


def _parse_step_checkpoints(payload: Any) -> Optional[list[dict[str, Any]]]:
    if not payload:
        return None
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Received invalid step_checkpoints payload: %s", payload)
            return None
    if isinstance(payload, (list, tuple)):
        return list(payload)
    logger.warning("Unsupported step_checkpoints payload type: %s", type(payload))
    return None


def _decode_input(input_data: bytes) -> Any:
    if not input_data:
        return {}
    try:
        text = input_data.decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Failed to decode input payload: %s", exc)
        raise RuntimeError("Invalid input payload; expected UTF-8 JSON") from exc


def _invoke_handler(handler: Any, ctx: Context, decoded_input: Any) -> Any:
    if isinstance(decoded_input, dict):
        result = handler(ctx, **decoded_input)
    elif isinstance(decoded_input, list):
        result = handler(ctx, *decoded_input)
    else:
        result = handler(ctx, decoded_input)

    if inspect.isawaitable(result):
        return _run_awaitable(result)
    return result


def _run_awaitable(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if loop.is_running():  # pragma: no cover - defensive path when nested loops exist
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(awaitable)
        finally:
            new_loop.close()
    return loop.run_until_complete(awaitable)


def _encode_output(result: Any) -> bytes:
    if result is None:
        return b""
    if isinstance(result, bytes):
        return result
    if isinstance(result, str):
        return result.encode("utf-8")
    try:
        return json.dumps(result).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Result serialization error: {exc}") from exc


__all__ = [
    "ComponentExecutionResult",
    "StateTransitionPayload",
    "StateUpdatePayload",
    "clear_registry",
    "execute_component",
    "function",
    "get_function_metadata",
    "get_registered_functions",
    "handler",
]
