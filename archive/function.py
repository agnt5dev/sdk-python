"""Function primitive - stateless durable operations with retry and checkpointing."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .context import Context
from .durable import DurableFunctionDefinition, function as durable_function, get_registry
from .tools import register_tool_metadata
from ._core import register_tool as _register_tool_core

logger = logging.getLogger(__name__)


@dataclass
class StateTransitionPayload:
    """Represents a single state transition in an entity."""

    operation: str
    key: str
    old_value: Optional[bytes]
    new_value: Optional[bytes]
    timestamp_ms: int


@dataclass
class StateUpdatePayload:
    """Entity state update with transitions."""

    new_state: bytes
    transitions: list[StateTransitionPayload]
    output_data: bytes = b""


@dataclass
class ComponentExecutionResult:
    """Result from executing a component (function, entity method, etc)."""

    output_data: bytes = b""
    metadata: Dict[str, str] = field(default_factory=dict)
    state_update: Optional[Any] = None


def function(*args, **kwargs):  # type: ignore[no-redef]
    """Decorator for stateless durable functions.

    Functions are the basic building block for stateless operations with
    automatic retries, checkpointing, and durability guarantees.

    Example:
        ```python
        @function
        async def process_payment(ctx, amount: float) -> dict:
            # Durable processing logic
            return {"status": "completed", "amount": amount}
        ```

    Args:
        *args: Positional arguments for decorator
        **kwargs: Keyword arguments including:
            - retry: RetryPolicy for error handling
            - backoff: BackoffPolicy for retry delays
            - name: Custom function name

    Returns:
        Decorated function with durability guarantees
    """
    durable_result = durable_function(*args, **kwargs)

    def _register_metadata(registered: Any) -> Any:
        definition: DurableFunctionDefinition = getattr(registered, "_agnt5_durable")
        register_tool_metadata(
            name=definition.name,
            description=getattr(registered, "__doc__", None),
            input_schema=None,
            output_schema=None,
        )
        _register_tool_core(
            definition.name,
            getattr(registered, "__doc__", None),
            None,
            None,
        )
        return registered

    if args and callable(args[0]) and kwargs == {}:
        # Decorator used without parentheses: @function
        registered = durable_result
        return _register_metadata(registered)

    def decorator(func: Any) -> Any:
        # Decorator used with parentheses: @function()
        registered = durable_result(func)
        return _register_metadata(registered)

    return decorator


# Alias for backward compatibility
handler = function


def get_registered_functions() -> Dict[str, Any]:
    """Get all registered functions.

    Returns:
        Dictionary mapping function name to handler
    """
    return {name: definition.handler for name, definition in get_registry().functions().items()}


def get_metadata(func: Any) -> Optional[Dict[str, Any]]:
    """Get metadata for a registered function.

    Args:
        func: Function to get metadata for

    Returns:
        Metadata dictionary or None if not found
    """
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
    """Clear the function registry (for testing)."""
    get_registry().clear()


# Alias for backward compatibility
get_function_metadata = get_metadata


def execute_component(
    name: str,
    input_data: bytes,
    context: Dict[str, Any],
    *,
    component_type: str = "function",
    method_name: Optional[str] = None,
    object_id: Optional[str] = None,
    state_snapshot: Optional[bytes] = None,
    journal_position: Optional[int] = None,
) -> ComponentExecutionResult:
    """Execute a component (function or entity method).

    Args:
        name: Name of the handler to execute
        input_data: Input data as bytes (UTF-8 JSON)
        context: Execution context dictionary
        component_type: Type of component ("function", "entity", "spawn", "task")
        method_name: Method name for entity components
        object_id: Object ID for entity components
        state_snapshot: State snapshot for stateful components
        journal_position: Journal position for replay

    Returns:
        ComponentExecutionResult with output and metadata

    Raises:
        ValueError: If handler not found or component type unsupported
        RuntimeError: If execution fails
    """
    component_key = (component_type or "function").lower()

    # Handle entity method execution
    if component_key == "entity":
        return _execute_entity_method(
            name=name,
            method_name=method_name or "handle",
            input_data=input_data,
            context=context,
            object_id=object_id,
            state_snapshot=state_snapshot,
            journal_position=journal_position,
        )

    # Handle function execution
    if component_key in {"function", "unspecified", "spawn", "task"}:
        return _execute_function(
            name=name,
            input_data=input_data,
            context=context,
            state_snapshot=state_snapshot,
            journal_position=journal_position,
            component_type="function",
        )

    raise ValueError(f"Unsupported component type '{component_type}' for handler '{name}'")


def _execute_function(
    *,
    name: str,
    input_data: bytes,
    context: Dict[str, Any],
    state_snapshot: Optional[bytes],
    journal_position: Optional[int],
    component_type: str,
) -> ComponentExecutionResult:
    """Execute a registered function."""
    registry = get_registry()
    definition = registry.get(name)
    if not definition:
        raise ValueError(f"Handler '{name}' not found in registry")

    ctx, outbound_metadata = _build_context(
        name=name,
        context=context,
        state_snapshot=state_snapshot,
        journal_position=journal_position,
        component_type=component_type,
    )

    decoded_input = _decode_input(input_data)

    try:
        result = _invoke_handler(definition.handler, ctx, decoded_input)
    except Exception as exc:
        logger.exception("Durable function '%s' failed", name)
        raise RuntimeError(f"Function execution failed: {exc}") from exc

    output_bytes = _encode_output(result)

    checkpoints = ctx.export_new_checkpoints()
    if checkpoints:
        outbound_metadata["step_checkpoints"] = json.dumps(checkpoints)
    else:
        outbound_metadata.pop("step_checkpoints", None)

    return ComponentExecutionResult(output_data=output_bytes, metadata=outbound_metadata)


def _execute_entity_method(
    *,
    name: str,
    method_name: str,
    input_data: bytes,
    context: Dict[str, Any],
    object_id: Optional[str],
    state_snapshot: Optional[bytes],
    journal_position: Optional[int],
) -> ComponentExecutionResult:
    """Execute an entity method."""
    from .entity import get_entity_registry

    registry = get_entity_registry()
    method_def = registry.get_method(name, method_name)
    if not method_def:
        raise ValueError(f"Entity method '{name}.{method_name}' not found in registry")

    ctx, outbound_metadata = _build_context(
        name=name,
        context=context,
        state_snapshot=state_snapshot,
        journal_position=journal_position,
        component_type="entity",
        object_id=object_id,
        method_name=method_name,
    )

    decoded_input = _decode_input(input_data)

    try:
        # Invoke the entity method handler
        handler = method_def.handler
        result = _invoke_handler(handler, ctx, decoded_input)
    except Exception as exc:
        logger.exception("Entity method '%s.%s' failed", name, method_name)
        raise RuntimeError(f"Entity method execution failed: {exc}") from exc

    output_bytes = _encode_output(result)

    checkpoints = ctx.export_new_checkpoints()
    if checkpoints:
        outbound_metadata["step_checkpoints"] = json.dumps(checkpoints)

    # TODO: Handle state updates for write methods
    # For now, return simple result
    return ComponentExecutionResult(output_data=output_bytes, metadata=outbound_metadata)


def _build_context(
    *,
    name: str,
    context: Dict[str, Any],
    state_snapshot: Optional[bytes],
    journal_position: Optional[int],
    component_type: str,
    object_id: Optional[str] = None,
    method_name: Optional[str] = None,
) -> tuple[Context, Dict[str, str]]:
    """Build execution context from request data."""
    invocation_metadata = dict(context.get("metadata") or {})
    run_id = invocation_metadata.get("run_id") or context.get("invocation_id")
    step_id = invocation_metadata.get("step_id") or name
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
        component_name=name,
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
    """Parse step checkpoints from payload."""
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
    """Decode input bytes to Python object."""
    if not input_data:
        return {}
    try:
        text = input_data.decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Failed to decode input payload: %s", exc)
        raise RuntimeError("Invalid input payload; expected UTF-8 JSON") from exc


def _invoke_handler(handler: Any, ctx: Context, decoded_input: Any) -> Any:
    """Invoke a handler function with context and input."""
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
    """Run an awaitable in the appropriate event loop."""
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
    """Encode Python object to output bytes."""
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
    "function",
    "handler",
    "get_registered_functions",
    "get_metadata",
    "clear_registry",
    "execute_component",
    "ComponentExecutionResult",
]
