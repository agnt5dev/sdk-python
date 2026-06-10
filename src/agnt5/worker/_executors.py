"""Executor mixin for component execution.

Supports functions, entities, workflows, agents, and tools.
"""

from __future__ import annotations, print_function

import asyncio
import inspect
import secrets
import time
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .._ids import generate_cid
from .._serialization import deserialize, serialize
from .._telemetry import setup_module_logger
from ._utils import create_failed_response, format_error_message

if TYPE_CHECKING:
    from .._core import PyExecuteComponentResponse

logger = setup_module_logger(__name__)


def _trace_id_from_request(request: Any) -> str:
    """Extract trace_id from request's runtime_context, or return 'none'."""
    rc = getattr(request, "runtime_context", None)
    if rc is not None:
        tid = getattr(rc, "trace_id", None)
        if tid:
            return tid
    return "none"


def _truncate_input(input_dict: Any, max_len: int = 200) -> str:
    """Return a truncated string repr of input for logging."""
    if isinstance(input_dict, dict) and not input_dict:
        return "{}"
    s = str(input_dict)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _ensure_input_dict(input_value: Any) -> dict:
    """Validate that component input is a JSON object after deserialization."""
    if not isinstance(input_value, dict):
        raise ValueError(
            f"Component input must be a JSON object, "
            f"got {type(input_value).__name__}: {input_value!r}"
        )
    return input_value


def _set_current_span_from_runtime_context(runtime_context: Any) -> Any | None:
    """Set the Python tracing contextvar from a runtime context, returning its token."""
    if runtime_context is None:
        return None
    trace_id = getattr(runtime_context, "trace_id", None)
    span_id = getattr(runtime_context, "span_id", None)
    if not trace_id or not span_id:
        return None

    from ..tracing import SpanInfo, _current_span

    return _current_span.set(SpanInfo(trace_id=trace_id, span_id=span_id))


def _reset_current_span_token(token: Any | None) -> None:
    """Reset a token returned by _set_current_span_from_runtime_context."""
    if token is None:
        return

    from ..tracing import _current_span

    _current_span.reset(token)


def _resolve_session_user_ids(request: Any, input_dict: Any) -> tuple[str, str | None]:
    """Resolve durable execution scope IDs from request metadata, payload, then run ID."""
    payload = input_dict if isinstance(input_dict, dict) else {}
    session_id = (
        getattr(request, "session_id", None)
        or payload.get("session_id")
        or request.invocation_id
    )
    user_id = getattr(request, "user_id", None) or payload.get("user_id") or None
    return session_id, user_id


class ExecutorMixin:
    """Mixin providing component execution methods for Worker.

    Uses a common execution pattern:
    1. Deserialize input
    2. Create context via factory
    3. Execute domain logic via callback
    4. Handle errors consistently
    5. Clean up context
    """

    # Expected attributes from Worker class
    _rust_worker: Any
    _entity_state_adapter: Any
    _checkpoint_client: Any
    service_name: str

    async def _execute_with_context(
        self,
        request: Any,
        context_factory: Callable[[dict, Any], Any],
        executor: Callable[[Any, dict, Any], Coroutine],
        component_type: str,
    ) -> "PyExecuteComponentResponse | None":
        """Common execution wrapper for all component types."""
        from .._core import PyExecuteComponentResponse
        from .._state_adapter import _entity_state_adapter_ctx
        from ..context import _current_context, get_current_context, set_current_context

        token = None
        span_token = None
        state_adapter_token = None
        try:
            input_dict = deserialize(request.input_data) if request.input_data else {}
            input_dict = _ensure_input_dict(input_dict)
            ctx = context_factory(input_dict, request)
            token = set_current_context(ctx)
            span_token = _set_current_span_from_runtime_context(
                getattr(request, "runtime_context", None)
            )
            state_adapter_token = _entity_state_adapter_ctx.set(
                getattr(self, "_entity_state_adapter", None)
            )

            return await executor(ctx, input_dict, request)

        except Exception as e:
            from ..events import ComponentType, Failed

            error_msg = format_error_message(e)
            current_ctx = get_current_context()
            error_logger = current_ctx.logger if current_ctx else logger
            error_logger.error(f"{component_type} execution failed: {error_msg}", exc_info=True)

            # Emit run.failed via event queue (not synchronous return)
            # This ensures proper event ordering: started -> failed
            if current_ctx is not None:
                failed_event = Failed(
                    name=component_type,
                    correlation_id=getattr(current_ctx, "correlation_id", generate_cid()),
                    parent_correlation_id=getattr(
                        current_ctx, "parent_correlation_id", generate_cid()
                    ),
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.debug(
                    f"[_execute_with_context] Emitting run.failed event: "
                    f"event_type={failed_event.event_type}, "
                    f"error={error_msg}"
                )
                current_ctx.emit(failed_event)
                return None

            # Fallback: if no context, return synchronous error response
            # This should be rare - only if context creation itself failed
            return create_failed_response(request, e, PyExecuteComponentResponse)

        finally:
            _reset_current_span_token(span_token)
            if state_adapter_token is not None:
                _entity_state_adapter_ctx.reset(state_adapter_token)
            if token is not None:
                _current_context.reset(token)

    def _create_error_response(
        self, request: Any, error_message: str
    ) -> "PyExecuteComponentResponse":
        """Create an error response for component not found."""
        from .._core import PyExecuteComponentResponse

        return PyExecuteComponentResponse(
            invocation_id=request.invocation_id,
            success=False,
            output_data=b"",
            state_update=None,
            error_message=error_message,
            metadata=None,
            event_type="run.failed",
            content_index=0,
            sequence=0,
            attempt=getattr(request, "attempt", 0),
        )

    # -------------------------------------------------------------------------
    # Function Execution
    # -------------------------------------------------------------------------

    async def _execute_function(
        self, config: Any, input_data: bytes, request: Any
    ) -> "PyExecuteComponentResponse | None":
        """Execute a function handler."""
        from ..events import Completed, ComponentType, Started
        from ..function import FunctionContext

        logger.debug(
            f"[_execute_function] Starting execution for component={config.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> FunctionContext:
            correlation_id = generate_cid()
            return FunctionContext(
                run_id=req.invocation_id,  # Use actual invocation_id for event routing
                correlation_id=correlation_id,
                parent_correlation_id="",
                attempt=getattr(req, "attempt", 0),
                runtime_context=req.runtime_context,
                retry_policy=config.retries,
                worker=self._rust_worker,
                trace_metadata=getattr(req, "metadata", None),
            )

        async def execute(ctx: FunctionContext, input_dict: dict, req: Any):
            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = ctx.run_id[:8]
            run_parent_correlation_id = ctx.parent_correlation_id
            start_time_ns = time.time_ns()
            fn_correlation_id = generate_cid()

            # Batch run.started + function.started into a single AppendBatch RPC
            run_started_event = Started(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=run_parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            fn_started_event = Started(
                name=config.name,
                correlation_id=fn_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.FUNCTION,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            await ctx.emit_batch_async([run_started_event, fn_started_event])
            ctx.correlation_id = fn_correlation_id
            ctx.parent_correlation_id = run_correlation_id

            trace_id = _trace_id_from_request(req)
            logger.info(
                f"run.started | run_id={req.invocation_id} component={config.name} "
                f"type=function trace_id={trace_id} input={_truncate_input(input_dict)}"
            )

            # Execute function with error handling for proper event emission
            try:
                result = (
                    config.handler(ctx, **input_dict)
                    if isinstance(input_dict, dict) and input_dict
                    else config.handler(ctx)
                )

                # Handle coroutine with optional timeout
                if inspect.iscoroutine(result):
                    if hasattr(config, "timeout_ms") and config.timeout_ms is not None:
                        try:
                            result = await asyncio.wait_for(result, timeout=config.timeout_ms / 1000.0)
                        except asyncio.TimeoutError:
                            raise asyncio.TimeoutError(
                                f"Function '{config.name}' timed out after {config.timeout_ms}ms"
                            )
                    else:
                        result = await result

                # Handle streaming
                if inspect.isasyncgen(result):
                    return await self._handle_streaming_function(ctx, result)

            except asyncio.CancelledError:
                # Cooperative cancellation (CancelExecution → task.cancel()).
                # The user's finally/async-with cleanup has already run as the
                # CancelledError propagated up. The gateway authored
                # run.cancelled as the terminal event, so do NOT emit
                # run.failed — stop cleanly with no response.
                duration_ms = (time.time_ns() - start_time_ns) // 1_000_000
                logger.info(
                    f"run.cancelled | run_id={req.invocation_id} component={config.name} "
                    f"type=function trace_id={trace_id} duration_ms={duration_ms}"
                )
                return None

            except Exception as e:
                # Calculate function duration even on failure
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_code = "TIMEOUT" if isinstance(e, (asyncio.TimeoutError, TimeoutError)) else type(e).__name__
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Emit function.failed (child of run)
                from ..events import Failed
                fn_failed_event = Failed(
                    name=config.name,
                    correlation_id=fn_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    error_code=error_code,
                    error_message=error_msg,
                    duration_ms=duration_ms,
                )
                await ctx.emit_async(fn_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=config.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=run_parent_correlation_id,
                    component_type=ComponentType.RUN,
                    error_code=error_code,
                    error_message=error_msg,
                )
                logger.info(
                    f"run.failed | run_id={req.invocation_id} component={config.name} "
                    f"type=function trace_id={trace_id} duration_ms={duration_ms} error={error_msg}"
                )
                await ctx.emit_async(run_failed_event)

                # Return None - the event queue handles delivery
                return None

            # Calculate function duration
            end_time_ns = time.time_ns()
            duration_ms = (end_time_ns - start_time_ns) // 1_000_000

            # Emit function.completed (child of run)
            fn_completed_event = Completed(
                name=config.name,
                correlation_id=fn_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.FUNCTION,
                output_data=result,
                duration_ms=duration_ms,
            )

            # Emit run.completed via event queue (not synchronous return)
            # This ensures proper event ordering: started -> completed
            run_completed_event = Completed(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=run_parent_correlation_id,
                component_type=ComponentType.RUN,
                output_data=result,
            )

            await ctx.emit_async(fn_completed_event)
            await ctx.emit_async(run_completed_event)

            logger.info(
                f"run.completed | run_id={req.invocation_id} component={config.name} "
                f"type=function trace_id={trace_id} duration_ms={duration_ms}"
            )

            # Return None - the event queue handles delivery
            return None

        return await self._execute_with_context(request, create_context, execute, "Function")

    async def _handle_streaming_function(self, ctx: Any, result: Any) -> None:
        """Handle streaming function by queueing deltas."""
        from ..events import Event

        sequence = 0
        has_typed_events = False
        first_chunk = True

        async for chunk in result:
            if isinstance(chunk, Event):
                has_typed_events = True
                event_fields = chunk.to_dict()
                output_data = event_fields.get("output_data", b"")

                if isinstance(output_data, bytes):
                    try:
                        event_data = deserialize(output_data)
                    except (ValueError, Exception):
                        event_data = {"content": output_data.decode("utf-8", errors="replace")}
                elif isinstance(output_data, dict):
                    event_data = output_data
                else:
                    event_data = {"content": str(output_data or "")}

                ctx.emit(
                    event_fields.get("event_type", "output.delta"),
                    event_data,
                    content_index=event_fields.get("content_index", 0),
                )
            else:
                if first_chunk:
                    ctx.emit("output.start", {}, content_index=0)
                    sequence += 1
                    first_chunk = False

                if isinstance(chunk, str):
                    chunk_content = chunk
                elif isinstance(chunk, bytes):
                    chunk_content = chunk.decode("utf-8")
                elif isinstance(chunk, dict):
                    chunk_content = chunk
                else:
                    chunk_content = serialize(chunk).decode("utf-8")

                if isinstance(chunk_content, dict):
                    ctx.emit("output.delta", chunk_content, content_index=0)
                else:
                    ctx.emit("output.delta", {"content": chunk_content}, content_index=0)
            sequence += 1

        if not has_typed_events and not first_chunk:
            ctx.emit("output.stop", {}, content_index=0)

        ctx.emit("run.completed", {}, content_index=0)
        logger.debug(f"Streaming function queued {sequence + 1} events")
        return None

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    async def _execute_tool(
        self, tool: Any, input_data: bytes, request: Any
    ) -> "PyExecuteComponentResponse | None":
        """Execute a tool handler."""
        from ..context import Context
        from ..events import Completed, ComponentType, Failed, Started

        logger.debug(
            f"[_execute_tool] Starting execution for tool={tool.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> Context:
            correlation_id = f"tool-{secrets.token_hex(5)}"
            return Context(
                run_id=req.invocation_id,
                correlation_id=correlation_id,
                parent_correlation_id=generate_cid(),
                attempt=getattr(req, "attempt", 0),
                runtime_context=req.runtime_context,
                worker=self._rust_worker,
            )

        async def execute(ctx: Context, input_dict: dict, req: Any):
            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = ctx.run_id[:8]

            # Emit run.started before executing handler
            run_started_event = Started(
                name=tool.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.debug(
                f"[_execute_tool] Emitting run.started event: "
                f"tool={tool.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            trace_id = _trace_id_from_request(req)
            logger.info(
                f"run.started | run_id={req.invocation_id} component={tool.name} "
                f"type=tool trace_id={trace_id} input={_truncate_input(input_dict)}"
            )

            # Emit tool.started (child of run)
            start_time_ns = time.time_ns()
            tool_correlation_id = f"tool-{secrets.token_hex(5)}"
            tool_started_event = Started(
                name=tool.name,
                correlation_id=tool_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.TOOL,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.debug(
                f"[_execute_tool] Emitting tool.started event: "
                f"tool={tool.name}, correlation_id={tool_correlation_id}"
            )
            ctx.emit(tool_started_event)

            # Execute tool with error handling for proper event emission
            try:
                result = (
                    await tool.invoke(ctx, **input_dict)
                    if isinstance(input_dict, dict) and input_dict
                    else await tool.invoke(ctx)
                )

            except Exception as e:
                # Calculate tool duration even on failure
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Emit tool.failed (child of run)
                tool_failed_event = Failed(
                    name=tool.name,
                    correlation_id=tool_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.TOOL,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                    duration_ms=duration_ms,
                )
                logger.debug(
                    f"[_execute_tool] Emitting tool.failed event: "
                    f"tool={tool.name}, error={error_msg}"
                )
                ctx.emit(tool_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=tool.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.debug(
                    f"[_execute_tool] Emitting run.failed event: "
                    f"tool={tool.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                logger.info(
                    f"run.failed | run_id={req.invocation_id} component={tool.name} "
                    f"type=tool trace_id={trace_id} duration_ms={duration_ms} error={error_msg}"
                )

                # Return None - the event queue handles delivery
                return None

            # Calculate tool duration
            end_time_ns = time.time_ns()
            duration_ms = (end_time_ns - start_time_ns) // 1_000_000

            # Emit tool.completed (child of run)
            tool_completed_event = Completed(
                name=tool.name,
                correlation_id=tool_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.TOOL,
                output_data=result,
                duration_ms=duration_ms,
            )
            logger.debug(
                f"[_execute_tool] Emitting tool.completed event: "
                f"tool={tool.name}, duration_ms={duration_ms}"
            )
            ctx.emit(tool_completed_event)

            # Emit run.completed via event queue (not synchronous return)
            run_completed_event = Completed(
                name=tool.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                output_data=result,
            )
            logger.debug(
                f"[_execute_tool] Emitting run.completed event: "
                f"tool={tool.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

            logger.info(
                f"run.completed | run_id={req.invocation_id} component={tool.name} "
                f"type=tool trace_id={trace_id} duration_ms={duration_ms}"
            )

            # Return None - the event queue handles delivery
            return None

        return await self._execute_with_context(request, create_context, execute, "Tool")

    # -------------------------------------------------------------------------
    # Entity Execution
    # -------------------------------------------------------------------------

    async def _execute_entity(
        self, entity_type: Any, input_data: bytes, request: Any
    ) -> "PyExecuteComponentResponse | None":
        """Execute an entity method with lifecycle events."""
        from ..context import Context
        from ..events import Completed, ComponentType, Failed, Started

        logger.debug(
            f"[_execute_entity] Starting execution for entity={entity_type.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> Context:
            entity_key = input_dict.get("key", "unknown")
            correlation_id = f"ent-{secrets.token_hex(5)}"
            return Context(
                run_id=req.invocation_id,
                correlation_id=correlation_id,
                parent_correlation_id=generate_cid(),
                attempt=getattr(req, "attempt", 0),
                runtime_context=req.runtime_context,
                worker=self._rust_worker,
            )

        async def execute(ctx: Context, input_dict: dict, req: Any):
            entity_key = input_dict.pop("key", None)
            method_name = input_dict.pop("method", None)

            if not entity_key:
                raise ValueError("Entity invocation requires 'key' parameter")
            if not method_name:
                raise ValueError("Entity invocation requires 'method' parameter")

            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = ctx.run_id[:8]

            # Emit run.started before executing entity method
            run_started_event = Started(
                name=entity_type.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data={"key": entity_key, "method": method_name, **input_dict},
                attempt=ctx.attempt,
            )
            logger.debug(
                f"[_execute_entity] Emitting run.started event: "
                f"entity={entity_type.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            trace_id = _trace_id_from_request(req)
            logger.info(
                f"run.started | run_id={req.invocation_id} component={entity_type.name} "
                f"type=entity trace_id={trace_id} input={_truncate_input({{'key': entity_key, 'method': method_name, **input_dict}})}"
            )

            # Emit entity.started (child of run)
            start_time_ns = time.time_ns()
            entity_correlation_id = f"ent-{secrets.token_hex(5)}"
            entity_started_event = Started(
                name=entity_type.name,
                correlation_id=entity_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.ENTITY,
                input_data={"key": entity_key, "method": method_name, **input_dict},
            )
            logger.debug(
                f"[_execute_entity] Emitting entity.started event: "
                f"entity={entity_type.name}, key={entity_key}, method={method_name}"
            )
            ctx.emit(entity_started_event)

            # Execute entity method with error handling
            try:
                entity_instance = entity_type.entity_class(key=entity_key)

                if not hasattr(entity_instance, method_name):
                    raise ValueError(f"Entity '{entity_type.name}' has no method '{method_name}'")

                method = getattr(entity_instance, method_name)
                result = (
                    await method(**input_dict)
                    if isinstance(input_dict, dict) and input_dict
                    else await method()
                )

            except Exception as e:
                # Calculate entity duration even on failure
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Emit entity.failed (child of run)
                entity_failed_event = Failed(
                    name=entity_type.name,
                    correlation_id=entity_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.ENTITY,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                    duration_ms=duration_ms,
                )
                logger.debug(
                    f"[_execute_entity] Emitting entity.failed event: "
                    f"entity={entity_type.name}, error={error_msg}"
                )
                ctx.emit(entity_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=entity_type.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.debug(
                    f"[_execute_entity] Emitting run.failed event: "
                    f"entity={entity_type.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                logger.info(
                    f"run.failed | run_id={req.invocation_id} component={entity_type.name} "
                    f"type=entity trace_id={trace_id} duration_ms={duration_ms} error={error_msg}"
                )

                # Return None - the event queue handles delivery
                return None

            # Calculate entity duration
            end_time_ns = time.time_ns()
            duration_ms = (end_time_ns - start_time_ns) // 1_000_000

            # Emit entity.completed (child of run)
            entity_completed_event = Completed(
                name=entity_type.name,
                correlation_id=entity_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.ENTITY,
                output_data=result,
                duration_ms=duration_ms,
            )
            logger.debug(
                f"[_execute_entity] Emitting entity.completed event: "
                f"entity={entity_type.name}, duration_ms={duration_ms}"
            )
            ctx.emit(entity_completed_event)

            # Emit run.completed via event queue
            run_completed_event = Completed(
                name=entity_type.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                output_data=result,
            )
            logger.debug(
                f"[_execute_entity] Emitting run.completed event: "
                f"entity={entity_type.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

            logger.info(
                f"run.completed | run_id={req.invocation_id} component={entity_type.name} "
                f"type=entity trace_id={trace_id} duration_ms={duration_ms}"
            )

            # Return None - the event queue handles delivery
            return None

        return await self._execute_with_context(request, create_context, execute, "Entity")

    # -------------------------------------------------------------------------
    # Agent Execution
    # -------------------------------------------------------------------------

    async def _execute_agent(
        self, agent: Any, input_data: bytes, request: Any
    ) -> "PyExecuteComponentResponse | None":
        """Execute an agent with session support."""
        from ..agent import AgentContext
        from ..agent.events import AgentCompleted, AgentFailed, AgentStarted
        from ..events import Completed, ComponentType, Event, Failed, Started

        logger.debug(
            f"[_execute_agent] Starting execution for agent={agent.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> AgentContext:
            session_id, user_id = _resolve_session_user_ids(req, input_dict)
            if getattr(req, "session_id", None) or input_dict.get("session_id"):
                logger.debug(f"Using agent session: {session_id}")
            else:
                logger.debug(f"Using invocation_id as ephemeral agent session: {session_id}")

            correlation_id = generate_cid()
            return AgentContext(
                run_id=req.invocation_id,
                agent_name=agent.name,
                session_id=session_id,
                user_id=user_id,
                runtime_context=req.runtime_context,
                is_streaming=getattr(req, "is_streaming", False),
                worker=self._rust_worker,
                correlation_id=correlation_id,
                parent_correlation_id=generate_cid(),
                trace_metadata=getattr(req, "metadata", None),
            )

        async def execute(ctx: AgentContext, input_dict: dict, req: Any):
            from .._core import PyExecuteComponentResponse

            user_message = input_dict.get("message", "")
            if not user_message:
                raise ValueError(
                    f"Agent invocation requires a 'message' key in the input dict. "
                    f"Received keys: {list(input_dict.keys())}. "
                    f"Check that your dataset input matches the component's expected schema."
                )

            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = ctx.run_id[:8]

            # Emit run.started before executing agent
            run_started_event = Started(
                name=agent.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=getattr(req, "attempt", 0),
            )
            logger.debug(
                f"[_execute_agent] Emitting run.started event: "
                f"agent={agent.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            trace_id = _trace_id_from_request(req)
            logger.info(
                f"run.started | run_id={req.invocation_id} component={agent.name} "
                f"type=agent trace_id={trace_id} input={_truncate_input(input_dict)}"
            )

            # Emit agent.started (child of run)
            start_time_ns = time.time_ns()
            agent_correlation_id = generate_cid()
            agent_started_event = Started(
                name=agent.name,
                correlation_id=agent_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.AGENT,
                input_data={"message": user_message},
            )
            logger.debug(
                f"[_execute_agent] Emitting agent.started event: "
                f"agent={agent.name}, correlation_id={agent_correlation_id}"
            )
            ctx.emit(agent_started_event)

            # Mark context as executor-managed so Agent._run_core() doesn't emit
            # duplicate agent.started/completed events
            ctx._executor_managed_lifecycle = True

            try:
                result = agent.run(user_message, context=ctx)

                if inspect.isasyncgen(result):
                    sequence = 0
                    final_output = None
                    final_tool_calls = []
                    handoff_to = None

                    async for event in result:
                        if isinstance(event, Event):
                            # Skip agent lifecycle events - executor already emits these
                            # to avoid duplicate agent.started/completed/failed events
                            if isinstance(event, (AgentStarted, AgentCompleted, AgentFailed)):
                                # Extract final results from AgentCompleted
                                if isinstance(event, AgentCompleted):
                                    if hasattr(event, 'output_data') and isinstance(event.output_data, dict):
                                        final_output = event.output_data.get("output", "")
                                        final_tool_calls = event.output_data.get("tool_calls", [])
                                        handoff_to = event.output_data.get("handoff_to")
                                continue

                            # Forward other events to the context
                            ctx.emit(event)
                            sequence += 1

                            # Check for completion event to extract final results
                            if hasattr(event, 'output_data') and isinstance(event.output_data, dict):
                                if event.output_data.get("output"):
                                    final_output = event.output_data.get("output", "")
                                    final_tool_calls = event.output_data.get("tool_calls", [])
                                    handoff_to = event.output_data.get("handoff_to")

                    # Calculate agent duration
                    end_time_ns = time.time_ns()
                    duration_ms = (end_time_ns - start_time_ns) // 1_000_000

                    # Emit agent.completed
                    agent_completed_event = Completed(
                        name=agent.name,
                        correlation_id=agent_correlation_id,
                        parent_correlation_id=run_correlation_id,
                        component_type=ComponentType.AGENT,
                        output_data={"output": final_output, "tool_calls": final_tool_calls},
                        duration_ms=duration_ms,
                    )
                    ctx.emit(agent_completed_event)

                    # Emit run.completed
                    final_result = {"output": final_output, "tool_calls": final_tool_calls}
                    if handoff_to:
                        final_result["handoff_to"] = handoff_to

                    run_completed_event = Completed(
                        name=agent.name,
                        correlation_id=run_correlation_id,
                        parent_correlation_id=ctx.parent_correlation_id,
                        component_type=ComponentType.RUN,
                        output_data=final_result,
                    )
                    ctx.emit(run_completed_event)

                    # Emit session.created so the session projection materializes
                    # this session for GET /v1/sessions/{id} queries.
                    from ..events import Event as _BaseEvent
                    session_event = _BaseEvent(
                        name=agent.name,
                        correlation_id=run_correlation_id,
                        parent_correlation_id="",
                    )
                    object.__setattr__(session_event, "event_type", "session.created")
                    session_event.metadata = {
                        "session_id": ctx.session_id or ctx.run_id,
                        "component_name": agent.name,
                        "session_type": "agent",
                    }
                    ctx.emit(session_event)

                    logger.info(
                        f"run.completed | run_id={req.invocation_id} component={agent.name} "
                        f"type=agent trace_id={trace_id} duration_ms={duration_ms}"
                    )
                    logger.debug(f"Agent streaming queued {sequence + 1} events")
                    return None

                # Non-streaming fallback
                if inspect.iscoroutine(result):
                    agent_result = await result
                else:
                    agent_result = result

                # Calculate agent duration
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000

                # Emit agent.completed
                agent_completed_event = Completed(
                    name=agent.name,
                    correlation_id=agent_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.AGENT,
                    output_data={"output": agent_result.output, "tool_calls": agent_result.tool_calls},
                    duration_ms=duration_ms,
                )
                logger.debug(
                    f"[_execute_agent] Emitting agent.completed event: "
                    f"agent={agent.name}, duration_ms={duration_ms}"
                )
                ctx.emit(agent_completed_event)

                # Emit run.completed
                run_completed_event = Completed(
                    name=agent.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.RUN,
                    output_data={"output": agent_result.output, "tool_calls": agent_result.tool_calls},
                )
                logger.debug(
                    f"[_execute_agent] Emitting run.completed event: "
                    f"agent={agent.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_completed_event)

                logger.info(
                    f"run.completed | run_id={req.invocation_id} component={agent.name} "
                    f"type=agent trace_id={trace_id} duration_ms={duration_ms}"
                )

                # Emit session.created so the session projection materializes
                # this session for GET /v1/sessions/{id} queries.
                from ..events import Event as _BaseEvent
                session_event = _BaseEvent(
                    name=agent.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id="",
                )
                object.__setattr__(session_event, "event_type", "session.created")
                session_event.metadata = {
                    "session_id": ctx.session_id or ctx.run_id,
                    "component_name": agent.name,
                    "session_type": "agent",
                }
                ctx.emit(session_event)

                return None

            except Exception as e:
                # Calculate agent duration even on failure
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Emit agent.failed (child of run)
                agent_failed_event = Failed(
                    name=agent.name,
                    correlation_id=agent_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.AGENT,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                    duration_ms=duration_ms,
                )
                logger.debug(
                    f"[_execute_agent] Emitting agent.failed event: "
                    f"agent={agent.name}, error={error_msg}"
                )
                ctx.emit(agent_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=agent.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.debug(
                    f"[_execute_agent] Emitting run.failed event: "
                    f"agent={agent.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                logger.info(
                    f"run.failed | run_id={req.invocation_id} component={agent.name} "
                    f"type=agent trace_id={trace_id} duration_ms={duration_ms} error={error_msg}"
                )

                # Return None - the event queue handles delivery
                return None

        return await self._execute_with_context(request, create_context, execute, "Agent")

    # -------------------------------------------------------------------------
    # Scorer Execution
    # -------------------------------------------------------------------------

    async def _execute_scorer(
        self, config: Any, input_data: bytes, request: Any
    ) -> "PyExecuteComponentResponse | None":
        """Execute a scorer handler.

        Scorers receive a ScorerRequest and return a ScorerResult.
        They are stateless evaluation functions.
        """
        from ..eval.types import ScorerRequest, ScorerResult
        from ..events import Completed, ComponentType, Failed, Started
        from ..scorer import ScorerContext

        logger.debug(
            f"[_execute_scorer] Starting execution for scorer={config.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> ScorerContext:
            correlation_id = f"scorer-{secrets.token_hex(5)}"
            return ScorerContext(
                run_id=req.invocation_id,
                correlation_id=correlation_id,
                parent_correlation_id=generate_cid(),
                attempt=getattr(req, "attempt", 0),
                runtime_context=req.runtime_context,
                worker=self._rust_worker,
                trace_metadata=getattr(req, "metadata", None),
                peer_scores=input_dict.get("peer_scores"),
            )

        async def execute(ctx: ScorerContext, input_dict: dict, req: Any):
            # Create short run correlation id
            run_correlation_id = ctx.run_id[:8]

            # Build ScorerRequest from input
            scorer_request = ScorerRequest(
                output=input_dict.get("output"),
                expected=input_dict.get("expected"),
                input=input_dict.get("input"),
                trace=input_dict.get("trace"),
                config=input_dict.get("config"),
                peer_scores=input_dict.get("peer_scores"),
            )

            # Emit run.started
            run_started_event = Started(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.debug(
                f"[_execute_scorer] Emitting run.started event: "
                f"scorer={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            trace_id = _trace_id_from_request(req)
            logger.info(
                f"run.started | run_id={req.invocation_id} component={config.name} "
                f"type=scorer trace_id={trace_id} input={_truncate_input(input_dict)}"
            )

            # Emit scorer.started (child of run)
            start_time_ns = time.time_ns()
            scorer_correlation_id = f"scorer-{secrets.token_hex(5)}"
            scorer_started_event = Started(
                name=config.name,
                correlation_id=scorer_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.SCORER,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.debug(
                f"[_execute_scorer] Emitting scorer.started event: "
                f"scorer={config.name}, correlation_id={scorer_correlation_id}"
            )
            ctx.emit(scorer_started_event)

            # Execute scorer with error handling
            try:
                # Scorer handlers can take (ctx, request) or just (request)
                sig = inspect.signature(config.handler)
                params = list(sig.parameters.values())
                needs_context = bool(params) and params[0].name == "ctx"

                if needs_context:
                    result = await config.handler(ctx, scorer_request)
                else:
                    result = await config.handler(scorer_request)

                # Ensure result is a ScorerResult
                if not isinstance(result, ScorerResult):
                    result = ScorerResult(
                        score=float(result) if isinstance(result, (int, float)) else 0.0,
                        passed=bool(result) if isinstance(result, bool) else None,
                    )

            except Exception as e:
                # Calculate scorer duration even on failure
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Emit scorer.failed (child of run)
                scorer_failed_event = Failed(
                    name=config.name,
                    correlation_id=scorer_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.SCORER,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                    duration_ms=duration_ms,
                )
                logger.debug(
                    f"[_execute_scorer] Emitting scorer.failed event: "
                    f"scorer={config.name}, error={error_msg}"
                )
                ctx.emit(scorer_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=config.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.debug(
                    f"[_execute_scorer] Emitting run.failed event: "
                    f"scorer={config.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                logger.info(
                    f"run.failed | run_id={req.invocation_id} component={config.name} "
                    f"type=scorer trace_id={trace_id} duration_ms={duration_ms} error={error_msg}"
                )

                return None

            # Calculate scorer duration
            end_time_ns = time.time_ns()
            duration_ms = (end_time_ns - start_time_ns) // 1_000_000

            # Convert result to dict for output
            result_dict = {
                "score": result.score,
                "passed": result.passed,
                "label": result.label,
                "explanation": result.explanation,
                "metadata": result.metadata,
            }

            # Emit scorer.completed (child of run)
            scorer_completed_event = Completed(
                name=config.name,
                correlation_id=scorer_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.SCORER,
                output_data=result_dict,
                duration_ms=duration_ms,
            )
            logger.debug(
                f"[_execute_scorer] Emitting scorer.completed event: "
                f"scorer={config.name}, duration_ms={duration_ms}"
            )
            ctx.emit(scorer_completed_event)

            # Emit run.completed
            run_completed_event = Completed(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                output_data=result_dict,
            )
            logger.debug(
                f"[_execute_scorer] Emitting run.completed event: "
                f"scorer={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

            logger.info(
                f"run.completed | run_id={req.invocation_id} component={config.name} "
                f"type=scorer trace_id={trace_id} duration_ms={duration_ms}"
            )

            return None

        return await self._execute_with_context(request, create_context, execute, "Scorer")

    # -------------------------------------------------------------------------
    # Workflow Execution
    # -------------------------------------------------------------------------

    async def _execute_workflow(
        self, config: Any, input_data: bytes, request: Any
    ) -> "PyExecuteComponentResponse | None":
        """Execute a workflow handler with automatic replay support.

        Uses ctx.emit() for ALL lifecycle events to ensure proper ordering:
        - run.started -> workflow.started -> workflow.step.* -> workflow.completed -> run.completed

        Returns None to let the event queue handle delivery.
        """
        import json
        import time as _time
        import traceback as _traceback

        from .._core import PyExecuteComponentResponse
        from .._state_adapter import _entity_state_adapter_ctx, _get_state_adapter
        from ..context import set_current_context
        from ..events import Completed, ComponentType, Failed, Started
        from ..exceptions import WaitingForUserInputException
        from ..workflow import WorkflowContext, WorkflowEntity, WorkflowState

        # Set entity state adapter in context so workflows can use Entities
        state_adapter_token = _entity_state_adapter_ctx.set(self._entity_state_adapter)

        # Variables that need to be accessible in exception handlers
        ctx = None
        token = None
        span_token = None
        session_id = None
        workflow_start_time = _time.time()
        start_time_ns = time.time_ns()

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}
            input_dict = _ensure_input_dict(input_dict)

            # Parse replay data from request metadata for crash recovery
            completed_steps = {}
            step_events = []
            initial_state = {}
            user_response = None
            resumed_workflow_correlation_id = None
            resumed_step_correlation_id = None
            resumed_step_name = None

            if hasattr(request, 'metadata') and request.metadata:
                # Parse completed steps for replay
                if "completed_steps" in request.metadata:
                    completed_steps_json = request.metadata["completed_steps"]
                    if completed_steps_json:
                        try:
                            completed_steps = json.loads(completed_steps_json)
                            logger.debug(f"Replaying workflow with {len(completed_steps)} cached steps")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse completed_steps from metadata")
                elif "step_events" in request.metadata:
                    step_events_json = request.metadata["step_events"]
                    if step_events_json:
                        try:
                            step_events_list = json.loads(step_events_json)
                            for event in step_events_list:
                                if "step_name" in event and "result" in event:
                                    completed_steps[event["step_name"]] = event["result"]
                            step_events = step_events_list
                            logger.debug(f"Resuming workflow with {len(completed_steps)} completed steps")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse step_events from metadata")

                # Parse initial workflow state
                if "workflow_state" in request.metadata:
                    workflow_state_json = request.metadata["workflow_state"]
                    if workflow_state_json:
                        try:
                            initial_state = json.loads(workflow_state_json)
                            logger.debug(f"Loaded workflow state: {len(initial_state)} keys")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse workflow_state from metadata")

                # Check for user response (resume after pause)
                if "user_response" in request.metadata:
                    user_response = request.metadata["user_response"]
                    logger.debug(f"Resuming workflow with user response: {user_response}")

                # Restore workflow correlation ID for resume
                # This ensures the same correlation ID is used after resume
                if "workflow_correlation_id" in request.metadata:
                    resumed_workflow_correlation_id = request.metadata["workflow_correlation_id"]
                    logger.debug(f"Restoring workflow correlation ID: {resumed_workflow_correlation_id}")

                # Restore step correlation info for proper event pairing on resume
                if "step_correlation_id" in request.metadata:
                    resumed_step_correlation_id = request.metadata["step_correlation_id"]
                    logger.debug(f"Restoring step correlation ID: {resumed_step_correlation_id}")
                if "step_name" in request.metadata:
                    resumed_step_name = request.metadata["step_name"]
                    logger.debug(f"Restoring step name: {resumed_step_name}")

            # Resolve session/user scopes for state and memory. Platform metadata
            # wins, payload fields are honored for legacy direct calls, and
            # invocation_id is the ephemeral fallback for non-session runs.
            session_id, user_id = _resolve_session_user_ids(request, input_dict)
            if getattr(request, "session_id", None) or input_dict.get("session_id"):
                logger.debug(f"Using workflow session: {session_id}")
            else:
                logger.debug(f"Using invocation_id as ephemeral workflow session: {session_id}")
            is_streaming = getattr(request, 'is_streaming', False)
            component_name = getattr(request, 'component_name', None)

            # Create WorkflowEntity for state management
            workflow_entity = WorkflowEntity(
                run_id=request.invocation_id,
                session_id=session_id,
                user_id=user_id,
                component_name=component_name,
            )

            # Load replay data into entity if provided
            if completed_steps:
                workflow_entity._completed_steps = completed_steps
                logger.debug(f"Loaded {len(completed_steps)} completed steps into workflow entity")

            if step_events:
                workflow_entity._step_events = step_events
                logger.debug(f"Restored {len(step_events)} step events into workflow entity")

            # Inject user response if resuming from pause
            if user_response:
                if hasattr(request, 'metadata') and request.metadata:
                    pause_index_str = request.metadata.get("pause_index", "0")
                    try:
                        workflow_entity._pause_index = int(pause_index_str)
                    except ValueError:
                        workflow_entity._pause_index = 0

                workflow_entity.inject_user_response(user_response)
                workflow_entity._pause_index = 0  # Reset for replay

                # Store resumed step info for proper event pairing
                if resumed_step_correlation_id:
                    workflow_entity._resumed_step_correlation_id = resumed_step_correlation_id
                if resumed_step_name:
                    workflow_entity._resumed_step_name = resumed_step_name

            if initial_state:
                state_adapter = _get_state_adapter()
                if hasattr(state_adapter, '_standalone_states'):
                    state_adapter._standalone_states[workflow_entity._state_key] = initial_state
                workflow_entity._state = WorkflowState(initial_state.copy(), workflow_entity)
                logger.debug(f"Initialized workflow entity state with {len(initial_state)} keys")

            wf_trace_id = _trace_id_from_request(request)

            # Create WorkflowContext
            ctx = WorkflowContext(
                workflow_entity=workflow_entity,
                run_id=request.invocation_id,
                session_id=session_id,
                user_id=user_id,
                runtime_context=request.runtime_context,
                is_streaming=is_streaming,
                worker=self._rust_worker,
                trace_metadata=getattr(request, "metadata", None),
            )

            # Set context in contextvar
            token = set_current_context(ctx)

            # Use restored correlation ID for resumed workflows, or generate a new one
            if resumed_workflow_correlation_id:
                workflow_correlation_id = resumed_workflow_correlation_id
                logger.debug(f"Using restored workflow correlation ID: {workflow_correlation_id}")
            else:
                workflow_correlation_id = generate_cid()

            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = ctx.run_id[:8]

            # Setup context fields for all workflow events
            ctx._correlation_id = workflow_correlation_id
            ctx._parent_correlation_id = run_correlation_id
            ctx._component_name = config.name
            ctx._workflow_name = config.name
            ctx._is_replay = bool(completed_steps)

            # Set up trace parent-child linking
            if request.runtime_context:
                span_token = _set_current_span_from_runtime_context(request.runtime_context)

            # Emit run.started and workflow.started events only for fresh executions.
            # For resumed workflows (HITL), the platform emits run.resumed instead,
            # and the workflow code will emit workflow.resumed when it detects the resume.
            if not ctx._is_replay:
                # Emit run.started event (like function executor does)
                run_started_event = Started(
                    name=config.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=None,  # Run events are top-level
                    component_type=ComponentType.RUN,
                    input_data=input_dict,
                    attempt=getattr(request, 'attempt', 0),
                )
                logger.debug(
                    f"[_execute_workflow] Emitting run.started event: "
                    f"component={config.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_started_event)

                wf_trace_id = _trace_id_from_request(request)
                logger.info(
                    f"run.started | run_id={request.invocation_id} component={config.name} "
                    f"type=workflow trace_id={wf_trace_id} input={_truncate_input(input_dict)}"
                )

                # Emit workflow.started event (child of run)
                workflow_started_event = Started(
                    name=config.name,
                    correlation_id=workflow_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.WORKFLOW,
                    input_data=input_dict,
                    attempt=getattr(request, 'attempt', 0),
                )
                logger.debug(
                    f"[_execute_workflow] Emitting workflow.started event: "
                    f"component={config.name}, correlation_id={workflow_correlation_id}"
                )
                ctx.emit(workflow_started_event)
            else:
                logger.debug(
                    f"[_execute_workflow] Skipping run.started and workflow.started for resumed workflow: "
                    f"component={config.name}"
                )

            # Execute workflow
            try:
                with ctx.as_parent():
                    if isinstance(input_dict, dict) and input_dict:
                        result = await config.handler(ctx, **input_dict)
                    else:
                        result = await config.handler(ctx)

            except WaitingForUserInputException:
                # Re-raise to be handled in the outer exception handler
                raise

            except Exception as workflow_error:
                # Calculate workflow duration on failure
                end_time_ns = time.time_ns()
                workflow_duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_msg = f"{type(workflow_error).__name__}: {str(workflow_error)}"

                logger.info(
                    f"run.failed | run_id={request.invocation_id} component={config.name} "
                    f"type=workflow trace_id={wf_trace_id} duration_ms={workflow_duration_ms} error={error_msg}"
                )
                logger.error(f"Workflow failed after {workflow_duration_ms}ms: {error_msg}", exc_info=True)

                # Emit workflow.failed (child of run)
                workflow_failed_event = Failed(
                    name=config.name,
                    correlation_id=workflow_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.WORKFLOW,
                    error_code=type(workflow_error).__name__,
                    error_message=error_msg,
                    duration_ms=workflow_duration_ms,
                )
                logger.debug(
                    f"[_execute_workflow] Emitting workflow.failed event: "
                    f"component={config.name}, error={error_msg}"
                )
                ctx.emit(workflow_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=config.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=None,  # Run events are top-level
                    component_type=ComponentType.RUN,
                    error_code=type(workflow_error).__name__,
                    error_message=error_msg,
                )
                logger.debug(
                    f"[_execute_workflow] Emitting run.failed event: "
                    f"component={config.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                # Return None - the event queue handles delivery
                return None

            # Calculate workflow duration
            end_time_ns = time.time_ns()
            workflow_duration_ms = (end_time_ns - start_time_ns) // 1_000_000

            logger.info(
                f"run.completed | run_id={request.invocation_id} component={config.name} "
                f"type=workflow trace_id={wf_trace_id} duration_ms={workflow_duration_ms}"
            )

            # Persist workflow entity state
            if hasattr(ctx, '_workflow_entity') and ctx._workflow_entity._state is not None:
                if ctx._workflow_entity._state.has_changes():
                    try:
                        await ctx._workflow_entity._persist_state()
                        logger.debug(f"Persisted WorkflowEntity state for run {request.invocation_id}")
                    except Exception as persist_error:
                        logger.error(f"Failed to persist WorkflowEntity state: {persist_error}", exc_info=True)

            # Emit workflow.completed (child of run)
            workflow_completed_event = Completed(
                name=config.name,
                correlation_id=workflow_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.WORKFLOW,
                output_data=result,
                duration_ms=workflow_duration_ms,
            )
            logger.debug(
                f"[_execute_workflow] Emitting workflow.completed event: "
                f"component={config.name}, duration_ms={workflow_duration_ms}"
            )
            ctx.emit(workflow_completed_event)

            # Emit run.completed via event queue (not synchronous return)
            # This ensures proper event ordering: started -> steps -> completed
            run_completed_event = Completed(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=None,  # Run events are top-level
                component_type=ComponentType.RUN,
                output_data=result,
            )
            logger.debug(
                f"[_execute_workflow] Emitting run.completed event: "
                f"component={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

            # Return None - the event queue handles delivery
            return None

        except WaitingForUserInputException:
            # Workflow paused for user input.
            # The workflow.paused event was already emitted via ctx.emit()
            # and flows through WriteCheckpoint to the Engine, which writes
            # run.paused. The coordinator's journal consumer watches for
            # run.paused and decrements active_invocations.
            logger.info("Workflow paused waiting for user input")
            return None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            stack_trace = ''.join(_traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Workflow execution failed: {error_msg}", exc_info=True)

            # If we have a context, emit events through the queue
            if ctx is not None:
                end_time_ns = time.time_ns()
                workflow_duration_ms = (end_time_ns - start_time_ns) // 1_000_000

                # Create short run correlation id (matches pattern of other events)
                outer_run_correlation_id = ctx.run_id[:8]

                # Emit run.failed via event queue
                run_failed_event = Failed(
                    name=config.name,
                    correlation_id=outer_run_correlation_id,
                    parent_correlation_id=None,  # Run events are top-level
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                ctx.emit(run_failed_event)

                outer_trace_id = _trace_id_from_request(request)
                logger.info(
                    f"run.failed | run_id={request.invocation_id} component={config.name} "
                    f"type=workflow trace_id={outer_trace_id} duration_ms={workflow_duration_ms} error={error_msg}"
                )
                return None

            # Fallback: if no context, return synchronous error response
            metadata = {
                "error_type": type(e).__name__,
                "stack_trace": stack_trace,
                "error": "true",
            }

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=metadata,
                event_type="run.failed",
                content_index=0,
                sequence=0,
                attempt=getattr(request, 'attempt', 0),
            )

        finally:
            _reset_current_span_token(span_token)
            if state_adapter_token is not None:
                _entity_state_adapter_ctx.reset(state_adapter_token)
            if token is not None:
                from ..context import _current_context
                _current_context.reset(token)
