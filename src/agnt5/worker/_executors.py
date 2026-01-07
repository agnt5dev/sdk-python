"""Executor mixin for component execution.

Supports functions, entities, workflows, agents, and tools.
"""

from __future__ import annotations, print_function

import asyncio
import inspect
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .._serialization import deserialize, serialize
from .._telemetry import setup_module_logger
from ._utils import create_failed_response, format_error_message

if TYPE_CHECKING:
    from .._core import PyExecuteComponentResponse

logger = setup_module_logger(__name__)


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
        from ..context import _current_context, get_current_context, set_current_context

        token = None
        try:
            input_dict = deserialize(request.input_data) if request.input_data else {}
            ctx = context_factory(input_dict, request)
            token = set_current_context(ctx)

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
                    correlation_id=getattr(current_ctx, "correlation_id", f"err-{request.invocation_id}"),
                    parent_correlation_id=getattr(
                        current_ctx, "parent_correlation_id", f"run-{request.invocation_id}"
                    ),
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.info(
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
        from ..tracing import SpanInfo, _current_span

        logger.debug(
            f"[_execute_function] Starting execution for component={config.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> FunctionContext:
            correlation_id = f"fn-{uuid.uuid4().hex[:12]}"
            return FunctionContext(
                run_id=req.invocation_id,  # Use actual invocation_id for event routing
                correlation_id=correlation_id,
                parent_correlation_id=f"run-{req.invocation_id}",
                attempt=getattr(req, "attempt", 0),
                runtime_context=req.runtime_context,
                retry_policy=config.retries,
                worker=self._rust_worker,
            )

        async def execute(ctx: FunctionContext, input_dict: dict, req: Any):
            # Set up trace parent-child linking
            if req.runtime_context:
                trace_id = req.runtime_context.trace_id
                span_id = req.runtime_context.span_id
                if trace_id and span_id:
                    _current_span.set(SpanInfo(trace_id=trace_id, span_id=span_id))

            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = f"run-{ctx.run_id[:8]}"

            # Emit run.started before executing handler
            run_started_event = Started(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.info(
                f"[_execute_function] Emitting run.started event: "
                f"component={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            # Emit function.started (child of run)
            start_time_ns = time.time_ns()
            fn_correlation_id = f"fn-{uuid.uuid4().hex[:8]}"
            fn_started_event = Started(
                name=config.name,
                correlation_id=fn_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.FUNCTION,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.info(
                f"[_execute_function] Emitting function.started event: "
                f"component={config.name}, correlation_id={fn_correlation_id}"
            )
            ctx.emit(fn_started_event)

            # Execute function with error handling for proper event emission
            try:
                result = config.handler(ctx, **input_dict) if input_dict else config.handler(ctx)

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

            except Exception as e:
                # Calculate function duration even on failure
                end_time_ns = time.time_ns()
                duration_ms = (end_time_ns - start_time_ns) // 1_000_000
                error_msg = f"{type(e).__name__}: {str(e)}"

                # Emit function.failed (child of run)
                from ..events import Failed
                fn_failed_event = Failed(
                    name=config.name,
                    correlation_id=fn_correlation_id,
                    parent_correlation_id=run_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                    duration_ms=duration_ms,
                )
                logger.info(
                    f"[_execute_function] Emitting function.failed event: "
                    f"component={config.name}, error={error_msg}"
                )
                ctx.emit(fn_failed_event)

                # Emit run.failed (parent event)
                run_failed_event = Failed(
                    name=config.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id=ctx.parent_correlation_id,
                    component_type=ComponentType.RUN,
                    error_code=type(e).__name__,
                    error_message=error_msg,
                )
                logger.info(
                    f"[_execute_function] Emitting run.failed event: "
                    f"component={config.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

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
            logger.info(
                f"[_execute_function] Emitting function.completed event: "
                f"component={config.name}, duration_ms={duration_ms}"
            )
            ctx.emit(fn_completed_event)

            # Emit run.completed via event queue (not synchronous return)
            # This ensures proper event ordering: started -> completed
            run_completed_event = Completed(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                output_data=result,
            )
            logger.info(
                f"[_execute_function] Emitting run.completed event: "
                f"component={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

            # Return None - the event queue handles delivery
            return None

        return await self._execute_with_context(request, create_context, execute, "Function")

    async def _handle_streaming_function(self, ctx: Any, result: Any) -> None:
        """Handle streaming function by queueing deltas."""
        from ..events import Event

        sequence = 0
        has_typed_events = False
        first_chunk = True

        print("=-----------------")
        print("Run Started..")

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
            correlation_id = f"tool-{uuid.uuid4().hex[:12]}"
            return Context(
                run_id=req.invocation_id,
                correlation_id=correlation_id,
                parent_correlation_id=f"run-{req.invocation_id}",
                attempt=getattr(req, "attempt", 0),
                runtime_context=req.runtime_context,
                worker=self._rust_worker,
            )

        async def execute(ctx: Context, input_dict: dict, req: Any):
            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = f"run-{ctx.run_id[:8]}"

            # Emit run.started before executing handler
            run_started_event = Started(
                name=tool.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.info(
                f"[_execute_tool] Emitting run.started event: "
                f"tool={tool.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            # Emit tool.started (child of run)
            start_time_ns = time.time_ns()
            tool_correlation_id = f"tool-{uuid.uuid4().hex[:8]}"
            tool_started_event = Started(
                name=tool.name,
                correlation_id=tool_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.TOOL,
                input_data=input_dict,
                attempt=ctx.attempt,
            )
            logger.info(
                f"[_execute_tool] Emitting tool.started event: "
                f"tool={tool.name}, correlation_id={tool_correlation_id}"
            )
            ctx.emit(tool_started_event)

            # Execute tool with error handling for proper event emission
            try:
                result = await tool.invoke(ctx, **input_dict)

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
                logger.info(
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
                logger.info(
                    f"[_execute_tool] Emitting run.failed event: "
                    f"tool={tool.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

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
            logger.info(
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
            logger.info(
                f"[_execute_tool] Emitting run.completed event: "
                f"tool={tool.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

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
        from ..entity import _entity_state_adapter_ctx
        from ..events import Completed, ComponentType, Failed, Started

        _entity_state_adapter_ctx.set(self._entity_state_adapter)

        logger.debug(
            f"[_execute_entity] Starting execution for entity={entity_type.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> Context:
            entity_key = input_dict.get("key", "unknown")
            correlation_id = f"ent-{uuid.uuid4().hex[:12]}"
            return Context(
                run_id=req.invocation_id,
                correlation_id=correlation_id,
                parent_correlation_id=f"run-{req.invocation_id}",
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
            run_correlation_id = f"run-{ctx.run_id[:8]}"

            # Emit run.started before executing entity method
            run_started_event = Started(
                name=entity_type.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data={"key": entity_key, "method": method_name, **input_dict},
                attempt=ctx.attempt,
            )
            logger.info(
                f"[_execute_entity] Emitting run.started event: "
                f"entity={entity_type.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            # Emit entity.started (child of run)
            start_time_ns = time.time_ns()
            entity_correlation_id = f"ent-{uuid.uuid4().hex[:8]}"
            entity_started_event = Started(
                name=entity_type.name,
                correlation_id=entity_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.ENTITY,
                input_data={"key": entity_key, "method": method_name, **input_dict},
            )
            logger.info(
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
                result = await method(**input_dict)

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
                logger.info(
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
                logger.info(
                    f"[_execute_entity] Emitting run.failed event: "
                    f"entity={entity_type.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

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
            logger.info(
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
            logger.info(
                f"[_execute_entity] Emitting run.completed event: "
                f"entity={entity_type.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

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
        from ..entity import _entity_state_adapter_ctx
        from ..events import Completed, ComponentType, Event, Failed, Started

        _entity_state_adapter_ctx.set(self._entity_state_adapter)

        logger.debug(
            f"[_execute_agent] Starting execution for agent={agent.name}, "
            f"invocation_id={getattr(request, 'invocation_id', 'unknown')}"
        )

        def create_context(input_dict: dict, req: Any) -> AgentContext:
            session_id = input_dict.get("session_id") or str(uuid.uuid4())

            if not input_dict.get("session_id"):
                logger.info(f"Created new agent session: {session_id}")
            else:
                logger.info(f"Using existing agent session: {session_id}")

            correlation_id = f"agent-{uuid.uuid4().hex[:12]}"
            return AgentContext(
                run_id=req.invocation_id,
                agent_name=agent.name,
                session_id=session_id,
                runtime_context=req.runtime_context,
                is_streaming=getattr(req, "is_streaming", False),
                worker=self._rust_worker,
                correlation_id=correlation_id,
                parent_correlation_id=f"run-{req.invocation_id}",
            )

        async def execute(ctx: AgentContext, input_dict: dict, req: Any):
            from .._core import PyExecuteComponentResponse

            user_message = input_dict.get("message", "")
            if not user_message:
                raise ValueError("Agent invocation requires 'message' parameter")

            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = f"run-{ctx.run_id[:8]}"

            # Emit run.started before executing agent
            run_started_event = Started(
                name=agent.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=ctx.parent_correlation_id,
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=getattr(req, "attempt", 0),
            )
            logger.info(
                f"[_execute_agent] Emitting run.started event: "
                f"agent={agent.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            # Emit agent.started (child of run)
            start_time_ns = time.time_ns()
            agent_correlation_id = f"agent-{uuid.uuid4().hex[:8]}"
            agent_started_event = Started(
                name=agent.name,
                correlation_id=agent_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.AGENT,
                input_data={"message": user_message},
            )
            logger.info(
                f"[_execute_agent] Emitting agent.started event: "
                f"agent={agent.name}, correlation_id={agent_correlation_id}"
            )
            ctx.emit(agent_started_event)

            try:
                result = agent.run(user_message, context=ctx)

                if inspect.isasyncgen(result):
                    sequence = 0
                    final_output = None
                    final_tool_calls = []
                    handoff_to = None

                    async for event in result:
                        if isinstance(event, Event):
                            # Forward the event to the context
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
                logger.info(
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
                logger.info(
                    f"[_execute_agent] Emitting run.completed event: "
                    f"agent={agent.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_completed_event)

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
                logger.info(
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
                logger.info(
                    f"[_execute_agent] Emitting run.failed event: "
                    f"agent={agent.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                # Return None - the event queue handles delivery
                return None

        return await self._execute_with_context(request, create_context, execute, "Agent")

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
        import uuid as _uuid

        from .._core import PyExecuteComponentResponse
        from ..context import set_current_context
        from ..entity import _entity_state_adapter_ctx, _get_state_adapter
        from ..events import Completed, ComponentType, Failed, Started
        from ..exceptions import WaitingForUserInputException
        from ..tracing import SpanInfo, _current_span
        from ..workflow import WorkflowContext, WorkflowEntity, WorkflowState

        # Set entity state adapter in context so workflows can use Entities
        _entity_state_adapter_ctx.set(self._entity_state_adapter)

        # Variables that need to be accessible in exception handlers
        ctx = None
        token = None
        session_id = None
        workflow_start_time = _time.time()
        start_time_ns = time.time_ns()

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Extract or generate session_id for multi-turn conversation support
            session_id = input_dict.get("session_id")
            if not session_id:
                session_id = str(_uuid.uuid4())
                logger.info(f"Created new workflow session: {session_id}")
            else:
                logger.info(f"Using existing workflow session: {session_id}")

            # Parse replay data from request metadata for crash recovery
            completed_steps = {}
            step_events = []
            initial_state = {}
            user_response = None

            if hasattr(request, 'metadata') and request.metadata:
                # Parse completed steps for replay
                if "completed_steps" in request.metadata:
                    completed_steps_json = request.metadata["completed_steps"]
                    if completed_steps_json:
                        try:
                            completed_steps = json.loads(completed_steps_json)
                            logger.info(f"Replaying workflow with {len(completed_steps)} cached steps")
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
                            logger.info(f"Resuming workflow with {len(completed_steps)} completed steps")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse step_events from metadata")

                # Parse initial workflow state
                if "workflow_state" in request.metadata:
                    workflow_state_json = request.metadata["workflow_state"]
                    if workflow_state_json:
                        try:
                            initial_state = json.loads(workflow_state_json)
                            logger.info(f"Loaded workflow state: {len(initial_state)} keys")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse workflow_state from metadata")

                # Check for user response (resume after pause)
                if "user_response" in request.metadata:
                    user_response = request.metadata["user_response"]
                    logger.info(f"Resuming workflow with user response: {user_response}")

            # Extract session_id and user_id for memory scoping
            session_id = request.session_id if hasattr(request, 'session_id') and request.session_id else request.invocation_id
            user_id = request.user_id if hasattr(request, 'user_id') and request.user_id else None
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

            if initial_state:
                state_adapter = _get_state_adapter()
                if hasattr(state_adapter, '_standalone_states'):
                    state_adapter._standalone_states[workflow_entity._state_key] = initial_state
                workflow_entity._state = WorkflowState(initial_state.copy(), workflow_entity)
                logger.info(f"Initialized workflow entity state with {len(initial_state)} keys")

            # Create WorkflowContext
            ctx = WorkflowContext(
                workflow_entity=workflow_entity,
                run_id=request.invocation_id,
                session_id=session_id,
                user_id=user_id,
                runtime_context=request.runtime_context,
                is_streaming=is_streaming,
                worker=self._rust_worker,
            )

            # Set context in contextvar
            token = set_current_context(ctx)

            workflow_correlation_id = f"wf-{_uuid.uuid4().hex[:8]}"

            # Create short run correlation id (matches pattern of other events)
            run_correlation_id = f"run-{ctx.run_id[:8]}"

            # Setup context fields for all workflow events
            ctx._correlation_id = workflow_correlation_id
            ctx._parent_correlation_id = run_correlation_id
            ctx._component_name = config.name
            ctx._workflow_name = config.name
            ctx._is_replay = bool(completed_steps)

            # Set up trace parent-child linking
            if request.runtime_context:
                trace_id = request.runtime_context.trace_id
                span_id = request.runtime_context.span_id
                if trace_id and span_id:
                    _current_span.set(SpanInfo(trace_id=trace_id, span_id=span_id))

            # Emit run.started event (like function executor does)
            run_started_event = Started(
                name=config.name,
                correlation_id=run_correlation_id,
                parent_correlation_id=None,  # Run events are top-level
                component_type=ComponentType.RUN,
                input_data=input_dict,
                attempt=getattr(request, 'attempt', 0),
            )
            logger.info(
                f"[_execute_workflow] Emitting run.started event: "
                f"component={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_started_event)

            # Emit workflow.started event (child of run)
            workflow_started_event = Started(
                name=config.name,
                correlation_id=workflow_correlation_id,
                parent_correlation_id=run_correlation_id,
                component_type=ComponentType.WORKFLOW,
                input_data=input_dict,
                attempt=getattr(request, 'attempt', 0),
            )
            logger.info(
                f"[_execute_workflow] Emitting workflow.started event: "
                f"component={config.name}, correlation_id={workflow_correlation_id}"
            )
            ctx.emit(workflow_started_event)

            # Execute workflow
            try:
                with ctx.as_parent():
                    if input_dict:
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
                logger.info(
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
                logger.info(
                    f"[_execute_workflow] Emitting run.failed event: "
                    f"component={config.name}, correlation_id={run_correlation_id}"
                )
                ctx.emit(run_failed_event)

                # Return None - the event queue handles delivery
                return None

            # Calculate workflow duration
            end_time_ns = time.time_ns()
            workflow_duration_ms = (end_time_ns - start_time_ns) // 1_000_000

            logger.info(f"Workflow completed in {workflow_duration_ms}ms")

            # Persist workflow entity state
            if hasattr(ctx, '_workflow_entity') and ctx._workflow_entity._state is not None:
                if ctx._workflow_entity._state.has_changes():
                    try:
                        await ctx._workflow_entity._persist_state()
                        logger.info(f"Persisted WorkflowEntity state for run {request.invocation_id}")
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
            logger.info(
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
            logger.info(
                f"[_execute_workflow] Emitting run.completed event: "
                f"component={config.name}, correlation_id={run_correlation_id}"
            )
            ctx.emit(run_completed_event)

            # Return None - the event queue handles delivery
            return None

        except WaitingForUserInputException as e:
            # Workflow paused for user input
            logger.info(f"Workflow paused waiting for user input: {e.question}")

            pause_metadata = {
                "status": "awaiting_user_input",
                "question": e.question,
                "input_type": e.input_type,
                "pause_index": str(e.pause_index),
            }

            if e.options:
                pause_metadata["options"] = json.dumps(e.options)
            if e.checkpoint_state:
                pause_metadata["checkpoint_state"] = json.dumps(e.checkpoint_state)
            if session_id:
                pause_metadata["session_id"] = session_id

            # Add step events to pause metadata
            if ctx is not None:
                step_events = ctx._workflow_entity._step_events
                if step_events:
                    pause_metadata["step_events"] = json.dumps(step_events)

                # Add current workflow state
                if hasattr(ctx, '_workflow_entity') and ctx._workflow_entity._state is not None:
                    if ctx._workflow_entity._state.has_changes():
                        state_snapshot = ctx._workflow_entity._state.get_state_snapshot()
                        pause_metadata["workflow_state"] = json.dumps(state_snapshot)

            output = {
                "question": e.question,
                "input_type": e.input_type,
                "options": e.options,
            }
            output_data = serialize(output)

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,
                error_message=None,
                metadata=pause_metadata,
                event_type="run.paused",
                content_index=0,
                sequence=0,
                attempt=getattr(request, 'attempt', 0),
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            stack_trace = ''.join(_traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"Workflow execution failed: {error_msg}", exc_info=True)

            # If we have a context, emit events through the queue
            if ctx is not None:
                end_time_ns = time.time_ns()
                workflow_duration_ms = (end_time_ns - start_time_ns) // 1_000_000

                # Create short run correlation id (matches pattern of other events)
                outer_run_correlation_id = f"run-{ctx.run_id[:8]}"

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
            if token is not None:
                from ..context import _current_context
                _current_context.reset(token)
