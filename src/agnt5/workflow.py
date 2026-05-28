"""Workflow component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import secrets
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, TypeVar, Union, cast

from ._ids import generate_cid
from ._schema_utils import extract_function_metadata, extract_function_schemas
from ._serialization import serialize_to_str
from ._state_adapter import StateInterface, _get_state_adapter
from .context import Context, set_current_context
from .function import FunctionContext
from .types import HandlerFunc, TriggerSpec, WorkflowConfig
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

T = TypeVar("T")

# Global workflow registry
_WORKFLOW_REGISTRY: Dict[str, WorkflowConfig] = {}


class WorkflowContext(Context):
    """
    Context for durable workflows.

    Extends base Context with:
    - State management via WorkflowEntity.state
    - Step tracking and replay
    - Orchestration (task, parallel, gather)
    - Checkpointing (step)
    - Memory scoping (session_id, user_id for multi-level memory)

    WorkflowContext delegates state to the underlying WorkflowEntity,
    which provides durability and state change tracking for AI workflows.

    Memory Scoping:
    - run_id: Unique workflow run identifier
    - session_id: For multi-turn conversations (optional)
    - user_id: For user-scoped long-term memory (optional)
    These identifiers enable agents to automatically select the appropriate
    memory scope (run/session/user) via context propagation.
    """

    def __init__(
        self,
        workflow_entity: "WorkflowEntity",  # Forward reference
        run_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        checkpoint_client: Optional[Any] = None,
        is_streaming: bool = False,
        worker: Optional[Any] = None,
        correlation_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
        trace_metadata: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Initialize workflow context.

        Args:
            workflow_entity: WorkflowEntity instance managing workflow state
            run_id: Unique workflow run identifier
            session_id: Session identifier for multi-turn conversations (default: run_id)
            user_id: User identifier for user-scoped memory (optional)
            attempt: Retry attempt number (0-indexed)
            runtime_context: RuntimeContext for trace correlation
            checkpoint_client: Optional CheckpointClient for platform-side memoization
            is_streaming: Whether this is a streaming request (for real-time SSE log delivery)
            worker: PyWorker instance for event queueing
            correlation_id: Unique identifier for this workflow execution
            parent_correlation_id: Parent's correlation ID for event hierarchy
        """
        import uuid
        super().__init__(
            run_id=run_id,
            correlation_id=correlation_id or generate_cid(),
            parent_correlation_id=parent_correlation_id or "",
            attempt=attempt,
            runtime_context=runtime_context,
            session_id=session_id,
            is_streaming=is_streaming,
            worker=worker,
            trace_metadata=trace_metadata,
        )
        self._is_streaming = is_streaming
        self._workflow_entity = workflow_entity
        self._step_counter: int = 0  # Track step sequence
        self._sequence_number: int = 0  # Global sequence for checkpoints
        self._checkpoint_client = checkpoint_client
        self._delta_sequence: int = 0  # Sequence for delta events (separate from checkpoint sequence)

        # Memory scoping identifiers (use private attrs since properties are read-only)
        self._session_id = session_id or run_id  # Default: session = run (ephemeral)
        self._user_id = user_id  # Optional: user-scoped memory

        # Step hierarchy tracking - for nested step visualization
        # Stack of event IDs for currently executing steps
        self._step_event_stack: List[str] = []

        # Workflow-specific metadata for events (set by worker during execution)
        self._workflow_name: Optional[str] = None
        self._is_replay: bool = False

    def get_event_metadata(self) -> Dict[str, str]:
        """Get workflow-specific metadata for events.

        Extends base Context metadata with:
        - workflow_name: Name of the workflow being executed
        - session_id: Session identifier for multi-turn conversations
        - is_replay: Whether this is a replay from cached steps
        """
        meta = super().get_event_metadata()
        if self._workflow_name:
            meta["workflow_name"] = self._workflow_name
        if self._session_id:
            meta["session_id"] = self._session_id
        meta["is_replay"] = str(self._is_replay).lower()
        return meta

    # === State Management ===

    def _forward_delta(self, event_type: str, output_data: str, content_index: int = 0, source_timestamp_ns: int = 0) -> None:
        """
        Forward a streaming delta event from a nested component.

        Used by step executors to forward events from streaming agents/functions
        to the client via the unified event queue.

        Args:
            event_type: Event type (e.g., "agent.started", "lm.message.delta")
            output_data: JSON-serialized event data
            content_index: Content index for parallel events (default: 0)
            source_timestamp_ns: Nanosecond timestamp when event was created (default: 0, will be generated if not provided)
        """
        import json
        from .events import ComponentType, Delta, OperationType

        try:
            # Parse output_data if it's a JSON string
            data = json.loads(output_data) if isinstance(output_data, str) else output_data
        except json.JSONDecodeError:
            data = {"raw": output_data}

        # Create Delta event for forwarding
        delta_event = Delta(
            name=self._workflow_name or "workflow",
            correlation_id=self._correlation_id,
            parent_correlation_id=self._parent_correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.OUTPUT,
            content=data,
            index=content_index,
        )
        # Preserve original event_type from nested component
        object.__setattr__(delta_event, "event_type", event_type)
        self.emit(delta_event)
        self._delta_sequence += 1

    async def _consume_streaming_result(self, async_gen: Any, step_name: str) -> Any:
        """
        Consume an async generator while forwarding streaming events to the client.

        This method handles streaming from nested agents and functions within
        workflow steps. Events are forwarded via the delta queue while the
        final result is collected and returned for the next step.

        For agents, the final output is extracted from the agent.completed event.
        For functions, the last yielded value (or collected output) is returned.

        Args:
            async_gen: Async generator yielding Event objects or raw values
            step_name: Name of the current step (for logging)

        Returns:
            The final result to pass to the next step:
            - For agents: The output from agent.completed event
            - For functions: The last yielded value or collected output
        """
        import json
        from .events import Event

        final_result = None
        collected_output = []  # For streaming functions that yield chunks

        async for item in async_gen:
            if isinstance(item, Event):
                # Forward typed Event via delta queue
                event_data = item.to_response_fields()
                output_data = event_data.get("output_data", b"")
                output_str = output_data.decode("utf-8") if isinstance(output_data, bytes) else str(output_data or "{}")

                self._forward_delta(
                    event_type=event_data.get("event_type", ""),
                    output_data=output_str,
                    content_index=event_data.get("content_index", 0),
                    source_timestamp_ns=item.source_timestamp_ns,
                )

                # Capture final result from specific event types
                if item.event_type == "agent.completed":
                    # For agents, extract the output from completed event
                    output_data_dict = getattr(item, 'output_data', {}) or {}
                    final_result = output_data_dict.get("output", "")
                    logger.debug(f"Step '{step_name}': Captured agent output from agent.completed")
                elif item.event_type == "output.stop":
                    # For streaming functions, the collected output is the result
                    # (already collected from delta events)
                    pass

            else:
                # Raw value (non-Event) - streaming function output
                # Forward as output.delta and collect for final result
                try:
                    chunk_json = serialize_to_str(item)
                except (TypeError, ValueError):
                    chunk_json = str(item)

                self._forward_delta(
                    event_type="output.delta",
                    output_data=chunk_json,
                    source_timestamp_ns=time.time_ns(),
                )
                collected_output.append(item)

        # Determine final result
        if final_result is not None:
            # Agent result was captured from agent.completed event
            return final_result
        elif collected_output:
            # Streaming function - return collected chunks
            # If single item, return it directly; otherwise return list
            if len(collected_output) == 1:
                return collected_output[0]
            return collected_output
        else:
            # Empty generator
            return None

    @property
    def state(self):
        """
        Delegate to WorkflowEntity.state for durable state management.

        Returns:
            WorkflowState instance from the workflow entity

        Example:
            ctx.state.set("status", "processing")
            status = ctx.state.get("status")
        """
        state = self._workflow_entity.state
        # Pass checkpoint callback to state for real-time streaming
        if hasattr(state, "_set_emitter"):
            state._set_emitter(self.emit)
        return state

    def _get_or_create_state_adapter(self):
        """Get the worker-bound state adapter, falling back to standalone mode."""
        if not hasattr(self, '_state_adapter'):
            try:
                self._state_adapter = _get_state_adapter()
            except RuntimeError:
                from ._state_adapter import StateAdapter
                self._state_adapter = StateAdapter()
        return self._state_adapter

    @property
    def session(self):
        """
        Get session context with state property.

        Session-scoped state persists across runs that share the same session_id.

        Example:
            await ctx.session.state.set("topic", "AI")
            topic = await ctx.session.state.get("topic")
        """
        from .state import SessionContext

        if not hasattr(self, '_session_context'):
            self._session_context = SessionContext(
                state_adapter=self._get_or_create_state_adapter(),
                session_id=self._session_id,
            )
        return self._session_context

    @property
    def user(self):
        """
        Get user context with state property.

        Returns None when this workflow was not started with a user_id.
        """
        if not self._user_id:
            return None

        from .state import UserContext

        if not hasattr(self, '_user_context'):
            self._user_context = UserContext(
                state_adapter=self._get_or_create_state_adapter(),
                user_id=self._user_id,
            )
        return self._user_context

    @property
    def memory(self):
        """
        Get unified memory accessor.

        Provides:
        - KV memory: ctx.memory.get/set/delete (session-scoped by default)
        - Scoped KV: ctx.memory.user(), ctx.memory.run(), ctx.memory.global_()
        - Working memory: ctx.memory.working
        - Semantic memory: ctx.memory.semantic (if configured at worker level)

        Returns:
            MemoryAccessor instance

        Example:
            await ctx.memory.set("theme", "dark")
            theme = await ctx.memory.get("theme", "light")
            wm = await ctx.memory.working.get()
        """
        from .memory import MemoryAccessor

        if not hasattr(self, '_memory_accessor'):
            # Get state adapter from worker context
            try:
                adapter = _get_state_adapter()
            except RuntimeError:
                from ._state_adapter import StateAdapter
                adapter = StateAdapter()

            self._memory_accessor = MemoryAccessor(
                state_adapter=adapter,
                session_id=self._session_id,
                user_id=self._user_id,
                run_id=self._run_id,
                semantic_provider=getattr(self, '_semantic_provider', None),
                tenant_id=(self._trace_metadata or {}).get("tenant_id"),
                deployment_id=(self._trace_metadata or {}).get("deployment_id"),
            )

        return self._memory_accessor

    @property
    def conversation(self):
        """
        Get conversation memory for this session.

        Provides message history management:
        - ctx.conversation.add(role, content) — record a message
        - ctx.conversation.get_messages(limit) — load history
        - ctx.conversation.get_as_lm_messages(limit) — history as LM Messages
        - ctx.conversation.clear() — wipe history

        Returns:
            ConversationAccessor instance

        Example:
            await ctx.conversation.add("user", user_input)
            history = await ctx.conversation.get_as_lm_messages(limit=20)
        """
        from .memory import ConversationAccessor

        if not hasattr(self, '_conversation_accessor'):
            try:
                adapter = _get_state_adapter()
            except RuntimeError:
                from ._state_adapter import StateAdapter
                adapter = StateAdapter()

            self._conversation_accessor = ConversationAccessor(
                state_adapter=adapter,
                session_id=self._session_id,
                component_name=self._workflow_entity.key if self._workflow_entity else "",
            )

        return self._conversation_accessor

    # === Orchestration ===

    async def step(
        self,
        name_or_handler: Union[str, Callable, Awaitable[T]],
        func_or_awaitable: Union[Callable[..., Awaitable[T]], Awaitable[T], Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a durable step with automatic checkpointing.

        Steps are the primary building block for durable workflows. Results are
        automatically persisted, so if the workflow crashes and restarts, completed
        steps return their cached result without re-executing.

        **Recommended Pattern** - Pass @function directly for clean, type-safe syntax:
        ```python
        result = await ctx.step(process_data, arg1, arg2, kwarg=value)
        # Or use ctx.run() alias:
        result = await ctx.run(process_data, arg1, arg2)
        ```

        Supports multiple calling patterns:

        1. **Call a @function (recommended - cleanest syntax)**:
           ```python
           result = await ctx.step(process_data, arg1, arg2, kwarg=value)
           ```
           Auto-generates step name from function. Full IDE support, type safety.

        2. **Checkpoint an awaitable with explicit name**:
           ```python
           result = await ctx.step("load_data", fetch_expensive_data())
           ```
           For arbitrary async operations that aren't @functions.

        3. **Checkpoint a callable with explicit name**:
           ```python
           result = await ctx.step("compute", my_function, arg1, arg2)
           ```

        4. **Legacy string-based @function call**:
           ```python
           result = await ctx.step("function_name", input=data)
           ```

        Args:
            name_or_handler: Step name (str), @function reference, or awaitable
            func_or_awaitable: Function/awaitable when name is provided, or first arg
            *args: Additional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The step result (cached on replay)

        Example (@function call):
            ```python
            @function
            async def process_data(ctx: FunctionContext, data: list, multiplier: int = 2):
                return [x * multiplier for x in data]

            @workflow
            async def my_workflow(ctx: WorkflowContext):
                result = await ctx.step(process_data, [1, 2, 3], multiplier=3)
                return result
            ```

        Example (checkpoint awaitable):
            ```python
            @workflow
            async def my_workflow(ctx: WorkflowContext):
                # Checkpoint expensive external call
                data = await ctx.step("fetch_api", fetch_from_external_api())
                return data
            ```
        """
        import inspect

        # Determine which calling pattern is being used
        if callable(name_or_handler) and hasattr(name_or_handler, "_agnt5_config"):
            # Pattern 1: step(handler, *args, **kwargs) - @function call
            return await self._step_function(name_or_handler, func_or_awaitable, *args, **kwargs)
        elif isinstance(name_or_handler, str):
            # Check if it's a registered function name (legacy pattern)
            from .function import FunctionRegistry
            if FunctionRegistry.get(name_or_handler) is not None:
                # Pattern 4: Legacy string-based function call
                return await self._step_function(name_or_handler, func_or_awaitable, *args, **kwargs)
            elif func_or_awaitable is not None:
                # Pattern 2/3: step("name", awaitable) or step("name", callable, *args)
                return await self._step_checkpoint(name_or_handler, func_or_awaitable, *args, **kwargs)
            else:
                # String without second arg and not a registered function
                raise ValueError(
                    f"Function '{name_or_handler}' not found in registry. "
                    f"Either register it with @function decorator, or use "
                    f"ctx.step('{name_or_handler}', awaitable) to checkpoint an arbitrary operation."
                )
        elif inspect.iscoroutine(name_or_handler) or inspect.isawaitable(name_or_handler):
            # Awaitable passed directly - auto-generate name
            coro_name = getattr(name_or_handler, '__name__', 'awaitable')
            return await self._step_checkpoint(coro_name, name_or_handler)
        elif callable(name_or_handler):
            # Callable without @function decorator
            raise ValueError(
                f"Function '{name_or_handler.__name__}' is not a registered @function. "
                f"Did you forget to add the @function decorator? "
                f"Or use ctx.step('name', callable) for non-decorated functions."
            )
        else:
            raise ValueError(
                f"step() first argument must be a @function, string name, or awaitable. "
                f"Got: {type(name_or_handler)}"
            )

    async def _step_function(
        self,
        handler: Union[str, Callable],
        first_arg: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Internal: Execute a @function as a durable step.

        This handles both function references and legacy string-based calls.
        """
        from .function import FunctionRegistry

        # Reconstruct args tuple (first_arg may have been split out by step())
        if first_arg is not None:
            args = (first_arg,) + args

        # Extract handler name from function reference or use string
        if callable(handler):
            if not hasattr(handler, "_agnt5_config"):
                raise ValueError(
                    f"Function '{handler.__name__}' is not a registered @function. "
                    f"Did you forget to add the @function decorator?"
                )
            # Use registered name (supports @function(name="custom"))
            handler_name = handler._agnt5_config.name
        else:
            handler_name = handler

        # Generate unique step name for durability
        step_name = f"{handler_name}_{self._step_counter}"
        self._step_counter += 1

        # Generate unique event_id for this step (for hierarchy tracking)
        step_event_id = str(uuid.uuid4())

        # Check if step already completed (for replay)
        if self._workflow_entity.has_completed_step(step_name):
            result = self._workflow_entity.get_completed_step(step_name)
            self._logger.info(f"🔄 Replaying cached step: {step_name}")
            # Emit a step.completed event with cache_hit metadata so tests can
            # prove replay happened (vs re-execution which emits started+completed)
            from .events import Completed, ComponentType, OperationType
            replay_event = Completed(
                name=step_name,
                correlation_id=generate_cid(),
                parent_correlation_id=self._correlation_id,
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                output_data=result,
                metadata={"cache_hit": "true"},
            )
            self.emit(replay_event)
            return result

        # Use step_event_id as correlation_id for pairing started ↔ completed
        step_correlation_id = generate_cid()

        # Emit workflow.step.started checkpoint
        from .events import ComponentType, OperationType, Started
        step_started = Started(
            name=step_name,
            correlation_id=step_correlation_id,
            parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.STEP,
            input_data={"step_name": step_name, "handler_name": handler_name, "input": args or kwargs},
            metadata={"name": step_name},
        )
        self.emit(step_started)

        # Push this step's event_id onto the stack for nested calls
        self._step_event_stack.append(step_event_id)

        # Execute function with OpenTelemetry span
        self._logger.info(f"▶️  Executing new step: {step_name}")
        func_config = FunctionRegistry.get(handler_name)
        if func_config is None:
            raise ValueError(f"Function '{handler_name}' not found in registry")

        # Import span creation utility (uses contextvar for async-safe parent-child linking)
        from .tracing import create_span

        # Serialize input data for span attributes
        input_repr = serialize_to_str({"args": args, "kwargs": kwargs}) if args or kwargs else "{}"

        # Create span for task execution (contextvar handles parent-child linking)
        with create_span(
            f"workflow.task.{handler_name}",
            "function",
            self._runtime_context,
            {
                "step_name": step_name,
                "handler_name": handler_name,
                "run_id": self.run_id,
                "input.data": input_repr,
            },
        ) as span:
            # Create FunctionContext for the function execution
            # Use step_correlation_id as parent so function events are nested under the step
            # The correlation_id uniquely identifies this function execution
            func_correlation_id = generate_cid()
            step_memo_namespace = self.allocate_memo_child_scope("step", step_name)
            func_ctx = FunctionContext(
                run_id=self.run_id,  # Pure run ID - correlation_id handles unique identification
                correlation_id=func_correlation_id,
                parent_correlation_id=step_correlation_id,
                runtime_context=self._runtime_context,
                worker=self._worker,
                trace_metadata=self._trace_metadata,
                memo_namespace=step_memo_namespace,
            )

            # Emit function.started event
            # Normalize input for event data
            if len(args) == 1 and isinstance(args[0], dict):
                func_input_data = args[0]
            elif kwargs.get("input") and isinstance(kwargs.get("input"), dict):
                func_input_data = kwargs["input"]
            elif kwargs:
                func_input_data = dict(kwargs)
            else:
                func_input_data = {"args": list(args)} if args else {}
            func_started = Started(
                name=handler_name,
                correlation_id=func_correlation_id,
                parent_correlation_id=step_correlation_id,
                component_type=ComponentType.FUNCTION,
                input_data=func_input_data,
                attempt=0,
            )
            self.emit(func_started)
            func_start_time = time.time_ns()

            context_token = set_current_context(func_ctx)
            try:
                try:
                    # Execute function with arguments
                    # Support legacy pattern: ctx.task("func_name", input=data) or ctx.task(func_ref, input=data)
                    if len(args) == 0 and "input" in kwargs:
                        # Legacy pattern - single input parameter
                        input_data = kwargs.pop("input")  # Remove from kwargs
                        handler_result = func_config.handler(func_ctx, input_data, **kwargs)
                    else:
                        # Type-safe pattern - pass all args/kwargs
                        handler_result = func_config.handler(func_ctx, *args, **kwargs)

                    # Check if result is an async generator (streaming function or agent)
                    # If so, consume it while forwarding events via delta queue
                    if inspect.isasyncgen(handler_result):
                        result = await self._consume_streaming_result(handler_result, step_name)
                    elif inspect.iscoroutine(handler_result):
                        result = await handler_result
                    else:
                        result = handler_result
                finally:
                    from .context import _current_context

                    _current_context.reset(context_token)

                # Add output data to span
                try:
                    output_repr = serialize_to_str(result)
                    span.set_attribute("output.data", output_repr)
                except (TypeError, ValueError):
                    # If result is not JSON serializable, use repr
                    span.set_attribute("output.data", repr(result))

                # Record step completion in WorkflowEntity
                self._workflow_entity.record_step_completion(
                    step_name, handler_name, args or kwargs, result
                )

                # Emit function.completed event (before workflow.step.completed)
                from .events import Completed
                func_duration_ms = (time.time_ns() - func_start_time) // 1_000_000
                func_completed = Completed(
                    name=handler_name,
                    correlation_id=func_correlation_id,
                    parent_correlation_id=step_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    output_data=result if isinstance(result, dict) else {"result": result},
                    duration_ms=func_duration_ms,
                )
                self.emit(func_completed)

                # Pop this step's event_id from the stack (execution complete)
                if self._step_event_stack:
                    popped_id = self._step_event_stack.pop()
                    if popped_id != step_event_id:
                        self._logger.warning(
                            f"Step event stack mismatch in task(): expected {step_event_id}, got {popped_id}"
                        )

                # Emit workflow.step.completed checkpoint
                step_completed = Completed(
                    name=step_name,
                    correlation_id=step_correlation_id,
                    parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
                    component_type=ComponentType.WORKFLOW,
                    operation=OperationType.STEP,
                    output_data={"step_name": step_name, "handler_name": handler_name, "result": result},
                    metadata={"name": step_name},
                )
                self.emit(step_completed)

                return result

            except Exception as e:
                # Emit function.failed event (before workflow.step.failed)
                from .events import Failed
                func_duration_ms = (time.time_ns() - func_start_time) // 1_000_000
                func_failed = Failed(
                    name=handler_name,
                    correlation_id=func_correlation_id,
                    parent_correlation_id=step_correlation_id,
                    component_type=ComponentType.FUNCTION,
                    error_code=type(e).__name__,
                    error_message=str(e),
                    duration_ms=func_duration_ms,
                )
                self.emit(func_failed)

                # Pop this step's event_id from the stack (execution failed)
                if self._step_event_stack:
                    popped_id = self._step_event_stack.pop()
                    if popped_id != step_event_id:
                        self._logger.warning(
                            f"Step event stack mismatch in task() error path: expected {step_event_id}, got {popped_id}"
                        )

                # Emit workflow.step.failed checkpoint
                step_failed = Failed(
                    name=step_name,
                    correlation_id=step_correlation_id,
                    parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
                    component_type=ComponentType.WORKFLOW,
                    operation=OperationType.STEP,
                    error_code=type(e).__name__,
                    error_message=str(e),
                    metadata={"name": step_name},
                )
                self.emit(step_failed)

                # Record error in span
                span.set_attribute("error", "true")
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)

                # Re-raise to propagate failure
                raise

    async def parallel(self, *tasks: Awaitable[T]) -> List[T]:
        """
        Run multiple tasks in parallel.

        Args:
            *tasks: Async tasks to run in parallel

        Returns:
            List of results in the same order as tasks

        Example:
            result1, result2 = await ctx.parallel(
                fetch_data(source1),
                fetch_data(source2)
            )
        """
        import asyncio

        return list(await asyncio.gather(*tasks))

    async def gather(self, **tasks: Awaitable[T]) -> Dict[str, T]:
        """
        Run tasks in parallel with named results.

        Args:
            **tasks: Named async tasks to run in parallel

        Returns:
            Dictionary mapping names to results

        Example:
            results = await ctx.gather(
                db=query_database(),
                api=fetch_api()
            )
        """
        import asyncio

        keys = list(tasks.keys())
        values = list(tasks.values())
        results = await asyncio.gather(*values)
        return dict(zip(keys, results))

    async def batch(
        self,
        func: Callable,
        items: List[Any],
        max_concurrency: int = 10,
        continue_on_failure: bool = True,
        timeout_per_item: Optional[float] = None,
    ) -> "BatchResult":
        """Execute a function across multiple inputs as a platform batch.

        This method sends all items to the AGNT5 platform for parallel
        execution with controlled concurrency. The platform handles:
        - Concurrent execution up to max_concurrency
        - Rate limiting and resource management
        - Individual item retries and error handling
        - Progress tracking and observability

        Args:
            func: The @function decorated callable to execute
            items: List of input items (each becomes a separate execution)
            max_concurrency: Maximum parallel executions (default: 10)
            continue_on_failure: Keep processing if items fail (default: True)
            timeout_per_item: Per-item timeout in seconds (optional)

        Returns:
            BatchResult containing all item results and statistics

        Raises:
            BatchError: If the batch execution fails entirely
            ValueError: If func is not a registered @function

        Example:
            ```python
            @function
            async def process_document(doc_id: str) -> dict:
                # Process a single document
                return {"processed": doc_id}

            @workflow
            async def batch_processor(ctx: WorkflowContext, doc_ids: list[str]):
                # Execute across all documents with controlled concurrency
                result = await ctx.batch(
                    process_document,
                    [{"doc_id": d} for d in doc_ids],
                    max_concurrency=20,
                )

                return {
                    "processed": result.stats.completed_items,
                    "failed": result.stats.failed_items,
                }
            ```
        """
        from .batch import BatchConfig, BatchResult, BatchError
        from .function import get_function_config

        # Get function name from registered function
        func_config = get_function_config(func)
        if func_config is None:
            raise ValueError(f"{func} is not a registered @function")

        function_name = func_config.name

        # Build batch items
        batch_items = []
        for i, item in enumerate(items):
            # If item is a dict, use it directly; otherwise wrap it
            if isinstance(item, dict):
                batch_items.append({"input": item, "index": i})
            else:
                batch_items.append({"input": {"value": item}, "index": i})

        # Build config
        config = BatchConfig(
            max_concurrency=max_concurrency,
            continue_on_failure=continue_on_failure,
            default_item_timeout_ms=int(timeout_per_item * 1000) if timeout_per_item else 30000,
        )

        # Get the runtime context to make the batch call
        runtime_ctx = self._runtime_context
        if runtime_ctx is None:
            raise RuntimeError("WorkflowContext not initialized with runtime context")

        # Call the platform's batch API via the runtime
        # For now, we use local parallel execution as a fallback
        # In production, this would call the gateway's batch endpoint
        try:
            result = await runtime_ctx.execute_batch(
                function_name=function_name,
                items=batch_items,
                config=config,
            )
            return result
        except AttributeError:
            # Fallback: local parallel execution if runtime doesn't support batch
            return await self._local_batch_fallback(
                func, items, max_concurrency, continue_on_failure, timeout_per_item
            )

    async def _local_batch_fallback(
        self,
        func: Callable,
        items: List[Any],
        max_concurrency: int,
        continue_on_failure: bool,
        timeout_per_item: Optional[float],
    ) -> "BatchResult":
        """Fallback implementation using local parallel execution.

        This is used when the runtime doesn't support native batch execution.
        """
        import asyncio
        from .batch import BatchResult, BatchItemResult, BatchStats, BatchError

        semaphore = asyncio.Semaphore(max_concurrency)
        results: List[BatchItemResult] = []
        start_time = time.time()

        async def execute_item(index: int, item: Any) -> BatchItemResult:
            async with semaphore:
                item_start = time.time()
                try:
                    # Execute the function
                    if isinstance(item, dict):
                        output = await func(**item)
                    else:
                        output = await func(item)

                    duration_ms = int((time.time() - item_start) * 1000)
                    return BatchItemResult(
                        index=index,
                        run_id=f"local-{index}",
                        status="completed",
                        output=output,
                        duration_ms=duration_ms,
                    )
                except Exception as e:
                    duration_ms = int((time.time() - item_start) * 1000)
                    from .batch import BatchItemError
                    return BatchItemResult(
                        index=index,
                        run_id=f"local-{index}",
                        status="failed",
                        error=BatchItemError(code="EXECUTION_ERROR", message=str(e)),
                        duration_ms=duration_ms,
                    )

        # Execute all items with timeout
        tasks = [execute_item(i, item) for i, item in enumerate(items)]

        if timeout_per_item:
            timeout = timeout_per_item * len(items) / max_concurrency + 60
        else:
            timeout = None

        try:
            if timeout:
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            else:
                results = await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise BatchError("Batch execution timed out")

        # Calculate stats
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        duration_ms = int((time.time() - start_time) * 1000)

        # Determine status
        if failed == 0:
            status = "completed"
        elif completed > 0:
            status = "partial_failure"
        else:
            status = "failed"

        return BatchResult(
            batch_id=f"local-batch-{self.run_id}",
            status=status,
            results=list(results),
            stats=BatchStats(
                total_items=len(items),
                completed_items=completed,
                failed_items=failed,
                cancelled_items=0,
                pending_items=0,
                duration_ms=duration_ms,
            ),
        )

    async def map(
        self,
        func: Callable,
        items: List[Any],
        max_concurrency: int = 10,
        **kwargs: Any,
    ) -> List[Any]:
        """Execute a function across items and return outputs.

        Convenience wrapper around batch() that returns just the outputs.
        Raises BatchError if any item fails.

        Args:
            func: The @function decorated callable to execute
            items: List of input items
            max_concurrency: Maximum parallel executions (default: 10)
            **kwargs: Additional arguments passed to batch()

        Returns:
            List of outputs in the same order as inputs

        Raises:
            BatchError: If any item fails

        Example:
            ```python
            @workflow
            async def process_all(ctx: WorkflowContext, ids: list[str]):
                # Process all IDs and get results
                results = await ctx.map(
                    process_item,
                    [{"id": i} for i in ids],
                    max_concurrency=10,
                )
                return {"results": results}
            ```
        """
        from .batch import BatchError

        result = await self.batch(
            func, items, max_concurrency=max_concurrency, continue_on_failure=False, **kwargs
        )

        if result.status == "failed":
            failed = result.failed_items()
            if failed:
                msg = f"Batch failed: {failed[0].error.message if failed[0].error else 'Unknown error'}"
            else:
                msg = "Batch failed"
            raise BatchError(msg, result)

        return result.outputs

    async def task(
        self,
        handler: Union[str, Callable],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function and wait for result.

        .. deprecated::
            Use :meth:`step` instead. ``task()`` will be removed in a future version.

        This method is an alias for :meth:`step` for backward compatibility.
        New code should use ``ctx.step()`` directly.

        Args:
            handler: Either a @function reference or string name
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function result
        """
        import warnings

        warnings.warn(
            "ctx.task() is deprecated, use ctx.step() instead. "
            "task() will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.step(handler, *args, **kwargs)

    async def run(
        self,
        handler: Union[str, Callable],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Simplified alias for ctx.step() that auto-generates step names.

        Matches Inngest/Restate API conventions for cleaner workflow code.
        This is the recommended method for calling @function decorated functions
        from workflows when you don't need custom step names.

        Args:
            handler: @function reference or string name
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function result

        Example (@function call with auto-naming):
            ```python
            @function
            async def process_data(ctx: FunctionContext, data: list) -> dict:
                return {"processed": [x * 2 for x in data]}

            @workflow
            async def my_workflow(ctx: WorkflowContext, data: list):
                # Clean syntax, auto-named steps
                result = await ctx.run(process_data, data)
                return result
            ```

        Example (multiple steps):
            ```python
            @workflow
            async def etl_workflow(ctx: WorkflowContext, dataset: str):
                extracted = await ctx.run(extract_data, dataset)
                transformed = await ctx.run(transform_data, extracted)
                loaded = await ctx.run(load_data, transformed, "warehouse")
                return loaded
            ```

        Note:
            This is an alias for ``ctx.step(handler, *args, **kwargs)`` which
            auto-generates step names from function names. Use ``ctx.step("custom_name", ...)``
            when you need explicit control over step naming.
        """
        return await self.step(handler, *args, **kwargs)

    async def _step_checkpoint(
        self,
        name: str,
        func_or_awaitable: Union[Callable[..., Awaitable[T]], Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Internal: Checkpoint an arbitrary awaitable or callable for durability.

        If workflow crashes, won't re-execute this step on retry.
        The step result is persisted to the platform for crash recovery.

        When a CheckpointClient is available, this method uses platform-side
        memoization via gRPC. The platform stores step results in the run_steps
        table, enabling replay even after worker crashes.

        Args:
            name: Unique name for this checkpoint (used as step_key for memoization)
            func_or_awaitable: Either an async function or awaitable
            *args: Arguments to pass if func_or_awaitable is callable
            **kwargs: Keyword arguments to pass if func_or_awaitable is callable

        Returns:
            The result of the function/awaitable
        """
        import inspect
        import json
        import time

        # Generate step key for platform memoization
        step_key = f"step:{name}:{self._step_counter}"
        self._step_counter += 1

        # Generate unique event_id for this step (for hierarchy tracking)
        step_event_id = str(uuid.uuid4())
        # Use step_event_id as correlation_id for pairing started ↔ completed
        step_correlation_id = generate_cid()

        # Check platform-side memoization first (Phase 3)
        if self._checkpoint_client:
            try:
                # Engine cache key is still (tenant_id, run_id). On worker
                # paths that tenant_id is a legacy alias for project identity,
                # so prefer project_id when present and fall back to tenant_id.
                project_or_tenant_id = (
                    (self._trace_metadata or {}).get("project_id", "")
                    or (self._trace_metadata or {}).get("tenant_id", "")
                )
                result = await self._checkpoint_client.step_started(
                    project_or_tenant_id,
                    self.run_id,
                    step_key,
                    name,
                    "checkpoint",
                )
                if result.memoized and result.cached_output:
                    # Deserialize cached output
                    cached_value = json.loads(result.cached_output.decode("utf-8"))
                    self._logger.info(f"🔄 Replaying memoized step from platform: {name}")
                    # Also record locally for consistency
                    self._workflow_entity.record_step_completion(name, "checkpoint", None, cached_value)
                    return cached_value
            except Exception as e:
                self._logger.warning(f"Platform memoization check failed, falling back to local: {e}")

        # Fall back to local memoization (for backward compatibility)
        if self._workflow_entity.has_completed_step(name):
            result = self._workflow_entity.get_completed_step(name)
            self._logger.info(f"🔄 Replaying checkpoint from local cache: {name}")
            return result

        # Emit workflow.step.started checkpoint for observability
        from .events import ComponentType, OperationType, Started
        step_started = Started(
            name=name,
            correlation_id=step_correlation_id,
            parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.STEP,
            input_data={"step_name": name, "handler_name": "checkpoint"},
            metadata={"name": name},
        )
        self.emit(step_started)

        # Push this step's event_id onto the stack for nested calls
        self._step_event_stack.append(step_event_id)

        start_time = time.time()
        try:
            # Execute and checkpoint
            with self.memo_child_scope("step", step_key):
                if inspect.isasyncgen(func_or_awaitable):
                    # Direct async generator - consume while forwarding events
                    result = await self._consume_streaming_result(func_or_awaitable, name)
                elif inspect.iscoroutine(func_or_awaitable) or inspect.isawaitable(func_or_awaitable):
                    result = await func_or_awaitable
                elif callable(func_or_awaitable):
                    # Call with args/kwargs if provided
                    call_result = func_or_awaitable(*args, **kwargs)
                    if inspect.isasyncgen(call_result):
                        # Callable returned async generator - consume while forwarding events
                        result = await self._consume_streaming_result(call_result, name)
                    elif inspect.iscoroutine(call_result) or inspect.isawaitable(call_result):
                        result = await call_result
                    else:
                        result = call_result
                else:
                    raise ValueError(f"step() second argument must be awaitable or callable, got {type(func_or_awaitable)}")

            latency_ms = int((time.time() - start_time) * 1000)

            # Record step completion locally for in-memory replay
            self._workflow_entity.record_step_completion(name, "checkpoint", None, result)

            # Record to platform for persistent memoization (Phase 3)
            if self._checkpoint_client:
                try:
                    output_bytes = serialize_to_str(result).encode("utf-8")
                    project_or_tenant_id = (
                        (self._trace_metadata or {}).get("project_id", "")
                        or (self._trace_metadata or {}).get("tenant_id", "")
                    )
                    await self._checkpoint_client.step_completed(
                        project_or_tenant_id,
                        self.run_id,
                        step_key,
                        name,
                        "checkpoint",
                        output_bytes,
                        latency_ms,
                    )
                except Exception as e:
                    self._logger.warning(f"Failed to record step completion to platform: {e}")

            # Pop this step's event_id from the stack (execution complete)
            if self._step_event_stack:
                popped_id = self._step_event_stack.pop()
                if popped_id != step_event_id:
                    self._logger.warning(
                        f"Step event stack mismatch in step(): expected {step_event_id}, got {popped_id}"
                    )

            # Emit workflow.step.completed checkpoint to journal for crash recovery
            from .events import Completed
            step_completed = Completed(
                name=name,
                correlation_id=step_correlation_id,
                parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                output_data={"step_name": name, "handler_name": "checkpoint", "result": result},
                duration_ms=latency_ms,
                metadata={"name": name},
            )
            self.emit(step_completed)

            self._logger.info(f"✅ Checkpoint completed: {name} ({latency_ms}ms)")
            return result

        except Exception as e:
            # Pop this step's event_id from the stack (execution failed)
            if self._step_event_stack:
                popped_id = self._step_event_stack.pop()
                if popped_id != step_event_id:
                    self._logger.warning(
                        f"Step event stack mismatch in step() error path: expected {step_event_id}, got {popped_id}"
                    )

            # Record failure to platform (Phase 3)
            if self._checkpoint_client:
                try:
                    project_or_tenant_id = (
                        (self._trace_metadata or {}).get("project_id", "")
                        or (self._trace_metadata or {}).get("tenant_id", "")
                    )
                    await self._checkpoint_client.step_failed(
                        project_or_tenant_id,
                        self.run_id,
                        step_key,
                        name,
                        "checkpoint",
                        str(e),
                        type(e).__name__,
                    )
                except Exception as cp_err:
                    self._logger.warning(f"Failed to record step failure to platform: {cp_err}")

            # Emit workflow.step.failed checkpoint
            from .events import Failed
            step_failed = Failed(
                name=name,
                correlation_id=step_correlation_id,
                parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                error_code=type(e).__name__,
                error_message=str(e),
                metadata={"name": name},
            )
            self.emit(step_failed)
            raise

    async def sleep(self, seconds: float, name: Optional[str] = None) -> None:
        """
        Durable sleep that survives workflow restarts.

        Unlike regular `asyncio.sleep()`, this sleep is checkpointed. If the
        workflow crashes and restarts, it will only sleep for the remaining
        duration (or skip entirely if the sleep period has already elapsed).

        Args:
            seconds: Duration to sleep in seconds
            name: Optional name for the sleep checkpoint (auto-generated if not provided)

        Example:
            ```python
            @workflow
            async def delayed_notification(ctx: WorkflowContext, user_id: str):
                # Send immediate acknowledgment
                await ctx.step(send_ack, user_id)

                # Wait 24 hours (survives restarts!)
                await ctx.sleep(24 * 60 * 60, name="wait_24h")

                # Send follow-up
                await ctx.step(send_followup, user_id)
            ```
        """
        import time

        # Generate unique step name for this sleep
        sleep_name = name or f"sleep_{self._step_counter}"
        self._step_counter += 1
        step_key = f"sleep:{sleep_name}"

        # Check if sleep was already started (replay scenario)
        if self._workflow_entity.has_completed_step(step_key):
            sleep_record = self._workflow_entity.get_completed_step(step_key)
            start_time = sleep_record.get("start_time", 0)
            duration = sleep_record.get("duration", seconds)
            elapsed = time.time() - start_time

            if elapsed >= duration:
                # Sleep period already elapsed
                self._logger.info(f"🔄 Sleep '{sleep_name}' already completed (elapsed: {elapsed:.1f}s)")
                return

            # Sleep for remaining duration
            remaining = duration - elapsed
            self._logger.info(f"⏰ Resuming sleep '{sleep_name}': {remaining:.1f}s remaining")
            await asyncio.sleep(remaining)
            return

        # Record sleep start time for replay
        sleep_record = {
            "start_time": time.time(),
            "duration": seconds,
        }
        self._workflow_entity.record_step_completion(step_key, "sleep", None, sleep_record)

        # Emit checkpoint for observability
        step_event_id = str(uuid.uuid4())
        # Use step_event_id as correlation_id for pairing started ↔ completed
        step_correlation_id = generate_cid()

        from .events import Completed, ComponentType, OperationType, Started
        step_started = Started(
            name=sleep_name,
            correlation_id=step_correlation_id,
            parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.STEP,
            input_data={"step_name": sleep_name, "handler_name": "sleep", "duration_seconds": seconds},
            metadata={"name": sleep_name},
        )
        self.emit(step_started)

        self._logger.info(f"💤 Starting durable sleep '{sleep_name}': {seconds}s")
        await asyncio.sleep(seconds)

        # Emit completion checkpoint
        duration_ms = int(seconds * 1000)
        step_completed = Completed(
            name=sleep_name,
            correlation_id=step_correlation_id,
            parent_correlation_id=self._step_event_stack[-1] if self._step_event_stack else self._correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.STEP,
            output_data={"step_name": sleep_name, "handler_name": "sleep", "duration_seconds": seconds},
            duration_ms=duration_ms,
            metadata={"name": sleep_name},
        )
        self.emit(step_completed)
        self._logger.info(f"⏰ Sleep '{sleep_name}' completed")

    async def wait_for_user(
        self,
        question: str,
        input_type: str = "text",
        options: Optional[List[Dict]] = None,
        allow_custom: bool = False,
        skippable: bool = False,
    ) -> Optional[str]:
        """
        Pause workflow execution and wait for user input.

        On replay (even after worker crash), resumes from this point
        with the user's response. This method enables human-in-the-loop
        workflows by pausing execution and waiting for user interaction.

        Args:
            question: Question to ask the user
            input_type: Type of input - "text", "approval", "select", or "multiselect".
                Legacy aliases: "choice" → "multiselect", "selection" → "select"
            options: For approval/select/multiselect, list of option dicts with 'id' and 'label'
            allow_custom: If True, adds a free-text "Something else" option to select/multiselect
            skippable: If True, adds a "Skip" button. Returns None when skipped.

        Returns:
            User's response string, or None if skipped

        Raises:
            WaitingForUserInputException: When no cached response exists (first call)

        Example (text input):
            ```python
            city = await ctx.wait_for_user("Which city?")
            ```

        Example (approval):
            ```python
            decision = await ctx.wait_for_user(
                "Approve this action?",
                input_type="approval",
                options=[
                    {"id": "approve", "label": "Approve"},
                    {"id": "reject", "label": "Reject"}
                ]
            )
            ```

        Example (select — single choice):
            ```python
            model = await ctx.wait_for_user(
                "Which model?",
                input_type="select",
                options=[
                    {"id": "gpt4", "label": "GPT-4"},
                    {"id": "claude", "label": "Claude"}
                ]
            )
            ```

        Example (multiselect — multiple choices):
            ```python
            topics = await ctx.wait_for_user(
                "Which topics?",
                input_type="multiselect",
                options=[
                    {"id": "ai", "label": "AI"},
                    {"id": "web", "label": "Web"},
                    {"id": "data", "label": "Data"}
                ]
            )
            ```

        Example (skippable with custom option):
            ```python
            preference = await ctx.wait_for_user(
                "Pick a theme:",
                input_type="select",
                options=[{"id": "dark", "label": "Dark"}, {"id": "light", "label": "Light"}],
                allow_custom=True,
                skippable=True,
            )
            if preference is None:
                # User skipped
                ...
            ```
        """
        import warnings

        # Normalize legacy type aliases
        if input_type == "choice":
            warnings.warn(
                'input_type="choice" is deprecated, use "multiselect" instead',
                DeprecationWarning,
                stacklevel=2,
            )
            input_type = "multiselect"
        elif input_type == "selection":
            warnings.warn(
                'input_type="selection" is deprecated, use "select" instead',
                DeprecationWarning,
                stacklevel=2,
            )
            input_type = "select"
        from .events import Completed, ComponentType, OperationType, Paused, Resumed, Started
        from .exceptions import WaitingForUserInputException

        # Generate unique step name for this user input request
        # Each wait_for_user call gets a unique key based on pause_index
        # This allows multi-step HITL workflows where each pause gets its own response
        pause_index = self._workflow_entity._pause_index
        response_key = f"user_response:{self.run_id}:{pause_index}"
        step_name = f"wait_for_user_{pause_index}"

        parent_correlation_id = self._step_event_stack[-1] if self._step_event_stack else self._correlation_id

        # Increment pause index for next call (whether we replay or pause)
        self._workflow_entity._pause_index += 1

        # IMPORTANT: Check for cached response FIRST (before emitting step.started)
        # On resume, we need to emit: workflow.resumed -> workflow.step.completed (with original ID)
        # NOT: workflow.step.started -> workflow.resumed -> workflow.step.completed
        if self._workflow_entity.has_completed_step(response_key):
            response = self._workflow_entity.get_completed_step(response_key)
            self._logger.debug(f"Replaying user response from checkpoint (pause {pause_index})")

            # Use the ORIGINAL step correlation ID from the pause, not a new one
            # This ensures step.started and step.completed have matching correlation IDs
            original_step_correlation_id = self._workflow_entity._resumed_step_correlation_id
            original_step_name = self._workflow_entity._resumed_step_name or step_name

            if original_step_correlation_id:
                self._logger.debug(f"Using restored step correlation ID: {original_step_correlation_id}")
            else:
                # Normal during multi-step HITL replay — only the latest paused step
                # has a correlation ID from the platform; earlier replayed steps generate new ones.
                step_event_id = str(uuid.uuid4())
                original_step_correlation_id = generate_cid()
                self._logger.debug(f"No restored step correlation ID, using new: {original_step_correlation_id}")

            # Emit workflow.resumed FIRST - workflow is continuing after pause
            workflow_resumed = Resumed(
                name=self._workflow_name or "workflow",
                correlation_id=self._correlation_id,
                parent_correlation_id=self.run_id[:8],
                component_type=ComponentType.WORKFLOW,
                resume_data={"response": response, "pause_index": pause_index},
            )
            self.emit(workflow_resumed)

            # Emit workflow.step.completed - complete the SAME step that was paused
            # Use the ORIGINAL correlation ID so started/completed events pair correctly
            step_completed = Completed(
                name=original_step_name,
                correlation_id=original_step_correlation_id,
                parent_correlation_id=parent_correlation_id,
                component_type=ComponentType.WORKFLOW,
                operation=OperationType.STEP,
                output_data={
                    "step_name": original_step_name,
                    "handler_name": "wait_for_user",
                    "result": response,
                },
                metadata={"name": original_step_name},
            )
            self.emit(step_completed)

            # Clear the resumed step info after use
            self._workflow_entity._resumed_step_correlation_id = None
            self._workflow_entity._resumed_step_name = None

            # Decode wire format
            if response == "__skipped__":
                return None
            if isinstance(response, str) and response.startswith("__custom__:"):
                return response[len("__custom__:"):]
            return response

        # No cached response - this is a fresh execution, emit step.started
        step_event_id = str(uuid.uuid4())
        step_correlation_id = generate_cid()

        # Emit workflow.step.started - this step is entering
        step_started = Started(
            name=step_name,
            correlation_id=step_correlation_id,
            parent_correlation_id=parent_correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.STEP,
            input_data={
                "step_name": step_name,
                "handler_name": "wait_for_user",
                "input": {"question": question, "input_type": input_type, "options": options},
            },
            metadata={"name": step_name, "type": "user_input"},
        )
        self.emit(step_started)

        # No response yet - pause execution
        # Collect current workflow state for checkpoint
        checkpoint_state = {}
        if hasattr(self._workflow_entity, "_state") and self._workflow_entity._state is not None:
            checkpoint_state = self._workflow_entity._state.get_state_snapshot()

        self._logger.info(f"⏸️  Pausing workflow for user input: {question}")

        # Emit approval.requested - explicit HITL event for UI to catch
        from .events import ApprovalRequested

        approval_requested = ApprovalRequested(
            name=step_name,
            correlation_id=step_correlation_id,
            parent_correlation_id=parent_correlation_id,
            question=question,
            input_type=input_type,
            options=options or [],
            step_key=step_name,
            allow_custom=allow_custom,
            skippable=skippable,
        )
        self.emit(approval_requested)

        # Emit workflow.step.paused - step is pausing for user input
        # This should arrive AFTER approval.requested in the event stream
        step_paused = Paused(
            name=step_name,
            correlation_id=step_correlation_id,
            parent_correlation_id=parent_correlation_id,
            component_type=ComponentType.WORKFLOW,
            operation=OperationType.STEP,
            reason="user_input_required",
            pause_data={
                "question": question,
                "input_type": input_type,
                "options": options,
                "pause_index": pause_index,
                "allow_custom": allow_custom,
                "skippable": skippable,
            },
        )
        self.emit(step_paused)

        # Emit workflow.paused - workflow is pausing for user input
        # This should arrive AFTER workflow.step.paused in the event stream
        # Include checkpoint metadata so the EE can store it in run.Metadata
        # for proper resume (step replay, state restoration)
        import json as _json

        checkpoint_metadata: dict[str, str] = {
            "question": question,
            "input_type": input_type,
            "pause_reason": "user_input_required",
            "pause_index": str(pause_index),
            "step_name": step_name,
            "step_correlation_id": step_correlation_id,
            "workflow_correlation_id": self._correlation_id,
            "allow_custom": str(allow_custom),
            "skippable": str(skippable),
        }
        if options:
            checkpoint_metadata["options"] = _json.dumps(options)
        if checkpoint_state:
            checkpoint_metadata["checkpoint_state"] = _json.dumps(checkpoint_state)
        # Include step events for replay on resume
        if self._workflow_entity._step_events:
            checkpoint_metadata["step_events"] = _json.dumps(self._workflow_entity._step_events)
        # Include workflow state snapshot
        if hasattr(self, '_workflow_entity') and self._workflow_entity._state is not None:
            if self._workflow_entity._state.has_changes():
                state_snapshot = self._workflow_entity._state.get_state_snapshot()
                checkpoint_metadata["workflow_state"] = _json.dumps(state_snapshot)

        workflow_paused = Paused(
            name=self._workflow_name or "workflow",
            correlation_id=self._correlation_id,
            parent_correlation_id=self.run_id[:8],
            component_type=ComponentType.WORKFLOW,
            reason="user_input_required",
            pause_data={
                "question": question,
                "input_type": input_type,
                "options": options,
                "pause_index": pause_index,
                "step_name": step_name,
                "step_correlation_id": step_correlation_id,
                "allow_custom": allow_custom,
                "skippable": skippable,
            },
            metadata=checkpoint_metadata,
        )
        self.emit(workflow_paused)

        raise WaitingForUserInputException(
            question=question,
            input_type=input_type,
            options=options,
            checkpoint_state=checkpoint_state,
            pause_index=pause_index,  # Pass the pause index for multi-step HITL
            step_name=step_name,  # Pass step info for proper event completion on resume
            step_correlation_id=step_correlation_id,
            allow_custom=allow_custom,
            skippable=skippable,
        )


# ============================================================================
# Helper functions for workflow execution
# ============================================================================


def _sanitize_for_json(obj: Any) -> Any:
    """
    Sanitize data for JSON serialization by removing or converting non-serializable objects.

    Specifically handles:
    - WorkflowContext objects (replaced with placeholder)
    - Nested structures (recursively sanitized)

    Args:
        obj: Object to sanitize

    Returns:
        JSON-serializable version of the object
    """
    # Handle None, primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle WorkflowContext - replace with placeholder
    if isinstance(obj, WorkflowContext):
        return "<WorkflowContext>"

    # Handle tuples/lists - recursively sanitize
    if isinstance(obj, (tuple, list)):
        sanitized = [_sanitize_for_json(item) for item in obj]
        return sanitized if isinstance(obj, list) else tuple(sanitized)

    # Handle dicts - recursively sanitize values
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}

    # For other objects, try to serialize or convert to string
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Not JSON serializable, use string representation
        return repr(obj)


# ============================================================================
# WorkflowEntity: Standalone workflow state management
# ============================================================================


class WorkflowEntity:
    """
    Workflow execution state management.

    Provides workflow-specific capabilities:
    - Step tracking for replay and crash recovery
    - State change tracking for debugging and audit (AI workflows)
    - Completed step cache for efficient replay
    - Automatic state persistence after workflow execution

    Workflow state is persisted to the database after successful execution,
    enabling crash recovery, replay, and cross-invocation state management.
    The workflow decorator automatically calls _persist_state() to ensure
    durability.
    """

    def __init__(
        self,
        run_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        component_name: Optional[str] = None,
    ):
        """
        Initialize workflow entity with memory scope.

        Args:
            run_id: Unique workflow run identifier
            session_id: Session identifier for multi-turn conversations (optional)
            user_id: User identifier for user-scoped memory (optional)
            component_name: Workflow component name for session-scoped entities (optional)

        Memory Scope Priority:
            - user_id present → key: user:{user_id} (shared across workflows)
            - session_id present (and != run_id) → key: workflow:{component_name}:session:{session_id}
            - else → key: run:{run_id}

        Note: For session scope, component_name enables listing sessions by workflow name.
        User scope is shared across all workflows (not per-workflow).
        """
        # Determine entity key based on memory scope priority
        if user_id:
            # User scope: shared across all workflows (not per-workflow)
            entity_key = f"user:{user_id}"
            memory_scope = "user"
        elif session_id and session_id != run_id:
            # Session scope: include workflow name for queryability
            if component_name:
                entity_key = f"workflow:{component_name}:session:{session_id}"
            else:
                # Fallback for backward compatibility
                entity_key = f"session:{session_id}"
            memory_scope = "session"
        else:
            entity_key = f"run:{run_id}"
            memory_scope = "run"

        # Store key and type info (previously inherited from Entity)
        self._key = entity_key
        self._entity_type = "WorkflowEntity"
        self._state_key = (self._entity_type, entity_key)

        # State will be initialized on first access
        self._state: Optional["WorkflowState"] = None

        # Store run_id separately for tracking (even if key is session/user scoped)
        self._run_id = run_id
        self._memory_scope = memory_scope
        self._component_name = component_name
        # Store scope identifiers for proper scope-based persistence
        self._session_id = session_id
        self._user_id = user_id

        # Step tracking for replay and recovery
        self._step_events: list[Dict[str, Any]] = []
        self._completed_steps: Dict[str, Any] = {}

        # HITL pause tracking - each wait_for_user gets unique index
        self._pause_index: int = 0

        # Resumed step info for proper event pairing after HITL resume
        self._resumed_step_correlation_id: Optional[str] = None
        self._resumed_step_name: Optional[str] = None

        # State change tracking for debugging/audit (AI workflows)
        self._state_changes: list[Dict[str, Any]] = []

        logger.debug(f"Created WorkflowEntity: run={run_id}, scope={memory_scope}, key={entity_key}, component={component_name}")

    @property
    def run_id(self) -> str:
        """Get run_id for this workflow execution."""
        return self._run_id

    @property
    def key(self) -> str:
        """Get entity key for this workflow."""
        return self._key

    def record_step_completion(
        self, step_name: str, handler_name: str, input_data: Any, result: Any
    ) -> None:
        """
        Record completed step for replay and recovery.

        Args:
            step_name: Unique step identifier
            handler_name: Function handler name
            input_data: Input data passed to function
            result: Function result
        """
        # Sanitize input_data and result to ensure JSON serializability
        # This removes WorkflowContext objects and other non-serializable types
        sanitized_input = _sanitize_for_json(input_data)
        sanitized_result = _sanitize_for_json(result)

        self._step_events.append(
            {
                "step_name": step_name,
                "handler_name": handler_name,
                "input": sanitized_input,
                "result": sanitized_result,
            }
        )
        self._completed_steps[step_name] = result
        logger.debug(f"Recorded step completion: {step_name}")

    def get_completed_step(self, step_name: str) -> Optional[Any]:
        """
        Get result of completed step (for replay).

        Args:
            step_name: Step identifier

        Returns:
            Step result if completed, None otherwise
        """
        return self._completed_steps.get(step_name)

    def has_completed_step(self, step_name: str) -> bool:
        """Check if step has been completed."""
        return step_name in self._completed_steps

    def inject_user_response(self, response: str) -> None:
        """
        Inject user response as a completed step for workflow resume.

        This method is called by the worker when resuming a paused workflow
        with the user's response. It stores the response as if it was a
        completed step, allowing wait_for_user() to retrieve it on replay.

        The response is injected at the current pause_index (which should be
        restored from metadata before calling this method for multi-step HITL).
        This matches the key format used by wait_for_user().

        Args:
            response: User's response to inject

        Example:
            # Platform resumes workflow with user response
            workflow_entity.inject_user_response("yes")
            # On replay, wait_for_user() returns "yes" from cache
        """
        # Inject at current pause_index (restored from metadata for multi-step HITL)
        response_key = f"user_response:{self.run_id}:{self._pause_index}"
        self._completed_steps[response_key] = response

        # Also add to step_events so it gets serialized to metadata on next pause
        # This ensures previous user responses are preserved across resumes
        self._step_events.append({
            "step_name": response_key,
            "handler_name": "user_response",
            "input": None,
            "result": response,
        })

        logger.info(f"Injected user response for {self.run_id} at pause {self._pause_index}: {response}")

    def get_agent_data(self, agent_name: str) -> Dict[str, Any]:
        """
        Get agent conversation data from workflow state.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary containing agent conversation data (messages, metadata)
            or empty dict if agent has no data yet

        Example:
            ```python
            agent_data = workflow_entity.get_agent_data("ResearchAgent")
            messages = agent_data.get("messages", [])
            ```
        """
        return self.state.get(f"agent.{agent_name}", {})

    def get_agent_messages(self, agent_name: str) -> list[Dict[str, Any]]:
        """
        Get agent messages from workflow state.

        Args:
            agent_name: Name of the agent

        Returns:
            List of message dictionaries

        Example:
            ```python
            messages = workflow_entity.get_agent_messages("ResearchAgent")
            for msg in messages:
                print(f"{msg['role']}: {msg['content']}")
            ```
        """
        agent_data = self.get_agent_data(agent_name)
        return agent_data.get("messages", [])

    def list_agents(self) -> list[str]:
        """
        List all agents with data in this workflow.

        Returns:
            List of agent names that have stored conversation data

        Example:
            ```python
            agents = workflow_entity.list_agents()
            # ['ResearchAgent', 'AnalysisAgent', 'SynthesisAgent']
            ```
        """
        agents = []
        for key in self.state._state.keys():
            if key.startswith("agent."):
                agents.append(key.replace("agent.", "", 1))
        return agents

    async def _persist_state(self) -> None:
        """
        Internal method to persist workflow state to entity storage.

        This is prefixed with _ so it won't be wrapped by the entity method wrapper.
        Called after workflow execution completes to ensure state is durable.
        """
        logger.debug(f"_persist_state() called for workflow {self.run_id}")

        try:
            # Get the state adapter (must be in Worker context)
            adapter = _get_state_adapter()

            # Get current state snapshot
            state_dict = self.state.get_state_snapshot()
            logger.debug(f"State snapshot has {len(state_dict)} keys: {list(state_dict.keys())}")

            # Determine scope and scope_id based on memory scope
            scope = self._memory_scope  # "session", "user", or "run"
            scope_id = ""
            if self._memory_scope == "session" and self._session_id:
                scope_id = self._session_id
            elif self._memory_scope == "user" and self._user_id:
                scope_id = self._user_id
            elif self._memory_scope == "run":
                scope_id = self._run_id

            # Load current version (for optimistic locking) with proper scope
            _, current_version = await adapter.load_with_version(
                self._entity_type, self._key, scope=scope, scope_id=scope_id
            )

            # Save state with version check and proper scope
            new_version = await adapter.save_state(
                self._entity_type, self._key, state_dict, current_version,
                scope=scope, scope_id=scope_id
            )

            logger.debug(
                f"Persisted WorkflowEntity state for {self.run_id} "
                f"(version {current_version} -> {new_version}, {len(state_dict)} keys)"
            )
        except Exception as e:
            logger.error(
                f"❌ ERROR: Failed to persist workflow state for {self.run_id}: {e}",
                exc_info=True
            )
            # Re-raise to let caller handle
            raise

    @property
    def state(self) -> "WorkflowState":
        """
        Get workflow state with change tracking.

        Returns WorkflowState which tracks all state mutations
        for debugging and replay of AI workflows.
        """
        if self._state is None:
            # Initialize with empty state dict - will be populated by entity system
            self._state = WorkflowState({}, self)
        return self._state


class WorkflowState(StateInterface):
    """
    State interface for WorkflowEntity with change tracking.

    Extends StateInterface to track all state mutations for:
    - AI workflow debugging
    - Audit trail
    - Replay capabilities
    """

    def __init__(self, state_dict: Dict[str, Any], workflow_entity: WorkflowEntity):
        """
        Initialize workflow state.

        Args:
            state_dict: Dictionary to use for state storage
            workflow_entity: Parent workflow entity for tracking
        """
        super().__init__(state_dict)
        self._workflow_entity = workflow_entity
        self._emitter: Optional[Any] = None  # EventEmitter for state change events

    def _set_emitter(self, emitter: Any) -> None:
        """
        Set the event emitter for real-time state change streaming.

        Args:
            emitter: EventEmitter instance for emitting state change events
        """
        self._emitter = emitter

    def set(self, key: str, value: Any) -> None:
        """Set value and track change."""
        super().set(key, value)
        # Track change for debugging/audit
        import time

        change_record = {"key": key, "value": value, "timestamp": time.time(), "deleted": False}
        self._workflow_entity._state_changes.append(change_record)

        # Emit event for real-time state streaming
        if self._emitter:
            from ._ids import generate_cid
            from .events import StateChanged
            state_event = StateChanged(
                name=self._workflow_entity._component_name or "workflow",
                correlation_id=generate_cid(),
                parent_correlation_id="",
                key=key,
                value=value,
                operation="set",
            )
            self._emitter(state_event)

    def delete(self, key: str) -> None:
        """Delete key and track change."""
        super().delete(key)
        # Track deletion
        import time

        change_record = {"key": key, "value": None, "timestamp": time.time(), "deleted": True}
        self._workflow_entity._state_changes.append(change_record)

        # Emit event for real-time state streaming
        if self._emitter:
            from ._ids import generate_cid
            from .events import StateChanged
            state_event = StateChanged(
                name=self._workflow_entity._component_name or "workflow",
                correlation_id=generate_cid(),
                parent_correlation_id="",
                key=key,
                operation="delete",
            )
            self._emitter(state_event)

    def clear(self) -> None:
        """Clear all state and track change."""
        super().clear()
        # Track clear operation
        import time

        change_record = {
            "key": "__clear__",
            "value": None,
            "timestamp": time.time(),
            "deleted": True,
        }
        self._workflow_entity._state_changes.append(change_record)

        # Emit event for real-time state streaming
        if self._emitter:
            from ._ids import generate_cid
            from .events import StateChanged
            state_event = StateChanged(
                name=self._workflow_entity._component_name or "workflow",
                correlation_id=generate_cid(),
                parent_correlation_id="",
                operation="clear",
            )
            self._emitter(state_event)

    def has_changes(self) -> bool:
        """Check if any state changes have been tracked."""
        return len(self._workflow_entity._state_changes) > 0

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state as a snapshot dictionary."""
        return dict(self._state)


class WorkflowRegistry:
    """Registry for workflow handlers."""

    @staticmethod
    def register(config: WorkflowConfig) -> None:
        """
        Register a workflow handler.

        Raises:
            ValueError: If a workflow with this name is already registered
        """
        if config.name in _WORKFLOW_REGISTRY:
            existing_workflow = _WORKFLOW_REGISTRY[config.name]
            logger.error(
                f"Workflow name collision detected: '{config.name}'\n"
                f"  First defined in:  {existing_workflow.handler.__module__}\n"
                f"  Also defined in:   {config.handler.__module__}\n"
                f"  This is a bug - workflows must have unique names."
            )
            raise ValueError(
                f"Workflow '{config.name}' is already registered. "
                f"Use @workflow(name='unique_name') to specify a different name."
            )

        _WORKFLOW_REGISTRY[config.name] = config
        logger.debug(f"Registered workflow '{config.name}'")

    @staticmethod
    def get(name: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration by name."""
        return _WORKFLOW_REGISTRY.get(name)

    @staticmethod
    def all() -> Dict[str, WorkflowConfig]:
        """Get all registered workflows."""
        return _WORKFLOW_REGISTRY.copy()

    @staticmethod
    def list_names() -> list[str]:
        """List all registered workflow names."""
        return list(_WORKFLOW_REGISTRY.keys())

    @staticmethod
    def clear() -> None:
        """Clear all registered workflows."""
        _WORKFLOW_REGISTRY.clear()


def workflow(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    chat: bool = False,
    cron: Optional[str] = None,
    webhook: bool = False,
    webhook_secret: Optional[str] = None,
    triggers: Optional[Sequence[TriggerSpec]] = None,
) -> Callable[..., Any]:
    """
    Decorator to mark a function as an AGNT5 durable workflow.

    Workflows use WorkflowEntity for state management and WorkflowContext
    for orchestration. State changes are automatically tracked for replay.

    Args:
        name: Custom workflow name (default: function's __name__)
        chat: Enable chat mode for multi-turn conversation workflows (default: False)
        cron: Cron expression for scheduled execution (e.g., "0 9 * * *" for daily at 9am)
        webhook: Enable webhook triggering for this workflow (default: False)
        webhook_secret: Optional secret for HMAC-SHA256 signature verification
        triggers: Typed trigger declarations such as event("user.created")

    Example (standard workflow):
        @workflow
        async def process_order(ctx: WorkflowContext, order_id: str) -> dict:
            # Durable state - survives crashes
            ctx.state.set("status", "processing")
            ctx.state.set("order_id", order_id)

            # Validate order
            order = await ctx.task(validate_order, input={"order_id": order_id})

            # Process payment (checkpointed - won't re-execute on crash)
            payment = await ctx.step("payment", process_payment(order["total"]))

            # Fulfill order
            await ctx.task(ship_order, input={"order_id": order_id})

            ctx.state.set("status", "completed")
            return {"status": ctx.state.get("status")}

    Example (chat workflow):
        @workflow(chat=True)
        async def customer_support(ctx: WorkflowContext, message: str) -> dict:
            # Initialize conversation state
            if not ctx.state.get("messages"):
                ctx.state.set("messages", [])

            # Add user message
            messages = ctx.state.get("messages")
            messages.append({"role": "user", "content": message})
            ctx.state.set("messages", messages)

            # Generate AI response
            response = await ctx.task(generate_response, messages=messages)

            # Add assistant response
            messages.append({"role": "assistant", "content": response})
            ctx.state.set("messages", messages)

            return {"response": response, "turn_count": len(messages) // 2}

    Example (scheduled workflow):
        @workflow(name="daily_report", cron="0 9 * * *")
        async def daily_report(ctx: WorkflowContext) -> dict:
            # Runs automatically every day at 9am
            sales = await ctx.task(get_sales_data, report_type="sales")
            report = await ctx.task(generate_pdf, input=sales)
            await ctx.task(send_email, to="team@company.com", attachment=report)
            return {"status": "sent", "report_id": report["id"]}

    Example (webhook workflow):
        @workflow(name="on_payment", webhook=True, webhook_secret="your_secret_key")
        async def on_payment(ctx: WorkflowContext, webhook_data: dict) -> dict:
            # Triggered by webhook POST /v1/webhooks/on_payment
            # webhook_data contains: payload, headers, source_ip, timestamp
            payment = webhook_data["payload"]

            if payment.get("status") == "succeeded":
                await ctx.task(fulfill_order, order_id=payment["order_id"])
                await ctx.task(send_receipt, customer_email=payment["email"])
                return {"status": "processed", "order_id": payment["order_id"]}

            return {"status": "skipped", "reason": "payment not successful"}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get workflow name
        workflow_name = name or func.__name__

        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params or params[0].name != "ctx":
            raise ValueError(
                f"Workflow '{workflow_name}' must have 'ctx: WorkflowContext' as first parameter"
            )

        # Convert sync to async if needed
        if inspect.iscoroutinefunction(func):
            handler_func = cast(HandlerFunc, func)
        else:
            # Wrap sync function in async
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            handler_func = cast(HandlerFunc, async_wrapper)

        # Extract schemas from type hints
        input_schema, output_schema = extract_function_schemas(func)

        # Extract metadata (description, etc.)
        metadata = extract_function_metadata(func)

        # Add chat metadata if chat mode is enabled
        if chat:
            metadata["chat"] = "true"

        # Add cron metadata if cron schedule is provided
        if cron:
            metadata["cron"] = cron

        # Add webhook metadata if webhook is enabled
        if webhook:
            metadata["webhook"] = "true"
            if webhook_secret:
                metadata["webhook_secret"] = webhook_secret

        # Register workflow
        config = WorkflowConfig(
            name=workflow_name,
            handler=handler_func,
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata,
            triggers=list(triggers or []),
        )
        WorkflowRegistry.register(config)

        # Create wrapper that provides context
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create WorkflowEntity and WorkflowContext if not provided
            if not args or not isinstance(args[0], WorkflowContext):
                # Auto-create workflow entity and context for direct workflow calls
                run_id = f"workflow-{uuid.uuid4().hex[:8]}"

                # Set up standalone state adapter if not already set
                # This enables direct workflow execution without Worker
                from ._state_adapter import _state_adapter_ctx, StateAdapter

                existing_adapter = _state_adapter_ctx.get()
                state_token = None
                if existing_adapter is None:
                    standalone_adapter = StateAdapter()  # In-memory mode
                    state_token = _state_adapter_ctx.set(standalone_adapter)

                # Create WorkflowEntity to manage state
                workflow_entity = WorkflowEntity(run_id=run_id)

                # Create WorkflowContext that wraps the entity
                ctx = WorkflowContext(
                    workflow_entity=workflow_entity,
                    run_id=run_id,
                )

                # Set context in task-local storage for automatic propagation
                token = set_current_context(ctx)
                try:
                    # Execute workflow - only pass kwargs, not original args
                    # The original args would contain a non-WorkflowContext that
                    # would incorrectly map to the first positional parameter
                    result = await handler_func(ctx, **kwargs)

                    # Persist workflow state after successful execution
                    try:
                        await workflow_entity._persist_state()
                    except Exception as e:
                        logger.error(f"Failed to persist workflow state (non-fatal): {e}", exc_info=True)
                        # Don't fail the workflow - persistence failure shouldn't break execution

                    return result
                finally:
                    # Always reset context to prevent leakage
                    from .context import _current_context

                    _current_context.reset(token)

                    # Clean up standalone state adapter if we created one
                    if state_token is not None:
                        _state_adapter_ctx.reset(state_token)
            else:
                # WorkflowContext provided - use it and set in contextvar
                ctx = args[0]
                token = set_current_context(ctx)
                try:
                    result = await handler_func(*args, **kwargs)

                    # Persist workflow state after successful execution
                    try:
                        await ctx._workflow_entity._persist_state()
                    except Exception as e:
                        logger.error(f"Failed to persist workflow state (non-fatal): {e}", exc_info=True)
                        # Don't fail the workflow - persistence failure shouldn't break execution

                    return result
                finally:
                    # Always reset context to prevent leakage
                    from .context import _current_context

                    _current_context.reset(token)

        # Store config on wrapper for introspection
        wrapper._agnt5_config = config  # type: ignore
        return wrapper

    # Handle both @workflow and @workflow(...) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
