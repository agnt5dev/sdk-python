"""Workflow component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union, cast

from ._schema_utils import extract_function_metadata, extract_function_schemas
from .context import Context, set_current_context
from .entity import Entity, EntityState, _get_state_adapter
from .function import FunctionContext
from .types import HandlerFunc, WorkflowConfig
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
        checkpoint_callback: Optional[Callable[[dict], None]] = None,
        checkpoint_client: Optional[Any] = None,
        is_streaming: bool = False,
        tenant_id: Optional[str] = None,
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
            checkpoint_callback: Optional callback for sending real-time checkpoints
            checkpoint_client: Optional CheckpointClient for platform-side memoization
            is_streaming: Whether this is a streaming request (for real-time SSE log delivery)
            tenant_id: Tenant identifier for multi-tenant deployments
        """
        super().__init__(run_id, attempt, runtime_context, is_streaming, tenant_id)
        self._workflow_entity = workflow_entity
        self._step_counter: int = 0  # Track step sequence
        self._sequence_number: int = 0  # Global sequence for checkpoints
        self._checkpoint_callback = checkpoint_callback
        self._checkpoint_client = checkpoint_client

        # Memory scoping identifiers
        self.session_id = session_id or run_id  # Default: session = run (ephemeral)
        self.user_id = user_id  # Optional: user-scoped memory

        # Step hierarchy tracking - for nested step visualization
        # Stack of event IDs for currently executing steps
        self._step_event_stack: List[str] = []

    # === State Management ===

    def _send_checkpoint(self, checkpoint_type: str, checkpoint_data: dict) -> None:
        """
        Send a checkpoint via the checkpoint callback.

        Automatically adds parent_event_id from the step event stack if we're
        currently executing inside a nested step call.

        Args:
            checkpoint_type: Type of checkpoint (e.g., "workflow.state.changed")
            checkpoint_data: Checkpoint payload (should include event_id if needed)
        """
        if self._checkpoint_callback:
            self._sequence_number += 1

            # Add parent_event_id if we're in a nested step
            if self._step_event_stack:
                checkpoint_data = {
                    **checkpoint_data,
                    "parent_event_id": self._step_event_stack[-1],
                }

            checkpoint = {
                "checkpoint_type": checkpoint_type,
                "checkpoint_data": checkpoint_data,
                "sequence_number": self._sequence_number,
                "source_timestamp_ns": time.time_ns(),  # Nanosecond timestamp for correct logical ordering
            }
            self._checkpoint_callback(checkpoint)

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
        if hasattr(state, "_set_checkpoint_callback"):
            state._set_checkpoint_callback(self._send_checkpoint)
        return state

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

        Supports multiple calling patterns:

        1. **Call a @function (recommended)**:
           ```python
           result = await ctx.step(process_data, arg1, arg2, kwarg=value)
           ```
           Auto-generates step name from function. Full IDE support.

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
            handler_name = handler.__name__
            if not hasattr(handler, "_agnt5_config"):
                raise ValueError(
                    f"Function '{handler_name}' is not a registered @function. "
                    f"Did you forget to add the @function decorator?"
                )
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
            return result

        # Emit workflow.step.started checkpoint
        self._send_checkpoint(
            "workflow.step.started",
            {
                "step_name": step_name,
                "handler_name": handler_name,
                "input": args or kwargs,
                "event_id": step_event_id,  # Include for hierarchy tracking
            },
        )

        # Push this step's event_id onto the stack for nested calls
        self._step_event_stack.append(step_event_id)

        # Execute function with OpenTelemetry span
        self._logger.info(f"▶️  Executing new step: {step_name}")
        func_config = FunctionRegistry.get(handler_name)
        if func_config is None:
            raise ValueError(f"Function '{handler_name}' not found in registry")

        # Import span creation utility and JSON serialization
        from ._core import create_span
        import json

        # Serialize input data for span attributes
        input_repr = json.dumps({"args": args, "kwargs": kwargs}) if args or kwargs else "{}"

        # Create span for task execution
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
            func_ctx = FunctionContext(
                run_id=f"{self.run_id}:task:{handler_name}",
                runtime_context=self._runtime_context,
            )

            try:
                # Execute function with arguments
                # Support legacy pattern: ctx.task("func_name", input=data) or ctx.task(func_ref, input=data)
                if len(args) == 0 and "input" in kwargs:
                    # Legacy pattern - single input parameter
                    input_data = kwargs.pop("input")  # Remove from kwargs
                    result = await func_config.handler(func_ctx, input_data, **kwargs)
                else:
                    # Type-safe pattern - pass all args/kwargs
                    result = await func_config.handler(func_ctx, *args, **kwargs)

                # Add output data to span
                try:
                    output_repr = json.dumps(result)
                    span.set_attribute("output.data", output_repr)
                except (TypeError, ValueError):
                    # If result is not JSON serializable, use repr
                    span.set_attribute("output.data", repr(result))

                # Record step completion in WorkflowEntity
                self._workflow_entity.record_step_completion(
                    step_name, handler_name, args or kwargs, result
                )

                # Pop this step's event_id from the stack (execution complete)
                if self._step_event_stack:
                    popped_id = self._step_event_stack.pop()
                    if popped_id != step_event_id:
                        self._logger.warning(
                            f"Step event stack mismatch in task(): expected {step_event_id}, got {popped_id}"
                        )

                # Emit workflow.step.completed checkpoint
                self._send_checkpoint(
                    "workflow.step.completed",
                    {
                        "step_name": step_name,
                        "handler_name": handler_name,
                        "input": args or kwargs,
                        "result": result,
                        "event_id": step_event_id,  # Include for consistency
                    },
                )

                return result

            except Exception as e:
                # Pop this step's event_id from the stack (execution failed)
                if self._step_event_stack:
                    popped_id = self._step_event_stack.pop()
                    if popped_id != step_event_id:
                        self._logger.warning(
                            f"Step event stack mismatch in task() error path: expected {step_event_id}, got {popped_id}"
                        )

                # Emit workflow.step.failed checkpoint
                self._send_checkpoint(
                    "workflow.step.failed",
                    {
                        "step_name": step_name,
                        "handler_name": handler_name,
                        "input": args or kwargs,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "event_id": step_event_id,  # Include for consistency
                    },
                )

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

        # Check platform-side memoization first (Phase 3)
        if self._checkpoint_client:
            try:
                result = await self._checkpoint_client.step_started(
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
        self._send_checkpoint(
            "workflow.step.started",
            {
                "step_name": name,
                "handler_name": "checkpoint",
                "event_id": step_event_id,  # Include for hierarchy tracking
            },
        )

        # Push this step's event_id onto the stack for nested calls
        self._step_event_stack.append(step_event_id)

        start_time = time.time()
        try:
            # Execute and checkpoint
            if inspect.iscoroutine(func_or_awaitable) or inspect.isawaitable(func_or_awaitable):
                result = await func_or_awaitable
            elif callable(func_or_awaitable):
                # Call with args/kwargs if provided
                call_result = func_or_awaitable(*args, **kwargs)
                if inspect.iscoroutine(call_result) or inspect.isawaitable(call_result):
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
                    output_bytes = json.dumps(result).encode("utf-8")
                    await self._checkpoint_client.step_completed(
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
            self._send_checkpoint(
                "workflow.step.completed",
                {
                    "step_name": name,
                    "handler_name": "checkpoint",
                    "result": result,
                    "event_id": step_event_id,  # Include for consistency
                },
            )

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
                    await self._checkpoint_client.step_failed(
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
            self._send_checkpoint(
                "workflow.step.failed",
                {
                    "step_name": name,
                    "handler_name": "checkpoint",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "event_id": step_event_id,  # Include for consistency
                },
            )
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
        self._send_checkpoint(
            "workflow.step.started",
            {
                "step_name": sleep_name,
                "handler_name": "sleep",
                "duration_seconds": seconds,
                "event_id": step_event_id,
            },
        )

        self._logger.info(f"💤 Starting durable sleep '{sleep_name}': {seconds}s")
        await asyncio.sleep(seconds)

        # Emit completion checkpoint
        self._send_checkpoint(
            "workflow.step.completed",
            {
                "step_name": sleep_name,
                "handler_name": "sleep",
                "duration_seconds": seconds,
                "event_id": step_event_id,
            },
        )
        self._logger.info(f"⏰ Sleep '{sleep_name}' completed")

    async def wait_for_user(
        self, question: str, input_type: str = "text", options: Optional[List[Dict]] = None
    ) -> str:
        """
        Pause workflow execution and wait for user input.

        On replay (even after worker crash), resumes from this point
        with the user's response. This method enables human-in-the-loop
        workflows by pausing execution and waiting for user interaction.

        Args:
            question: Question to ask the user
            input_type: Type of input - "text", "approval", or "choice"
            options: For approval/choice, list of option dicts with 'id' and 'label'

        Returns:
            User's response string

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

        Example (choice):
            ```python
            model = await ctx.wait_for_user(
                "Which model?",
                input_type="choice",
                options=[
                    {"id": "gpt4", "label": "GPT-4"},
                    {"id": "claude", "label": "Claude"}
                ]
            )
            ```
        """
        from .exceptions import WaitingForUserInputException

        # Generate unique step name for this user input request
        # Using run_id ensures uniqueness across workflow execution
        response_key = f"user_response:{self.run_id}"

        # Check if we already have the user's response (replay scenario)
        if self._workflow_entity.has_completed_step(response_key):
            response = self._workflow_entity.get_completed_step(response_key)
            self._logger.info("🔄 Replaying user response from checkpoint")
            return response

        # No response yet - pause execution
        # Collect current workflow state for checkpoint
        checkpoint_state = {}
        if hasattr(self._workflow_entity, "_state") and self._workflow_entity._state is not None:
            checkpoint_state = self._workflow_entity._state.get_state_snapshot()

        self._logger.info(f"⏸️  Pausing workflow for user input: {question}")

        raise WaitingForUserInputException(
            question=question,
            input_type=input_type,
            options=options,
            checkpoint_state=checkpoint_state,
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
# WorkflowEntity: Entity specialized for workflow execution state
# ============================================================================


class WorkflowEntity(Entity):
    """
    Entity specialized for workflow execution state.

    Extends Entity with workflow-specific capabilities:
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
    ):
        """
        Initialize workflow entity with memory scope.

        Args:
            run_id: Unique workflow run identifier
            session_id: Session identifier for multi-turn conversations (optional)
            user_id: User identifier for user-scoped memory (optional)

        Memory Scope Priority:
            - user_id present → key: user:{user_id}
            - session_id present (and != run_id) → key: session:{session_id}
            - else → key: run:{run_id}
        """
        # Determine entity key based on memory scope priority
        if user_id:
            entity_key = f"user:{user_id}"
            memory_scope = "user"
        elif session_id and session_id != run_id:
            entity_key = f"session:{session_id}"
            memory_scope = "session"
        else:
            entity_key = f"run:{run_id}"
            memory_scope = "run"

        # Initialize as entity with scoped key pattern
        super().__init__(key=entity_key)

        # Store run_id separately for tracking (even if key is session/user scoped)
        self._run_id = run_id
        self._memory_scope = memory_scope

        # Step tracking for replay and recovery
        self._step_events: list[Dict[str, Any]] = []
        self._completed_steps: Dict[str, Any] = {}

        # State change tracking for debugging/audit (AI workflows)
        self._state_changes: list[Dict[str, Any]] = []

        logger.debug(f"Created WorkflowEntity: run={run_id}, scope={memory_scope}, key={entity_key}")

    @property
    def run_id(self) -> str:
        """Get run_id for this workflow execution."""
        return self._run_id

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

        Args:
            response: User's response to inject

        Example:
            # Platform resumes workflow with user response
            workflow_entity.inject_user_response("yes")
            # On replay, wait_for_user() returns "yes" from cache
        """
        response_key = f"user_response:{self.run_id}"
        self._completed_steps[response_key] = response
        logger.info(f"Injected user response for {self.run_id}: {response}")

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
        logger.info(f"🔍 DEBUG: _persist_state() CALLED for workflow {self.run_id}")

        try:
            from .entity import _get_state_adapter

            logger.info(f"🔍 DEBUG: Getting state adapter...")
            # Get the state adapter (must be in Worker context)
            adapter = _get_state_adapter()
            logger.info(f"🔍 DEBUG: Got state adapter: {type(adapter).__name__}")

            logger.info(f"🔍 DEBUG: Getting state snapshot...")
            # Get current state snapshot
            state_dict = self.state.get_state_snapshot()
            logger.info(f"🔍 DEBUG: State snapshot has {len(state_dict)} keys: {list(state_dict.keys())}")

            logger.info(f"🔍 DEBUG: Loading current version for optimistic locking...")
            # Load current version (for optimistic locking)
            _, current_version = await adapter.load_with_version(self._entity_type, self._key)
            logger.info(f"🔍 DEBUG: Current version: {current_version}")

            logger.info(f"🔍 DEBUG: Saving state to database...")
            # Save state with version check
            new_version = await adapter.save_state(
                self._entity_type, self._key, state_dict, current_version
            )

            logger.info(
                f"✅ SUCCESS: Persisted WorkflowEntity state for {self.run_id} "
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


class WorkflowState(EntityState):
    """
    State interface for WorkflowEntity with change tracking.

    Extends EntityState to track all state mutations for:
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
        self._checkpoint_callback: Optional[Callable[[str, dict], None]] = None

    def _set_checkpoint_callback(self, callback: Callable[[str, dict], None]) -> None:
        """
        Set the checkpoint callback for real-time state change streaming.

        Args:
            callback: Function to call when state changes
        """
        self._checkpoint_callback = callback

    def set(self, key: str, value: Any) -> None:
        """Set value and track change."""
        super().set(key, value)
        # Track change for debugging/audit
        import time

        change_record = {"key": key, "value": value, "timestamp": time.time(), "deleted": False}
        self._workflow_entity._state_changes.append(change_record)

        # Emit checkpoint for real-time state streaming
        if self._checkpoint_callback:
            self._checkpoint_callback(
                "workflow.state.changed", {"key": key, "value": value, "operation": "set"}
            )

    def delete(self, key: str) -> None:
        """Delete key and track change."""
        super().delete(key)
        # Track deletion
        import time

        change_record = {"key": key, "value": None, "timestamp": time.time(), "deleted": True}
        self._workflow_entity._state_changes.append(change_record)

        # Emit checkpoint for real-time state streaming
        if self._checkpoint_callback:
            self._checkpoint_callback("workflow.state.changed", {"key": key, "operation": "delete"})

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

        # Emit checkpoint for real-time state streaming
        if self._checkpoint_callback:
            self._checkpoint_callback("workflow.state.changed", {"operation": "clear"})

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
        )
        WorkflowRegistry.register(config)

        # Create wrapper that provides context
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create WorkflowEntity and WorkflowContext if not provided
            if not args or not isinstance(args[0], WorkflowContext):
                # Auto-create workflow entity and context for direct workflow calls
                run_id = f"workflow-{uuid.uuid4().hex[:8]}"

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
                    # Execute workflow
                    result = await handler_func(ctx, *args, **kwargs)

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
