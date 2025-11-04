"""Workflow component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
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
        """
        super().__init__(run_id, attempt, runtime_context)
        self._workflow_entity = workflow_entity
        self._step_counter: int = 0  # Track step sequence
        self._sequence_number: int = 0  # Global sequence for checkpoints
        self._checkpoint_callback = checkpoint_callback

        # Memory scoping identifiers
        self.session_id = session_id or run_id  # Default: session = run (ephemeral)
        self.user_id = user_id  # Optional: user-scoped memory

    # === State Management ===

    def _send_checkpoint(self, checkpoint_type: str, checkpoint_data: dict) -> None:
        """
        Send a checkpoint via the checkpoint callback.

        Args:
            checkpoint_type: Type of checkpoint (e.g., "workflow.state.changed")
            checkpoint_data: Checkpoint payload
        """
        if self._checkpoint_callback:
            self._sequence_number += 1
            checkpoint = {
                "checkpoint_type": checkpoint_type,
                "checkpoint_data": checkpoint_data,
                "sequence_number": self._sequence_number,
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

    async def task(
        self,
        handler: Union[str, Callable],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function and wait for result.

        Supports two calling patterns:

        1. **Type-safe with function reference (recommended)**:
           ```python
           result = await ctx.task(process_data, arg1, arg2, kwarg=value)
           ```
           Full IDE support, type checking, and refactoring safety.

        2. **Legacy string-based (backward compatible)**:
           ```python
           result = await ctx.task("function_name", input=data)
           ```
           String lookup without type safety.

        Args:
            handler: Either a @function reference (recommended) or string name (legacy)
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function result

        Example (type-safe):
            ```python
            @function
            async def process_data(ctx: FunctionContext, data: list, multiplier: int = 2):
                return [x * multiplier for x in data]

            @workflow
            async def my_workflow(ctx: WorkflowContext):
                # Type-safe call with positional and keyword args
                result = await ctx.task(process_data, [1, 2, 3], multiplier=3)
                return result
            ```

        Example (legacy):
            ```python
            result = await ctx.task("process_data", input={"data": [1, 2, 3]})
            ```
        """
        from .function import FunctionRegistry

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
            },
        )

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

                # Emit workflow.step.completed checkpoint
                self._send_checkpoint(
                    "workflow.step.completed",
                    {
                        "step_name": step_name,
                        "handler_name": handler_name,
                        "input": args or kwargs,
                        "result": result,
                    },
                )

                return result

            except Exception as e:
                # Emit workflow.step.error checkpoint
                self._send_checkpoint(
                    "workflow.step.error",
                    {
                        "step_name": step_name,
                        "handler_name": handler_name,
                        "input": args or kwargs,
                        "error": str(e),
                        "error_type": type(e).__name__,
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

    async def step(
        self, name: str, func_or_awaitable: Union[Callable[[], Awaitable[T]], Awaitable[T]]
    ) -> T:
        """
        Checkpoint expensive operations for durability.

        If workflow crashes, won't re-execute this step on retry.

        Args:
            name: Unique name for this checkpoint
            func_or_awaitable: Either an async function or awaitable

        Returns:
            The result of the function/awaitable

        Example:
            result = await ctx.step("load", load_data())
        """
        import inspect

        # Check if step already completed (for replay)
        if self._workflow_entity.has_completed_step(name):
            result = self._workflow_entity.get_completed_step(name)
            self._logger.info(f"🔄 Replaying checkpoint: {name}")
            return result

        # Execute and checkpoint
        if inspect.iscoroutine(func_or_awaitable) or inspect.isawaitable(func_or_awaitable):
            result = await func_or_awaitable
        else:
            result = await func_or_awaitable()

        # Record step completion
        self._workflow_entity.record_step_completion(name, "checkpoint", None, result)

        return result

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
        self._step_events.append(
            {
                "step_name": step_name,
                "handler_name": handler_name,
                "input": input_data,
                "result": result,
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
) -> Callable[..., Any]:
    """
    Decorator to mark a function as an AGNT5 durable workflow.

    Workflows use WorkflowEntity for state management and WorkflowContext
    for orchestration. State changes are automatically tracked for replay.

    Args:
        name: Custom workflow name (default: function's __name__)
        chat: Enable chat mode for multi-turn conversation workflows (default: False)

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
