"""Workflow component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from ._schema_utils import extract_function_metadata, extract_function_schemas
from .context import Context
from .entity import Entity, EntityState, _get_state_manager
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

    WorkflowContext delegates state to the underlying WorkflowEntity,
    which provides durability and state change tracking for AI workflows.
    """

    def __init__(
        self,
        workflow_entity: "WorkflowEntity",  # Forward reference
        run_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
    ) -> None:
        """
        Initialize workflow context.

        Args:
            workflow_entity: WorkflowEntity instance managing workflow state
            run_id: Unique workflow run identifier
            attempt: Retry attempt number (0-indexed)
            runtime_context: RuntimeContext for trace correlation
        """
        super().__init__(run_id, attempt, runtime_context)
        self._workflow_entity = workflow_entity
        self._step_counter: int = 0  # Track step sequence

    # === State Management ===

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
        return self._workflow_entity.state

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
            if not hasattr(handler, '_agnt5_config'):
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

        # Execute function
        self._logger.info(f"▶️  Executing new step: {step_name}")
        func_config = FunctionRegistry.get(handler_name)
        if func_config is None:
            raise ValueError(f"Function '{handler_name}' not found in registry")

        # Create FunctionContext for the function execution
        func_ctx = FunctionContext(
            run_id=f"{self.run_id}:task:{handler_name}",
            runtime_context=self._runtime_context,
        )

        # Execute function with arguments
        # Support legacy pattern: ctx.task("func_name", input=data) or ctx.task(func_ref, input=data)
        if len(args) == 0 and "input" in kwargs:
            # Legacy pattern - single input parameter
            input_data = kwargs.pop("input")  # Remove from kwargs
            result = await func_config.handler(func_ctx, input_data, **kwargs)
        else:
            # Type-safe pattern - pass all args/kwargs
            result = await func_config.handler(func_ctx, *args, **kwargs)

        # Record step completion in WorkflowEntity
        self._workflow_entity.record_step_completion(step_name, handler_name, args or kwargs, result)

        return result

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
        self,
        name: str,
        func_or_awaitable: Union[Callable[[], Awaitable[T]], Awaitable[T]]
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

    Workflows are temporary entities - they exist for the duration of
    execution and their state is used for coordination between steps.
    """

    def __init__(self, run_id: str):
        """
        Initialize workflow entity.

        Args:
            run_id: Unique workflow run identifier
        """
        # Initialize as entity with workflow key pattern
        super().__init__(key=f"workflow:{run_id}")

        # Step tracking for replay and recovery
        self._step_events: list[Dict[str, Any]] = []
        self._completed_steps: Dict[str, Any] = {}

        # State change tracking for debugging/audit (AI workflows)
        self._state_changes: list[Dict[str, Any]] = []

        logger.debug(f"Created WorkflowEntity: {run_id}")

    @property
    def run_id(self) -> str:
        """Extract run_id from workflow key."""
        return self._key.split(":", 1)[1]

    def record_step_completion(
        self,
        step_name: str,
        handler_name: str,
        input_data: Any,
        result: Any
    ) -> None:
        """
        Record completed step for replay and recovery.

        Args:
            step_name: Unique step identifier
            handler_name: Function handler name
            input_data: Input data passed to function
            result: Function result
        """
        self._step_events.append({
            "step_name": step_name,
            "handler_name": handler_name,
            "input": input_data,
            "result": result
        })
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

    @property
    def state(self) -> "WorkflowState":
        """
        Get workflow state with change tracking.

        Returns WorkflowState which tracks all state mutations
        for debugging and replay of AI workflows.
        """
        if self._state is None:
            # Get state dict from state manager
            state_manager = _get_state_manager()
            state_dict = state_manager.get_or_create_state(self._state_key)
            self._state = WorkflowState(state_dict, self)
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

    def set(self, key: str, value: Any) -> None:
        """Set value and track change."""
        super().set(key, value)
        # Track change for debugging/audit
        import time
        self._workflow_entity._state_changes.append({
            "key": key,
            "value": value,
            "timestamp": time.time(),
            "deleted": False
        })

    def delete(self, key: str) -> None:
        """Delete key and track change."""
        super().delete(key)
        # Track deletion
        import time
        self._workflow_entity._state_changes.append({
            "key": key,
            "value": None,
            "timestamp": time.time(),
            "deleted": True
        })

    def clear(self) -> None:
        """Clear all state and track change."""
        super().clear()
        # Track clear operation
        import time
        self._workflow_entity._state_changes.append({
            "key": "__clear__",
            "value": None,
            "timestamp": time.time(),
            "deleted": True
        })


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

                # Execute workflow
                return await handler_func(ctx, *args, **kwargs)
            else:
                # WorkflowContext provided - use it
                return await handler_func(*args, **kwargs)

        # Store config on wrapper for introspection
        wrapper._agnt5_config = config  # type: ignore
        return wrapper

    # Handle both @workflow and @workflow(...) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)


