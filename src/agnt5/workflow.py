"""Workflow component implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import functools
import inspect
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from .context import Context
from .types import HandlerFunc, WorkflowConfig

T = TypeVar("T")

# Global workflow registry
_WORKFLOW_REGISTRY: Dict[str, WorkflowConfig] = {}


class WorkflowRegistry:
    """Registry for workflow handlers."""

    @staticmethod
    def register(config: WorkflowConfig) -> None:
        """Register a workflow handler."""
        _WORKFLOW_REGISTRY[config.name] = config

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
) -> Callable[..., Any]:
    """
    Decorator to mark a function as an AGNT5 durable workflow.

    Phase 1: In-memory orchestration with asyncio primitives.
    Phase 2: Durable execution with checkpoint/replay and distributed tasks.

    Args:
        name: Custom workflow name (default: function's __name__)

    Example:
        @workflow
        async def process_order(ctx: Context, order_id: str) -> dict:
            # Validate order
            order = await ctx.task("orders", "validate", input={"order_id": order_id})

            # Process payment
            payment = await ctx.task("payments", "charge", input={"amount": order["total"]})

            # Fulfill order
            await ctx.task("fulfillment", "ship", input={"order_id": order_id})

            return {"status": "completed"}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Get workflow name
        workflow_name = name or func.__name__

        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params or params[0].name != "ctx":
            raise ValueError(
                f"Workflow '{workflow_name}' must have 'ctx: Context' as first parameter"
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

        # Register workflow
        config = WorkflowConfig(
            name=workflow_name,
            handler=handler_func,
        )
        WorkflowRegistry.register(config)

        # Create wrapper that provides context
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create context if not provided
            if not args or not isinstance(args[0], Context):
                # Auto-create context for direct workflow calls
                ctx = Context(
                    run_id=f"workflow-{uuid.uuid4().hex[:8]}",
                    component_type="workflow",
                )
                # Execute workflow
                return await handler_func(ctx, *args, **kwargs)
            else:
                # Context provided - use it
                return await handler_func(*args, **kwargs)

        # Store config on wrapper for introspection
        wrapper._agnt5_config = config  # type: ignore
        return wrapper

    # Handle both @workflow and @workflow(...) syntax
    if _func is None:
        return decorator
    else:
        return decorator(_func)
