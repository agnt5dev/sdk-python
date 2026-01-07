"""
Tool component for Agent capabilities with automatic schema extraction.

Tools extend what agents can do by providing structured interfaces to functions,
with automatic schema generation from Python type hints and docstrings.
"""

import asyncio
import dataclasses as dc
import functools
import inspect
import json
import logging
import uuid as _uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, get_args, get_origin

from docstring_parser import parse as parse_docstring

from .context import Context, set_current_context
from .exceptions import ConfigurationError
from ._telemetry import setup_module_logger
from .agent.events import ToolCallFailed

logger = setup_module_logger(__name__)

T = TypeVar("T")
ToolHandler = Callable[..., Awaitable[T]]


def _serialize_for_span(value: Any) -> str:
    """Serialize a value to JSON string for span attributes.

    Handles Pydantic models, dataclasses, and other complex types.

    Args:
        value: The value to serialize

    Returns:
        JSON string representation of the value
    """
    if value is None:
        return "null"

    # Handle Pydantic models (v2 API)
    if hasattr(value, 'model_dump'):
        return json.dumps(value.model_dump())

    # Handle Pydantic models (v1 API)
    if hasattr(value, 'dict') and hasattr(value, '__fields__'):
        return json.dumps(value.dict())

    # Handle dataclasses
    if dc.is_dataclass(value) and not isinstance(value, type):
        return json.dumps(dc.asdict(value))

    # Default JSON serialization
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """
    Convert Python type to JSON Schema.

    Args:
        py_type: Python type annotation

    Returns:
        JSON Schema dict with type and potentially items/properties
    """
    # Handle None/NoneType
    if py_type is None or py_type is type(None):
        return {"type": "null"}

    # Get origin and args for generic types
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle Optional[T] -> unwrap to T
    if origin is type(None.__class__):  # Union type
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        # Multiple non-None types -> just use first one
        if non_none_args:
            return _python_type_to_json_schema(non_none_args[0])
        return {"type": "null"}

    # Handle List[T] - need to specify items
    if origin is list or (origin is None and py_type is list):
        if args:
            # List[T] - extract T and create items schema
            item_type = args[0]
            return {
                "type": "array",
                "items": _python_type_to_json_schema(item_type)
            }
        else:
            # Bare list without type parameter
            return {
                "type": "array",
                "items": {"type": "string"}  # Default to string items
            }

    # Handle Dict[K, V] - basic object schema
    if origin is dict or (origin is None and py_type is dict):
        return {"type": "object"}

    # Handle basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        Any: "string",  # Default to string for Any
    }

    # Direct type match
    json_type = type_map.get(py_type, "string")
    return {"type": json_type}


def _extract_schema_from_function(func: Callable) -> Dict[str, Any]:
    """
    Extract JSON schema from function signature and docstring.

    Args:
        func: Function to extract schema from

    Returns:
        Dict containing input_schema and output_schema
    """
    # Parse function signature
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    parsed_doc = parse_docstring(docstring)

    # Build parameter schemas
    properties = {}
    required = []

    # Build mapping from param name to docstring description
    param_descriptions = {}
    if parsed_doc.params:
        for param_doc in parsed_doc.params:
            param_descriptions[param_doc.arg_name] = param_doc.description or ""

    for param_name, param in sig.parameters.items():
        # Skip 'ctx' parameter (Context is auto-injected)
        if param_name == "ctx":
            continue

        # Get type annotation
        param_type = param.annotation
        if param_type == inspect.Parameter.empty:
            param_type = Any

        # Get description from docstring
        description = param_descriptions.get(param_name, "")

        # Build parameter schema using full JSON Schema conversion
        param_schema = _python_type_to_json_schema(param_type)
        param_schema["description"] = description

        properties[param_name] = param_schema

        # Check if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Build input schema
    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }

    # Extract return type for output schema (optional for basic tool functionality)
    return_type = sig.return_annotation
    output_schema = None
    if return_type != inspect.Parameter.empty:
        output_schema = _python_type_to_json_schema(return_type)

    return {
        "input_schema": input_schema,
        "output_schema": output_schema
    }


class Tool:
    """
    Represents a tool that agents can use.

    Tools wrap functions with automatic schema extraction and provide
    a structured interface for agent invocation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: Optional[Dict[str, Any]] = None,
        confirmation: bool = False,
        auto_schema: bool = False
    ):
        """
        Initialize a Tool.

        Args:
            name: Tool name
            description: Tool description for agents
            handler: Function that implements the tool
            input_schema: Manual JSON schema for input parameters
            confirmation: Whether tool requires human confirmation before execution
            auto_schema: Whether to automatically extract schema from handler
        """
        self.name = name
        self.description = description
        self.handler = handler
        self.confirmation = confirmation

        # Extract or use provided schema
        if auto_schema:
            schemas = _extract_schema_from_function(handler)
            self.input_schema = schemas["input_schema"]
            self.output_schema = schemas.get("output_schema")
        else:
            self.input_schema = input_schema or {"type": "object", "properties": {}}
            self.output_schema = None

        # Validate handler signature
        self._validate_handler()

        logger.debug(f"Created tool '{name}' with auto_schema={auto_schema}")

    def _validate_handler(self) -> None:
        """Validate that handler has correct signature."""
        sig = inspect.signature(self.handler)
        params = list(sig.parameters.values())

        if not params:
            raise ConfigurationError(
                f"Tool handler '{self.name}' must have at least one parameter (ctx: Context)"
            )

        first_param = params[0]
        if first_param.annotation != Context and first_param.annotation != inspect.Parameter.empty:
            logger.warning(
                f"Tool handler '{self.name}' first parameter should be 'ctx: Context', "
                f"got '{first_param.annotation}'"
            )

    async def invoke(self, ctx: Context, **kwargs) -> Any:
        """
        Invoke the tool with given arguments.

        Args:
            ctx: Execution context
            **kwargs: Tool arguments matching input_schema

        Returns:
            Tool execution result

        Raises:
            ConfigurationError: If tool requires confirmation (not yet implemented)

        Note:
            If memoization is enabled on the context, this method will check
            the journal for cached results before executing and cache results
            after successful execution.
        """
        # Check for memoization before tool execution
        step_key = None
        content_hash = None
        memo = None

        if hasattr(ctx, '_memo') and ctx._memo:
            memo = ctx._memo
            step_key, content_hash = memo.tool_call_key(self.name, kwargs)

            # Check cache first - skip execution if cached
            found, cached_result = await memo.get_cached_tool_result(step_key, content_hash)
            if found:
                logger.debug(f"Tool call {step_key} served from memoization cache")
                return cached_result

        if self.confirmation:
            # TODO: Implement actual confirmation workflow
            # For now, just log a warning
            logger.warning(
                f"Tool '{self.name}' requires confirmation but confirmation is not yet implemented"
            )

        # Emit tool event from any context (not just workflow)
        from .context import get_current_context

        context = get_current_context()
        # Generate correlation_id for pairing tool.invoked ↔ tool.completed/failed
        tool_correlation_id = f"tool-{_uuid.uuid4().hex[:12]}"
        # TODO: Add ToolInvoked typed event (different from ToolCallStarted which is for agent→tool)
        # if context:
        #     context.emit(ToolInvoked(...))

        # Set context in task-local storage for automatic propagation to nested calls
        token = set_current_context(ctx)
        try:
            try:
                # Create span for tool execution (uses contextvar for async-safe parent-child linking)
                from .tracing import create_span

                logger.debug(f"Invoking tool '{self.name}' with args: {list(kwargs.keys())}")

                # Create span with runtime_context for parent-child span linking
                # contextvar handles proper nesting for parallel async operations
                with create_span(
                    self.name,
                    "tool",
                    ctx._runtime_context if hasattr(ctx, "_runtime_context") else None,
                    {
                        "tool.name": self.name,
                        "tool.args": ",".join(kwargs.keys()),
                        "input.data": _serialize_for_span(kwargs),
                    },
                ) as span:
                    # Handler is already async (validated in tool() decorator)
                    result = await self.handler(ctx, **kwargs)

                    # Add output data to span for trace visibility
                    span.set_attribute("output.data", _serialize_for_span(result))

                    logger.debug(f"Tool '{self.name}' completed successfully")

                    # TODO: Add ToolCompleted typed event
                    # if context:
                    #     context.emit(ToolCompleted(...))

                    # Cache result for replay if memoization is enabled
                    if memo and step_key:
                        await memo.cache_tool_result(step_key, content_hash, result)

                    return result
            except Exception as e:
                # Emit error event for observability
                if context:
                    context.emit(ToolCallFailed(
                        name=self.name,
                        correlation_id=tool_correlation_id,
                        parent_correlation_id=context._correlation_id,
                        error_code=type(e).__name__,
                        error_message=str(e),
                        tool_name=self.name,
                        tool_call_id=tool_correlation_id,
                        metadata={"name": self.name},
                    ))
                raise
        finally:
            # Always reset context to prevent leakage
            from .context import _current_context
            _current_context.reset(token)

    def get_schema(self) -> Dict[str, Any]:
        """
        Get complete tool schema for agent consumption.

        Returns:
            Dict with name, description, and input_schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "requires_confirmation": self.confirmation
        }


class ToolRegistry:
    """Global registry for tools."""

    _tools: Dict[str, Tool] = {}

    @classmethod
    def register(cls, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in cls._tools:
            logger.warning(f"Overwriting existing tool '{tool.name}'")
        cls._tools[tool.name] = tool
        logger.debug(f"Registered tool '{tool.name}'")

    @classmethod
    def get(cls, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return cls._tools.get(name)

    @classmethod
    def all(cls) -> Dict[str, Tool]:
        """Get all registered tools."""
        return cls._tools.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (for testing)."""
        cls._tools.clear()
        logger.debug("Cleared tool registry")

    @classmethod
    def list_names(cls) -> List[str]:
        """Get list of all tool names."""
        return list(cls._tools.keys())


def tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    auto_schema: bool = True,
    confirmation: bool = False,
    input_schema: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to mark a function as a tool with automatic schema extraction.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to first line of docstring)
        auto_schema: Automatically extract schema from type hints and docstring
        confirmation: Whether tool requires confirmation before execution
        input_schema: Manual schema (only if auto_schema=False)

    Returns:
        Decorated function that can be invoked as a tool

    Example:
        ```python
        @tool(auto_schema=True)
        def search_web(ctx: Context, query: str, max_results: int = 10) -> List[Dict]:
            \"\"\"Search the web for information.

            Args:
                query: The search query string
                max_results: Maximum number of results to return

            Returns:
                List of search results
            \"\"\"
            # Implementation
            return results
        ```
    """
    def decorator(func: Callable) -> Callable:
        # Determine tool name
        tool_name = name or func.__name__

        # Extract description from docstring if not provided
        tool_description = description
        if tool_description is None:
            docstring = inspect.getdoc(func)
            if docstring:
                parsed_doc = parse_docstring(docstring)
                tool_description = parsed_doc.short_description or parsed_doc.long_description or ""
            else:
                tool_description = ""

        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ConfigurationError(
                f"Tool function '{func.__name__}' must have at least one parameter (ctx: Context)"
            )

        first_param = params[0]
        if first_param.annotation != Context and first_param.annotation != inspect.Parameter.empty:
            raise ConfigurationError(
                f"Tool function '{func.__name__}' first parameter must be 'ctx: Context', "
                f"got '{first_param.annotation}'"
            )

        # Convert sync to async if needed
        if not asyncio.iscoroutinefunction(func):
            original_func = func

            @functools.wraps(original_func)
            async def async_wrapper(*args, **kwargs):
                # Run sync function in thread pool to prevent blocking event loop
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: original_func(*args, **kwargs))

            handler_func = async_wrapper
        else:
            handler_func = func

        # Create Tool instance
        tool_instance = Tool(
            name=tool_name,
            description=tool_description,
            handler=handler_func,
            input_schema=input_schema,
            confirmation=confirmation,
            auto_schema=auto_schema
        )

        # Register tool
        ToolRegistry.register(tool_instance)

        # Return wrapper that invokes tool
        @functools.wraps(func)
        async def tool_wrapper(*args, **kwargs) -> Any:
            """Wrapper that invokes tool with context."""
            # If called with Context as first arg, use tool.invoke
            if args and isinstance(args[0], Context):
                ctx = args[0]
                return await tool_instance.invoke(ctx, **kwargs)

            # Otherwise, direct call (for testing)
            return await handler_func(*args, **kwargs)

        # Attach tool instance to wrapper for inspection
        tool_wrapper._tool = tool_instance

        return tool_wrapper

    if _func is None:
        return decorator
    return decorator(_func)


# ============================================================================
# Built-in Human-in-the-Loop Tools
# ============================================================================

class AskUserTool(Tool):
    """
    Built-in tool that agents can use to request text input from users.

    This tool pauses the workflow execution and waits for the user to provide
    a text response. The workflow resumes when the user submits their input.

    Example:
        ```python
        from agnt5 import Agent, workflow, WorkflowContext
        from agnt5.tool import AskUserTool

        @workflow(chat=True)
        async def agent_with_hitl(ctx: WorkflowContext, query: str) -> dict:
            agent = Agent(
                name="research_agent",
                model="openai/gpt-4o-mini",
                instructions="You are a research assistant.",
                tools=[AskUserTool(ctx)]
            )

            result = await agent.run_sync(query, context=ctx)
            return {"response": result.output}
        ```
    """

    def __init__(self, context: Optional["WorkflowContext"] = None):  # type: ignore
        """
        Initialize AskUserTool.

        Args:
            context: Optional workflow context with wait_for_user capability.
                     If not provided, will attempt to get from task-local contextvar.
        """
        # Import here to avoid circular dependency
        from .workflow import WorkflowContext

        if context is not None and not isinstance(context, WorkflowContext):
            raise ConfigurationError(
                "AskUserTool requires a WorkflowContext. "
                "This tool can only be used within workflows."
            )

        super().__init__(
            name="ask_user",
            description="Ask the user a question and wait for their text response",
            handler=self._handler,
            auto_schema=True
        )
        self.context = context

    async def _handler(self, ctx: Context, question: str) -> str:
        """
        Ask user a question and wait for their response.

        Args:
            ctx: Execution context (may contain WorkflowContext via contextvar)
            question: Question to ask the user

        Returns:
            User's text response
        """
        # Import here to avoid circular dependency
        from .workflow import WorkflowContext
        from .context import get_current_context

        # Use explicit context if provided during __init__
        workflow_ctx = self.context

        # If not provided, try to get from task-local contextvar
        if workflow_ctx is None:
            current = get_current_context()
            if isinstance(current, WorkflowContext):
                workflow_ctx = current
            elif hasattr(current, '_workflow_entity'):
                # Current context has workflow entity (is WorkflowContext)
                workflow_ctx = current  # type: ignore

        if workflow_ctx is None:
            raise ConfigurationError(
                "AskUserTool requires WorkflowContext. "
                "Either pass context to __init__ or ensure tool is used within a workflow."
            )

        return await workflow_ctx.wait_for_user(question, input_type="text")


class RequestApprovalTool(Tool):
    """
    Built-in tool that agents can use to request approval from users.

    This tool pauses the workflow execution and presents an approval request
    to the user with approve/reject options. The workflow resumes when the
    user makes a decision.

    Example:
        ```python
        from agnt5 import Agent, workflow, WorkflowContext
        from agnt5.tool import RequestApprovalTool

        @workflow(chat=True)
        async def deployment_agent(ctx: WorkflowContext, changes: dict) -> dict:
            agent = Agent(
                name="deploy_agent",
                model="openai/gpt-4o-mini",
                instructions="You help deploy code changes safely.",
                tools=[RequestApprovalTool(ctx)]
            )

            result = await agent.run_sync(
                f"Review and deploy these changes: {changes}",
                context=ctx
            )
            return {"response": result.output}
        ```
    """

    def __init__(self, context: Optional["WorkflowContext"] = None):  # type: ignore
        """
        Initialize RequestApprovalTool.

        Args:
            context: Optional workflow context with wait_for_user capability.
                     If not provided, will attempt to get from task-local contextvar.
        """
        # Import here to avoid circular dependency
        from .workflow import WorkflowContext

        if context is not None and not isinstance(context, WorkflowContext):
            raise ConfigurationError(
                "RequestApprovalTool requires a WorkflowContext. "
                "This tool can only be used within workflows."
            )

        super().__init__(
            name="request_approval",
            description="Request user approval for an action before proceeding",
            handler=self._handler,
            auto_schema=True
        )
        self.context = context

    async def _handler(self, ctx: Context, action: str, details: str = "") -> str:
        """
        Request approval from user for an action.

        Args:
            ctx: Execution context (may contain WorkflowContext via contextvar)
            action: The action requiring approval
            details: Additional details about the action

        Returns:
            "approve" or "reject" based on user's decision
        """
        # Import here to avoid circular dependency
        from .workflow import WorkflowContext
        from .context import get_current_context

        # Use explicit context if provided during __init__
        workflow_ctx = self.context

        # If not provided, try to get from task-local contextvar
        if workflow_ctx is None:
            current = get_current_context()
            if isinstance(current, WorkflowContext):
                workflow_ctx = current
            elif hasattr(current, '_workflow_entity'):
                # Current context has workflow entity (is WorkflowContext)
                workflow_ctx = current  # type: ignore

        if workflow_ctx is None:
            raise ConfigurationError(
                "RequestApprovalTool requires WorkflowContext. "
                "Either pass context to __init__ or ensure tool is used within a workflow."
            )

        question = f"Action: {action}"
        if details:
            question += f"\n\nDetails:\n{details}"
        question += "\n\nDo you approve?"

        return await workflow_ctx.wait_for_user(
            question,
            input_type="approval",
            options=[
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"}
            ]
        )
