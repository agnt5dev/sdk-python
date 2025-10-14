"""
Tool component for Agent capabilities with automatic schema extraction.

Tools extend what agents can do by providing structured interfaces to functions,
with automatic schema generation from Python type hints and docstrings.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, get_args, get_origin

from docstring_parser import parse as parse_docstring

from .context import Context
from .exceptions import ConfigurationError
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

T = TypeVar("T")
ToolHandler = Callable[..., Awaitable[T]]


def _python_type_to_json_schema_type(py_type: Any) -> str:
    """
    Convert Python type to JSON Schema type.

    Args:
        py_type: Python type annotation

    Returns:
        JSON Schema type string
    """
    # Handle None/NoneType
    if py_type is None or py_type is type(None):
        return "null"

    # Handle string types
    origin = get_origin(py_type)

    # Handle Optional[T] -> unwrap to T
    if origin is type(None.__class__):  # Union type
        args = get_args(py_type)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema_type(non_none_args[0])
        # Multiple non-None types -> just use first one
        if non_none_args:
            return _python_type_to_json_schema_type(non_none_args[0])
        return "null"

    # Handle basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        List: "array",
        dict: "object",
        Dict: "object",
        Any: "string",  # Default to string for Any
    }

    # Check origin for generic types
    if origin is not None:
        return type_map.get(origin, "string")

    # Direct type match
    return type_map.get(py_type, "string")


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

        # Build parameter schema
        param_schema = {
            "type": _python_type_to_json_schema_type(param_type),
            "description": description
        }

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
        output_schema = {
            "type": _python_type_to_json_schema_type(return_type)
        }

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
        """
        if self.confirmation:
            # TODO: Implement actual confirmation workflow
            # For now, just log a warning
            logger.warning(
                f"Tool '{self.name}' requires confirmation but confirmation is not yet implemented"
            )

        # Create span for tool execution with trace linking
        from ._core import create_span

        logger.debug(f"Invoking tool '{self.name}' with args: {list(kwargs.keys())}")

        # Create span with runtime_context for parent-child span linking
        with create_span(
            self.name,
            "tool",
            ctx._runtime_context if hasattr(ctx, "_runtime_context") else None,
            {
                "tool.name": self.name,
                "tool.args": ",".join(kwargs.keys()),
            },
        ) as span:
            # Handler is already async (validated in tool() decorator)
            result = await self.handler(ctx, **kwargs)

            logger.debug(f"Tool '{self.name}' completed successfully")
            return result

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
