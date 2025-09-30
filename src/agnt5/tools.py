"""Tool decorator and registry for agent capabilities."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ._core import (
    clear_tools as _core_clear_tools,
    get_registered_tools as _core_get_registered_tools,
    register_tool as _core_register_tool,
)

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, "ToolMetadata"] = {}


@dataclass
class ToolMetadata:
    name: str
    description: Optional[str]
    input_schema: Optional[str]
    output_schema: Optional[str]


def register_tool_metadata(
    *,
    name: str,
    description: Optional[str],
    input_schema: Optional[str],
    output_schema: Optional[str],
) -> None:
    if name in _REGISTRY:
        return
    _REGISTRY[name] = ToolMetadata(
        name=name,
        description=description.strip() if description else None,
        input_schema=input_schema,
        output_schema=output_schema,
    )
    _core_register_tool(name, description, input_schema, output_schema)


def get_registered_tools() -> Dict[str, ToolMetadata]:
    tools = dict(_REGISTRY)
    for definition in _core_get_registered_tools():
        tools.setdefault(
            definition.name,
            ToolMetadata(
                name=definition.name,
                description=definition.description,
                input_schema=definition.input_schema,
                output_schema=definition.output_schema,
            ),
        )
    return tools


def clear_registered_tools() -> None:
    _REGISTRY.clear()
    _core_clear_tools()


def tool(
    auto_schema: bool = True,
    confirmation: bool = False,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    """Decorator for creating tools with automatic schema extraction.

    Args:
        auto_schema: Extract schema from type hints and docstrings (default: True)
        confirmation: Require user approval before execution (default: False)
        name: Override tool name (default: function name)
        description: Override tool description (default: from docstring)

    Example:
        ```python
        @tool(auto_schema=True)
        def search_web(query: str, max_results: int = 10) -> List[Dict[str, str]]:
            '''Search the web for information.

            Args:
                query: The search query string
                max_results: Maximum number of results to return

            Returns:
                List of search results with title, url, and snippet
            '''
            return search_results
        ```
    """

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_description = description

        # Extract description from docstring if not provided
        if not tool_description and func.__doc__:
            # Take first line of docstring as description
            doc_lines = func.__doc__.strip().split("\n")
            tool_description = doc_lines[0].strip() if doc_lines else None

        # Extract schema if auto_schema is enabled
        input_schema_dict = None
        output_schema_dict = None

        if auto_schema:
            try:
                from .schema_extractor import extract_function_schema

                schema_info = extract_function_schema(func)
                input_schema_dict = schema_info.get("input_schema", {})
                output_schema_dict = schema_info.get("output_schema", {})

                # Use long description from schema if available
                if not tool_description:
                    tool_description = schema_info.get("description", "")

            except Exception as e:
                logger.warning(f"Failed to extract schema for tool '{tool_name}': {e}")

        # Serialize schemas to JSON strings
        input_schema = json.dumps(input_schema_dict) if input_schema_dict else None
        output_schema = json.dumps(output_schema_dict) if output_schema_dict else None

        # Register tool metadata
        register_tool_metadata(
            name=tool_name,
            description=tool_description,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        # Mark function with tool metadata
        func._tool_name = tool_name  # type: ignore[attr-defined]
        func._tool_auto_schema = auto_schema  # type: ignore[attr-defined]
        func._tool_confirmation = confirmation  # type: ignore[attr-defined]
        func._tool_description = tool_description  # type: ignore[attr-defined]
        func._tool_input_schema = input_schema_dict  # type: ignore[attr-defined]
        func._tool_output_schema = output_schema_dict  # type: ignore[attr-defined]

        logger.debug(
            f"Registered tool '{tool_name}' with auto_schema={auto_schema}, confirmation={confirmation}"
        )

        return func

    return decorator


class Tool:
    """Tool class for manual tool construction.

    Use this when you need more control than the @tool decorator provides.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable,
        confirmation: bool = False,
    ):
        """Create a tool manually.

        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for input validation
            handler: Callable that implements the tool
            confirmation: Require user approval before execution
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler
        self.confirmation = confirmation

        # Register with metadata system
        register_tool_metadata(
            name=name,
            description=description,
            input_schema=json.dumps(input_schema) if input_schema else None,
            output_schema=None,
        )


__all__ = [
    "tool",
    "Tool",
    "ToolMetadata",
    "register_tool_metadata",
    "get_registered_tools",
    "clear_registered_tools",
]
