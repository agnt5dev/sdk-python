"""
Tests for Tool component.

Tests cover:
- Tool creation with auto-schema extraction
- Manual schema definition
- Tool registration and discovery
- Schema extraction from type hints and docstrings
- Tool invocation
- Error handling
"""

import json

import pytest

from agnt5 import Context, tool
from agnt5.exceptions import ConfigurationError
from agnt5.tool import (
    Tool,
    ToolRegistry,
    _extract_schema_from_function,
    _python_type_to_json_schema,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear tool registry before each test."""
    ToolRegistry.clear()
    yield
    ToolRegistry.clear()


# Test Schema Type Conversion

def test_python_type_to_json_schema():
    """Test Python type to JSON schema type conversion."""
    assert _python_type_to_json_schema(str) == {"type": "string"}
    assert _python_type_to_json_schema(int) == {"type": "integer"}
    assert _python_type_to_json_schema(float) == {"type": "number"}
    assert _python_type_to_json_schema(bool) == {"type": "boolean"}
    # list without type args defaults to array of strings
    assert _python_type_to_json_schema(list) == {"type": "array", "items": {"type": "string"}}
    assert _python_type_to_json_schema(dict) == {"type": "object"}


def test_python_type_optional():
    """Test Optional type handling."""
    from typing import Optional

    # Optional[str] should resolve to string type
    result = _python_type_to_json_schema(Optional[str])
    assert result == {"type": "string"}


# Test Schema Extraction

def test_extract_schema_basic():
    """Test schema extraction from simple function."""
    def simple_func(ctx: Context, name: str, age: int) -> str:
        """A simple function.

        Args:
            name: Person's name
            age: Person's age
        """
        return f"{name} is {age}"

    schema = _extract_schema_from_function(simple_func)

    assert "input_schema" in schema
    input_schema = schema["input_schema"]

    # Check properties
    assert "name" in input_schema["properties"]
    assert input_schema["properties"]["name"]["type"] == "string"
    assert "Person's name" in input_schema["properties"]["name"]["description"]

    assert "age" in input_schema["properties"]
    assert input_schema["properties"]["age"]["type"] == "integer"

    # Check required fields
    assert set(input_schema["required"]) == {"name", "age"}


def test_extract_schema_with_defaults():
    """Test schema extraction with default parameters."""
    def func_with_defaults(ctx: Context, query: str, limit: int = 10) -> list:
        """Function with defaults.

        Args:
            query: Search query
            limit: Maximum results
        """
        return []

    schema = _extract_schema_from_function(func_with_defaults)
    input_schema = schema["input_schema"]

    # Only 'query' should be required (limit has default)
    assert input_schema["required"] == ["query"]
    assert "limit" in input_schema["properties"]


def test_extract_schema_no_docstring():
    """Test schema extraction without docstring."""
    def no_doc_func(ctx: Context, param1: str, param2: int):
        return None

    schema = _extract_schema_from_function(no_doc_func)
    input_schema = schema["input_schema"]

    # Should still extract parameters
    assert "param1" in input_schema["properties"]
    assert "param2" in input_schema["properties"]

    # Descriptions will be empty
    assert input_schema["properties"]["param1"]["description"] == ""


# Test Tool Creation with @tool Decorator

def test_tool_decorator_auto_schema():
    """Test @tool decorator with auto_schema=True."""
    @tool
    def search_web(ctx: Context, query: str, max_results: int = 10) -> list:
        """Search the web for information.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
        """
        return []

    assert "search_web" in ToolRegistry.list_names()

    tool_instance = ToolRegistry.get("search_web")
    assert tool_instance is not None
    assert tool_instance.name == "search_web"
    assert "Search the web" in tool_instance.description

    # Check schema
    schema = tool_instance.get_schema()
    assert "query" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["query"]


def test_tool_decorator_custom_name():
    """Test @tool with custom name."""
    @tool(name="custom_search")
    def search_func(ctx: Context, query: str):
        """Search function."""
        return []

    assert "custom_search" in ToolRegistry.list_names()
    assert ToolRegistry.get("custom_search") is not None


def test_tool_decorator_custom_description():
    """Test @tool with custom description."""
    @tool(description="Custom description")
    def my_tool(ctx: Context, param: str):
        return None

    tool_instance = ToolRegistry.get("my_tool")
    assert tool_instance.description == "Custom description"


def test_tool_decorator_confirmation():
    """Test @tool with confirmation flag."""
    @tool(confirmation=True)
    def dangerous_tool(ctx: Context, action: str):
        """Perform dangerous action."""
        return "done"

    tool_instance = ToolRegistry.get("dangerous_tool")
    assert tool_instance.confirmation is True

    schema = tool_instance.get_schema()
    assert schema["requires_confirmation"] is True


def test_tool_decorator_sync_function():
    """Test @tool with sync function (auto-converts to async)."""
    @tool
    def sync_tool(ctx: Context, value: int) -> int:
        """A sync tool."""
        return value * 2

    tool_instance = ToolRegistry.get("sync_tool")
    assert tool_instance is not None

    # Handler should be async
    import asyncio
    assert asyncio.iscoroutinefunction(tool_instance.handler)


def test_tool_decorator_without_context_fails():
    """Test that @tool without Context parameter fails."""
    with pytest.raises(ConfigurationError, match="must have at least one parameter"):

        @tool
        def bad_tool():
            pass


def test_tool_decorator_wrong_first_param_fails():
    """Test that @tool with wrong first parameter fails."""
    with pytest.raises(ConfigurationError, match="first parameter must be 'ctx: Context'"):

        @tool
        def bad_tool(wrong_param: str):
            pass


# Test Tool Invocation

@pytest.mark.asyncio
async def test_tool_invocation():
    """Test invoking a tool."""
    @tool
    async def add_numbers(ctx: Context, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")
    tool_instance = ToolRegistry.get("add_numbers")

    result = await tool_instance.invoke(ctx, a=5, b=3)
    assert result == 8


@pytest.mark.asyncio
async def test_eval_tool_fault_injection_from_runtime_metadata():
    """Eval target metadata can inject deterministic tool faults."""
    @tool
    async def search(ctx: Context, query: str) -> dict:
        return {"title": query}

    ctx = Context(
        run_id="test-123",
        correlation_id="corr-123",
        parent_correlation_id="parent-123",
        trace_metadata={
            "agnt5_eval_role": "target",
            "agnt5.eval.tool_faults": json.dumps([
                {
                    "tool": "search",
                    "error_code": "SIMULATED_TIMEOUT",
                    "message": "search timed out",
                    "times": 1,
                }
            ]),
        },
    )
    with pytest.raises(RuntimeError, match="SIMULATED_TIMEOUT: search timed out"):
        await search.invoke(ctx, query="first")
    assert await search.invoke(ctx, query="second") == {"title": "second"}

    non_target = Context(
        run_id="test-456",
        correlation_id="corr-456",
        parent_correlation_id="parent-456",
        trace_metadata={
            "agnt5_eval_role": "scorer",
            "agnt5.eval.tool_faults": json.dumps([{"tool": "search"}]),
        },
    )
    assert await search.invoke(non_target, query="safe") == {"title": "safe"}


@pytest.mark.asyncio
async def test_tool_invocation_with_context_access():
    """Test tool accessing context properties."""
    @tool
    async def context_aware_tool(ctx: Context, key: str, value: str) -> dict:
        """Tool that accesses context."""
        # Context provides run_id, component_type, etc.
        return {
            "run_id": ctx.run_id,
            "stored_key": key,
            "stored_value": value
        }

    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")
    tool_instance = ToolRegistry.get("context_aware_tool")

    result = await tool_instance.invoke(ctx, key="test_key", value="test_value")

    assert result["run_id"] == "test-123"
    assert result["stored_key"] == "test_key"
    assert result["stored_value"] == "test_value"


@pytest.mark.asyncio
async def test_tool_direct_call():
    """Test calling tool function directly."""
    @tool
    async def direct_call_tool(ctx: Context, message: str) -> str:
        """A tool for direct calling."""
        return f"Echo: {message}"

    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")

    # Call through wrapper
    result = await direct_call_tool(ctx, message="Hello")
    assert result == "Echo: Hello"


# Test Tool Registry

def test_tool_registry_registration():
    """Test tool registration."""
    @tool
    def tool1(ctx: Context, param: str):
        pass

    @tool
    def tool2(ctx: Context, param: int):
        pass

    assert len(ToolRegistry.list_names()) == 2
    assert "tool1" in ToolRegistry.list_names()
    assert "tool2" in ToolRegistry.list_names()


def test_tool_registry_get():
    """Test retrieving tools from registry."""
    @tool
    def my_tool(ctx: Context, param: str):
        """My tool."""
        pass

    retrieved = ToolRegistry.get("my_tool")
    assert retrieved is not None
    assert retrieved.name == "my_tool"

    missing = ToolRegistry.get("nonexistent")
    assert missing is None


def test_tool_registry_all():
    """Test getting all tools."""
    @tool
    def tool1(ctx: Context, p: str):
        pass

    @tool
    def tool2(ctx: Context, p: int):
        pass

    all_tools = ToolRegistry.all()
    assert len(all_tools) == 2
    assert "tool1" in all_tools
    assert "tool2" in all_tools


def test_tool_registry_clear():
    """Test clearing registry."""
    @tool
    def temp_tool(ctx: Context, p: str):
        pass

    assert len(ToolRegistry.list_names()) > 0

    ToolRegistry.clear()
    assert len(ToolRegistry.list_names()) == 0


def test_tool_registry_overwrite():
    """Test that registering same name overwrites the previous tool."""
    @tool(name="duplicate")
    def tool1(ctx: Context, p: str) -> str:
        """First tool."""
        return "tool1"

    @tool(name="duplicate")
    def tool2(ctx: Context, p: int) -> int:
        """Second tool."""
        return 42

    # Second registration should overwrite first
    retrieved = ToolRegistry.get("duplicate")
    assert retrieved is not None

    # Should be the second tool (returns int, not str)
    schema = retrieved.get_schema()
    assert "Second tool" in schema["description"]


# Test Manual Schema

def test_tool_manual_schema():
    """Test creating tool with manual schema."""
    manual_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"}
        },
        "required": ["query"]
    }

    async def handler(ctx: Context, **kwargs):
        return kwargs

    tool_instance = Tool(
        name="manual_tool",
        description="Manually defined tool",
        handler=handler,
        input_schema=manual_schema,
        auto_schema=False
    )

    schema = tool_instance.get_schema()
    assert schema["input_schema"] == manual_schema
    assert schema["name"] == "manual_tool"


# Test Complex Type Annotations

def test_tool_with_list_type():
    """Test tool with List type annotation."""
    from typing import List

    @tool
    def process_items(ctx: Context, items: List[str]) -> int:
        """Process list of items.

        Args:
            items: List of items to process
        """
        return len(items)

    tool_instance = ToolRegistry.get("process_items")
    schema = tool_instance.get_schema()

    assert schema["input_schema"]["properties"]["items"]["type"] == "array"


def test_tool_with_dict_type():
    """Test tool with Dict type annotation."""
    from typing import Dict

    @tool
    def process_config(ctx: Context, config: Dict[str, str]) -> bool:
        """Process configuration.

        Args:
            config: Configuration dictionary
        """
        return True

    tool_instance = ToolRegistry.get("process_config")
    schema = tool_instance.get_schema()

    assert schema["input_schema"]["properties"]["config"]["type"] == "object"


def test_tool_with_optional_type():
    """Test tool with Optional type."""
    from typing import Optional

    @tool
    def optional_param(ctx: Context, required: str, optional: Optional[str] = None) -> str:
        """Tool with optional parameter.

        Args:
            required: Required parameter
            optional: Optional parameter
        """
        return required

    tool_instance = ToolRegistry.get("optional_param")
    schema = tool_instance.get_schema()

    # Only 'required' should be in required list
    assert schema["input_schema"]["required"] == ["required"]
    assert "optional" in schema["input_schema"]["properties"]


# Test Get Schema

def test_get_schema_format():
    """Test that get_schema returns correct format."""
    @tool(confirmation=True)
    def test_tool(ctx: Context, param: str) -> None:
        """Test tool for schema."""
        pass

    tool_instance = ToolRegistry.get("test_tool")
    schema = tool_instance.get_schema()

    # Check schema structure
    assert "name" in schema
    assert "description" in schema
    assert "input_schema" in schema
    assert "requires_confirmation" in schema

    assert schema["name"] == "test_tool"
    assert schema["requires_confirmation"] is True


# Test Edge Cases

def test_tool_no_parameters_except_context():
    """Test tool with only context parameter."""
    @tool
    def no_params(ctx: Context) -> str:
        """Tool with no parameters."""
        return "done"

    tool_instance = ToolRegistry.get("no_params")
    schema = tool_instance.get_schema()

    # Should have empty properties and required
    assert schema["input_schema"]["properties"] == {}
    assert schema["input_schema"]["required"] == []


def test_tool_complex_docstring():
    """Test tool with complex docstring."""
    @tool
    def complex_doc(ctx: Context, param1: str, param2: int) -> dict:
        """
        This is a complex tool with a detailed description.

        It does many things and has a long explanation.
        Multiple paragraphs are supported.

        Args:
            param1: First parameter with long description
                that spans multiple lines
            param2: Second parameter

        Returns:
            A dictionary with results

        Examples:
            >>> complex_doc(ctx, "test", 42)
            {"result": "success"}
        """
        return {}

    tool_instance = ToolRegistry.get("complex_doc")

    # Should extract first line as short description
    assert "complex tool" in tool_instance.description.lower()

    schema = tool_instance.get_schema()
    # Should have extracted parameter descriptions
    assert "param1" in schema["input_schema"]["properties"]
    assert "param2" in schema["input_schema"]["properties"]


# Additional Edge Cases

@pytest.mark.asyncio
async def test_tool_sync_function_execution():
    """Test that sync functions actually execute correctly when converted to async."""
    execution_log = []

    @tool
    def sync_execution_tool(ctx: Context, value: str) -> str:
        """Sync tool that logs execution."""
        execution_log.append(f"executed_{value}")
        return f"result_{value}"

    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")
    tool_instance = ToolRegistry.get("sync_execution_tool")

    result = await tool_instance.invoke(ctx, value="test")

    assert result == "result_test"
    assert "executed_test" in execution_log


def test_tool_with_any_type():
    """Test tool with Any type annotation."""
    from typing import Any

    @tool
    def any_type_tool(ctx: Context, data: Any) -> str:
        """Tool accepting any type.

        Args:
            data: Data of any type
        """
        return "processed"

    tool_instance = ToolRegistry.get("any_type_tool")
    schema = tool_instance.get_schema()

    # Any should default to string
    assert schema["input_schema"]["properties"]["data"]["type"] == "string"


def test_tool_decorator_returns_tool_instance():
    """Test that @tool decorator returns Tool instance directly."""
    @tool
    async def inspectable_tool(ctx: Context, param: str) -> str:
        """Inspectable tool."""
        return param

    # @tool decorator now returns Tool instance directly
    assert isinstance(inspectable_tool, Tool)
    assert inspectable_tool.name == "inspectable_tool"
    # Function metadata is copied to Tool
    assert inspectable_tool.__name__ == "inspectable_tool"
    assert inspectable_tool.__doc__ == "Inspectable tool."


def test_tool_without_return_type():
    """Test tool without return type annotation."""
    @tool
    def no_return_type(ctx: Context, param: str):
        """Tool without return type."""
        return "done"

    tool_instance = ToolRegistry.get("no_return_type")

    # Should still work, just no output_schema
    assert tool_instance is not None
    schema = tool_instance.get_schema()
    assert "input_schema" in schema
