"""Tests for Worker component registration."""

import tempfile
from pathlib import Path
from typing import List

import pytest

from agnt5 import Agent, Worker, function, tool, workflow
from agnt5.context import Context
from agnt5.entity import Entity
from agnt5.function import FunctionRegistry
from agnt5.workflow import WorkflowRegistry
from agnt5.entity import EntityRegistry
from agnt5.agent import AgentRegistry
from agnt5.tool import ToolRegistry


class TestWorkerExplicitRegistration:
    """Test explicit component registration."""

    def setup_method(self) -> None:
        """Clear all registries before each test."""
        FunctionRegistry.clear()
        WorkflowRegistry.clear()
        EntityRegistry.clear()
        AgentRegistry.clear()
        ToolRegistry.clear()

    def test_worker_with_no_components(self) -> None:
        """Test worker initialization with no components."""
        worker = Worker(
            service_name="test-service",
            coordinator_endpoint="http://localhost:34186"
        )

        assert worker.service_name == "test-service"
        assert worker._explicit_components['functions'] == []
        assert worker._explicit_components['workflows'] == []
        assert worker._explicit_components['entities'] == []
        assert worker._explicit_components['agents'] == []
        assert worker._explicit_components['tools'] == []

    def test_worker_with_explicit_functions(self) -> None:
        """Test worker with explicit function registration."""
        @function
        async def test_func(ctx: Context, value: str) -> str:
            return f"Result: {value}"

        worker = Worker(
            service_name="test-service",
            functions=[test_func]
        )

        assert len(worker._explicit_components['functions']) == 1
        assert test_func in worker._explicit_components['functions']

    def test_worker_with_explicit_workflows(self) -> None:
        """Test worker with explicit workflow registration."""
        @workflow
        async def test_workflow(ctx, data: str) -> str:
            return f"Processed: {data}"

        worker = Worker(
            service_name="test-service",
            workflows=[test_workflow]
        )

        assert len(worker._explicit_components['workflows']) == 1
        assert test_workflow in worker._explicit_components['workflows']

    def test_worker_with_explicit_entities(self) -> None:
        """Test worker with explicit entity registration."""
        class TestEntity(Entity):
            async def get_value(self) -> str:
                return self.state.get("value", "default")

        worker = Worker(
            service_name="test-service",
            entities=[TestEntity]
        )

        assert len(worker._explicit_components['entities']) == 1
        assert TestEntity in worker._explicit_components['entities']

    def test_worker_with_explicit_agents(self) -> None:
        """Test worker with explicit agent registration."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="Test agent"
        )

        worker = Worker(
            service_name="test-service",
            agents=[agent]
        )

        assert len(worker._explicit_components['agents']) == 1
        assert agent in worker._explicit_components['agents']

    def test_worker_with_explicit_tools(self) -> None:
        """Test worker with explicit tool registration."""
        @tool
        async def test_tool(ctx: Context, value: int) -> int:
            """Test tool."""
            return value * 2

        worker = Worker(
            service_name="test-service",
            tools=[test_tool._tool]  # Access the tool instance
        )

        assert len(worker._explicit_components['tools']) == 1

    def test_worker_with_all_component_types(self) -> None:
        """Test worker with all component types registered."""
        @function
        async def func1(ctx: Context) -> str:
            return "test"

        @workflow
        async def workflow1(ctx) -> str:
            return "test"

        class Entity1(Entity):
            async def method1(self) -> str:
                return "test"

        agent1 = Agent(name="agent1", model="openai/gpt-4o-mini", instructions="Test")

        @tool
        async def tool1(ctx: Context) -> str:
            """Tool 1."""
            return "test"

        worker = Worker(
            service_name="test-service",
            functions=[func1],
            workflows=[workflow1],
            entities=[Entity1],
            agents=[agent1],
            tools=[tool1._tool]
        )

        assert len(worker._explicit_components['functions']) == 1
        assert len(worker._explicit_components['workflows']) == 1
        assert len(worker._explicit_components['entities']) == 1
        assert len(worker._explicit_components['agents']) == 1
        assert len(worker._explicit_components['tools']) == 1


class TestWorkerIncrementalRegistration:
    """Test incremental component registration via register_components()."""

    def setup_method(self) -> None:
        """Clear all registries before each test."""
        FunctionRegistry.clear()
        WorkflowRegistry.clear()
        EntityRegistry.clear()
        AgentRegistry.clear()
        ToolRegistry.clear()

    def test_incremental_function_registration(self) -> None:
        """Test adding functions after worker initialization."""
        @function
        async def func1(ctx: Context) -> str:
            return "1"

        @function
        async def func2(ctx: Context) -> str:
            return "2"

        worker = Worker(service_name="test-service", functions=[func1])
        assert len(worker._explicit_components['functions']) == 1

        worker.register_components(functions=[func2])
        assert len(worker._explicit_components['functions']) == 2
        assert func1 in worker._explicit_components['functions']
        assert func2 in worker._explicit_components['functions']

    def test_incremental_multiple_component_types(self) -> None:
        """Test adding multiple component types incrementally."""
        @function
        async def func1(ctx: Context) -> str:
            return "test"

        @workflow
        async def workflow1(ctx) -> str:
            return "test"

        worker = Worker(service_name="test-service")
        assert len(worker._explicit_components['functions']) == 0
        assert len(worker._explicit_components['workflows']) == 0

        worker.register_components(functions=[func1], workflows=[workflow1])
        assert len(worker._explicit_components['functions']) == 1
        assert len(worker._explicit_components['workflows']) == 1


class TestWorkerHybridAutoInclude:
    """Test hybrid auto-include of agent tools."""

    def setup_method(self) -> None:
        """Clear all registries before each test."""
        FunctionRegistry.clear()
        WorkflowRegistry.clear()
        EntityRegistry.clear()
        AgentRegistry.clear()
        ToolRegistry.clear()

    def test_agent_tools_auto_included(self) -> None:
        """Test that agent tools are automatically included."""
        @tool
        async def agent_tool(ctx: Context, x: int) -> int:
            """Agent's tool."""
            return x * 2

        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="Test agent",
            tools=[agent_tool]
        )

        worker = Worker(
            service_name="test-service",
            agents=[agent]
        )

        # Agent should be registered
        assert len(worker._explicit_components['agents']) == 1

        # Tools should be empty (not explicitly registered)
        assert len(worker._explicit_components['tools']) == 0

        # But when we discover components, tools should be auto-included
        components = worker._discover_components()

        # Should have 1 agent + 1 auto-included tool
        tool_components = [c for c in components if c.component_type == "tool"]
        assert len(tool_components) == 1
        assert tool_components[0].name == "agent_tool"

    def test_multiple_agents_with_shared_tools(self) -> None:
        """Test multiple agents can share tools (no duplicates)."""
        @tool
        async def shared_tool(ctx: Context) -> str:
            """Shared tool."""
            return "shared"

        agent1 = Agent(
            name="agent1",
            model="openai/gpt-4o-mini",
            instructions="Agent 1",
            tools=[shared_tool]
        )

        agent2 = Agent(
            name="agent2",
            model="openai/gpt-4o-mini",
            instructions="Agent 2",
            tools=[shared_tool]
        )

        worker = Worker(
            service_name="test-service",
            agents=[agent1, agent2]
        )

        components = worker._discover_components()

        # Should have 2 agents but only 1 tool (deduplicated)
        agent_components = [c for c in components if c.component_type == "agent"]
        tool_components = [c for c in components if c.component_type == "tool"]

        assert len(agent_components) == 2
        assert len(tool_components) == 1

    def test_explicit_tools_plus_agent_tools(self) -> None:
        """Test explicit tools + auto-included agent tools."""
        @tool
        async def standalone_tool(ctx: Context) -> str:
            """Standalone tool."""
            return "standalone"

        @tool
        async def agent_tool(ctx: Context) -> str:
            """Agent's tool."""
            return "agent"

        agent = Agent(
            name="agent",
            model="openai/gpt-4o-mini",
            instructions="Test",
            tools=[agent_tool]
        )

        worker = Worker(
            service_name="test-service",
            agents=[agent],
            tools=[standalone_tool._tool]
        )

        components = worker._discover_components()
        tool_components = [c for c in components if c.component_type == "tool"]

        # Should have 2 tools: 1 explicit + 1 auto-included
        assert len(tool_components) == 2
        tool_names = {c.name for c in tool_components}
        assert "standalone_tool" in tool_names
        assert "agent_tool" in tool_names


class TestWorkerAutoRegistration:
    """Test auto-registration from pyproject.toml."""

    def setup_method(self) -> None:
        """Clear all registries before each test."""
        FunctionRegistry.clear()
        WorkflowRegistry.clear()
        EntityRegistry.clear()
        AgentRegistry.clear()
        ToolRegistry.clear()

    def test_discover_source_paths_from_hatch(self) -> None:
        """Test discovering source paths from Hatch config."""
        # Create temporary pyproject.toml with Hatch config
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject_path = Path(tmpdir) / "pyproject.toml"
            pyproject_path.write_text("""
[tool.hatch.build.targets.wheel]
packages = ["src/mypackage"]
            """)

            worker = Worker(service_name="test")
            paths = worker._discover_source_paths(str(pyproject_path))

            assert paths == ["src/mypackage"]

    def test_discover_source_paths_from_maturin(self) -> None:
        """Test discovering source paths from Maturin config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject_path = Path(tmpdir) / "pyproject.toml"
            pyproject_path.write_text("""
[tool.maturin]
python-source = "python"
            """)

            worker = Worker(service_name="test")
            paths = worker._discover_source_paths(str(pyproject_path))

            assert paths == ["python"]

    def test_discover_source_paths_fallback_to_src(self) -> None:
        """Test fallback to 'src' when no config found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject_path = Path(tmpdir) / "pyproject.toml"
            pyproject_path.write_text("""
[project]
name = "test"
            """)

            worker = Worker(service_name="test")
            paths = worker._discover_source_paths(str(pyproject_path))

            assert paths == ["src"]

    def test_discover_source_paths_missing_file(self) -> None:
        """Test fallback when pyproject.toml doesn't exist."""
        worker = Worker(service_name="test")
        paths = worker._discover_source_paths("/nonexistent/pyproject.toml")

        assert paths == ["src"]

    def test_auto_discover_components_with_explicit_paths(self) -> None:
        """Test auto-discovery with explicit paths."""
        # Create temporary package structure
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "testpkg"
            pkg_dir.mkdir()

            # Create __init__.py
            (pkg_dir / "__init__.py").write_text("")

            # Create module with decorated function
            (pkg_dir / "funcs.py").write_text("""
from agnt5 import function, Context

@function
async def test_func(ctx: Context) -> str:
    return "test"
            """)

            worker = Worker(
                service_name="test",
                auto_register=True,
                auto_register_paths=[str(pkg_dir)]
            )

            # Function should be auto-discovered and registered
            assert len(worker._explicit_components['functions']) >= 1
            func_names = [f.__name__ for f in worker._explicit_components['functions']]
            assert "test_func" in func_names


class TestWorkerDiscoverComponents:
    """Test _discover_components() method."""

    def setup_method(self) -> None:
        """Clear all registries before each test."""
        FunctionRegistry.clear()
        WorkflowRegistry.clear()
        EntityRegistry.clear()
        AgentRegistry.clear()
        ToolRegistry.clear()

    def test_discover_components_with_functions(self) -> None:
        """Test discovering function components."""
        @function
        async def func1(ctx: Context) -> str:
            return "test"

        worker = Worker(service_name="test", functions=[func1])
        components = worker._discover_components()

        func_components = [c for c in components if c.component_type == "function"]
        assert len(func_components) == 1
        assert func_components[0].name == "func1"

    def test_discover_components_with_workflows(self) -> None:
        """Test discovering workflow components."""
        @workflow
        async def wf1(ctx) -> str:
            return "test"

        worker = Worker(service_name="test", workflows=[wf1])
        components = worker._discover_components()

        wf_components = [c for c in components if c.component_type == "workflow"]
        assert len(wf_components) == 1
        assert wf_components[0].name == "wf1"

    def test_discover_components_with_entities(self) -> None:
        """Test discovering entity components."""
        class MyEntity(Entity):
            async def my_method(self) -> str:
                return "test"

        worker = Worker(service_name="test", entities=[MyEntity])
        components = worker._discover_components()

        entity_components = [c for c in components if c.component_type == "entity"]
        assert len(entity_components) == 1
        assert entity_components[0].name == "MyEntity"

    def test_discover_components_with_agents(self) -> None:
        """Test discovering agent components."""
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="Test"
        )

        worker = Worker(service_name="test", agents=[agent])
        components = worker._discover_components()

        agent_components = [c for c in components if c.component_type == "agent"]
        assert len(agent_components) == 1
        assert agent_components[0].name == "test_agent"

    def test_discover_components_schemas_serialized(self) -> None:
        """Test that component schemas are properly serialized to JSON."""
        @function
        async def typed_func(ctx: Context, value: str) -> dict:
            return {"result": value}

        worker = Worker(service_name="test", functions=[typed_func])
        components = worker._discover_components()

        func_components = [c for c in components if c.component_type == "function"]
        assert len(func_components) == 1

        # Should have serialized JSON schemas
        assert func_components[0].input_schema is not None
        assert func_components[0].output_schema is not None

        # Should be JSON strings
        import json
        input_schema = json.loads(func_components[0].input_schema)
        assert "properties" in input_schema
        assert "value" in input_schema["properties"]


class TestWorkerConfigurationParameters:
    """Test worker configuration parameters."""

    def test_worker_service_name_required(self) -> None:
        """Test that service_name is required."""
        with pytest.raises(TypeError):
            Worker()  # type: ignore

    def test_worker_default_parameters(self) -> None:
        """Test default parameter values."""
        worker = Worker(service_name="test")

        assert worker.service_name == "test"
        assert worker.service_version == "1.0.0"
        assert worker.runtime == "standalone"
        assert worker.metadata == {}

    def test_worker_custom_parameters(self) -> None:
        """Test custom parameter values."""
        worker = Worker(
            service_name="custom-service",
            service_version="2.0.0",
            coordinator_endpoint="http://custom:8080",
            runtime="docker",
            metadata={"env": "production"}
        )

        assert worker.service_name == "custom-service"
        assert worker.service_version == "2.0.0"
        assert worker.coordinator_endpoint == "http://custom:8080"
        assert worker.runtime == "docker"
        assert worker.metadata == {"env": "production"}

    def test_worker_auto_register_default_false(self) -> None:
        """Test that auto_register defaults to False."""
        worker = Worker(service_name="test")

        # Should use explicit registration (empty components)
        assert worker._explicit_components['functions'] == []
        assert worker._explicit_components['workflows'] == []
