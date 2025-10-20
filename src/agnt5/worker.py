"""Worker implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any, Dict, List, Optional

from .function import FunctionRegistry
from .workflow import WorkflowRegistry
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Context variable to store trace metadata for propagation to LM calls
# This allows Rust LM layer to access traceparent without explicit parameter passing
_trace_metadata: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    '_trace_metadata', default={}
)


class Worker:
    """AGNT5 Worker for registering and running functions/workflows with the coordinator.

    The Worker class manages the lifecycle of your service, including:
    - Registration with the AGNT5 coordinator
    - Automatic discovery of @function and @workflow decorated handlers
    - Message handling and execution
    - Health monitoring

    Example:
        ```python
        from agnt5 import Worker, function

        @function
        async def process_data(ctx: Context, data: str) -> dict:
            return {"result": data.upper()}

        async def main():
            worker = Worker(
                service_name="data-processor",
                service_version="1.0.0",
                coordinator_endpoint="http://localhost:34186"
            )
            await worker.run()

        if __name__ == "__main__":
            asyncio.run(main())
        ```
    """

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        coordinator_endpoint: Optional[str] = None,
        runtime: str = "standalone",
        metadata: Optional[Dict[str, str]] = None,
        # Explicit component registration (hybrid approach)
        functions: Optional[List] = None,
        workflows: Optional[List] = None,
        entities: Optional[List] = None,
        agents: Optional[List] = None,
        tools: Optional[List] = None,
        # Auto-registration (development mode)
        auto_register: bool = False,
        auto_register_paths: Optional[List[str]] = None,
        pyproject_path: Optional[str] = None,
    ):
        """Initialize a new Worker with explicit or automatic component registration.

        The Worker supports two registration modes:

        **Explicit Mode (default, production):**
        - Register workflows/agents explicitly, their dependencies are auto-included
        - Optionally register standalone functions/tools for direct API invocation

        **Auto-Registration Mode (development):**
        - Automatically discovers all decorated components in source paths
        - Reads source paths from pyproject.toml or uses explicit paths
        - No need to maintain import lists

        Args:
            service_name: Unique name for this service
            service_version: Version string (semantic versioning recommended)
            coordinator_endpoint: Coordinator endpoint URL (default: from env AGNT5_COORDINATOR_ENDPOINT)
            runtime: Runtime type - "standalone", "docker", "kubernetes", etc.
            metadata: Optional service-level metadata
            functions: List of @function decorated handlers (explicit mode)
            workflows: List of @workflow decorated handlers (explicit mode)
            entities: List of Entity classes (explicit mode)
            agents: List of Agent instances (explicit mode)
            tools: List of Tool instances (explicit mode)
            auto_register: Enable automatic component discovery (default: False)
            auto_register_paths: Explicit source paths to scan (overrides pyproject.toml discovery)
            pyproject_path: Path to pyproject.toml (default: current directory)

        Example (explicit mode - production):
            ```python
            from agnt5 import Worker
            from my_service import greet_user, order_fulfillment, ShoppingCart, analyst_agent

            worker = Worker(
                service_name="my-service",
                workflows=[order_fulfillment],
                entities=[ShoppingCart],
                agents=[analyst_agent],
                functions=[greet_user],
            )
            await worker.run()
            ```

        Example (auto-register mode - development):
            ```python
            from agnt5 import Worker

            worker = Worker(
                service_name="my-service",
                auto_register=True,  # Discovers from pyproject.toml
            )
            await worker.run()
            ```
        """
        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        self.runtime = runtime
        self.metadata = metadata or {}

        # Import Rust worker
        try:
            from ._core import PyWorker, PyWorkerConfig, PyComponentInfo
            self._PyWorker = PyWorker
            self._PyWorkerConfig = PyWorkerConfig
            self._PyComponentInfo = PyComponentInfo
        except ImportError as e:
            raise ImportError(
                f"Failed to import Rust core worker: {e}. "
                "Make sure agnt5 is properly installed with: pip install agnt5"
            )

        # Create Rust worker config
        self._rust_config = self._PyWorkerConfig(
            service_name=service_name,
            service_version=service_version,
            service_type=runtime,
        )

        # Create Rust worker instance
        self._rust_worker = self._PyWorker(self._rust_config)

        # Create worker-scoped entity state manager
        from .entity import EntityStateManager
        self._entity_state_manager = EntityStateManager()

        # Component registration: auto-discover or explicit
        if auto_register:
            # Auto-registration mode: discover from source paths
            if auto_register_paths:
                source_paths = auto_register_paths
                logger.info(f"Auto-registration with explicit paths: {source_paths}")
            else:
                source_paths = self._discover_source_paths(pyproject_path)
                logger.info(f"Auto-registration with discovered paths: {source_paths}")

            # Auto-discover components (will populate _explicit_components)
            self._auto_discover_components(source_paths)
        else:
            # Explicit registration from constructor kwargs
            self._explicit_components = {
                'functions': list(functions or []),
                'workflows': list(workflows or []),
                'entities': list(entities or []),
                'agents': list(agents or []),
                'tools': list(tools or []),
            }

            # Count explicitly registered components
            total_explicit = sum(len(v) for v in self._explicit_components.values())
            logger.info(
                f"Worker initialized: {service_name} v{service_version} (runtime: {runtime}), "
                f"{total_explicit} components explicitly registered"
            )

    def register_components(
        self,
        functions=None,
        workflows=None,
        entities=None,
        agents=None,
        tools=None,
    ):
        """Register additional components after Worker initialization.

        This method allows incremental registration of components after the Worker
        has been created. Useful for conditional or dynamic component registration.

        Args:
            functions: List of functions decorated with @function
            workflows: List of workflows decorated with @workflow
            entities: List of entity classes
            agents: List of agent instances
            tools: List of tool instances

        Example:
            ```python
            worker = Worker(service_name="my-service")

            # Register conditionally
            if feature_enabled:
                worker.register_components(workflows=[advanced_workflow])
            ```
        """
        if functions:
            self._explicit_components['functions'].extend(functions)
            logger.debug(f"Incrementally registered {len(functions)} functions")

        if workflows:
            self._explicit_components['workflows'].extend(workflows)
            logger.debug(f"Incrementally registered {len(workflows)} workflows")

        if entities:
            self._explicit_components['entities'].extend(entities)
            logger.debug(f"Incrementally registered {len(entities)} entities")

        if agents:
            self._explicit_components['agents'].extend(agents)
            logger.debug(f"Incrementally registered {len(agents)} agents")

        if tools:
            self._explicit_components['tools'].extend(tools)
            logger.debug(f"Incrementally registered {len(tools)} tools")

        total = sum(len(v) for v in self._explicit_components.values())
        logger.info(f"Total components now registered: {total}")

    def _discover_source_paths(self, pyproject_path: Optional[str] = None) -> List[str]:
        """Discover source paths from pyproject.toml.

        Reads pyproject.toml to find package source directories using:
        - Hatch: [tool.hatch.build.targets.wheel] packages
        - Maturin: [tool.maturin] python-source
        - Fallback: ["src"] if not found

        Args:
            pyproject_path: Path to pyproject.toml (default: current directory)

        Returns:
            List of directory paths to scan (e.g., ["src/agnt5_benchmark"])
        """
        from pathlib import Path

        # Python 3.11+ has tomllib in stdlib
        try:
            import tomllib
        except ImportError:
            logger.error("tomllib not available (Python 3.11+ required for auto-registration)")
            return ["src"]

        # Determine pyproject.toml location
        if pyproject_path:
            pyproject_file = Path(pyproject_path)
        else:
            # Look in current directory
            pyproject_file = Path.cwd() / "pyproject.toml"

        if not pyproject_file.exists():
            logger.warning(
                f"pyproject.toml not found at {pyproject_file}, "
                f"defaulting to 'src/' directory"
            )
            return ["src"]

        # Parse pyproject.toml
        try:
            with open(pyproject_file, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {e}")
            return ["src"]

        # Extract source paths based on build system
        source_paths = []

        # Try Hatch configuration
        if "tool" in config and "hatch" in config["tool"]:
            hatch_config = config["tool"]["hatch"]
            if "build" in hatch_config and "targets" in hatch_config["build"]:
                wheel_config = hatch_config["build"]["targets"].get("wheel", {})
                packages = wheel_config.get("packages", [])
                source_paths.extend(packages)

        # Try Maturin configuration
        if not source_paths and "tool" in config and "maturin" in config["tool"]:
            maturin_config = config["tool"]["maturin"]
            python_source = maturin_config.get("python-source")
            if python_source:
                source_paths.append(python_source)

        # Fallback to src/
        if not source_paths:
            logger.info("No source paths in pyproject.toml, defaulting to 'src/'")
            source_paths = ["src"]

        logger.info(f"Discovered source paths from pyproject.toml: {source_paths}")
        return source_paths

    def _auto_discover_components(self, source_paths: List[str]) -> None:
        """Auto-discover components by importing all Python files in source paths.

        Args:
            source_paths: List of directory paths to scan
        """
        import importlib.util
        import sys
        from pathlib import Path

        logger.info(f"Auto-discovering components in paths: {source_paths}")

        total_modules = 0

        for source_path in source_paths:
            path = Path(source_path)

            if not path.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue

            # Recursively find all .py files
            for py_file in path.rglob("*.py"):
                # Skip __pycache__ and test files
                if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                    continue

                # Convert path to module name
                # e.g., src/agnt5_benchmark/functions.py -> agnt5_benchmark.functions
                relative_path = py_file.relative_to(path.parent)
                module_parts = list(relative_path.parts[:-1])  # Remove .py extension part
                module_parts.append(relative_path.stem)  # Add filename without .py
                module_name = ".".join(module_parts)

                # Import module (triggers decorators)
                try:
                    if module_name in sys.modules:
                        logger.debug(f"Module already imported: {module_name}")
                    else:
                        spec = importlib.util.spec_from_file_location(module_name, py_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            logger.debug(f"Auto-imported: {module_name}")
                            total_modules += 1
                except Exception as e:
                    logger.warning(f"Failed to import {module_name}: {e}")

        logger.info(f"Auto-imported {total_modules} modules")

        # Collect components from registries
        from .agent import AgentRegistry
        from .entity import EntityRegistry
        from .tool import ToolRegistry

        # Extract actual objects from registries
        functions = [cfg.handler for cfg in FunctionRegistry.all().values()]
        workflows = [cfg.handler for cfg in WorkflowRegistry.all().values()]
        entities = [et.entity_class for et in EntityRegistry.all().values()]
        agents = list(AgentRegistry.all().values())
        tools = list(ToolRegistry.all().values())

        self._explicit_components = {
            'functions': functions,
            'workflows': workflows,
            'entities': entities,
            'agents': agents,
            'tools': tools,
        }

        logger.info(
            f"Auto-discovered components: "
            f"{len(functions)} functions, "
            f"{len(workflows)} workflows, "
            f"{len(entities)} entities, "
            f"{len(agents)} agents, "
            f"{len(tools)} tools"
        )

    def _discover_components(self):
        """Discover explicit components and auto-include their dependencies.

        Hybrid approach:
        - Explicitly registered workflows/agents are processed
        - Functions called by workflows are auto-included (TODO: implement)
        - Tools used by agents are auto-included
        - Standalone functions/tools can be explicitly registered

        Returns:
            List of PyComponentInfo instances for all components
        """
        components = []
        import json

        # Import registries
        from .entity import EntityRegistry
        from .tool import ToolRegistry

        # Track all components (explicit + auto-included)
        all_functions = set(self._explicit_components['functions'])
        all_tools = set(self._explicit_components['tools'])

        # Auto-include agent tool dependencies
        for agent in self._explicit_components['agents']:
            if hasattr(agent, 'tools') and agent.tools:
                # Agent.tools is a dict of {tool_name: tool_instance}
                all_tools.update(agent.tools.values())
                logger.debug(
                    f"Auto-included {len(agent.tools)} tools from agent '{agent.name}'"
                )

        # Log registration summary
        explicit_func_count = len(self._explicit_components['functions'])
        explicit_tool_count = len(self._explicit_components['tools'])
        auto_func_count = len(all_functions) - explicit_func_count
        auto_tool_count = len(all_tools) - explicit_tool_count

        logger.info(
            f"Component registration summary: "
            f"{len(all_functions)} functions ({explicit_func_count} explicit, {auto_func_count} auto-included), "
            f"{len(self._explicit_components['workflows'])} workflows, "
            f"{len(self._explicit_components['entities'])} entities, "
            f"{len(self._explicit_components['agents'])} agents, "
            f"{len(all_tools)} tools ({explicit_tool_count} explicit, {auto_tool_count} auto-included)"
        )

        # Process functions (explicit + auto-included)
        for func in all_functions:
            config = FunctionRegistry.get(func.__name__)
            if not config:
                logger.warning(f"Function '{func.__name__}' not found in FunctionRegistry")
                continue

            input_schema_str = json.dumps(config.input_schema) if config.input_schema else None
            output_schema_str = json.dumps(config.output_schema) if config.output_schema else None
            metadata = config.metadata if config.metadata else {}

            component_info = self._PyComponentInfo(
                name=config.name,
                component_type="function",
                metadata=metadata,
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)

        # Process workflows
        for workflow in self._explicit_components['workflows']:
            config = WorkflowRegistry.get(workflow.__name__)
            if not config:
                logger.warning(f"Workflow '{workflow.__name__}' not found in WorkflowRegistry")
                continue

            input_schema_str = json.dumps(config.input_schema) if config.input_schema else None
            output_schema_str = json.dumps(config.output_schema) if config.output_schema else None
            metadata = config.metadata if config.metadata else {}

            component_info = self._PyComponentInfo(
                name=config.name,
                component_type="workflow",
                metadata=metadata,
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)

        # Process entities
        for entity_class in self._explicit_components['entities']:
            entity_type = EntityRegistry.get(entity_class.__name__)
            if not entity_type:
                logger.warning(f"Entity '{entity_class.__name__}' not found in EntityRegistry")
                continue

            # Build complete entity definition with state schema and method schemas
            entity_definition = entity_type.build_entity_definition()
            definition_str = json.dumps(entity_definition)

            # Keep minimal metadata for backward compatibility
            metadata_dict = {
                "methods": json.dumps(list(entity_type._method_schemas.keys())),
            }

            component_info = self._PyComponentInfo(
                name=entity_type.name,
                component_type="entity",
                metadata=metadata_dict,
                config={},
                input_schema=None,  # Entities don't have single input/output schemas
                output_schema=None,
                definition=definition_str,  # Complete entity definition with state and methods
            )
            components.append(component_info)
            logger.debug(f"Registered entity '{entity_type.name}' with definition")

        # Process agents
        from .agent import AgentRegistry

        for agent in self._explicit_components['agents']:
            # Register agent in AgentRegistry for execution lookup
            AgentRegistry.register(agent)
            logger.debug(f"Registered agent '{agent.name}' in AgentRegistry for execution")

            input_schema_str = json.dumps(agent.input_schema) if hasattr(agent, 'input_schema') and agent.input_schema else None
            output_schema_str = json.dumps(agent.output_schema) if hasattr(agent, 'output_schema') and agent.output_schema else None

            metadata_dict = agent.metadata if hasattr(agent, 'metadata') else {}
            if hasattr(agent, 'tools'):
                metadata_dict["tools"] = json.dumps(list(agent.tools.keys()))

            component_info = self._PyComponentInfo(
                name=agent.name,
                component_type="agent",
                metadata=metadata_dict,
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)

        # Process tools (explicit + auto-included)
        for tool in all_tools:
            input_schema_str = json.dumps(tool.input_schema) if hasattr(tool, 'input_schema') and tool.input_schema else None
            output_schema_str = json.dumps(tool.output_schema) if hasattr(tool, 'output_schema') and tool.output_schema else None

            component_info = self._PyComponentInfo(
                name=tool.name,
                component_type="tool",
                metadata={},
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)

        logger.info(f"Discovered {len(components)} total components")
        return components

    def _create_message_handler(self):
        """Create the message handler that will be called by Rust worker."""

        def handle_message(request):
            """Handle incoming execution requests - returns coroutine for Rust to await."""
            # Extract request details
            component_name = request.component_name
            component_type = request.component_type
            input_data = request.input_data

            logger.debug(
                f"Handling {component_type} request: {component_name}, input size: {len(input_data)} bytes"
            )

            # Import all registries
            from .tool import ToolRegistry
            from .entity import EntityRegistry
            from .agent import AgentRegistry

            # Route based on component type and return coroutines
            if component_type == "tool":
                tool = ToolRegistry.get(component_name)
                if tool:
                    logger.debug(f"Found tool: {component_name}")
                    # Return coroutine, don't await it
                    return self._execute_tool(tool, input_data, request)

            elif component_type == "entity":
                entity_type = EntityRegistry.get(component_name)
                if entity_type:
                    logger.debug(f"Found entity: {component_name}")
                    # Return coroutine, don't await it
                    return self._execute_entity(entity_type, input_data, request)

            elif component_type == "agent":
                agent = AgentRegistry.get(component_name)
                if agent:
                    logger.debug(f"Found agent: {component_name}")
                    # Return coroutine, don't await it
                    return self._execute_agent(agent, input_data, request)

            elif component_type == "workflow":
                workflow_config = WorkflowRegistry.get(component_name)
                if workflow_config:
                    logger.debug(f"Found workflow: {component_name}")
                    # Return coroutine, don't await it
                    return self._execute_workflow(workflow_config, input_data, request)

            elif component_type == "function":
                function_config = FunctionRegistry.get(component_name)
                if function_config:
                    logger.info(f"🔥 WORKER: Received request for function: {component_name}")
                    # Return coroutine, don't await it
                    return self._execute_function(function_config, input_data, request)

            # Not found - need to return an async error response
            error_msg = f"Component '{component_name}' of type '{component_type}' not found"
            logger.error(error_msg)

            # Create async wrapper for error response
            async def error_response():
                return self._create_error_response(request, error_msg)

            return error_response()

        return handle_message

    async def _execute_function(self, config, input_data: bytes, request):
        """Execute a function handler (supports both regular and streaming functions)."""
        import json
        import inspect
        from .context import Context
        from ._core import PyExecuteComponentResponse

        logger.info(f"🔥 WORKER: Executing function {config.name}")

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Store trace metadata in contextvar for LM calls to access
            # The Rust worker injects traceparent into request.metadata for trace propagation
            if hasattr(request, 'metadata') and request.metadata:
                _trace_metadata.set(dict(request.metadata))
                logger.debug(f"Trace metadata stored: traceparent={request.metadata.get('traceparent', 'N/A')}")

            # Create context with runtime_context for trace correlation
            ctx = Context(
                run_id=f"{self.service_name}:{config.name}",
                runtime_context=request.runtime_context,
            )

            # Create span for function execution with trace linking
            from ._core import create_span, flush_telemetry_py

            with create_span(
                config.name,
                "function",
                request.runtime_context,
                {
                    "function.name": config.name,
                    "service.name": self.service_name,
                },
            ) as span:
                # Execute function
                if input_dict:
                    result = config.handler(ctx, **input_dict)
                else:
                    result = config.handler(ctx)

                # Debug: Log what type result is
                logger.info(f"🔥 WORKER: Function result type: {type(result).__name__}, isasyncgen: {inspect.isasyncgen(result)}, iscoroutine: {inspect.iscoroutine(result)}")

            # Flush telemetry after span ends to ensure it's exported
            try:
                flush_telemetry_py()
                logger.debug("Telemetry flushed after function execution")
            except Exception as e:
                logger.warning(f"Failed to flush telemetry: {e}")

            # Check if result is an async generator (streaming function)
            if inspect.isasyncgen(result):
                # Streaming function - return list of responses
                # Rust bridge will send each response separately to coordinator
                responses = []
                chunk_index = 0

                async for chunk in result:
                    # Serialize chunk
                    chunk_data = json.dumps(chunk).encode("utf-8")

                    responses.append(PyExecuteComponentResponse(
                        invocation_id=request.invocation_id,
                        success=True,
                        output_data=chunk_data,
                        state_update=None,
                        error_message=None,
                        metadata=None,
                        is_chunk=True,
                        done=False,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1

                # Add final "done" marker
                responses.append(PyExecuteComponentResponse(
                    invocation_id=request.invocation_id,
                    success=True,
                    output_data=b"",
                    state_update=None,
                    error_message=None,
                    metadata=None,
                    is_chunk=True,
                    done=True,
                    chunk_index=chunk_index,
                ))

                logger.debug(f"Streaming function produced {len(responses)} chunks")
                return responses
            else:
                # Regular function - await and return single response
                if inspect.iscoroutine(result):
                    result = await result

                # Serialize result
                output_data = json.dumps(result).encode("utf-8")

                return PyExecuteComponentResponse(
                    invocation_id=request.invocation_id,
                    success=True,
                    output_data=output_data,
                    state_update=None,
                    error_message=None,
                    metadata=None,
                    is_chunk=False,
                    done=True,
                    chunk_index=0,
                )

        except Exception as e:
            # Include exception type for better error messages
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Function execution failed: {error_msg}", exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

    async def _execute_workflow(self, config, input_data: bytes, request):
        """Execute a workflow handler with automatic replay support."""
        import json
        from .workflow import WorkflowEntity, WorkflowContext
        from .entity import _get_state_manager
        from ._core import PyExecuteComponentResponse

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Parse replay data from request metadata for crash recovery
            completed_steps = {}
            initial_state = {}

            if hasattr(request, 'metadata') and request.metadata:
                # Parse completed steps for replay
                if "completed_steps" in request.metadata:
                    completed_steps_json = request.metadata["completed_steps"]
                    if completed_steps_json:
                        try:
                            completed_steps = json.loads(completed_steps_json)
                            logger.info(f"🔄 Replaying workflow with {len(completed_steps)} cached steps")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse completed_steps from metadata")

                # Parse initial workflow state for replay
                if "workflow_state" in request.metadata:
                    workflow_state_json = request.metadata["workflow_state"]
                    if workflow_state_json:
                        try:
                            initial_state = json.loads(workflow_state_json)
                            logger.info(f"🔄 Loaded workflow state: {len(initial_state)} keys")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse workflow_state from metadata")

            # Create WorkflowEntity for state management
            workflow_entity = WorkflowEntity(run_id=f"{self.service_name}:{config.name}")

            # Load replay data into entity if provided
            if completed_steps:
                workflow_entity._completed_steps = completed_steps
                logger.debug(f"Loaded {len(completed_steps)} completed steps into workflow entity")

            if initial_state:
                # Load initial state into entity's state manager
                state_manager = _get_state_manager()
                state_manager._states[workflow_entity._state_key] = initial_state
                logger.debug(f"Loaded initial state with {len(initial_state)} keys into workflow entity")

            # Create WorkflowContext with entity and runtime_context for trace correlation
            ctx = WorkflowContext(
                workflow_entity=workflow_entity,
                run_id=f"{self.service_name}:{config.name}",
                runtime_context=request.runtime_context,
            )

            # Create span for workflow execution with trace linking
            from ._core import create_span, flush_telemetry_py

            with create_span(
                config.name,
                "workflow",
                request.runtime_context,
                {
                    "workflow.name": config.name,
                    "service.name": self.service_name,
                },
            ) as span:
                # Execute workflow
                if input_dict:
                    result = await config.handler(ctx, **input_dict)
                else:
                    result = await config.handler(ctx)

            # Flush telemetry after span ends to ensure it's exported
            try:
                flush_telemetry_py()
                logger.debug("Telemetry flushed after workflow execution")
            except Exception as e:
                logger.warning(f"Failed to flush telemetry: {e}")

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            # Collect workflow execution metadata for durability
            metadata = {}

            # Add step events to metadata (for workflow durability)
            # Access _step_events from the workflow entity, not the context
            step_events = ctx._workflow_entity._step_events
            if step_events:
                metadata["step_events"] = json.dumps(step_events)
                logger.debug(f"Workflow has {len(step_events)} recorded steps")

            # Add final state snapshot to metadata (if state was used)
            # Check if _state was initialized without triggering property getter
            if hasattr(ctx, '_workflow_entity') and ctx._workflow_entity._state is not None:
                if ctx._workflow_entity._state.has_changes():
                    state_snapshot = ctx._workflow_entity._state.get_state_snapshot()
                    metadata["workflow_state"] = json.dumps(state_snapshot)
                    logger.debug(f"Workflow state snapshot: {state_snapshot}")

            logger.info(f"Workflow completed successfully with {len(step_events)} steps")

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,  # Not used for workflows (use metadata instead)
                error_message=None,
                metadata=metadata if metadata else None,  # Include step events + state
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

        except Exception as e:
            # Include exception type for better error messages
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Workflow execution failed: {error_msg}", exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

    async def _execute_tool(self, tool, input_data: bytes, request):
        """Execute a tool handler."""
        import json
        from .context import Context
        from ._core import PyExecuteComponentResponse

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Create context with runtime_context for trace correlation
            ctx = Context(
                run_id=f"{self.service_name}:{tool.name}",
                runtime_context=request.runtime_context,
            )

            # Execute tool
            result = await tool.invoke(ctx, **input_dict)

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,
                error_message=None,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

        except Exception as e:
            # Include exception type for better error messages
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Tool execution failed: {error_msg}", exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

    async def _execute_entity(self, entity_type, input_data: bytes, request):
        """Execute an entity method."""
        import json
        from .context import Context
        from .entity import EntityType, Entity, _entity_state_manager_ctx
        from ._core import PyExecuteComponentResponse

        # Set entity state manager in context for Entity instances to access
        _entity_state_manager_ctx.set(self._entity_state_manager)

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Extract entity key and method name from input
            entity_key = input_dict.pop("key", None)
            method_name = input_dict.pop("method", None)

            if not entity_key:
                raise ValueError("Entity invocation requires 'key' parameter")
            if not method_name:
                raise ValueError("Entity invocation requires 'method' parameter")

            # Load state from platform if provided in request metadata
            state_key = (entity_type.name, entity_key)
            if hasattr(request, 'metadata') and request.metadata:
                if "entity_state" in request.metadata:
                    platform_state_json = request.metadata["entity_state"]
                    platform_version = int(request.metadata.get("state_version", "0"))

                    # Load platform state into state manager
                    self._entity_state_manager.load_state_from_platform(
                        state_key,
                        platform_state_json,
                        platform_version
                    )
                    logger.info(
                        f"Loaded entity state from platform: {entity_type.name}/{entity_key} "
                        f"(version {platform_version})"
                    )

            # Create entity instance using the stored class reference
            entity_instance = entity_type.entity_class(key=entity_key)

            # Get method
            if not hasattr(entity_instance, method_name):
                raise ValueError(f"Entity '{entity_type.name}' has no method '{method_name}'")

            method = getattr(entity_instance, method_name)

            # Execute method
            result = await method(**input_dict)

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            # Capture entity state after execution with version tracking
            state_dict, expected_version, new_version = \
                self._entity_state_manager.get_state_for_persistence(state_key)

            metadata = {}
            if state_dict:
                # Serialize state as JSON string for platform persistence
                state_json = json.dumps(state_dict)
                # Pass in metadata for Worker Coordinator to publish
                metadata = {
                    "entity_state": state_json,
                    "entity_type": entity_type.name,
                    "entity_key": entity_key,
                    "expected_version": str(expected_version),
                    "new_version": str(new_version),
                }
                logger.info(
                    f"Captured entity state: {entity_type.name}/{entity_key} "
                    f"(version {expected_version} → {new_version})"
                )

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,  # TODO: Use structured StateUpdate object
                error_message=None,
                metadata=metadata,  # Include state in metadata for Worker Coordinator
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

        except Exception as e:
            # Include exception type for better error messages
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Entity execution failed: {error_msg}", exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

    async def _execute_agent(self, agent, input_data: bytes, request):
        """Execute an agent."""
        import json
        from .context import Context
        from ._core import PyExecuteComponentResponse

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Extract user message
            user_message = input_dict.get("message", "")
            if not user_message:
                raise ValueError("Agent invocation requires 'message' parameter")

            # Create context with runtime_context for trace correlation
            ctx = Context(
                run_id=f"{self.service_name}:{agent.name}",
                runtime_context=request.runtime_context,
            )

            # Execute agent
            agent_result = await agent.run(user_message, context=ctx)

            # Build response
            result = {
                "output": agent_result.output,
                "tool_calls": agent_result.tool_calls,
            }

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,
                error_message=None,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

        except Exception as e:
            # Include exception type for better error messages
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Agent execution failed: {error_msg}", exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
                is_chunk=False,
                done=True,
                chunk_index=0,
            )

    def _create_error_response(self, request, error_message: str):
        """Create an error response."""
        from ._core import PyExecuteComponentResponse

        return PyExecuteComponentResponse(
            invocation_id=request.invocation_id,
            success=False,
            output_data=b"",
            state_update=None,
            error_message=error_message,
            metadata=None,
            is_chunk=False,
            done=True,
            chunk_index=0,
        )

    async def run(self):
        """Run the worker (register and start message loop).

        This method will:
        1. Discover all registered @function and @workflow handlers
        2. Register with the coordinator
        3. Enter the message processing loop
        4. Block until shutdown

        This is the main entry point for your worker service.
        """
        logger.info(f"Starting worker: {self.service_name}")

        # Discover components
        components = self._discover_components()

        # Set components on Rust worker
        self._rust_worker.set_components(components)

        # Set metadata
        if self.metadata:
            self._rust_worker.set_service_metadata(self.metadata)

        # Set message handler
        handler = self._create_message_handler()
        self._rust_worker.set_message_handler(handler)

        # Initialize worker
        self._rust_worker.initialize()

        logger.info("Worker registered successfully, entering message loop...")

        # Run worker (this will block until shutdown)
        await self._rust_worker.run()

        logger.info("Worker shutdown complete")
