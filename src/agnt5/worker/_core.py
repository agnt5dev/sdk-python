"""Worker implementation for AGNT5 SDK.

Supports functions, entities, workflows, agents, and tools.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .. import _sentry
from .._serialization import serialize_to_str
from .._telemetry import setup_module_logger
from ..function import FunctionRegistry

# from ..workflow import WorkflowRegistry  # COMMENTED OUT - functions only for now
from ._executors import ExecutorMixin

logger = setup_module_logger(__name__)


class Worker(ExecutorMixin):
    """AGNT5 Worker for registering and executing the components.

    The Worker class manages the lifecycle of your service, including:
    - Registration with the AGNT5 platform
    - Automatic discovery of components
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
        coordinator_endpoint: str | None = None,
        runtime: str = "standalone",
        metadata: dict[str, str] | None = None,
        functions: list | None = None,
        workflows: list | None = None,
        entities: list | None = None,
        agents: list | None = None,
        tools: list | None = None,
        auto_register: bool = False,
        auto_register_paths: list[str] | None = None,
        pyproject_path: str | None = None,
    ):
        """Initialize a new Worker with explicit or automatic component registration.

        The Worker supports two registration modes:

        **Explicit Mode:**
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
        """
        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        self.runtime = runtime
        self.metadata = metadata or {}

        # Import Rust worker
        try:
            from .._core import PyComponentInfo, PyWorker, PyWorkerConfig

            self._PyWorker = PyWorker
            self._PyWorkerConfig = PyWorkerConfig
            self._PyComponentInfo = PyComponentInfo
        except ImportError as e:
            _sentry.capture_exception(
                e,
                context={
                    "service_name": service_name,
                    "service_version": service_version,
                    "error_location": "Worker.__init__",
                    "error_phase": "rust_core_import",
                },
                tags={
                    "sdk_error": "true",
                    "error_type": "import_error",
                    "component": "rust_core",
                },
                level="error",
            )
            raise ImportError(
                f"Failed to import Rust core worker: {e}. "
                "Make sure agnt5 is properly installed with: pip install agnt5"
            ) from e

        # Create Rust worker
        self._rust_config = self._PyWorkerConfig(
            service_name=service_name,
            service_version=service_version,
            service_type=runtime,
        )
        self._rust_worker = self._PyWorker(self._rust_config)

        # Create entity state adapter
        # TODO: tenant_id should be handled by Rust core, but it still requires it for now
        from .._core import EntityStateManager as RustEntityStateManager
        from ..entity import EntityStateAdapter

        rust_core = RustEntityStateManager(tenant_id="default")
        self._entity_state_adapter = EntityStateAdapter(rust_core=rust_core)
        logger.info("Created EntityStateAdapter with Rust core for state management")

        # Create CheckpointClient for step-level memoization
        try:
            from ..checkpoint import CheckpointClient

            self._checkpoint_client = CheckpointClient()
            logger.info("Created CheckpointClient for step-level memoization")
        except Exception as e:
            logger.warning(f"Failed to create CheckpointClient (memoization disabled): {e}")
            self._checkpoint_client = None

        # Initialize Sentry
        from ..version import _get_version

        sdk_version = _get_version()
        sentry_enabled = _sentry.initialize_sentry(
            service_name=service_name,
            service_version=service_version,
            sdk_version=sdk_version,
        )
        if sentry_enabled:
            _sentry.set_context(
                "service",
                {
                    "name": service_name,
                    "version": service_version,
                    "runtime": runtime,
                },
            )
        else:
            logger.debug("SDK telemetry not enabled")

        # Component registration
        if auto_register:
            if any([functions, workflows, entities, agents, tools]):
                logger.warning(
                    "auto_register=True ignores explicit functions/workflows/entities/agents/tools parameters. "
                    "Remove explicit params or set auto_register=False to use explicit registration."
                )

            if auto_register_paths:
                source_paths = auto_register_paths
                logger.info(f"Auto-registration with explicit paths: {source_paths}")
            else:
                source_paths = self._discover_source_paths(pyproject_path)
                logger.info(f"Auto-registration with discovered paths: {source_paths}")

            self._auto_discover_components(source_paths)
        else:
            self._explicit_components = {
                "functions": list(functions or []),
                "workflows": list(workflows or []),
                "entities": list(entities or []),
                "agents": list(agents or []),
                "tools": list(tools or []),
            }

            total_explicit = sum(len(v) for v in self._explicit_components.values())
            logger.info(
                f"Worker initialized: {service_name} v{service_version} (runtime: {runtime}), "
                f"{total_explicit} components explicitly registered"
            )

    def register_components(
        self,
        functions: list | None = None,
        workflows: list | None = None,
        entities: list | None = None,
        agents: list | None = None,
        tools: list | None = None,
    ) -> None:
        """Register additional components after Worker initialization.

        This method allows incremental registration of components after the Worker
        has been created. Useful for conditional or dynamic component registration.

        Args:
            functions: List of functions decorated with @function
            workflows: List of workflows decorated with @workflow
            entities: List of entity classes
            agents: List of agent instances
            tools: List of tool instances
        """
        if functions:
            self._explicit_components["functions"].extend(functions)
            logger.debug(f"Registered {len(functions)} functions")

        if workflows:
            self._explicit_components["workflows"].extend(workflows)
            logger.debug(f"Registered {len(workflows)} workflows")

        if entities:
            self._explicit_components["entities"].extend(entities)
            logger.debug(f"Registered {len(entities)} entities")

        if agents:
            self._explicit_components["agents"].extend(agents)
            logger.debug(f"Registered {len(agents)} agents")

        if tools:
            self._explicit_components["tools"].extend(tools)
            logger.debug(f"Registered {len(tools)} tools")

        total = sum(len(v) for v in self._explicit_components.values())
        logger.info(f"Total components now registered: {total}")

    def _discover_source_paths(self, pyproject_path: str | None = None) -> list[str]:
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
        try:
            import tomllib
        except ImportError:
            logger.error("tomllib not available (Python 3.11+ required for auto-registration)")
            return ["src"]

        if pyproject_path:
            pyproject_file = Path(pyproject_path)
        else:
            pyproject_file = Path.cwd() / "pyproject.toml"

        if not pyproject_file.exists():
            logger.warning(
                f"pyproject.toml not found at {pyproject_file}, defaulting to 'src/' directory"
            )
            return ["src"]

        try:
            with open(pyproject_file, "rb") as f:
                import tomllib

                config = tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to parse pyproject.toml: {e}")
            return ["src"]

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

        if not source_paths:
            logger.info("No source paths in pyproject.toml, defaulting to 'src/'")
            source_paths = ["src"]

        logger.info(f"Discovered source paths from pyproject.toml: {source_paths}")
        return source_paths

    def _auto_discover_components(self, source_paths: list[str]) -> None:
        """Auto-discover components by importing all Python files in source paths.

        Args:
            source_paths: List of directory paths to scan
        """
        import importlib.util
        import sys

        logger.info(f"Auto-discovering components in paths: {source_paths}")

        total_modules = 0

        for source_path in source_paths:
            path = Path(source_path)

            if not path.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue

            for py_file in path.rglob("*.py"):
                if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                    continue

                relative_path = py_file.relative_to(path.parent)
                module_parts = list(relative_path.parts[:-1])
                module_parts.append(relative_path.stem)
                module_name = ".".join(module_parts)

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
                    _sentry.capture_exception(
                        e,
                        context={
                            "service_name": self.service_name,
                            "module_name": module_name,
                            "source_path": str(py_file),
                            "error_location": "_auto_discover_components",
                        },
                        tags={
                            "sdk_error": "true",
                            "error_type": "auto_registration_failure",
                        },
                        level="warning",
                    )

        logger.info(f"Auto-imported {total_modules} modules")

        # Collect components from registries
        from ..agent import AgentRegistry
        from ..entity import EntityRegistry
        from ..tool import ToolRegistry
        from ..workflow import WorkflowRegistry

        functions = [cfg.handler for cfg in FunctionRegistry.all().values()]
        workflows = [cfg.handler for cfg in WorkflowRegistry.all().values()]
        entities = [et.entity_class for et in EntityRegistry.all().values()]
        agents = list(AgentRegistry.all().values())
        tools = list(ToolRegistry.all().values())

        self._explicit_components = {
            "functions": functions,
            "workflows": workflows,
            "entities": entities,
            "agents": agents,
            "tools": tools,
        }

        logger.info(
            f"Auto-discovered components: {len(functions)} functions, {len(entities)} entities, "
            f"{len(workflows)} workflows, {len(agents)} agents, {len(tools)} tools"
        )

    def _serialize_schema(self, schema: Any) -> str | None:
        """Serialize a schema to JSON string, returning None if empty."""
        return serialize_to_str(schema) if schema else None

    def _create_component_info(
        self,
        name: str,
        component_type: str,
        metadata: dict | None = None,
        config: dict | None = None,
        input_schema: Any = None,
        output_schema: Any = None,
        definition: Any = None,
    ) -> Any:
        """Create a PyComponentInfo with serialized schemas."""
        return self._PyComponentInfo(
            name=name,
            component_type=component_type,
            metadata=metadata or {},
            config=config or {},
            input_schema=self._serialize_schema(input_schema),
            output_schema=self._serialize_schema(output_schema),
            definition=self._serialize_schema(definition),
        )

    def _discover_components(self) -> list:
        """Discover explicit components (functions, entities, workflows, agents, tools).

        Returns:
            List of PyComponentInfo instances for all components
        """
        components = []

        # Process functions only
        for func in self._explicit_components["functions"]:
            config = FunctionRegistry.get(func.__name__)
            if not config:
                logger.warning(f"Function '{func.__name__}' not found in FunctionRegistry")
                continue

            config_dict = {}
            if config.retries:
                config_dict.update({
                    "max_attempts": str(config.retries.max_attempts),
                    "initial_interval_ms": str(config.retries.initial_interval_ms),
                    "max_interval_ms": str(config.retries.max_interval_ms),
                })
            if config.backoff:
                config_dict.update({
                    "backoff_type": config.backoff.type.value,
                    "backoff_multiplier": str(config.backoff.multiplier),
                })

            components.append(self._create_component_info(
                name=config.name,
                component_type="function",
                metadata=config.metadata,
                config=config_dict,
                input_schema=config.input_schema,
                output_schema=config.output_schema,
            ))

        # Process entities
        from ..entity import EntityRegistry
        for entity_class in self._explicit_components["entities"]:
            entity_type = EntityRegistry.get(entity_class.__name__)
            if not entity_type:
                logger.warning(f"Entity '{entity_class.__name__}' not found in EntityRegistry")
                continue

            # Build entity definition with method schemas
            definition = entity_type.build_entity_definition()

            components.append(self._create_component_info(
                name=entity_type.name,
                component_type="entity",
                metadata={},
                config={},
                definition=definition,
            ))

        # Process workflows
        from ..workflow import WorkflowRegistry
        for workflow_handler in self._explicit_components["workflows"]:
            config = WorkflowRegistry.get(workflow_handler.__name__)
            if not config:
                logger.warning(f"Workflow '{workflow_handler.__name__}' not found in WorkflowRegistry")
                continue

            components.append(self._create_component_info(
                name=config.name,
                component_type="workflow",
                metadata=config.metadata,
                config={},
                input_schema=config.input_schema,
                output_schema=config.output_schema,
            ))

        # Process agents
        from ..agent import AgentRegistry
        for agent in self._explicit_components["agents"]:
            # Build agent definition with tool schemas
            tool_schemas = []
            for tool_name, tool in agent.tools.items():
                tool_schemas.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })

            definition = {
                "instructions": agent.instructions,
                "model": agent.model,
                "max_iterations": agent.max_iterations,
                "tools": tool_schemas,
                "handoffs": [h.agent.name for h in agent.handoffs] if agent.handoffs else [],
            }

            components.append(self._create_component_info(
                name=agent.name,
                component_type="agent",
                metadata={},
                config={
                    "model": agent.model,
                    "max_iterations": str(agent.max_iterations),
                },
                definition=definition,
            ))

        # Process tools
        from ..tool import ToolRegistry
        for tool in self._explicit_components["tools"]:
            components.append(self._create_component_info(
                name=tool.name,
                component_type="tool",
                metadata={},
                config={
                    "confirmation": str(tool.confirmation),
                },
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
            ))

        logger.info(f"Discovered {len(components)} components (functions + entities + workflows + agents + tools)")
        return components

    def _create_message_handler(self) -> Any:
        """Create the message handler that will be called by Rust worker.

        Handles function, entity, and workflow components.
        """
        def handle_message(request: Any) -> Any:
            """Handle incoming execution requests - returns coroutine for Rust to await."""
            component_name = request.component_name
            component_type = request.component_type
            input_data = request.input_data

            logger.debug(
                f"Handling {component_type} request: {component_name}, "
                f"input size: {len(input_data)} bytes"
            )

            # Functions
            if component_type == "function":
                function_config = FunctionRegistry.get(component_name)
                if function_config:
                    return self._execute_function(function_config, input_data, request)

            # Entities
            elif component_type == "entity":
                from ..entity import EntityRegistry
                entity_type = EntityRegistry.get(component_name)
                if entity_type:
                    return self._execute_entity(entity_type, input_data, request)

            # Workflows
            elif component_type == "workflow":
                from ..workflow import WorkflowRegistry
                workflow_config = WorkflowRegistry.get(component_name)
                if workflow_config:
                    return self._execute_workflow(workflow_config, input_data, request)

            # Tools
            elif component_type == "tool":
                from ..tool import ToolRegistry
                tool = ToolRegistry.get(component_name)
                if tool:
                    return self._execute_tool(tool, input_data, request)

            # Agents
            elif component_type == "agent":
                from ..agent import AgentRegistry
                agent = AgentRegistry.get(component_name)
                if agent:
                    return self._execute_agent(agent, input_data, request)

            # Not found or unsupported
            error_msg = f"Component '{component_name}' of type '{component_type}' not found"
            logger.error(error_msg)

            async def error_response():
                return self._create_error_response(request, error_msg)

            return error_response()

        return handle_message

    async def run(self) -> None:
        """Run the worker (register and start message loop).

        This method will:
        1. Discover all registered @function and @workflow handlers
        2. Register with the coordinator
        3. Create a shared Python event loop for all function executions
        4. Enter the message processing loop
        5. Block until shutdown

        This is the main entry point for your worker service.
        """
        try:
            logger.info(f"Starting worker: {self.service_name}")

            components = self._discover_components()
            self._rust_worker.set_components(components)

            if self.metadata:
                self._rust_worker.set_service_metadata(self.metadata)

            # Configure entity state manager
            logger.info("Configuring Rust EntityStateManager for database persistence")
            if (
                hasattr(self._entity_state_adapter, "_rust_core")
                and self._entity_state_adapter._rust_core
            ):
                self._rust_worker.set_entity_state_manager(self._entity_state_adapter._rust_core)
                logger.info("Successfully configured Rust EntityStateManager")

            loop = asyncio.get_running_loop()
            logger.info("Passing Python event loop to Rust worker for concurrent execution")

            self._rust_worker.set_event_loop(loop)
            handler = self._create_message_handler()
            self._rust_worker.set_message_handler(handler)
            self._rust_worker.initialize()

            logger.info("Worker registered successfully, entering message loop...")

            await self._rust_worker.run()

        except Exception as e:
            logger.error(
                f"Worker failed to start or encountered critical error: {e}",
                exc_info=True,
            )
            _sentry.capture_exception(
                e,
                context={
                    "service_name": self.service_name,
                    "service_version": self.service_version,
                    "error_location": "Worker.run",
                    "error_phase": "worker_lifecycle",
                },
                tags={
                    "sdk_error": "true",
                    "error_type": "worker_failure",
                    "severity": "critical",
                },
                level="error",
            )
            raise

        finally:
            logger.info("Flushing Sentry events before shutdown...")
            _sentry.flush(timeout=5.0)
            logger.info("Worker shutdown complete")
