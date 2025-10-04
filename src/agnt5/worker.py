"""Worker implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from .function import FunctionRegistry
from .workflow import WorkflowRegistry

logger = logging.getLogger(__name__)


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
    ):
        """Initialize a new Worker.

        Args:
            service_name: Unique name for this service
            service_version: Version string (semantic versioning recommended)
            coordinator_endpoint: Coordinator endpoint URL (default: from env AGNT5_COORDINATOR_ENDPOINT)
            runtime: Runtime type - "standalone", "docker", "kubernetes", etc.
            metadata: Optional service-level metadata
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

        logger.info(
            f"Worker initialized: {service_name} v{service_version} (runtime: {runtime})"
        )

    def _discover_components(self):
        """Discover all registered components across all registries."""
        components = []

        # Import all registries
        from .tool import ToolRegistry
        from .entity import EntityRegistry
        from .agent import AgentRegistry

        # Discover functions
        import json
        for name, config in FunctionRegistry.all().items():
            # Serialize schemas to JSON strings
            input_schema_str = None
            if config.input_schema:
                input_schema_str = json.dumps(config.input_schema)

            output_schema_str = None
            if config.output_schema:
                output_schema_str = json.dumps(config.output_schema)

            # Get metadata with description
            metadata = config.metadata if config.metadata else {}

            component_info = self._PyComponentInfo(
                name=name,
                component_type="function",
                metadata=metadata,
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered function: {name}")

        # Discover workflows
        for name, config in WorkflowRegistry.all().items():
            # Serialize schemas to JSON strings
            input_schema_str = None
            if config.input_schema:
                input_schema_str = json.dumps(config.input_schema)

            output_schema_str = None
            if config.output_schema:
                output_schema_str = json.dumps(config.output_schema)

            # Get metadata with description
            metadata = config.metadata if config.metadata else {}

            component_info = self._PyComponentInfo(
                name=name,
                component_type="workflow",
                metadata=metadata,
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered workflow: {name}")

        # Discover tools
        for name, tool in ToolRegistry.all().items():
            # Serialize schemas to JSON strings
            input_schema_str = None
            if hasattr(tool, 'input_schema') and tool.input_schema:
                input_schema_str = json.dumps(tool.input_schema)

            output_schema_str = None
            if hasattr(tool, 'output_schema') and tool.output_schema:
                output_schema_str = json.dumps(tool.output_schema)

            component_info = self._PyComponentInfo(
                name=name,
                component_type="tool",
                metadata={},
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered tool: {name}")

        # Discover entities
        for name, entity_type in EntityRegistry.all().items():
            # Build method schemas and metadata for each method
            method_schemas = {}
            for method_name, (input_schema, output_schema) in entity_type._method_schemas.items():
                method_metadata = entity_type._method_metadata.get(method_name, {})
                method_schemas[method_name] = {
                    "input_schema": input_schema,
                    "output_schema": output_schema,
                    "metadata": method_metadata
                }

            # Build metadata dict with methods list and schemas
            metadata_dict = {
                "methods": json.dumps(list(entity_type._methods.keys())),
                "method_schemas": json.dumps(method_schemas)
            }

            component_info = self._PyComponentInfo(
                name=name,
                component_type="entity",
                metadata=metadata_dict,
                config={},
                input_schema=None,  # Entities have per-method schemas in metadata
                output_schema=None,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered entity: {name} with methods: {list(entity_type._methods.keys())}")

        # Discover agents
        for name, agent in AgentRegistry.all().items():
            # Serialize schemas to JSON strings
            input_schema_str = None
            if hasattr(agent, 'input_schema') and agent.input_schema:
                input_schema_str = json.dumps(agent.input_schema)

            output_schema_str = None
            if hasattr(agent, 'output_schema') and agent.output_schema:
                output_schema_str = json.dumps(agent.output_schema)

            # Get metadata (includes description and model info)
            metadata_dict = agent.metadata if hasattr(agent, 'metadata') else {}
            # Add tools list to metadata
            if hasattr(agent, 'tools'):
                metadata_dict["tools"] = json.dumps(list(agent.tools.keys()))

            component_info = self._PyComponentInfo(
                name=name,
                component_type="agent",
                metadata=metadata_dict,
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered agent: {name}")

        # Discover tools
        for name, tool in ToolRegistry.all().items():
            # Serialize schemas to JSON strings
            input_schema_str = None
            if hasattr(tool, 'input_schema') and tool.input_schema:
                input_schema_str = json.dumps(tool.input_schema)

            output_schema_str = None
            if hasattr(tool, 'output_schema') and tool.output_schema:
                output_schema_str = json.dumps(tool.output_schema)

            component_info = self._PyComponentInfo(
                name=name,
                component_type="tool",
                metadata={},
                config={},
                input_schema=input_schema_str,
                output_schema=output_schema_str,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered tool: {name}")

        # Discover entities
        for name, entity_type in EntityRegistry.all().items():
            # Build metadata dict with methods list as JSON string
            metadata_dict = {
                "methods": json.dumps(list(entity_type._methods.keys()))
            }

            component_info = self._PyComponentInfo(
                name=name,
                component_type="entity",
                metadata=metadata_dict,
                config={},
                input_schema=None,
                output_schema=None,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered entity: {name} with methods: {list(entity_type._methods.keys())}")

        # Discover agents
        for name, agent in AgentRegistry.all().items():
            # Build metadata dict with agent info
            metadata_dict = {
                "model": agent.model_name,
                "tools": json.dumps(list(agent.tools.keys()) if hasattr(agent, 'tools') else [])
            }

            component_info = self._PyComponentInfo(
                name=name,
                component_type="agent",
                metadata=metadata_dict,
                config={},
                input_schema=None,
                output_schema=None,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered agent: {name}")

        logger.info(f"Discovered {len(components)} components")
        return components

    def _create_message_handler(self):
        """Create the message handler that will be called by Rust worker."""

        def handle_message(request):
            """Handle incoming execution requests."""
            try:
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

                # Route based on component type
                if component_type == "tool":
                    tool = ToolRegistry.get(component_name)
                    if tool:
                        logger.debug(f"Found tool: {component_name}")
                        result = asyncio.run(self._execute_tool(tool, input_data, request))
                        return result

                elif component_type == "entity":
                    entity_type = EntityRegistry.get(component_name)
                    if entity_type:
                        logger.debug(f"Found entity: {component_name}")
                        result = asyncio.run(self._execute_entity(entity_type, input_data, request))
                        return result

                elif component_type == "agent":
                    agent = AgentRegistry.get(component_name)
                    if agent:
                        logger.debug(f"Found agent: {component_name}")
                        result = asyncio.run(self._execute_agent(agent, input_data, request))
                        return result

                elif component_type == "workflow":
                    workflow_config = WorkflowRegistry.get(component_name)
                    if workflow_config:
                        logger.debug(f"Found workflow: {component_name}")
                        result = asyncio.run(self._execute_workflow(workflow_config, input_data, request))
                        return result

                elif component_type == "function":
                    function_config = FunctionRegistry.get(component_name)
                    if function_config:
                        logger.debug(f"Found function: {component_name}")
                        result = asyncio.run(self._execute_function(function_config, input_data, request))
                        return result

                # Not found
                error_msg = f"Component '{component_name}' of type '{component_type}' not found"
                logger.error(error_msg)
                return self._create_error_response(request, error_msg)

            except Exception as e:
                error_msg = f"Handler error: {e}"
                logger.error(error_msg, exc_info=True)
                return self._create_error_response(request, error_msg)

        return handle_message

    async def _execute_function(self, config, input_data: bytes, request):
        """Execute a function handler."""
        import json
        from .context import Context
        from ._core import PyExecuteComponentResponse

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Create context
            ctx = Context(
                run_id=f"{self.service_name}:{config.name}",
                component_type="function",
            )

            # Execute function
            if input_dict:
                result = await config.handler(ctx, **input_dict)
            else:
                result = await config.handler(ctx)

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,
                error_message=None,
                metadata=None,
            )

        except Exception as e:
            error_msg = f"Function execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
            )

    async def _execute_workflow(self, config, input_data: bytes, request):
        """Execute a workflow handler."""
        import json
        from .context import Context
        from ._core import PyExecuteComponentResponse

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Create context
            ctx = Context(
                run_id=f"{self.service_name}:{config.name}",
                component_type="workflow",
            )

            # Execute workflow
            if input_dict:
                result = await config.handler(ctx, **input_dict)
            else:
                result = await config.handler(ctx)

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,
                error_message=None,
                metadata=None,
            )

        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
            )

    async def _execute_tool(self, tool, input_data: bytes, request):
        """Execute a tool handler."""
        import json
        from .context import Context
        from ._core import PyExecuteComponentResponse

        try:
            # Parse input data
            input_dict = json.loads(input_data.decode("utf-8")) if input_data else {}

            # Create context
            ctx = Context(
                run_id=f"{self.service_name}:{tool.name}",
                component_type="tool",
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
            )

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
            )

    async def _execute_entity(self, entity_type, input_data: bytes, request):
        """Execute an entity method."""
        import json
        from .context import Context
        from ._core import PyExecuteComponentResponse

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

            # Create entity instance
            entity_instance = entity_type(entity_key)

            # Get method
            if not hasattr(entity_instance, method_name):
                raise ValueError(f"Entity '{entity_type.name}' has no method '{method_name}'")

            method = getattr(entity_instance, method_name)

            # Execute method
            result = await method(**input_dict)

            # Serialize result
            output_data = json.dumps(result).encode("utf-8")

            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=True,
                output_data=output_data,
                state_update=None,
                error_message=None,
                metadata=None,
            )

        except Exception as e:
            error_msg = f"Entity execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
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

            # Create context
            ctx = Context(
                run_id=f"{self.service_name}:{agent.name}",
                component_type="agent",
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
            )

        except Exception as e:
            error_msg = f"Agent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return PyExecuteComponentResponse(
                invocation_id=request.invocation_id,
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata=None,
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
