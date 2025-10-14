"""Worker implementation for AGNT5 SDK."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any, Dict, Optional

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

        # Create worker-scoped entity state manager
        from .entity import EntityStateManager
        self._entity_state_manager = EntityStateManager()

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
                "methods": json.dumps(list(entity_type._method_schemas.keys())),
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
            logger.debug(f"Discovered entity: {name} with methods: {list(entity_type._method_schemas.keys())}")

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

        logger.info(f"Discovered {len(components)} components")
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
