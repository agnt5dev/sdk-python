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
        """Discover all registered functions and workflows."""
        components = []

        # Discover functions
        for name, config in FunctionRegistry.all().items():
            component_info = self._PyComponentInfo(
                name=name,
                component_type="function",
                metadata={},
                config={},
                input_schema=None,
                output_schema=None,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered function: {name}")

        # Discover workflows
        for name, config in WorkflowRegistry.all().items():
            component_info = self._PyComponentInfo(
                name=name,
                component_type="workflow",
                metadata={},
                config={},
                input_schema=None,
                output_schema=None,
                definition=None,
            )
            components.append(component_info)
            logger.debug(f"Discovered workflow: {name}")

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

                # Look up the component - try both registries
                # Check workflow registry first, then function registry
                workflow_config = WorkflowRegistry.get(component_name)
                if workflow_config:
                    logger.debug(f"Found workflow: {component_name}")
                    result = asyncio.run(self._execute_workflow(workflow_config, input_data, request))
                    return result

                function_config = FunctionRegistry.get(component_name)
                if function_config:
                    logger.debug(f"Found function: {component_name}")
                    result = asyncio.run(self._execute_function(function_config, input_data, request))
                    return result

                # Not found in either registry
                error_msg = f"Component '{component_name}' not found in function or workflow registry"
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
