"""
High-level Worker manager that integrates function decorators with the Rust core.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ._compat import _rust_available, _import_error

# from .agent import get_registered_agents  # REMOVED
from .function import (
    ComponentExecutionResult,
    execute_component,
    get_registered_functions,
    get_function_metadata,
)
from .workflows import get_registered_workflows
from .runtimes import WorkerRuntime, ASGIRuntime
from .logging import install_opentelemetry_logging
from .schema_extractor import extract_function_schema, extract_workflow_schema

# Core functionality import from Rust extension
from ._compat import _rust_available

if _rust_available:
    from ._core import (
        PyComponentInfo,
        PyExecuteComponentRequest,
        PyExecuteComponentResponse,
        PyStateTransition,
        PyStateUpdate,
        PyWorker,
        PyWorkerConfig,
    )

logger = logging.getLogger(__name__)


def _decode_checkpoint_result(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        if not raw:
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return raw


def _serialize_step_checkpoints(py_checkpoints: List[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for checkpoint in py_checkpoints:
        try:
            result = _decode_checkpoint_result(getattr(checkpoint, "result", b""))
            serialized.append(
                {
                    "name": getattr(checkpoint, "name", ""),
                    "key": getattr(checkpoint, "key", None),
                    "status": getattr(checkpoint, "status", ""),
                    "attempt": getattr(checkpoint, "attempt", 0),
                    "updated_at": getattr(checkpoint, "updated_at", None),
                    "result": result,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to serialize step checkpoint %s: %s", checkpoint, exc)
    return serialized


class Worker:
    """
    High-level AGNT5 Worker that automatically registers decorated functions.

    This class wraps the low-level Rust PyWorker and provides automatic
    registration of @function decorated handlers.
    """

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
        coordinator_endpoint: str = "http://localhost:34186",
        runtime: str = "standalone",
    ):
        """
        Initialize the worker.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            coordinator_endpoint: Endpoint of the coordinator service
            runtime: Runtime mode - "standalone" or "asgi"
        """
        if not _rust_available:
            raise RuntimeError(f"Rust core is required but not available: {_import_error}")

        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        self.runtime_mode = runtime

        # Create runtime adapter
        if runtime == "asgi":
            self.runtime_adapter = ASGIRuntime(worker=self)
        elif runtime == "standalone":
            self.runtime_adapter = WorkerRuntime()
        else:
            raise ValueError(f"Unknown runtime: {runtime}. Supported: 'standalone', 'asgi'")

        # Import and create Rust worker
        config = PyWorkerConfig(service_name, service_version, "python")
        self._rust_worker = PyWorker(config)

        # Note: Telemetry initialization deferred to run() method due to Tokio runtime requirement

        # Set up OpenTelemetry logging integration (handler is resilient to timing issues)
        try:
            self._otel_handler = install_opentelemetry_logging(
                logger=None,  # Install on root logger to capture all Python logs
                level=logging.INFO,
            )
            logger.info("OpenTelemetry logging integration enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry logging: {e}")
            self._otel_handler = None

        # Set the message handler - this is the simple FFI boundary
        self._rust_worker.set_message_handler(self._handle_message)

        self._running = False

        logger.info(f"Worker created: {service_name} v{service_version} (runtime: {runtime})")

    async def run(self):
        """
        Run the worker and handle decorated function invocations.

        This will:
        1. Register all decorated functions
        2. Start the underlying Rust worker
        3. Handle incoming invocations
        """
        logger.info(f"Starting worker {self.service_name}...")

        # Register all decorated functions and workflows first
        self._register_components()

        # Run the Rust worker (this will block until shutdown)
        try:
            self._running = True
            await self._rust_worker.run()
        except Exception as e:
            logger.error(f"Worker {self.service_name} failed: {e}")
            raise
        finally:
            self._running = False
            logger.info(f"Worker {self.service_name} stopped")

            # Clean up OpenTelemetry logging handler
            if hasattr(self, "_otel_handler") and self._otel_handler:
                try:
                    from .logging import remove_opentelemetry_logging

                    remove_opentelemetry_logging()
                    logger.info("OpenTelemetry logging integration cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to cleanup OpenTelemetry logging: {e}")

    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running

    def _register_components(self):
        """Register decorated functions and workflows with the Worker Coordinator."""

        from .entity import get_entity_registry

        functions = get_registered_functions()
        entities = get_entity_registry().list_entities()
        workflows = get_registered_workflows()
        service_metadata: Dict[str, str] = {}

        if not functions and not workflows and not entities:
            logger.warning("No components registered via decorators")
            return

        py_components = []

        if functions:
            logger.info(
                "Registering %d function handlers: %s", len(functions), list(functions.keys())
            )
            for handler_name, func in functions.items():
                metadata = get_function_metadata(func)
                if not metadata:
                    continue

                # Extract schema information
                schema_info = extract_function_schema(func)

                component_metadata = {
                    "handler_name": metadata.get("handler", handler_name),
                    "module": metadata.get("module", func.__module__),
                    "signature": metadata.get("signature", ""),
                    "retry_policy": json.dumps(metadata.get("retry_policy", {})),
                    "backoff_policy": json.dumps(metadata.get("backoff_policy", {})),
                    "parameters": json.dumps(metadata.get("parameters", [])),
                    "description": schema_info.get("description", ""),
                    "long_description": schema_info.get("long_description", ""),
                }

                # Extract function source as definition
                definition = None
                try:
                    import inspect

                    definition = inspect.getsource(func)
                except Exception:
                    # If we can't get source, that's okay
                    pass

                py_components.append(
                    PyComponentInfo(
                        name=handler_name,
                        component_type="function",
                        metadata=component_metadata,
                        config={},
                        input_schema=json.dumps(schema_info.get("input_schema", {})),
                        output_schema=json.dumps(schema_info.get("output_schema", {})),
                        definition=definition,
                    )
                )

        if entities:
            entity_names = list(entities.keys())
            logger.info("Registering %d entities: %s", len(entity_names), entity_names)
            for entity_name, entity_type in entities.items():
                component_metadata = {
                    "entity_type": entity_name,
                    "write_methods": list(entity_type._write_methods.keys()),
                    "shared_methods": list(entity_type._shared_methods.keys()),
                }

                py_components.append(
                    PyComponentInfo(
                        name=entity_name,
                        component_type="entity",
                        metadata=component_metadata,
                        config={},
                        input_schema=None,  # TODO: Extract entity method schemas
                        output_schema=None,
                        definition=None,
                    )
                )

        if workflows:
            workflow_names = list(workflows.keys())
            logger.info("Registering %d workflows: %s", len(workflow_names), workflow_names)
            for flow_name, definition in workflows.items():
                try:
                    flow_json = definition.to_json()
                    flow_def = definition.to_dict()
                except (TypeError, ValueError) as exc:
                    logger.error("Failed to serialize workflow '%s': %s", flow_name, exc)
                    continue

                # Extract workflow schema
                workflow_schema = extract_workflow_schema(flow_def)

                component_metadata = {
                    "flow_definition": flow_json,
                    "step_count": str(len(definition.steps)),
                    "description": definition.__doc__ or "",
                }

                service_metadata[f"workflow:{flow_name}"] = flow_json

                py_components.append(
                    PyComponentInfo(
                        name=flow_name,
                        component_type="workflow",
                        metadata=component_metadata,
                        config={},
                        input_schema=json.dumps(workflow_schema.get("input_schema", {})),
                        output_schema=json.dumps(workflow_schema.get("output_schema", {})),
                        definition=flow_json,
                    )
                )

        try:
            self._rust_worker.set_service_metadata(service_metadata)
        except Exception as exc:
            logger.error("Failed to set service metadata: %s", exc)

        if py_components:
            self._rust_worker.set_components(py_components)
            logger.info("Registered %d components with Rust worker", len(py_components))
        else:
            logger.warning("No components were registered after serialization step")

        # Function invocations are now handled through the message handler

    def _handle_message(self, request: "PyExecuteComponentRequest") -> "PyExecuteComponentResponse":
        """Handle incoming function invocation requests."""
        try:
            # Extract request data
            invocation_id = request.invocation_id
            handler_name = request.component_name
            component_type = getattr(request, "component_type", "function") or "function"
            input_data = bytes(request.input_data)

            logger.info(
                "Processing %s invocation - Handler: %s, ID: %s, Data size: %d bytes",
                component_type,
                handler_name,
                invocation_id,
                len(input_data),
            )
            if request.metadata:
                logger.debug(f"Request metadata: {dict(request.metadata)}")

            # Log input data preview for debugging
            if input_data:
                try:
                    # Try to show a preview of the input data
                    if len(input_data) > 100:
                        preview = input_data[:100].hex() + "..."
                    else:
                        preview = input_data.hex()
                    logger.debug(f"Input data hex preview: {preview}")
                except Exception:
                    logger.debug(f"Input data (raw bytes): {len(input_data)} bytes")

            # Prepare step checkpoints for the durable context
            raw_step_checkpoints = list(getattr(request, "step_checkpoints", []))
            step_checkpoints = _serialize_step_checkpoints(raw_step_checkpoints)

            # Create context for the function
            context = {
                "invocation_id": invocation_id,
                "service_name": request.service_name,
                "handler_name": handler_name,
                "metadata": request.metadata,
                "step_checkpoints": step_checkpoints,
                "component_type": component_type,
                "object_id": getattr(request, "object_id", None),
                "method_name": getattr(request, "method_name", None),
                "state_snapshot": getattr(request, "state_snapshot", b""),
                "journal_position": getattr(request, "journal_position", 0),
                "flow_instance_id": getattr(request, "flow_instance_id", None),
                "flow_step": getattr(request, "flow_step", None),
            }

            # Call the function through the decorator system
            # RuntimeAdapter is used internally by execute_component if needed
            try:
                execution = execute_component(
                    name=handler_name,
                    input_data=input_data,
                    context=context,
                    component_type=component_type,
                    method_name=getattr(request, "method_name", None),
                    object_id=getattr(request, "object_id", None),
                    state_snapshot=getattr(request, "state_snapshot", b""),
                    journal_position=getattr(request, "journal_position", 0),
                )

                logger.info("Component %s completed successfully", handler_name)

                state_update_obj = None
                if execution.state_update is not None:
                    py_transitions = [
                        PyStateTransition(
                            operation=transition.operation,
                            key=transition.key,
                            old_value=transition.old_value if transition.old_value else None,
                            new_value=transition.new_value if transition.new_value else None,
                            timestamp=transition.timestamp_ms,
                        )
                        for transition in execution.state_update.transitions
                    ]
                    state_update_obj = PyStateUpdate(
                        new_state=execution.state_update.new_state,
                        transitions=py_transitions,
                        output_data=execution.state_update.output_data,
                    )

                # Return successful response
                return PyExecuteComponentResponse(
                    invocation_id=invocation_id,
                    success=True,
                    output_data=execution.output_data,
                    state_update=state_update_obj,
                    error_message=None,
                    metadata=execution.metadata,
                )

            except Exception as e:
                error_msg = f"{component_type.title()} {handler_name} failed: {str(e)}"
                logger.error(error_msg)

                # Return error response
                return PyExecuteComponentResponse(
                    invocation_id=invocation_id,
                    success=False,
                    output_data=b"",
                    state_update=None,
                    error_message=error_msg,
                    metadata={},
                )

        except Exception as e:
            error_msg = f"Message handling failed: {str(e)}"
            logger.error(error_msg)

            # Return error response with fallback invocation_id
            return PyExecuteComponentResponse(
                invocation_id=getattr(request, "invocation_id", "unknown"),
                success=False,
                output_data=b"",
                state_update=None,
                error_message=error_msg,
                metadata={},
            )

    async def __call__(self, scope, receive, send):
        """
        ASGI application interface.

        This makes the Worker itself callable as an ASGI app when using ASGI runtime.
        """
        if self.runtime_mode != "asgi":
            raise RuntimeError("ASGI interface only available when runtime='asgi'")

        return await self.runtime_adapter(scope, receive, send)

    def enable_cors(self, origins: List[str] = None):
        """Enable CORS for ASGI runtime."""
        if self.runtime_mode == "asgi" and hasattr(self.runtime_adapter, "enable_cors"):
            self.runtime_adapter.enable_cors(origins)
        else:
            logger.warning("CORS can only be enabled for ASGI runtime")

    def disable_cors(self):
        """Disable CORS for ASGI runtime."""
        if self.runtime_mode == "asgi" and hasattr(self.runtime_adapter, "disable_cors"):
            self.runtime_adapter.disable_cors()
        else:
            logger.warning("CORS can only be disabled for ASGI runtime")
