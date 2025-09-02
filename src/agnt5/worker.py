"""
High-level Worker manager that integrates function decorators with the Rust core.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ._compat import _rust_available, _import_error
from .decorators import get_registered_functions, get_function_metadata, invoke_function
from .runtimes import WorkerRuntime, ASGIRuntime
from .logging import install_opentelemetry_logging

# Core functionality import from Rust extension
from ._compat import _rust_available

if _rust_available:
    from ._core import PyWorker, PyWorkerConfig, PyInvokeFunctionRequest, PyInvokeFunctionResponse, PyComponentInfo

logger = logging.getLogger(__name__)


class Worker:
    """
    High-level AGNT5 Worker that automatically registers decorated functions.
    
    This class wraps the low-level Rust PyWorker and provides automatic
    registration of @function decorated handlers.
    """
    
    def __init__(self, 
                 service_name: str,
                 service_version: str = "1.0.0",
                 coordinator_endpoint: str = "http://localhost:9091",
                 runtime: str = "standalone"):
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
                level=logging.INFO
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
        
        # Register all decorated functions first
        self._register_functions()
        
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
            if hasattr(self, '_otel_handler') and self._otel_handler:
                try:
                    from .logging import remove_opentelemetry_logging
                    remove_opentelemetry_logging()
                    logger.info("OpenTelemetry logging integration cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to cleanup OpenTelemetry logging: {e}")
        
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running
        
    def _register_functions(self):
        """Register all decorated functions with the Worker Coordinator."""
        functions = get_registered_functions()
        
        if not functions:
            logger.warning("No @function decorated handlers found")
            return
            
        logger.info(f"Registering {len(functions)} function handlers: {list(functions.keys())}")
        
        # Build component list for registration
        py_components = []
        for handler_name, func in functions.items():
            metadata = get_function_metadata(func)
            if metadata:
                # Create PyComponentInfo for the Rust worker
                component_metadata = {
                    'handler_name': handler_name,
                    'return_type': metadata.get('return_type', 'any'),
                    'parameters': str(len(metadata.get('parameters', [])))
                }
                
                py_component = PyComponentInfo(
                    name=handler_name,
                    component_type='function',
                    metadata=component_metadata
                )
                py_components.append(py_component)
                
        # Set components on the Rust worker
        if py_components:
            self._rust_worker.set_components(py_components)
            logger.info(f"Registered {len(py_components)} components with Rust worker")
        
        # Function invocations are now handled through the message handler
    
    def _handle_message(self, request: 'PyInvokeFunctionRequest') -> 'PyInvokeFunctionResponse':
        """Handle incoming function invocation requests."""
        try:
            # Extract request data
            invocation_id = request.invocation_id
            handler_name = request.component_name
            input_data = bytes(request.input_data)
            
            logger.info(f"Processing function invocation - Handler: {handler_name}, ID: {invocation_id}, Data size: {len(input_data)} bytes")
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
            
            # Create context for the function
            context = {
                'invocation_id': invocation_id,
                'service_name': request.service_name,
                'handler_name': handler_name,
                'metadata': request.metadata
            }
            
            # Call the function through the decorator system
            # RuntimeAdapter is used internally by invoke_function if needed
            try:
                result_data = invoke_function(
                    handler_name=handler_name,
                    input_data=input_data,
                    context=context
                )
                
                logger.info(f"Function {handler_name} completed successfully")
                
                # Return successful response
                return PyInvokeFunctionResponse(
                    invocation_id=invocation_id,
                    success=True,
                    output_data=list(result_data),  # Convert bytes to list for PyO3
                    error_message=None,
                    metadata={}
                )
                
            except Exception as e:
                error_msg = f"Function {handler_name} failed: {str(e)}"
                logger.error(error_msg)
                
                # Return error response
                return PyInvokeFunctionResponse(
                    invocation_id=invocation_id,
                    success=False,
                    output_data=[],
                    error_message=error_msg,
                    metadata={}
                )
                
        except Exception as e:
            error_msg = f"Message handling failed: {str(e)}"
            logger.error(error_msg)
            
            # Return error response with fallback invocation_id
            return PyInvokeFunctionResponse(
                invocation_id=getattr(request, 'invocation_id', 'unknown'),
                success=False,
                output_data=[],
                error_message=error_msg,
                metadata={}
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
        if self.runtime_mode == "asgi" and hasattr(self.runtime_adapter, 'enable_cors'):
            self.runtime_adapter.enable_cors(origins)
        else:
            logger.warning("CORS can only be enabled for ASGI runtime")
            
    def disable_cors(self):
        """Disable CORS for ASGI runtime."""
        if self.runtime_mode == "asgi" and hasattr(self.runtime_adapter, 'disable_cors'):
            self.runtime_adapter.disable_cors()
        else:
            logger.warning("CORS can only be disabled for ASGI runtime")