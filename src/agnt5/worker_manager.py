"""
High-level Worker manager that integrates function decorators with the Rust core.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ._compat import _rust_available, _import_error
from .decorators import get_registered_functions, get_function_metadata, invoke_function

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
                 coordinator_endpoint: str = "http://localhost:9091"):
        """
        Initialize the worker.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service  
            coordinator_endpoint: Endpoint of the coordinator service
        """
        if not _rust_available:
            raise RuntimeError(f"Rust core is required but not available: {_import_error}")
            
        self.service_name = service_name
        self.service_version = service_version
        self.coordinator_endpoint = coordinator_endpoint
        
        # Import and create Rust worker
        from ._core import PyWorker
        self._rust_worker = PyWorker(
            coordinator_endpoint, 
            service_name, 
            service_version, 
            "python"
        )
        
        self._running = False
        
        logger.info(f"Worker created: {service_name} v{service_version}")
        
    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._rust_worker.worker_id()
    
    @property 
    def tenant_id(self) -> Optional[str]:
        """Get the tenant ID."""
        return self._rust_worker.tenant_id()
    
    @property
    def deployment_id(self) -> Optional[str]:
        """Get the deployment ID."""
        return self._rust_worker.deployment_id()
    
    def start(self):
        """
        Start the worker and register all decorated functions.
        
        This will:
        1. Start the underlying Rust worker
        2. Wait for successful registration with coordinator
        3. Collect all @function decorated handlers
        """
        logger.info(f"Starting worker {self.service_name}...")
        
        # Start the Rust worker first
        self._rust_worker.start()
        
        # Give it a moment to connect and register
        # TODO: Replace with proper callback mechanism from Rust core
        time.sleep(1.0)  # Increased timeout for registration
        
        # Check if worker is still running (registration successful)
        if not self._rust_worker.is_running():
            raise RuntimeError(f"Worker {self.service_name} failed to start or register")
        
        # Register all decorated functions
        self._register_functions()
        
        self._running = True
        logger.info(f"Worker {self.service_name} connected and registered successfully")
        
    def stop(self):
        """Stop the worker."""
        logger.info(f"Stopping worker {self.service_name}...")
        
        self._running = False
        self._rust_worker.stop()
        
        logger.info(f"Worker {self.service_name} stopped")
        
    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running and self._rust_worker.is_running()
        
    def _register_functions(self):
        """Register all decorated functions with the Worker Coordinator."""
        functions = get_registered_functions()
        
        if not functions:
            logger.warning("No @function decorated handlers found")
            return
            
        logger.info(f"Registering {len(functions)} function handlers: {list(functions.keys())}")
        
        # Build component list for registration
        components = []
        for handler_name, func in functions.items():
            metadata = get_function_metadata(func)
            if metadata:
                components.append({
                    'name': handler_name,
                    'type': 'function',
                    'metadata': metadata
                })
                
        # TODO: Use the Rust worker to send registration message
        # For now, just log the components that would be registered
        logger.info(f"Would register components: {components}")
        
    def handle_invocation(self, handler_name: str, input_data: bytes) -> bytes:
        """
        Handle a function invocation request.
        
        Args:
            handler_name: Name of the handler to invoke
            input_data: Input data as bytes
            
        Returns:
            Function result as bytes
        """
        logger.info(f"Handling invocation: {handler_name}")
        
        try:
            # Create a basic context object
            context = {
                'worker_id': self.worker_id,
                'service_name': self.service_name,
                'handler_name': handler_name
            }
            
            result = invoke_function(handler_name, input_data, context)
            logger.info(f"Invocation {handler_name} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Invocation {handler_name} failed: {e}")
            raise