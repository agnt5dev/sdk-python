"""
Base runtime adapter class.
"""

from typing import Any, Dict, Optional, Protocol


class InvocationRequest:
    """Request for function invocation."""
    
    def __init__(
        self, 
        invocation_id: str,
        service_name: str,
        handler_name: str,
        input_data: bytes,
        metadata: Optional[Dict[str, str]] = None
    ):
        self.invocation_id = invocation_id
        self.service_name = service_name
        self.handler_name = handler_name
        self.input_data = input_data
        self.metadata = metadata or {}


class InvocationResponse:
    """Response from function invocation."""
    
    def __init__(
        self,
        invocation_id: str,
        output_data: bytes,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ):
        self.invocation_id = invocation_id
        self.output_data = output_data
        self.success = success
        self.error_message = error_message
        self.metadata = metadata or {}


class RuntimeContext:
    """Runtime execution context."""
    
    def __init__(
        self,
        invocation_id: str,
        service_name: str,
        component_name: str,
        tenant_id: str = "default",
        deployment_id: str = "default",
        metadata: Optional[Dict[str, str]] = None
    ):
        self.invocation_id = invocation_id
        self.service_name = service_name
        self.component_name = component_name
        self.tenant_id = tenant_id
        self.deployment_id = deployment_id
        self.metadata = metadata or {}


class RuntimeAdapter(Protocol):
    """Protocol for runtime adapters.
    
    Any class implementing this protocol can be used as a RuntimeAdapter.
    The Protocol pattern uses duck typing - if an object has the required
    methods with the correct signature, it satisfies the protocol.
    """
    
    async def handle_request(
        self, 
        ctx: RuntimeContext, 
        request: InvocationRequest
    ) -> InvocationResponse:
        """Handle a function invocation request."""
        ...