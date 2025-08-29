"""
Standalone runtime adapter for direct worker execution.

This adapter is used when running workers directly with asyncio.run(main()).
"""

import json
import logging
import uuid
from typing import Any, Dict

from .base import RuntimeAdapter, RuntimeContext, InvocationRequest, InvocationResponse
from ..decorators import invoke_function

logger = logging.getLogger(__name__)


class WorkerRuntime(RuntimeAdapter):
    """Runtime adapter for standalone worker execution."""
    
    def __init__(self):
        self.name = "standalone"
        
    async def handle_request(
        self, 
        ctx: RuntimeContext, 
        request: InvocationRequest
    ) -> InvocationResponse:
        """
        Handle function invocation by directly calling the decorated function.
        
        Args:
            ctx: Runtime execution context
            request: Invocation request with handler name and input data
            
        Returns:
            InvocationResponse with function result or error
        """
        logger.info(f"Handling standalone invocation: {request.handler_name}")
        
        try:
            # Create context dict for the function
            function_context = {
                'invocation_id': ctx.invocation_id,
                'service_name': ctx.service_name,
                'handler_name': request.handler_name,
                'tenant_id': ctx.tenant_id,
                'deployment_id': ctx.deployment_id,
                'metadata': {**ctx.metadata, **request.metadata}
            }
            
            # Call the function through the decorator system
            result_data = invoke_function(
                handler_name=request.handler_name,
                input_data=request.input_data, 
                context=function_context
            )
            
            logger.info(f"Standalone invocation {request.handler_name} completed successfully")
            
            return InvocationResponse(
                invocation_id=request.invocation_id,
                output_data=result_data,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Function {request.handler_name} failed: {str(e)}"
            logger.error(error_msg)
            
            # Return error response with empty output
            return InvocationResponse(
                invocation_id=request.invocation_id,
                output_data=b'',
                success=False,
                error_message=error_msg
            )