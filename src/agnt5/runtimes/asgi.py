"""
ASGI runtime adapter for web framework integration.

This adapter creates a pure ASGI application that can be run with any ASGI server
like uvicorn, hypercorn, or daphne without requiring FastAPI or other frameworks.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import parse_qs

from .base import RuntimeAdapter, RuntimeContext, InvocationRequest, InvocationResponse
from ..decorators import invoke_function, get_registered_functions

logger = logging.getLogger(__name__)


class ASGIRuntime(RuntimeAdapter):
    """Pure ASGI runtime adapter."""
    
    def __init__(self, worker=None):
        self.name = "asgi"
        self.worker = worker
        self._cors_enabled = True
        self._cors_origins = ["*"]
        
    def enable_cors(self, origins: List[str] = None):
        """Enable CORS with specified origins."""
        self._cors_enabled = True
        self._cors_origins = origins or ["*"]
        
    def disable_cors(self):
        """Disable CORS."""
        self._cors_enabled = False
        
    async def __call__(self, scope: dict, receive, send):
        """ASGI application entry point."""
        assert scope['type'] == 'http'
        
        # Parse request
        method = scope['method']
        path = scope['path']
        
        # Handle CORS preflight
        if method == 'OPTIONS' and self._cors_enabled:
            await self._handle_cors_preflight(scope, receive, send)
            return
            
        # Route to appropriate handler
        if path.startswith('/invoke/'):
            await self._handle_invocation(scope, receive, send)
        elif path == '/health':
            await self._handle_health(scope, receive, send)
        elif path == '/functions':
            await self._handle_list_functions(scope, receive, send)
        else:
            await self._handle_not_found(scope, receive, send)
            
    async def _handle_invocation(self, scope: dict, receive, send):
        """Handle function invocation requests."""
        path = scope['path']
        method = scope['method']
        
        if method != 'POST':
            await self._send_error(send, 405, "Method not allowed")
            return
            
        # Extract function name from path: /invoke/{function_name}
        path_parts = path.split('/')
        if len(path_parts) < 3:
            await self._send_error(send, 400, "Invalid path format")
            return
            
        function_name = path_parts[2]
        
        try:
            # Read request body
            body = b''
            while True:
                message = await receive()
                if message['type'] == 'http.request':
                    body += message.get('body', b'')
                    if not message.get('more_body', False):
                        break
                        
            # Parse JSON body
            try:
                if body:
                    input_data = json.loads(body.decode('utf-8'))
                    input_bytes = json.dumps(input_data).encode('utf-8')
                else:
                    input_bytes = b'{}'
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                await self._send_error(send, 400, f"Invalid JSON: {str(e)}")
                return
                
            # Create invocation request
            invocation_id = str(uuid.uuid4())
            request = InvocationRequest(
                invocation_id=invocation_id,
                service_name=self.worker.service_name if self.worker else "unknown",
                handler_name=function_name,
                input_data=input_bytes
            )
            
            # Create runtime context
            ctx = RuntimeContext(
                invocation_id=invocation_id,
                service_name=request.service_name,
                component_name=function_name
            )
            
            # Handle the request
            response = await self.handle_request(ctx, request)
            
            if response.success:
                # Parse output data back to JSON for HTTP response
                try:
                    if response.output_data:
                        output = json.loads(response.output_data.decode('utf-8'))
                    else:
                        output = None
                        
                    await self._send_json_response(send, 200, {
                        "result": output,
                        "invocation_id": response.invocation_id
                    })
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If output is not JSON, send as raw bytes (base64 encoded)
                    import base64
                    await self._send_json_response(send, 200, {
                        "result": base64.b64encode(response.output_data).decode('utf-8'),
                        "invocation_id": response.invocation_id,
                        "encoding": "base64"
                    })
            else:
                await self._send_error(send, 500, response.error_message or "Function execution failed")
                
        except Exception as e:
            logger.exception(f"Error handling invocation for {function_name}")
            await self._send_error(send, 500, f"Internal server error: {str(e)}")
            
    async def _handle_health(self, scope: dict, receive, send):
        """Handle health check requests."""
        await self._send_json_response(send, 200, {
            "status": "healthy",
            "runtime": self.name,
            "service": self.worker.service_name if self.worker else "unknown"
        })
        
    async def _handle_list_functions(self, scope: dict, receive, send):
        """Handle function listing requests."""
        functions = get_registered_functions()
        function_info = []
        
        for name, func in functions.items():
            info = {
                "name": name,
                "type": "function"
            }
            if hasattr(func, '__doc__') and func.__doc__:
                info["description"] = func.__doc__.strip()
            function_info.append(info)
            
        await self._send_json_response(send, 200, {
            "functions": function_info,
            "count": len(function_info)
        })
        
    async def _handle_not_found(self, scope: dict, receive, send):
        """Handle 404 responses."""
        await self._send_error(send, 404, "Not found")
        
    async def _handle_cors_preflight(self, scope: dict, receive, send):
        """Handle CORS preflight requests."""
        headers = [
            (b'access-control-allow-origin', b'*'),
            (b'access-control-allow-methods', b'GET, POST, OPTIONS'),
            (b'access-control-allow-headers', b'content-type'),
            (b'access-control-max-age', b'3600'),
        ]
        
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': headers
        })
        await send({
            'type': 'http.response.body',
            'body': b''
        })
        
    async def _send_json_response(self, send, status: int, data: dict):
        """Send JSON response with optional CORS headers."""
        headers = [(b'content-type', b'application/json')]
        
        if self._cors_enabled:
            headers.extend([
                (b'access-control-allow-origin', b'*'),
                (b'access-control-allow-methods', b'GET, POST, OPTIONS'),
                (b'access-control-allow-headers', b'content-type'),
            ])
            
        body = json.dumps(data).encode('utf-8')
        
        await send({
            'type': 'http.response.start',
            'status': status,
            'headers': headers
        })
        await send({
            'type': 'http.response.body',
            'body': body
        })
        
    async def _send_error(self, send, status: int, message: str):
        """Send error response."""
        await self._send_json_response(send, status, {
            "error": message,
            "status": status
        })
        
    async def handle_request(
        self, 
        ctx: RuntimeContext, 
        request: InvocationRequest
    ) -> InvocationResponse:
        """
        Handle function invocation using the decorator system.
        """
        logger.info(f"Handling ASGI invocation: {request.handler_name}")
        
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
            
            logger.info(f"ASGI invocation {request.handler_name} completed successfully")
            
            return InvocationResponse(
                invocation_id=request.invocation_id,
                output_data=result_data,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Function {request.handler_name} failed: {str(e)}"
            logger.error(error_msg)
            
            return InvocationResponse(
                invocation_id=request.invocation_id,
                output_data=b'',
                success=False,
                error_message=error_msg
            )