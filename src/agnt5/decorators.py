"""
Function decorators for AGNT5 workers.

This module provides decorators for registering functions as handlers
that can be invoked through the AGNT5 platform.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global registry of decorated functions
_function_registry: Dict[str, Callable] = {}


def function(name: str = None):
    """
    Decorator to register a function as an AGNT5 handler.
    
    Args:
        name: The name to register the function under. If None, uses the function's name.
        
    Usage:
        @function("add_numbers")
        def add_numbers(ctx, a: int, b: int) -> int:
            return a + b
            
        @function()
        def greet_user(ctx, name: str) -> str:
            return f"Hello, {name}!"
    """
    def decorator(func: Callable) -> Callable:
        handler_name = name if name is not None else func.__name__
        
        # Store function metadata
        func._agnt5_handler_name = handler_name
        func._agnt5_is_function = True
        
        # Register in global registry
        _function_registry[handler_name] = func
        
        logger.debug(f"Registered function handler: {handler_name}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Copy metadata to wrapper
        wrapper._agnt5_handler_name = handler_name
        wrapper._agnt5_is_function = True
        
        return wrapper
    
    return decorator


def get_registered_functions() -> Dict[str, Callable]:
    """
    Get all registered function handlers.
    
    Returns:
        Dictionary mapping handler names to functions
    """
    return _function_registry.copy()


def get_function_metadata(func: Callable) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a decorated function.
    
    Args:
        func: The function to extract metadata from
        
    Returns:
        Dictionary with function metadata or None if not decorated
    """
    if not hasattr(func, '_agnt5_is_function'):
        return None
        
    signature = inspect.signature(func)
    parameters = []
    param_items = list(signature.parameters.items())
    
    for i, (param_name, param) in enumerate(param_items):
        if i == 0 and param_name == 'ctx':  # Skip context parameter if it's the first one
            continue
            
        param_info = {
            'name': param_name,
            'type': 'any'  # Default type, could be enhanced with type hints
        }
        
        # Extract type information if available
        if param.annotation != inspect.Parameter.empty:
            param_info['type'] = str(param.annotation.__name__ if hasattr(param.annotation, '__name__') else param.annotation)
            
        if param.default != inspect.Parameter.empty:
            param_info['default'] = param.default
            
        parameters.append(param_info)
    
    return {
        'name': func._agnt5_handler_name,
        'type': 'function',
        'parameters': parameters,
        'return_type': str(signature.return_annotation.__name__ if signature.return_annotation != inspect.Parameter.empty else 'any')
    }


# Alias for more intuitive usage
handler = function


def clear_registry():
    """Clear the function registry. Mainly for testing."""
    global _function_registry
    _function_registry.clear()


def invoke_function(handler_name: str, input_data: bytes, context: Any = None) -> bytes:
    """
    Invoke a registered function handler.
    
    Args:
        handler_name: Name of the handler to invoke
        input_data: Input data as bytes (will be decoded from JSON)
        context: Execution context
        
    Returns:
        Function result as bytes (JSON encoded)
        
    Raises:
        ValueError: If handler is not found
        RuntimeError: If function execution fails
    """
    import json
    
    if handler_name not in _function_registry:
        raise ValueError(f"Handler '{handler_name}' not found")
    
    func = _function_registry[handler_name]
    
    try:
        # Decode input data
        if input_data:
            print(f"📨 Received function invocation: {handler_name}")
            
            # Check if this is protobuf data by looking for the pattern
            try:
                raw_data = input_data.decode('utf-8')
                input_params = json.loads(raw_data)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # This is protobuf data - extract the JSON payload
                # The JSON is embedded after the \x1a<length> pattern
                start_idx = input_data.find(b'\x1a')
                if start_idx != -1 and start_idx + 1 < len(input_data):
                    # The byte after \x1a indicates the length of the JSON data
                    json_length = input_data[start_idx + 1]
                    json_start = start_idx + 2
                    
                    if json_start + json_length <= len(input_data):
                        json_bytes = input_data[json_start:json_start + json_length]
                        raw_data = json_bytes.decode('utf-8')
                        print(f"📋 Extracted JSON from protobuf: {raw_data}")
                        input_params = json.loads(raw_data)
                    else:
                        raise ValueError("Invalid protobuf structure - JSON length exceeds available data")
                else:
                    raise ValueError("Could not find JSON data in protobuf message")
        else:
            input_params = {}
            
        logger.debug(f"Invoking function {handler_name} with params: {input_params}")
        
        # Call function - check if it expects context as first parameter
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if params and params[0] == 'ctx':
            # Function expects context as first parameter
            if isinstance(input_params, dict):
                result = func(context, **input_params)
            else:
                result = func(context, input_params)
        else:
            # Function doesn't expect context
            if isinstance(input_params, dict):
                result = func(**input_params)
            else:
                result = func(input_params)
            
        # Encode result
        if result is None:
            result_data = b""
        else:
            result_json = json.dumps(result)
            result_data = result_json.encode('utf-8')
            
        logger.debug(f"Function {handler_name} completed successfully")
        return result_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"📋 Failed to parse: {repr(raw_data if 'raw_data' in locals() else 'No raw_data available')}")
        logger.error(f"JSON decode error for {handler_name}: {e}")
        raise RuntimeError(f"Invalid JSON input: {e}")
    except Exception as e:
        print(f"❌ Function '{handler_name}' failed: {type(e).__name__}: {e}")
        logger.error(f"Function {handler_name} failed: {e}")
        raise RuntimeError(f"Function execution failed: {e}")