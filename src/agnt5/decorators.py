"""
Function decorators for AGNT5 workers.

This module provides decorators for registering functions as handlers
that can be invoked through the AGNT5 platform.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

# Set default logging level to DEBUG
logging.getLogger().setLevel(logging.DEBUG)
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
    import traceback
    
    # Input validation
    if not handler_name:
        error_msg = "Empty handler name provided"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if handler_name not in _function_registry:
        error_msg = f"Handler '{handler_name}' not found in registry. Available handlers: {list(_function_registry.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    func = _function_registry[handler_name]
    logger.info(f"Invoking handler: {handler_name}")
    
    try:
        # Decode input data
        if input_data:
            logger.debug(f"Processing {len(input_data)} bytes for {handler_name}")
            
            # Try direct JSON first
            try:
                raw_data = input_data.decode('utf-8')
                input_params = json.loads(raw_data)
                logger.info(f"Decoded JSON input for {handler_name}: {type(input_params)} with keys: {list(input_params.keys()) if isinstance(input_params, dict) else 'non-dict'}")
                logger.debug(f"Input parameters: {input_params}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Fallback to protobuf extraction
                logger.debug(f"JSON decoding failed, trying protobuf extraction for {handler_name}")
                start_idx = input_data.find(b'\x1a')
                if start_idx == -1 or start_idx + 1 >= len(input_data):
                    logger.error(f"Invalid data format for {handler_name}. Length: {len(input_data)}, Hex: {input_data.hex()}")
                    raise RuntimeError("Invalid input data - not JSON and no protobuf marker found")
                
                json_length = input_data[start_idx + 1]
                json_start = start_idx + 2
                
                if json_start + json_length > len(input_data):
                    raise RuntimeError(f"Protobuf structure invalid - length {json_length} exceeds data")
                
                json_bytes = input_data[json_start:json_start + json_length]
                raw_data = json_bytes.decode('utf-8')
                input_params = json.loads(raw_data)
                logger.info(f"Extracted from protobuf for {handler_name}: {type(input_params)} with keys: {list(input_params.keys()) if isinstance(input_params, dict) else 'non-dict'}")
                logger.debug(f"Extracted parameters: {input_params}")
                
        else:
            input_params = {}
            logger.debug(f"No input data provided for {handler_name}")
            
        # Execute function
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            logger.info(f"Calling {handler_name} with signature: {sig}")
            
            if params and params[0] == 'ctx':
                if isinstance(input_params, dict):
                    logger.debug(f"Calling {handler_name}(ctx, **{input_params})")
                    result = func(context, **input_params)
                else:
                    logger.debug(f"Calling {handler_name}(ctx, {input_params})")
                    result = func(context, input_params)
            else:
                if isinstance(input_params, dict):
                    logger.debug(f"Calling {handler_name}(**{input_params})")
                    result = func(**input_params)
                else:
                    logger.debug(f"Calling {handler_name}({input_params})")
                    result = func(input_params)
                    
        except TypeError as e:
            logger.error(f"Signature mismatch in {handler_name}: {e}. Expected: {sig}, Got: {input_params}")
            raise RuntimeError(f"Function signature mismatch: {e}")
            
        except Exception as e:
            logger.error(f"Function {handler_name} failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"Function execution failed: {e}")
            
        # Encode result
        if result is None:
            return b""
        
        try:
            result_json = json.dumps(result)
            return result_json.encode('utf-8')
        except (TypeError, ValueError, UnicodeEncodeError) as e:
            logger.error(f"Cannot serialize/encode result from {handler_name}: {type(result)} - {e}")
            raise RuntimeError(f"Result serialization/encoding error: {e}")
        
    except RuntimeError:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in {handler_name}: {type(e).__name__}: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise RuntimeError(f"Unexpected error: {e}")