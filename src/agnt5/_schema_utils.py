"""Schema conversion utilities for structured output support.

This module provides utilities to convert Python dataclasses and Pydantic models
to JSON Schema format for LLM structured output generation, function signatures,
and tool definitions.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, get_args, get_origin, get_type_hints

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None  # type: ignore
    PYDANTIC_AVAILABLE = False


def detect_format_type(response_format: Any) -> Tuple[str, Dict[str, Any]]:
    """Auto-detect format type and convert to JSON schema.

    Args:
        response_format: Pydantic model, dataclass, or dict

    Returns:
        Tuple of (format_type, json_schema)
        - format_type: "pydantic", "dataclass", or "raw"
        - json_schema: JSON schema dictionary

    Raises:
        ValueError: If format type is not supported
    """
    # Check for Pydantic model
    if PYDANTIC_AVAILABLE and isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return 'pydantic', pydantic_to_json_schema(response_format)

    # Check for dataclass
    if dataclasses.is_dataclass(response_format):
        return 'dataclass', dataclass_to_json_schema(response_format)

    # Check for raw dict
    if isinstance(response_format, dict):
        return 'raw', response_format

    raise ValueError(
        f"Unsupported response_format type: {type(response_format)}. "
        f"Expected Pydantic model, dataclass, or dict."
    )


def pydantic_to_json_schema(model: type) -> Dict[str, Any]:
    """Convert Pydantic model to JSON schema.

    Supports both Pydantic v1 and v2 APIs.

    Args:
        model: Pydantic BaseModel class

    Returns:
        JSON schema dictionary
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is not installed. Install with: pip install pydantic")

    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(f"Expected Pydantic BaseModel class, got {type(model)}")

    try:
        # Try Pydantic v2 API first
        if hasattr(model, 'model_json_schema'):
            schema = model.model_json_schema()
        # Fall back to Pydantic v1 API
        elif hasattr(model, 'schema'):
            schema = model.schema()
        else:
            # Fallback for edge cases
            schema = {"type": "object"}
    except Exception:
        # If schema generation fails, return basic object schema
        schema = {"type": "object"}

    # Ensure we have the required fields
    if "type" not in schema:
        schema["type"] = "object"

    return schema


def dataclass_to_json_schema(cls: type) -> Dict[str, Any]:
    """Convert Python dataclass to JSON schema.

    Args:
        cls: Dataclass type

    Returns:
        JSON schema dictionary
    """
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"Expected dataclass, got {type(cls)}")

    properties: Dict[str, Any] = {}
    required: list[str] = []

    for field in dataclasses.fields(cls):
        # Convert field type to JSON schema
        field_schema = _type_to_schema(field.type)
        properties[field.name] = field_schema

        # Check if field is required (no default value)
        if field.default == dataclasses.MISSING and field.default_factory == dataclasses.MISSING:  # type: ignore
            required.append(field.name)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False
    }

    return schema


def _type_to_schema(python_type: Any) -> Dict[str, Any]:
    """Convert Python type hint to JSON schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON schema type definition
    """
    # Handle Optional types
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional[X] which is Union[X, None]
    if origin is type(None) or python_type is type(None):
        return {"type": "null"}

    # Handle Union types (including Optional)
    if origin is Union:  # type: ignore
        # Filter out None from union types
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            # Optional[X] case
            return _type_to_schema(non_none_types[0])
        else:
            # True Union - use anyOf
            return {"anyOf": [_type_to_schema(t) for t in non_none_types]}

    # Handle List types
    if origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _type_to_schema(item_type)
        }

    # Handle Dict types
    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": _type_to_schema(value_type)
        }

    # Handle basic types
    if python_type == str:
        return {"type": "string"}
    elif python_type == int:
        return {"type": "integer"}
    elif python_type == float:
        return {"type": "number"}
    elif python_type == bool:
        return {"type": "boolean"}
    elif python_type == dict:
        return {"type": "object"}
    elif python_type == list:
        return {"type": "array"}
    elif python_type == Any:
        return {}  # Any type - no restrictions

    # Fallback for unknown types
    return {"type": "string", "description": f"Type: {python_type}"}


# Import Union for type checking
try:
    from typing import Union
except ImportError:
    Union = None  # type: ignore


def is_pydantic_model(type_hint: Any) -> bool:
    """Check if a type hint is a Pydantic model.

    Args:
        type_hint: Type annotation to check

    Returns:
        True if type_hint is a Pydantic BaseModel subclass
    """
    if not PYDANTIC_AVAILABLE:
        return False

    try:
        return isinstance(type_hint, type) and issubclass(type_hint, BaseModel)
    except TypeError:
        return False


def extract_function_schemas(func: Callable[..., Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract input and output schemas from function type hints.

    Supports both plain Python types and Pydantic models.
    Pydantic models provide richer validation and schema generation.

    Args:
        func: Function to extract schemas from

    Returns:
        Tuple of (input_schema, output_schema) where each is a JSON Schema dict or None
    """
    try:
        # Get type hints
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Build input schema from parameters (excluding 'ctx')
        input_properties = {}
        required_params = []

        for param_name, param in sig.parameters.items():
            if param_name == "ctx":
                continue

            # Get type hint for this parameter
            if param_name in hints:
                param_type = hints[param_name]

                # Check if it's a Pydantic model
                if is_pydantic_model(param_type):
                    # Use Pydantic's schema generation
                    input_properties[param_name] = pydantic_to_json_schema(param_type)
                else:
                    # Use basic type conversion
                    input_properties[param_name] = _type_to_schema(param_type)
            else:
                # No type hint, use generic object
                input_properties[param_name] = {"type": "object"}

            # Check if parameter is required (no default value)
            if param.default is inspect.Parameter.empty:
                required_params.append(param_name)

        input_schema = None
        if input_properties:
            input_schema = {
                "type": "object",
                "properties": input_properties,
            }
            if required_params:
                input_schema["required"] = required_params

            # Add description from docstring if available
            if func.__doc__:
                docstring = inspect.cleandoc(func.__doc__)
                first_line = docstring.split('\n')[0].strip()
                if first_line:
                    input_schema["description"] = first_line

        # Build output schema from return type hint
        output_schema = None
        if "return" in hints:
            return_type = hints["return"]

            # Check if return type is a Pydantic model
            if is_pydantic_model(return_type):
                output_schema = pydantic_to_json_schema(return_type)
            else:
                output_schema = _type_to_schema(return_type)

        return input_schema, output_schema

    except Exception:
        # If schema extraction fails, return None schemas
        return None, None


def extract_function_metadata(func: Callable[..., Any]) -> Dict[str, str]:
    """Extract metadata from function including description from docstring.

    Args:
        func: Function to extract metadata from

    Returns:
        Dictionary with metadata fields like 'description'
    """
    metadata = {}

    # Extract description from docstring
    if func.__doc__:
        # Get first line of docstring as description
        docstring = inspect.cleandoc(func.__doc__)
        first_line = docstring.split('\n')[0].strip()
        if first_line:
            metadata["description"] = first_line

    return metadata
