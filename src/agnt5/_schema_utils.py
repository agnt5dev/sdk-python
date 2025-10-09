"""Schema conversion utilities for structured output support.

This module provides utilities to convert Python dataclasses and Pydantic models
to JSON Schema format for LLM structured output generation.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Tuple, get_args, get_origin

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

    Args:
        model: Pydantic BaseModel class

    Returns:
        JSON schema dictionary
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError("Pydantic is not installed. Install with: pip install pydantic>=2.0")

    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(f"Expected Pydantic BaseModel class, got {type(model)}")

    # Pydantic v2 has model_json_schema() method
    schema = model.model_json_schema()

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
    elif python_type == Any:
        return {}  # Any type - no restrictions

    # Fallback for unknown types
    return {"type": "string", "description": f"Type: {python_type}"}


# Import Union for type checking
try:
    from typing import Union
except ImportError:
    Union = None  # type: ignore
