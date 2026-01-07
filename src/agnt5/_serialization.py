"""Centralized serialization utilities for AGNT5 SDK.

This module provides consistent JSON serialization across the SDK,
handling Pydantic models, dataclasses, and other complex types.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import orjson


def serialize(obj: Any) -> bytes:
    """Serialize any Python object to JSON bytes.

    Handles:
    - Pydantic models (v1 and v2)
    - Dataclasses
    - Bytes (decoded as UTF-8)
    - Sets (converted to lists)
    - All standard JSON types

    Args:
        obj: Any Python object to serialize

    Returns:
        JSON-encoded bytes
    """
    return orjson.dumps(obj, default=_default_serializer)


def serialize_to_str(obj: Any) -> str:
    """Serialize any Python object to JSON string.

    Args:
        obj: Any Python object to serialize

    Returns:
        JSON string
    """
    return serialize(obj).decode("utf-8")


def deserialize(data: bytes | str) -> Any:
    """Deserialize JSON bytes or string to Python object.

    Args:
        data: JSON bytes or string

    Returns:
        Deserialized Python object
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return orjson.loads(data)


def _default_serializer(obj: Any) -> Any:
    """Default serializer for orjson to handle complex types."""
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict") and hasattr(obj, "__fields__"):
        return obj.dict()
    # Dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    # Bytes
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Sets
    if isinstance(obj, set):
        return list(obj)
    # Let orjson raise for unknown types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Convert metadata dict to string values for Rust FFI compatibility.

    PyO3 requires HashMap<String, String>, but Python code may include
    booleans, integers, or other types.

    Args:
        metadata: Dictionary with potentially mixed types

    Returns:
        Dictionary with all string values
    """
    result: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            result[key] = value
        elif isinstance(value, bool):
            result[key] = str(value).lower()
        elif value is None:
            result[key] = ""
        else:
            result[key] = str(value)
    return result
