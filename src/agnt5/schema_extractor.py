"""
Schema extraction utilities for Python components.
Extracts JSON Schema from Python type hints, docstrings, and complex types.
"""

import inspect
import json
from dataclasses import fields, is_dataclass, MISSING
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints, get_origin, get_args

try:
    from docstring_parser import parse as parse_docstring

    DOCSTRING_PARSER_AVAILABLE = True
except ImportError:
    DOCSTRING_PARSER_AVAILABLE = False

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class SchemaExtractor:
    """Extract JSON schemas from Python functions and types."""

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self._seen_types: Dict[Type, Dict] = {}

    def extract_function_schema(self, func: callable) -> Dict[str, Any]:
        """
        Extract complete schema information from a Python function.

        Returns:
            {
                "description": "Short description from docstring",
                "long_description": "Detailed description",
                "input_schema": {...},  # JSON Schema
                "output_schema": {...}, # JSON Schema
                "raises": [...],        # Exception information
                "examples": [...]       # Usage examples
            }
        """
        # Parse docstring
        docstring_info = self._parse_docstring(func)

        # Extract type hints
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build schemas
        input_schema = self._build_input_schema(signature, type_hints, docstring_info)
        output_schema = self._build_output_schema(
            type_hints.get("return"), docstring_info.get("returns")
        )

        return {
            "description": docstring_info.get("short_description", ""),
            "long_description": docstring_info.get("long_description", ""),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "raises": docstring_info.get("raises", []),
            "examples": docstring_info.get("examples", []),
        }

    def _parse_docstring(self, func: callable) -> Dict[str, Any]:
        """Parse function docstring using docstring_parser if available."""
        if not func.__doc__ or not DOCSTRING_PARSER_AVAILABLE:
            return {
                "short_description": func.__doc__.split("\n")[0] if func.__doc__ else "",
                "long_description": func.__doc__ or "",
                "params": {},
                "returns": None,
                "raises": [],
                "examples": [],
            }

        parsed = parse_docstring(func.__doc__)

        result = {
            "short_description": parsed.short_description or "",
            "long_description": parsed.long_description or "",
            "params": {},
            "returns": None,
            "raises": [],
            "examples": [],
        }

        # Extract parameter descriptions
        for param in parsed.params:
            result["params"][param.arg_name] = {
                "description": param.description,
                "type": param.type_name,
            }

        # Extract return description
        if parsed.returns:
            result["returns"] = {
                "description": parsed.returns.description,
                "type": parsed.returns.type_name,
            }

        # Extract raises information
        for raises in parsed.raises:
            result["raises"].append(
                {"exception": raises.type_name, "description": raises.description}
            )

        # Extract examples
        for example in parsed.examples:
            result["examples"].append(example.description)

        return result

    def _build_input_schema(
        self,
        signature: inspect.Signature,
        type_hints: Dict[str, Type],
        docstring_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build JSON Schema for function inputs."""
        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            # Skip self, cls, ctx, *args, **kwargs
            if param_name in ("self", "cls", "ctx"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get type and convert to schema
            param_type = type_hints.get(param_name, Any)
            param_schema = self.type_to_json_schema(param_type)

            # Add description from docstring
            param_doc = docstring_info.get("params", {}).get(param_name, {})
            if param_doc.get("description"):
                param_schema["description"] = param_doc["description"]

            # Add default value if present
            if param.default != param.empty:
                if param.default is not None:
                    param_schema["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = param_schema

        return {"type": "object", "properties": properties, "required": required}

    def _build_output_schema(
        self, return_type: Optional[Type], return_doc: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Build JSON Schema for function output."""
        if return_type is None or return_type == type(None):
            return {"type": "null"}

        schema = self.type_to_json_schema(return_type)

        if return_doc and return_doc.get("description"):
            schema["description"] = return_doc["description"]

        return schema

    def type_to_json_schema(self, python_type: Type, depth: int = 0) -> Dict[str, Any]:
        """
        Convert Python type to JSON Schema (pure, no references).
        """
        if depth >= self.max_depth:
            return {
                "type": "object",
                "description": "Truncated due to depth limit",
                "additionalProperties": True,
            }

        # Handle None type
        if python_type is None or python_type == type(None):
            return {"type": "null"}

        # Basic types
        if python_type == str:
            return {"type": "string"}
        elif python_type == int:
            return {"type": "integer"}
        elif python_type == float:
            return {"type": "number"}
        elif python_type == bool:
            return {"type": "boolean"}
        elif python_type == bytes:
            return {"type": "string", "contentEncoding": "base64"}
        elif python_type == dict:
            return {"type": "object", "additionalProperties": True}
        elif python_type == list:
            return {"type": "array"}
        elif python_type == Any:
            return {"description": "Any type"}

        # Get origin and args for generic types
        origin = get_origin(python_type)
        args = get_args(python_type)

        # Handle Dict types
        if origin == dict:
            if len(args) == 2:
                # Dict[str, ValueType]
                value_schema = self.type_to_json_schema(args[1], depth + 1)
                return {"type": "object", "additionalProperties": value_schema}
            return {"type": "object", "additionalProperties": True}

        # Handle List types
        elif origin in (list, List):
            if args:
                item_schema = self.type_to_json_schema(args[0], depth + 1)
                return {"type": "array", "items": item_schema}
            return {"type": "array"}

        # Handle Optional/Union types
        elif origin == Union:
            # Check if it's Optional (Union with None)
            if len(args) == 2 and type(None) in args:
                # Optional[T]
                non_none_type = args[0] if args[1] == type(None) else args[1]
                schema = self.type_to_json_schema(non_none_type, depth + 1)
                return {"oneOf": [{"type": "null"}, schema]}
            else:
                # General Union
                return {"oneOf": [self.type_to_json_schema(arg, depth + 1) for arg in args]}

        # Handle dataclass
        elif is_dataclass(python_type):
            return self._dataclass_to_schema(python_type, depth)

        # Handle Pydantic models
        elif (
            PYDANTIC_AVAILABLE
            and isinstance(python_type, type)
            and issubclass(python_type, BaseModel)
        ):
            return self._pydantic_to_schema(python_type, depth)

        # Default fallback
        return {"type": "object", "additionalProperties": True}

    def _dataclass_to_schema(self, dataclass_type: Type, depth: int) -> Dict[str, Any]:
        """Convert dataclass to pure JSON Schema."""
        if dataclass_type in self._seen_types:
            return {
                "type": "object",
                "description": f"Circular reference to {dataclass_type.__name__}",
                "additionalProperties": True,
            }

        self._seen_types[dataclass_type] = True

        properties = {}
        required = []

        for field in fields(dataclass_type):
            field_schema = self.type_to_json_schema(field.type, depth + 1)

            # Add field metadata if available
            if field.metadata:
                if "description" in field.metadata:
                    field_schema["description"] = field.metadata["description"]

            # Handle defaults
            if field.default != MISSING:
                field_schema["default"] = field.default
            elif field.default_factory != MISSING:
                # Can't serialize factory, just mark as optional
                pass
            else:
                required.append(field.name)

            properties[field.name] = field_schema

        del self._seen_types[dataclass_type]

        return {
            "type": "object",
            "title": dataclass_type.__name__,
            "description": dataclass_type.__doc__ or "",
            "properties": properties,
            "required": required,
        }

    def _pydantic_to_schema(self, model_type: Type, depth: int) -> Dict[str, Any]:
        """Convert Pydantic model to pure JSON Schema."""
        # Get Pydantic's generated schema
        raw_schema = model_type.model_json_schema()

        # Resolve all references to create pure schema
        return self._resolve_refs(raw_schema, raw_schema.get("$defs", {}))

    def _resolve_refs(self, schema: Any, definitions: Dict[str, Any]) -> Any:
        """Recursively resolve all $ref references."""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref = schema["$ref"]
                if ref.startswith("#/$defs/"):
                    def_name = ref.split("/")[-1]
                    if def_name in definitions:
                        return self._resolve_refs(definitions[def_name], definitions)
                return {"type": "object", "additionalProperties": True}

            result = {}
            for key, value in schema.items():
                if key not in ("$defs", "definitions"):
                    result[key] = self._resolve_refs(value, definitions)
            return result

        elif isinstance(schema, list):
            return [self._resolve_refs(item, definitions) for item in schema]

        return schema


# Singleton instance
_schema_extractor = SchemaExtractor()


def extract_function_schema(func: callable) -> Dict[str, Any]:
    """Extract schema information from a function."""
    return _schema_extractor.extract_function_schema(func)


def extract_workflow_schema(workflow_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract schema information from a workflow definition.

    For workflows, we analyze the input requirements of the first step
    and the output of the final step.
    """
    # TODO: Implement workflow schema extraction
    # This would analyze the workflow steps to determine overall input/output
    return {"input_schema": {"type": "object"}, "output_schema": {"type": "object"}}
