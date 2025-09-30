"""
Tests for schema extraction functionality.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from agnt5.schema_extractor import SchemaExtractor, extract_function_schema


@dataclass
class UserProfile:
    """User profile data."""

    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = None


@dataclass
class ProcessingResult:
    """Result of processing operation."""

    success: bool
    message: str
    data: Dict[str, Any]


class TestSchemaExtractor:
    """Test the SchemaExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = SchemaExtractor()

    def test_basic_types(self):
        """Test conversion of basic Python types to JSON Schema."""
        assert self.extractor.type_to_json_schema(str) == {"type": "string"}
        assert self.extractor.type_to_json_schema(int) == {"type": "integer"}
        assert self.extractor.type_to_json_schema(float) == {"type": "number"}
        assert self.extractor.type_to_json_schema(bool) == {"type": "boolean"}
        assert self.extractor.type_to_json_schema(bytes) == {
            "type": "string",
            "contentEncoding": "base64",
        }

    def test_collection_types(self):
        """Test conversion of collection types."""
        # Basic collections
        assert self.extractor.type_to_json_schema(dict) == {
            "type": "object",
            "additionalProperties": True,
        }
        assert self.extractor.type_to_json_schema(list) == {"type": "array"}

        # Typed dict
        dict_schema = self.extractor.type_to_json_schema(Dict[str, int])
        expected = {"type": "object", "additionalProperties": {"type": "integer"}}
        assert dict_schema == expected

        # Typed list
        list_schema = self.extractor.type_to_json_schema(List[str])
        expected = {"type": "array", "items": {"type": "string"}}
        assert list_schema == expected

    def test_optional_types(self):
        """Test Optional type handling."""
        optional_schema = self.extractor.type_to_json_schema(Optional[str])
        expected = {"oneOf": [{"type": "null"}, {"type": "string"}]}
        assert optional_schema == expected

    def test_union_types(self):
        """Test Union type handling."""
        union_schema = self.extractor.type_to_json_schema(Union[str, int])
        expected = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        assert union_schema == expected

    def test_dataclass_conversion(self):
        """Test dataclass to schema conversion."""
        schema = self.extractor.type_to_json_schema(UserProfile)

        assert schema["type"] == "object"
        assert schema["title"] == "UserProfile"
        assert schema["description"] == "User profile data."

        # Check properties
        props = schema["properties"]
        assert props["name"]["type"] == "string"
        assert props["age"]["type"] == "integer"

        # Check optional field (email)
        assert props["email"]["oneOf"] == [{"type": "null"}, {"type": "string"}]

        # Check list field
        assert props["tags"]["oneOf"] == [
            {"type": "null"},
            {"type": "array", "items": {"type": "string"}},
        ]

        # Check required fields
        assert set(schema["required"]) == {"name", "age"}

    def test_nested_dataclass(self):
        """Test nested dataclass handling."""

        @dataclass
        class NestedData:
            profile: UserProfile
            result: ProcessingResult

        schema = self.extractor.type_to_json_schema(NestedData)

        assert schema["type"] == "object"
        assert "profile" in schema["properties"]
        assert "result" in schema["properties"]

        # Verify nested schemas are expanded
        profile_schema = schema["properties"]["profile"]
        assert profile_schema["type"] == "object"
        assert "name" in profile_schema["properties"]

        result_schema = schema["properties"]["result"]
        assert result_schema["type"] == "object"
        assert "success" in result_schema["properties"]

    def test_circular_reference_handling(self):
        """Test that circular references are handled gracefully."""

        @dataclass
        class Node:
            value: str
            parent: Optional["Node"] = None

        schema = self.extractor.type_to_json_schema(Node)

        # Should not crash and should handle the circular reference
        assert schema["type"] == "object"
        assert "value" in schema["properties"]
        assert "parent" in schema["properties"]

    def test_depth_limiting(self):
        """Test that deeply nested types are truncated."""
        extractor = SchemaExtractor(max_depth=2)

        # Create a deeply nested type
        nested_dict = Dict[str, Dict[str, Dict[str, str]]]
        schema = extractor.type_to_json_schema(nested_dict)

        # Should be truncated due to depth limit
        assert schema["type"] == "object"


class TestFunctionSchemaExtraction:
    """Test function schema extraction."""

    def test_simple_function(self):
        """Test schema extraction for a simple function."""

        def greet(name: str) -> str:
            """Greet a user by name."""
            return f"Hello, {name}!"

        schema = extract_function_schema(greet)

        assert schema["description"] == "Greet a user by name."

        # Check input schema
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert input_schema["properties"]["name"]["type"] == "string"
        assert input_schema["required"] == ["name"]

        # Check output schema
        output_schema = schema["output_schema"]
        assert output_schema["type"] == "string"

    def test_function_with_defaults(self):
        """Test function with default parameters."""

        def process_data(
            data: str, count: int = 1, optional_flag: Optional[bool] = None
        ) -> Dict[str, Any]:
            """Process some data with optional parameters."""
            return {"processed": True}

        schema = extract_function_schema(process_data)

        input_schema = schema["input_schema"]
        props = input_schema["properties"]

        # Required parameter
        assert "data" in input_schema["required"]
        assert props["data"]["type"] == "string"

        # Parameter with default
        assert "count" not in input_schema["required"]
        assert props["count"]["type"] == "integer"
        assert props["count"]["default"] == 1

        # Optional parameter
        assert "optional_flag" not in input_schema["required"]

    def test_function_with_complex_types(self):
        """Test function with complex parameter types."""

        def process_user(
            user_id: str, profile: UserProfile, options: Dict[str, str] = None
        ) -> ProcessingResult:
            """Process a user profile."""
            return ProcessingResult(True, "Done", {})

        schema = extract_function_schema(process_user)

        input_schema = schema["input_schema"]
        props = input_schema["properties"]

        # Check dataclass parameter
        profile_prop = props["profile"]
        assert profile_prop["type"] == "object"
        assert profile_prop["title"] == "UserProfile"
        assert "name" in profile_prop["properties"]

        # Check output is also properly extracted
        output_schema = schema["output_schema"]
        assert output_schema["type"] == "object"
        assert output_schema["title"] == "ProcessingResult"

    def test_function_without_annotations(self):
        """Test function without type annotations."""

        def legacy_function(data, count=1):
            """A legacy function without type hints."""
            return "processed"

        schema = extract_function_schema(legacy_function)

        input_schema = schema["input_schema"]
        props = input_schema["properties"]

        # Parameters without type hints should get generic schemas
        assert "data" in props
        assert "count" in props

    def test_function_with_docstring_params(self):
        """Test function with detailed docstring."""

        def advanced_process(input_data: str, max_retries: int = 3) -> bool:
            """
            Process input data with retry logic.

            This function processes the input data and retries on failure.

            Args:
                input_data: The data to process
                max_retries: Maximum number of retry attempts

            Returns:
                True if processing succeeded, False otherwise

            Raises:
                ValueError: If input_data is invalid
                RuntimeError: If processing fails after max retries
            """
            return True

        schema = extract_function_schema(advanced_process)

        # Should extract descriptions (if docstring_parser is available)
        assert "Process input data with retry logic." in schema["description"]

        # Check that we have the expected structure
        assert "input_schema" in schema
        assert "output_schema" in schema
        assert "raises" in schema
        assert "examples" in schema

    def test_async_function(self):
        """Test schema extraction for async functions."""

        async def async_process(data: str) -> Dict[str, Any]:
            """Async processing function."""
            return {"result": "processed"}

        schema = extract_function_schema(async_process)

        # Should work the same as sync functions
        assert schema["description"] == "Async processing function."
        assert schema["input_schema"]["properties"]["data"]["type"] == "string"
        assert schema["output_schema"]["type"] == "object"

    def test_function_with_context_parameter(self):
        """Test function that has ctx parameter (should be skipped)."""
        from agnt5.context import Context

        def handler_function(ctx: Context, user_id: str) -> str:
            """Handler with context parameter."""
            return "handled"

        schema = extract_function_schema(handler_function)

        input_schema = schema["input_schema"]

        # ctx parameter should be skipped
        assert "ctx" not in input_schema["properties"]
        assert "user_id" in input_schema["properties"]
        assert input_schema["required"] == ["user_id"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_function_without_docstring(self):
        """Test function without docstring."""

        def no_docs(x: int) -> int:
            return x * 2

        schema = extract_function_schema(no_docs)

        # Should still work, just with empty descriptions
        assert schema["description"] == ""
        assert "input_schema" in schema
        assert "output_schema" in schema

    def test_malformed_annotations(self):
        """Test function with malformed or missing annotations."""

        def mixed_annotations(a: str, b, c: int = 5) -> str:
            return "result"

        schema = extract_function_schema(mixed_annotations)

        input_schema = schema["input_schema"]
        props = input_schema["properties"]

        # Should handle mixed annotations gracefully
        assert "a" in props
        assert "b" in props  # Should get Any type
        assert "c" in props

    def test_very_complex_nested_types(self):
        """Test very complex nested type structures."""

        def complex_function(
            data: Dict[str, List[Optional[UserProfile]]],
        ) -> List[Dict[str, Union[str, int]]]:
            """Function with very complex types."""
            return []

        schema = extract_function_schema(complex_function)

        # Should not crash and should produce reasonable schema
        assert "input_schema" in schema
        assert "output_schema" in schema

        input_props = schema["input_schema"]["properties"]
        assert "data" in input_props
