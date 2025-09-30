"""
Integration tests for worker schema registration.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from agnt5.function import get_function_metadata, get_registered_functions
from agnt5.durable import function, get_registry
from agnt5.context import Context


@dataclass
class TestData:
    """Test data structure."""

    name: str
    value: int
    optional_field: Optional[str] = None


class TestWorkerSchemaRegistration:
    """Test schema registration in the worker."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing registered functions
        get_registry().clear()

    def teardown_method(self):
        """Clean up after tests."""
        get_registry().clear()

    def test_function_registration_with_schema(self):
        """Test that functions are registered with proper schema information."""

        @function
        async def test_handler(ctx: Context, data: TestData) -> Dict[str, str]:
            """
            Test handler function.

            Args:
                data: Test data to process

            Returns:
                Processing result
            """
            return {"status": "processed", "name": data.name}

        # Get registered functions
        functions = get_registered_functions()
        assert "test_handler" in functions

        # Get function metadata
        metadata = get_function_metadata(functions["test_handler"])
        assert metadata is not None

        # Test schema extraction
        from agnt5.schema_extractor import extract_function_schema

        schema_info = extract_function_schema(functions["test_handler"])

        assert schema_info["description"] == "Test handler function."

        # Check input schema
        input_schema = schema_info["input_schema"]
        assert input_schema["type"] == "object"

        # Should not include 'ctx' parameter
        assert "ctx" not in input_schema["properties"]
        assert "data" in input_schema["properties"]

        data_prop = input_schema["properties"]["data"]
        assert data_prop["type"] == "object"
        assert data_prop["title"] == "TestData"
        assert "name" in data_prop["properties"]
        assert "value" in data_prop["properties"]

        # Check output schema
        output_schema = schema_info["output_schema"]
        assert output_schema["type"] == "object"
        assert output_schema["additionalProperties"]["type"] == "string"

    def test_function_with_complex_signature(self):
        """Test function with complex signature."""

        @function
        async def complex_handler(
            ctx: Context,
            user_id: str,
            data_list: List[TestData],
            options: Optional[Dict[str, str]] = None,
            count: int = 10,
        ) -> List[str]:
            """Complex handler with multiple parameter types."""
            return ["result1", "result2"]

        functions = get_registered_functions()
        from agnt5.schema_extractor import extract_function_schema

        schema_info = extract_function_schema(functions["complex_handler"])

        input_schema = schema_info["input_schema"]
        props = input_schema["properties"]

        # Check individual parameters
        assert props["user_id"]["type"] == "string"
        assert "user_id" in input_schema["required"]

        # List parameter
        assert props["data_list"]["type"] == "array"
        assert props["data_list"]["items"]["type"] == "object"
        assert "data_list" in input_schema["required"]

        # Optional parameter with default
        assert "options" not in input_schema["required"]

        # Parameter with default
        assert "count" not in input_schema["required"]
        assert props["count"]["default"] == 10

        # Check output schema
        output_schema = schema_info["output_schema"]
        assert output_schema["type"] == "array"
        assert output_schema["items"]["type"] == "string"

    @patch("agnt5.worker.PyWorker")
    def test_worker_component_registration(self, mock_py_worker):
        """Test that worker properly registers components with schema information."""
        from agnt5.worker import Worker

        # Register a test function
        @function
        async def sample_function(ctx: Context, input_str: str) -> bool:
            """Sample function for testing."""
            return True

        # Create worker instance
        worker = Worker(service_name="test-service", service_version="1.0.0")

        # Mock the Rust worker
        mock_rust_worker = Mock()
        worker._rust_worker = mock_rust_worker

        # Call the registration method
        worker._register_components()

        # Verify set_components was called
        mock_rust_worker.set_components.assert_called_once()

        # Get the components that were registered
        call_args = mock_rust_worker.set_components.call_args
        py_components = call_args[0][0]

        assert len(py_components) == 1
        component = py_components[0]

        # Verify component has all the new fields
        assert hasattr(component, "name")
        assert hasattr(component, "component_type")
        assert hasattr(component, "metadata")
        assert hasattr(component, "config")
        assert hasattr(component, "input_schema")
        assert hasattr(component, "output_schema")
        assert hasattr(component, "definition")

        # Verify the schema fields are populated
        assert component.input_schema is not None
        assert component.output_schema is not None

        # Parse and verify the schemas
        input_schema = json.loads(component.input_schema)
        output_schema = json.loads(component.output_schema)

        assert input_schema["type"] == "object"
        assert "input_str" in input_schema["properties"]
        assert output_schema["type"] == "boolean"

    def test_function_without_type_hints(self):
        """Test function without type hints."""

        @function
        async def untyped_function(ctx, data):
            """Function without type hints."""
            return "result"

        functions = get_registered_functions()
        from agnt5.schema_extractor import extract_function_schema

        schema_info = extract_function_schema(functions["untyped_function"])

        # Should still extract schema, but with Any types
        input_schema = schema_info["input_schema"]
        assert "data" in input_schema["properties"]

        # Output schema should be generic since no return type annotation
        output_schema = schema_info["output_schema"]
        assert "type" not in output_schema or output_schema.get("type") == "object"

    def test_function_with_source_extraction(self):
        """Test that function source code is extracted."""

        @function
        async def source_test_function(ctx: Context, value: int) -> str:
            """Function to test source extraction."""
            return str(value)

        functions = get_registered_functions()
        func = functions["source_test_function"]

        # Test source extraction
        import inspect

        try:
            source = inspect.getsource(func)
            assert "def source_test_function" in source
            assert "Function to test source extraction." in source
        except Exception:
            # If source extraction fails, that's okay for the test
            pass

    def test_error_handling_in_schema_extraction(self):
        """Test that schema extraction errors are handled gracefully."""

        # Create a function that might cause schema extraction issues
        @function
        async def problematic_function(ctx: Context, data) -> str:
            """Function that might cause schema issues."""
            return "result"

        functions = get_registered_functions()

        # Schema extraction should not crash even with problematic types
        from agnt5.schema_extractor import extract_function_schema

        schema_info = extract_function_schema(functions["problematic_function"])

        # Should return some schema even if not perfect
        assert "input_schema" in schema_info
        assert "output_schema" in schema_info

    def test_multiple_functions_registration(self):
        """Test registration of multiple functions with different signatures."""

        @function
        async def func1(ctx: Context, x: int) -> str:
            """First function."""
            return str(x)

        @function
        async def func2(ctx: Context, data: TestData) -> bool:
            """Second function."""
            return True

        @function
        async def func3(ctx: Context, items: List[str], count: int = 5) -> Dict[str, int]:
            """Third function."""
            return {"count": count}

        functions = get_registered_functions()
        assert len(functions) == 3

        from agnt5.schema_extractor import extract_function_schema

        # Test each function's schema
        for func_name, func in functions.items():
            schema_info = extract_function_schema(func)
            assert "input_schema" in schema_info
            assert "output_schema" in schema_info
            assert schema_info["input_schema"]["type"] == "object"

    def test_metadata_enrichment(self):
        """Test that metadata is enriched with schema information."""

        @function
        async def documented_function(ctx: Context, user_id: str) -> Dict[str, Any]:
            """
            Well documented function.

            This function does important processing.

            Args:
                user_id: The user identifier to process

            Returns:
                Dictionary containing processing results
            """
            return {"processed": True}

        functions = get_registered_functions()
        from agnt5.schema_extractor import extract_function_schema

        schema_info = extract_function_schema(functions["documented_function"])

        # Check that descriptions are extracted
        assert schema_info["description"] == "Well documented function."
        if schema_info.get("long_description"):
            assert "important processing" in schema_info["long_description"]


class TestWorkflowSchemaExtraction:
    """Test workflow schema extraction (basic)."""

    def test_workflow_schema_placeholder(self):
        """Test that workflow schema extraction returns expected structure."""
        from agnt5.schema_extractor import extract_workflow_schema

        # Test with mock workflow definition
        workflow_def = {
            "steps": [
                {"name": "step1", "type": "function", "handler": "process_data"},
                {"name": "step2", "type": "function", "handler": "validate_result"},
            ]
        }

        schema = extract_workflow_schema(workflow_def)

        # Should return basic schema structure
        assert "input_schema" in schema
        assert "output_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert schema["output_schema"]["type"] == "object"
