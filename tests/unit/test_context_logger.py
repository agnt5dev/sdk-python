"""Unit tests for ContextLogger."""

import importlib.util
import logging
import weakref
from pathlib import Path

# Import _telemetry module directly without triggering the full agnt5 package
# This avoids dependencies on docstring_parser, etc.
_telemetry_path = Path(__file__).parent.parent.parent / "src" / "agnt5" / "_telemetry.py"
spec = importlib.util.spec_from_file_location("_telemetry", _telemetry_path)
_telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_telemetry)

ContextLogger = _telemetry.ContextLogger


class TestContextLogger:
    """Test ContextLogger functionality."""

    def test_process_extracts_custom_attributes(self):
        """Test that custom attributes are extracted from kwargs."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger)

        msg, kwargs = logger.process("test message", {"attr1": "value1", "attr2": "value2"})

        assert msg == "test message"
        assert "extra" in kwargs
        assert "agnt5_attrs" in kwargs["extra"]
        assert kwargs["extra"]["agnt5_attrs"] == {"attr1": "value1", "attr2": "value2"}

    def test_process_preserves_standard_kwargs(self):
        """Test that standard logging kwargs are preserved."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger)

        msg, kwargs = logger.process(
            "test message",
            {"attr1": "value1", "exc_info": True, "stack_info": False}
        )

        assert msg == "test message"
        assert kwargs.get("exc_info") is True
        assert kwargs.get("stack_info") is False
        assert "extra" in kwargs
        assert kwargs["extra"]["agnt5_attrs"] == {"attr1": "value1"}

    def test_process_converts_non_string_values(self):
        """Test that non-string values are converted to strings."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger)

        msg, kwargs = logger.process(
            "test message",
            {"count": 42, "active": True, "ratio": 3.14}
        )

        assert msg == "test message"
        attrs = kwargs["extra"]["agnt5_attrs"]
        assert attrs["count"] == "42"
        assert attrs["active"] == "True"
        assert attrs["ratio"] == "3.14"

    def test_process_serializes_dict_and_list(self):
        """Test that dict and list values are JSON serialized."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger)

        msg, kwargs = logger.process(
            "test message",
            {
                "data": {"key": "value"},
                "items": [1, 2, 3]
            }
        )

        assert msg == "test message"
        attrs = kwargs["extra"]["agnt5_attrs"]
        assert attrs["data"] == '{"key": "value"}'
        assert attrs["items"] == "[1, 2, 3]"

    def test_process_merges_with_existing_extra(self):
        """Test that custom attrs are merged with existing extra dict."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger)

        msg, kwargs = logger.process(
            "test message",
            {
                "attr1": "value1",
                "extra": {"existing_key": "existing_value"}
            }
        )

        assert msg == "test message"
        assert kwargs["extra"]["existing_key"] == "existing_value"
        assert kwargs["extra"]["agnt5_attrs"] == {"attr1": "value1"}

    def test_process_attaches_adapter_extra_as_record_fields(self):
        """Test that run-scoped adapter fields become structured log fields."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(
            base_logger,
            {
                "run_id": "run-123",
                "trace_id": "trace-123",
                "span_id": "span-123",
            },
        )

        msg, kwargs = logger.process("test message", {"attr1": "value1"})

        assert msg == "test message"
        assert kwargs["extra"]["run_id"] == "run-123"
        assert kwargs["extra"]["trace_id"] == "trace-123"
        assert kwargs["extra"]["span_id"] == "span-123"
        assert kwargs["extra"]["agnt5_attrs"] == {"attr1": "value1"}

    def test_process_empty_kwargs(self):
        """Test that empty kwargs are handled correctly."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger)

        msg, kwargs = logger.process("test message", {})

        assert msg == "test message"
        assert "extra" in kwargs
        # No custom attrs, so agnt5_attrs should not be present
        assert "agnt5_attrs" not in kwargs["extra"]

    def test_logging_methods_work(self):
        """Test that standard logging methods work with custom attributes."""
        base_logger = logging.getLogger("test_logging_methods")
        base_logger.handlers = []  # Clear existing handlers
        base_logger.setLevel(logging.DEBUG)

        # Create a mock handler to capture log records
        handler = logging.Handler()
        handler.setLevel(logging.DEBUG)
        records = []
        handler.emit = lambda r: records.append(r)
        base_logger.addHandler(handler)

        logger = ContextLogger(base_logger)

        # Test info
        logger.info("info message", request_id="abc123")
        assert len(records) == 1
        assert records[0].msg == "info message"
        assert hasattr(records[0], "agnt5_attrs")
        assert records[0].agnt5_attrs == {"request_id": "abc123"}

        # Test debug
        records.clear()
        logger.debug("debug message", count=42)
        assert len(records) == 1
        assert records[0].msg == "debug message"
        assert records[0].agnt5_attrs == {"count": "42"}

        # Test warning
        records.clear()
        logger.warning("warning message", severity="high")
        assert len(records) == 1
        assert records[0].msg == "warning message"
        assert records[0].agnt5_attrs == {"severity": "high"}

        # Test error
        records.clear()
        logger.error("error message", error_code="E001")
        assert len(records) == 1
        assert records[0].msg == "error message"
        assert records[0].agnt5_attrs == {"error_code": "E001"}

    def test_real_world_usage(self):
        """Test a real-world usage pattern like the issue description."""
        base_logger = logging.getLogger("test_real_world")
        base_logger.handlers = []
        base_logger.setLevel(logging.DEBUG)

        records = []
        handler = logging.Handler()
        handler.setLevel(logging.DEBUG)
        handler.emit = lambda r: records.append(r)
        base_logger.addHandler(handler)

        logger = ContextLogger(base_logger)

        # This is the usage pattern from the issue
        logger.info(
            "Workflow execution started",
            scoping_prompt="Generate a comprehensive analysis",
            topic="AI research"
        )

        assert len(records) == 1
        assert records[0].msg == "Workflow execution started"
        assert hasattr(records[0], "agnt5_attrs")
        attrs = records[0].agnt5_attrs
        assert attrs["scoping_prompt"] == "Generate a comprehensive analysis"
        assert attrs["topic"] == "AI research"

    def test_context_logger_does_not_register_per_run_loggers(self):
        """Context loggers should keep run_id as a field, not in logger names."""
        from agnt5.context import Context

        execution_logger = logging.getLogger("agnt5.execution")
        records = []
        capture = logging.Handler()
        capture.setLevel(logging.DEBUG)
        capture.emit = lambda record: records.append(record)
        execution_logger.addHandler(capture)

        run_ids = [f"logger-leak-run-{idx}" for idx in range(20)]
        try:
            for run_id in run_ids:
                ctx = Context(
                    run_id=run_id,
                    correlation_id=f"corr-{run_id}",
                    parent_correlation_id="parent",
                )
                ctx.logger.debug("context log", marker=run_id)
        finally:
            execution_logger.removeHandler(capture)

        logger_keys = set(logging.Logger.manager.loggerDict.keys())
        assert "agnt5.execution" in logger_keys
        assert not any(f"agnt5.{run_id}" in logger_keys for run_id in run_ids)

        assert len(records) == len(run_ids)
        for run_id, record in zip(run_ids, records):
            assert record.name == "agnt5.execution"
            assert record.run_id == run_id
            assert isinstance(record._agnt5_context_ref, weakref.ReferenceType)
            assert record.agnt5_attrs == {"marker": run_id}

    def test_truncate_span_attribute_value_bounds_large_values(self, monkeypatch):
        """Large span attributes should be bounded but keep original size visible."""
        monkeypatch.setenv("AGNT5_SPAN_ATTRIBUTE_VALUE_MAX_CHARS", "8")

        value = _telemetry.truncate_span_attribute_value("x" * 20)

        assert value == "xxxxxxxx...[truncated, original_chars=20]"
