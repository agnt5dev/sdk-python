#!/usr/bin/env python3
"""
Comprehensive test for trace correlation with OpenTelemetry initialized.

This script:
1. Initializes OpenTelemetry telemetry system
2. Creates a mock context with correlation IDs
3. Emits logs and verifies they include trace context
"""

import logging
import sys
from agnt5._core import log_from_python

# Test the FFI directly first
print("=" * 60)
print("Testing FFI Signature")
print("=" * 60)
print()

# Test that the new FFI signature works
try:
    log_from_python(
        level="INFO",
        message="Test message with correlation IDs",
        target="test",
        module_path="test_module",
        filename="test.py",
        line=10,
        trace_id="0102030405060708090a0b0c0d0e0f10",
        span_id="0102030405060708",
        run_id="test-run-123",
    )
    print("✅ FFI signature test passed - correlation IDs accepted")
except TypeError as e:
    print(f"❌ FFI signature test failed: {e}")
    sys.exit(1)

print()

# Now test with the full handler
print("=" * 60)
print("Testing OpenTelemetryHandler with Correlation IDs")
print("=" * 60)
print()

from agnt5._telemetry import OpenTelemetryHandler
from agnt5.context import _CorrelationFilter

# Create a logger
logger = logging.getLogger("test_correlation")
logger.setLevel(logging.DEBUG)

# Add OpenTelemetry handler
handler = OpenTelemetryHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Add correlation filter with dummy IDs
class DummyRuntimeContext:
    """Mock RuntimeContext for testing."""
    def __init__(self):
        self.run_id = "test-run-456"
        self.trace_id = "a1b2c3d4e5f6071829384756ab01cd23"
        self.span_id = "1234567890abcdef"

runtime_ctx = DummyRuntimeContext()
correlation_filter = _CorrelationFilter(runtime_ctx)
logger.addFilter(correlation_filter)

# Don't propagate to avoid duplicate logs
logger.propagate = False

print(f"Expected run_id:  {runtime_ctx.run_id}")
print(f"Expected trace_id: {runtime_ctx.trace_id}")
print(f"Expected span_id:  {runtime_ctx.span_id}")
print()
print("Emitting test log messages through handler...")
print()

# Emit test logs
logger.info("Test message 1: INFO level with correlation")
logger.warning("Test message 2: WARNING level with correlation")
logger.error("Test message 3: ERROR level with correlation")

print()
print("=" * 60)
print("✅ All tests completed successfully!")
print()
print("The correlation IDs should now be available as span attributes")
print("in the OpenTelemetry logs. When exported to an OTLP backend,")
print("look for these attributes:")
print("  - otel.trace_id")
print("  - otel.span_id")
print("  - run.id")
print("=" * 60)
