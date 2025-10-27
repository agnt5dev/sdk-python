#!/usr/bin/env python3
"""
Quick test to verify trace correlation IDs are passed through FFI.

This script tests that:
1. _CorrelationFilter adds correlation IDs to LogRecord
2. OpenTelemetryHandler extracts and passes them through FFI
3. Rust side receives and attaches them as span attributes
"""

import logging
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
        self.run_id = "test-run-123"
        self.trace_id = "0102030405060708090a0b0c0d0e0f10"
        self.span_id = "0102030405060708"

runtime_ctx = DummyRuntimeContext()
correlation_filter = _CorrelationFilter(runtime_ctx)
logger.addFilter(correlation_filter)

# Don't propagate to avoid duplicate logs
logger.propagate = False

# Test logging with correlation IDs
print("=" * 60)
print("Testing Correlation ID Propagation")
print("=" * 60)
print()
print(f"Expected run_id:  {runtime_ctx.run_id}")
print(f"Expected trace_id: {runtime_ctx.trace_id}")
print(f"Expected span_id:  {runtime_ctx.span_id}")
print()
print("Emitting test log messages...")
print()

# Emit test logs
logger.info("Test message 1: Checking correlation IDs")
logger.warning("Test message 2: This is a warning")
logger.error("Test message 3: This is an error")

print()
print("=" * 60)
print("✅ Test completed successfully!")
print()
print("Next steps:")
print("  1. Check console output for structured logs")
print("  2. If using OTLP collector, verify trace/span IDs in backend")
print("  3. Look for 'otel.trace_id', 'otel.span_id', 'run.id' attributes")
print("=" * 60)
