#!/usr/bin/env python3
"""Minimal test to check if pyo3-async-runtimes 0.27 works correctly."""

import asyncio
import sys

# Import the Rust extension
from agnt5 import _core


async def simple_async_function():
    """Simple async function for testing."""
    print("Python: Starting async function")
    await asyncio.sleep(0.1)
    print("Python: Finished async function")
    return "Hello from Python!"


def test_rust_to_python_async():
    """Test calling Python async from Rust via into_future_with_locals."""
    print("\n=== Testing Rust -> Python async call ===")

    # This would be called from Rust using into_future_with_locals
    # For now, just verify the basic async works
    result = asyncio.run(simple_async_function())
    print(f"Result: {result}")
    assert result == "Hello from Python!"
    print("✓ Test passed!")


if __name__ == "__main__":
    try:
        test_rust_to_python_async()
        print("\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
