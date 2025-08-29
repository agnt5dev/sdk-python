#!/usr/bin/env python3
"""
Example of ASGI runtime usage.

Run with: uvicorn asgi_example:app --reload
"""

import sys
import os

# Add the SDK to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from agnt5 import Worker, function


@function()
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


@function("add_numbers") 
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@function()
def get_status() -> dict:
    """Get service status."""
    return {
        "status": "running",
        "service": "test-service",
        "runtime": "asgi"
    }


# Create worker as ASGI application
app = Worker("test-service", runtime="asgi")
app.enable_cors()  # Enable CORS for web access

if __name__ == "__main__":
    print("AGNT5 ASGI Runtime Example")
    print("==========================")
    print("To run this example:")
    print("1. Install uvicorn: pip install uvicorn")
    print("2. Run: uvicorn asgi_example:app --reload")
    print("3. Open browser to: http://localhost:8000/health")
    print("4. Test functions:")
    print("   POST http://localhost:8000/invoke/greet {'name': 'Alice'}")
    print("   POST http://localhost:8000/invoke/add_numbers {'a': 5, 'b': 3}")
    print("   GET http://localhost:8000/functions")