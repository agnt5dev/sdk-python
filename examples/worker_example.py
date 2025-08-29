#!/usr/bin/env python3
"""
Simple AGNT5 Python SDK worker example.

Run with: python worker_example.py
"""

import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def process_data(data: dict) -> dict:
    """Process some data."""
    return {"processed": True, "input": data}

async def main():
    logger.info("Starting AGNT5 worker...")
    worker = Worker("test-service", runtime="standalone")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())