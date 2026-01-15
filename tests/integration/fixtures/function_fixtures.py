"""
Function fixtures for integration tests.

Start simple - only add fixtures as tests need them.
"""

import asyncio

from agnt5 import FunctionContext, function


@function
async def add(ctx: FunctionContext, a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@function
async def divide(ctx: FunctionContext, a: int, b: int) -> float:
    """Divide two numbers. Raises ZeroDivisionError if b is 0."""
    return a / b


@function
async def slow_function(ctx: FunctionContext, delay_seconds: int) -> str:
    """Sleep for specified seconds. Used for timeout testing."""
    await asyncio.sleep(delay_seconds)
    return f"Completed after {delay_seconds}s"


@function
async def type_strict_function(ctx: FunctionContext, value: int) -> int:
    """Function that expects strict int type."""
    if not isinstance(value, int):
        raise TypeError(f"Expected int, got {type(value).__name__}")
    return value * 2


@function
async def failing_function(ctx: FunctionContext, message: str) -> str:
    """Function that always fails with RuntimeError."""
    raise RuntimeError(f"Intentional failure: {message}")
