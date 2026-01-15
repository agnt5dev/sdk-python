"""
Test fixtures for integration tests.

Simple, minimal fixtures for testing. Add only what's needed.
"""

from .function_fixtures import (
    add,
    divide,
    failing_function,
    slow_function,
    type_strict_function,
)

__all__ = [
    "add",
    "divide",
    "failing_function",
    "slow_function",
    "type_strict_function",
]
