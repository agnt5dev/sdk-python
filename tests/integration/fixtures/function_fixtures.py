"""
Function fixtures for integration tests.

This module provides reusable function backends for testing various scenarios:
- Happy path: Basic operations that should succeed
- Error scenarios: Functions that test failure modes
- Edge cases: Boundary conditions and corner cases

These fixtures are separate from pedagogical examples, allowing tests to cover
complex scenarios without cluttering the user-facing examples.
"""

import asyncio
import random
import time
from typing import Any, Optional

from agnt5 import function, FunctionContext


# =============================================================================
# HAPPY PATH FIXTURES
# =============================================================================


@function
async def add_numbers(ctx: FunctionContext, a: int, b: int) -> int:
    """
    Simple addition for basic tests.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    ctx.logger.info(f"Adding {a} + {b}")
    return a + b


@function
async def multiply_numbers(ctx: FunctionContext, a: int, b: int) -> int:
    """
    Simple multiplication for basic tests.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    ctx.logger.info(f"Multiplying {a} * {b}")
    return a * b


@function(timeout_ms=30000)  # 30 second timeout
async def slow_function(ctx: FunctionContext, duration: float = 5.0) -> dict:
    """
    Function that takes time (for timeout tests).

    Args:
        duration: Seconds to sleep (default: 5.0)

    Returns:
        Dict with completion info
    """
    ctx.logger.info(f"Sleeping for {duration} seconds")
    await asyncio.sleep(duration)
    ctx.logger.info(f"Woke up after {duration} seconds")
    return {"slept": duration, "completed": True}


@function
async def large_payload_function(ctx: FunctionContext, size_kb: int = 100) -> dict:
    """
    Generate large payloads for size limit tests.

    Args:
        size_kb: Size of payload in kilobytes

    Returns:
        Dict with data field containing size_kb KB of data
    """
    ctx.logger.info(f"Generating {size_kb}KB payload")
    data = "x" * (size_kb * 1024)
    return {
        "data": data,
        "size_kb": size_kb,
        "length": len(data),
    }


@function
async def concurrent_function(ctx: FunctionContext, task_id: str, delay: float = 0.1) -> dict:
    """
    Function for testing concurrent execution.

    Args:
        task_id: Unique identifier for this task
        delay: Artificial delay in seconds

    Returns:
        Dict with task info and completion time
    """
    ctx.logger.info(f"Task {task_id} starting")
    await asyncio.sleep(delay)
    ctx.logger.info(f"Task {task_id} completed")
    return {
        "task_id": task_id,
        "delay": delay,
        "completed": True,
    }


@function
async def unicode_function(ctx: FunctionContext, text: str) -> dict:
    """
    Function for testing Unicode/internationalization.

    Args:
        text: Text with Unicode characters

    Returns:
        Dict with processed text and metadata
    """
    ctx.logger.info(f"Processing Unicode text: {text[:50]}...")
    return {
        "original": text,
        "upper": text.upper(),
        "lower": text.lower(),
        "length": len(text),
        "byte_length": len(text.encode("utf-8")),
    }


@function
async def nested_function_call(ctx: FunctionContext, depth: int = 1) -> dict:
    """
    Function that calls itself (via client) for nested execution tests.

    Args:
        depth: Current recursion depth

    Returns:
        Dict with depth info
    """
    ctx.logger.info(f"Nested call at depth {depth}")

    if depth <= 0:
        return {"depth": 0, "message": "Base case reached"}

    # This would require client to be passed in context
    # For now, just return depth info
    return {
        "depth": depth,
        "message": f"Depth {depth} processed",
    }


@function
async def echo_function(ctx: FunctionContext, value: Any) -> Any:
    """
    Echo back the input value (for type testing).

    Args:
        value: Any value to echo

    Returns:
        The same value
    """
    ctx.logger.info(f"Echoing value: {value}")
    return value


@function
async def process_dict(ctx: FunctionContext, data: dict) -> dict:
    """
    Process dictionary input (for complex type testing).

    Args:
        data: Input dictionary

    Returns:
        Processed dictionary with metadata
    """
    ctx.logger.info(f"Processing dict with {len(data)} keys")
    return {
        "original": data,
        "key_count": len(data),
        "keys": list(data.keys()),
        "processed": True,
    }


@function
async def process_list(ctx: FunctionContext, items: list) -> dict:
    """
    Process list input (for array type testing).

    Args:
        items: Input list

    Returns:
        Processed data with list statistics
    """
    ctx.logger.info(f"Processing list with {len(items)} items")
    return {
        "original": items,
        "count": len(items),
        "sum": sum(items) if all(isinstance(x, (int, float)) for x in items) else None,
        "processed": True,
    }


# =============================================================================
# ERROR SCENARIO FIXTURES
# =============================================================================


@function
async def sometimes_fails(ctx: FunctionContext, fail_probability: float = 0.5) -> dict:
    """
    Randomly fails for flakiness testing.

    Args:
        fail_probability: Probability of failure (0.0 to 1.0)

    Returns:
        Success dict if doesn't fail

    Raises:
        RuntimeError: Random failure based on probability
    """
    ctx.logger.info(f"Running with {fail_probability*100}% failure rate")

    if random.random() < fail_probability:
        raise RuntimeError(f"Random failure occurred (probability: {fail_probability})")

    return {"success": True, "message": "No failure occurred"}


@function
async def always_fails(ctx: FunctionContext, error_type: str = "RuntimeError") -> dict:
    """
    Always fails with specified error type.

    Args:
        error_type: Type of error to raise

    Returns:
        Never returns (always raises)

    Raises:
        Various error types based on error_type parameter
    """
    ctx.logger.info(f"Raising {error_type}")

    if error_type == "ValueError":
        raise ValueError("This is a ValueError for testing")
    elif error_type == "TypeError":
        raise TypeError("This is a TypeError for testing")
    elif error_type == "RuntimeError":
        raise RuntimeError("This is a RuntimeError for testing")
    elif error_type == "KeyError":
        raise KeyError("missing_key")
    elif error_type == "AttributeError":
        raise AttributeError("missing_attribute")
    else:
        raise Exception(f"Generic exception: {error_type}")


@function
async def parameter_validator(ctx: FunctionContext, value: int) -> dict:
    """
    Strict parameter validation for testing error handling.

    Args:
        value: Must be between 0 and 1000

    Returns:
        Dict with validated value

    Raises:
        ValueError: If value is out of range
    """
    ctx.logger.info(f"Validating value: {value}")

    if value < 0:
        raise ValueError(f"Value must be non-negative, got {value}")

    if value > 1000:
        raise ValueError(f"Value exceeds maximum (1000), got {value}")

    return {
        "value": value,
        "doubled": value * 2,
        "valid": True,
    }


@function
async def invalid_return_type(ctx: FunctionContext) -> Any:
    """
    Returns a type that cannot be serialized (for serialization error tests).

    Returns:
        Non-serializable object (if possible)
    """
    ctx.logger.info("Returning potentially non-serializable type")

    # Return a lambda (not JSON serializable)
    # In practice, the SDK should catch this before return
    return {"lambda": lambda x: x + 1}  # This should fail serialization


@function(timeout_ms=1000)
async def timeout_function(ctx: FunctionContext, duration: float = 5.0) -> dict:
    """
    Function with short timeout for timeout testing.

    Args:
        duration: Seconds to sleep (default exceeds timeout)

    Returns:
        Dict if completes (won't if duration > 1 second)
    """
    ctx.logger.info(f"Sleeping for {duration}s (timeout: 1000ms)")
    await asyncio.sleep(duration)
    return {"completed": True, "duration": duration}


@function
async def memory_intensive_function(ctx: FunctionContext, size_mb: int = 100) -> dict:
    """
    Allocates large amounts of memory (for resource exhaustion tests).

    Args:
        size_mb: Megabytes of memory to allocate

    Returns:
        Dict with allocation info
    """
    ctx.logger.info(f"Allocating {size_mb}MB")

    # Allocate large list
    data = [0] * (size_mb * 1024 * 1024 // 8)  # Roughly size_mb MB

    return {
        "allocated_mb": size_mb,
        "list_length": len(data),
        "success": True,
    }


@function(retries={"max_attempts": 3, "initial_interval_ms": 100})
async def retry_eventually_succeeds(ctx: FunctionContext) -> dict:
    """
    Fails on first attempt, succeeds on retry.

    Returns:
        Dict with attempt count

    Raises:
        RuntimeError: On first attempt only
    """
    ctx.logger.info(f"Attempt {ctx.attempt + 1}/3")

    if ctx.attempt < 1:  # Fail on first attempt (attempt 0)
        raise RuntimeError(f"Simulated failure on attempt {ctx.attempt + 1}")

    return {"success": True, "attempts": ctx.attempt + 1}


@function(retries={"max_attempts": 2, "initial_interval_ms": 100})
async def retry_always_fails(ctx: FunctionContext) -> dict:
    """
    Always fails, exhausting all retries.

    Returns:
        Never returns (always raises)

    Raises:
        RuntimeError: On every attempt
    """
    ctx.logger.info(f"Attempt {ctx.attempt + 1}/2 - will fail")
    raise RuntimeError(f"Always fails, attempt {ctx.attempt + 1}")


@function
async def wrong_parameter_count_function(ctx: FunctionContext) -> dict:
    """
    Function signature expects no parameters (for parameter validation tests).

    Returns:
        Success dict
    """
    return {"success": True}


@function
async def missing_required_param(ctx: FunctionContext, required: str, optional: str = "default") -> dict:
    """
    Function with required parameters (for missing param tests).

    Args:
        required: Required parameter
        optional: Optional parameter with default

    Returns:
        Dict with parameters
    """
    return {
        "required": required,
        "optional": optional,
    }


@function
async def type_strict_function(ctx: FunctionContext, count: int, name: str, active: bool) -> dict:
    """
    Function with strict types (for type validation tests).

    Args:
        count: Must be int
        name: Must be str
        active: Must be bool

    Returns:
        Dict with validated parameters
    """
    # Type validation happens at SDK level
    return {
        "count": count,
        "name": name,
        "active": active,
        "types_valid": True,
    }


@function
async def null_handling_function(ctx: FunctionContext, value: Optional[str] = None) -> dict:
    """
    Function that handles None/null values.

    Args:
        value: Optional string (can be None)

    Returns:
        Dict indicating if value was None
    """
    return {
        "value": value,
        "was_none": value is None,
        "length": len(value) if value else 0,
    }


@function
async def division_function(ctx: FunctionContext, numerator: float, denominator: float) -> dict:
    """
    Division function for testing divide-by-zero and edge cases.

    Args:
        numerator: Number to divide
        denominator: Divisor

    Returns:
        Dict with result

    Raises:
        ZeroDivisionError: If denominator is zero
    """
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero")

    result = numerator / denominator

    return {
        "numerator": numerator,
        "denominator": denominator,
        "result": result,
    }


@function
async def context_access_function(ctx: FunctionContext) -> dict:
    """
    Function that accesses context (for context testing).

    Returns:
        Dict with context information
    """
    return {
        "run_id": ctx.run_id,
        "correlation_id": ctx.correlation_id,
        "attempt": ctx.attempt,
        "has_logger": hasattr(ctx, "logger"),
    }


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================


@function
async def empty_string_function(ctx: FunctionContext, text: str = "") -> dict:
    """
    Function for testing empty string handling.

    Args:
        text: String that may be empty

    Returns:
        Dict with text info
    """
    return {
        "text": text,
        "is_empty": len(text) == 0,
        "length": len(text),
    }


@function
async def very_long_string_function(ctx: FunctionContext, length: int = 10000) -> dict:
    """
    Generate very long strings (for size testing).

    Args:
        length: Length of string to generate

    Returns:
        Dict with long string
    """
    text = "a" * length
    return {
        "text": text,
        "length": len(text),
    }


@function
async def negative_number_function(ctx: FunctionContext, value: int) -> dict:
    """
    Function for testing negative number handling.

    Args:
        value: Any integer (including negative)

    Returns:
        Dict with value info
    """
    return {
        "value": value,
        "is_negative": value < 0,
        "is_positive": value > 0,
        "is_zero": value == 0,
        "absolute": abs(value),
    }


@function
async def float_precision_function(ctx: FunctionContext, value: float) -> dict:
    """
    Function for testing floating point precision.

    Args:
        value: Float value

    Returns:
        Dict with float operations
    """
    return {
        "original": value,
        "squared": value * value,
        "sqrt": value ** 0.5 if value >= 0 else None,
        "rounded": round(value, 2),
    }


@function
async def special_characters_function(ctx: FunctionContext, text: str) -> dict:
    """
    Function for testing special characters (emoji, unicode, etc).

    Args:
        text: Text with special characters

    Returns:
        Dict with character analysis
    """
    return {
        "text": text,
        "length": len(text),
        "byte_length": len(text.encode("utf-8")),
        "has_emoji": any(ord(c) > 127 for c in text),
    }
