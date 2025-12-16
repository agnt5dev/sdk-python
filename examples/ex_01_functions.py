"""
Example: AGNT5 Functions

This example demonstrates basic function features:
- Basic function definition with @function decorator
- Context usage (logging)
- Error handling and different error types

Functions are atomic units - they don't support steps or state management.
For durable multi-step operations, use workflows.
For stateful operations, use entities.

Each function can be:
- Run standalone: python ex_01_functions.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("greet", {"name": "Alice"})
"""

import asyncio

from agnt5 import function, FunctionContext


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================


@function
async def greet(ctx: FunctionContext, name: str) -> str:
    """
    Simple greeting function.

    Args:
        name: Name to greet

    Returns:
        Greeting message

    Example:
        result = client.run("greet", {"name": "Alice"})
        # Returns: "Hello, Alice!"
    """
    ctx.logger.info(f"Greeting user: {name}")
    return f"Hello, {name}!"


@function
async def add(ctx: FunctionContext, a: int, b: int) -> int:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b

    Example:
        result = client.run("add", {"a": 5, "b": 3})
        # Returns: 8
    """
    ctx.logger.info(f"Adding {a} + {b}")
    return a + b


@function
async def process_data(ctx: FunctionContext, data: str) -> dict:
    """
    Process data and return results.

    Args:
        data: Input data string

    Returns:
        Dict with processed data and length

    Example:
        result = client.run("process_data", {"data": "hello"})
        # Returns: {"processed": "HELLO", "length": 5}
    """
    ctx.logger.info(f"Processing data: {data}")
    return {"processed": data.upper(), "length": len(data)}


# =============================================================================
# ERROR HANDLING
# =============================================================================


@function
async def failing_function(ctx: FunctionContext, should_fail: bool, error_type: str = "ValueError") -> dict:
    """
    Function that can raise different error types for testing.

    Args:
        should_fail: Whether to raise an error
        error_type: Type of error to raise (ValueError, TypeError, RuntimeError)

    Returns:
        Success dict if should_fail is False

    Raises:
        ValueError, TypeError, RuntimeError, or Exception based on error_type

    Example:
        # Success case
        result = client.run("failing_function", {"should_fail": False})

        # Error case
        try:
            client.run("failing_function", {"should_fail": True, "error_type": "ValueError"})
        except RunError as e:
            print(f"Caught error: {e}")
    """
    ctx.logger.info(f"should_fail={should_fail}, error_type={error_type}")

    if should_fail:
        if error_type == "ValueError":
            raise ValueError("This is a ValueError for testing")
        elif error_type == "TypeError":
            raise TypeError("This is a TypeError for testing")
        elif error_type == "RuntimeError":
            raise RuntimeError("This is a RuntimeError for testing")
        else:
            raise Exception(f"Generic exception: {error_type}")

    return {"success": True, "message": "No error was raised"}


# =============================================================================
# RETRY FUNCTIONS
# =============================================================================


@function(retries={"max_attempts": 3, "initial_interval_ms": 100})
async def retry_eventually_succeeds(ctx: FunctionContext) -> dict:
    """
    Function that fails on first attempt, succeeds on second.

    This tests platform-level retry orchestration:
    - Fails on attempt 0 with RuntimeError
    - Succeeds on attempt 1+

    Args:
        None (uses ctx.attempt to track retries)

    Returns:
        Dict with success status and attempt count

    Example:
        result = client.run("retry_eventually_succeeds", {})
        # Returns: {"success": True, "attempts": 2}
    """
    ctx.logger.info(f"Attempt {ctx.attempt + 1}/3")

    if ctx.attempt < 1:  # Fail on first attempt (attempt 0)
        raise RuntimeError(f"Simulated failure on attempt {ctx.attempt + 1}")

    return {"success": True, "attempts": ctx.attempt + 1}


@function(retries={"max_attempts": 2, "initial_interval_ms": 100})
async def retry_always_fails(ctx: FunctionContext) -> dict:
    """
    Function that always fails, exhausting all retries.

    This tests retry exhaustion behavior:
    - Fails on every attempt
    - Should result in error after 2 attempts

    Args:
        None

    Returns:
        Never returns (always raises)

    Example:
        try:
            client.run("retry_always_fails", {})
        except RunError as e:
            print(f"Failed after retries: {e}")
    """
    ctx.logger.info(f"Attempt {ctx.attempt + 1}/2 - will fail")
    raise RuntimeError(f"Always fails, attempt {ctx.attempt + 1}")


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging
    from agnt5 import Context

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("=" * 60)
    print("AGNT5 Functions Example")
    print("=" * 60)

    # Create a context for standalone execution
    ctx = Context(run_id="example-run")

    # Basic functions
    print("\n--- Basic Functions ---")
    print(f"greet: {await greet(ctx, 'Alice')}")
    print(f"add: {await add(ctx, 5, 3)}")
    print(f"process_data: {await process_data(ctx, 'hello')}")

    # Error handling
    print("\n--- Error Handling ---")
    print(f"failing_function (success): {await failing_function(ctx, False)}")
    try:
        await failing_function(ctx, True, "ValueError")
    except ValueError as e:
        print(f"failing_function (error): Caught {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
