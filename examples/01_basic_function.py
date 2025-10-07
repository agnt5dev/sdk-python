"""
Example 1: Basic AGNT5 Function

This example demonstrates:
- Simple function definition with @function decorator
- Context parameter usage
- Basic state management (get/set)
- Checkpoint usage with ctx.step()
"""

import asyncio

from agnt5 import Context, function


@function
async def greet_user(ctx: Context, name: str) -> str:
    """Simple greeting function."""
    ctx.logger.info(f"Greeting user: {name}")
    return f"Hello, {name}!"


@function
async def process_data(ctx: Context, data: str) -> dict:
    """Function with state management."""
    # Store in state
    ctx.set("input_data", data)

    # Retrieve from state
    stored = ctx.get("input_data")
    ctx.logger.info(f"Stored data: {stored}")

    return {"processed": data.upper(), "length": len(data)}


@function
async def expensive_pipeline(ctx: Context, dataset_id: str) -> dict:
    """Function with checkpointing."""

    # Step 1: Load data (checkpointed)
    data = await ctx.step(
        "load_data", lambda: asyncio.sleep(0.1) or f"data-{dataset_id}"  # type: ignore
    )
    ctx.logger.info(f"Loaded: {data}")

    # Step 2: Transform data (checkpointed)
    transformed = await ctx.step(
        "transform", lambda: asyncio.sleep(0.1) or data.upper()  # type: ignore
    )
    ctx.logger.info(f"Transformed: {transformed}")

    # Step 3: Validate (checkpointed)
    valid = await ctx.step(
        "validate", lambda: asyncio.sleep(0.1) or len(transformed) > 0  # type: ignore
    )

    return {
        "dataset_id": dataset_id,
        "result": transformed,
        "valid": valid,
    }


@function
async def failing_function(ctx: Context, should_fail: bool, error_type: str = "ValueError") -> dict:
    """Function that can raise different types of errors for testing."""
    ctx.logger.info(f"Testing error handling: should_fail={should_fail}, error_type={error_type}")

    if should_fail:
        if error_type == "ValueError":
            raise ValueError("This is a user-thrown ValueError for testing")
        elif error_type == "TypeError":
            raise TypeError("This is a user-thrown TypeError for testing")
        elif error_type == "RuntimeError":
            raise RuntimeError("This is a user-thrown RuntimeError for testing")
        else:
            raise Exception(f"Generic exception: {error_type}")

    return {"success": True, "message": "No error was raised"}


@function
async def long_running_task(ctx: Context, duration_seconds: int = 30) -> dict:
    """
    Long-running task for testing durability and crash recovery.

    This function simulates work that takes a while to complete. It's useful for:
    - Testing crash recovery (kill server during execution)
    - Testing async execution patterns
    - Verifying event sourcing durability
    """
    ctx.logger.info(f"Starting long-running task: {duration_seconds} seconds")

    # Checkpoint at 25% progress
    await ctx.step(
        "quarter", lambda: asyncio.sleep(duration_seconds * 0.25)  # type: ignore
    )
    ctx.logger.info("25% complete")

    # Checkpoint at 50% progress
    await ctx.step(
        "half", lambda: asyncio.sleep(duration_seconds * 0.25)  # type: ignore
    )
    ctx.logger.info("50% complete")

    # Checkpoint at 75% progress
    await ctx.step(
        "three_quarters", lambda: asyncio.sleep(duration_seconds * 0.25)  # type: ignore
    )
    ctx.logger.info("75% complete")

    # Final 25%
    await ctx.step(
        "complete", lambda: asyncio.sleep(duration_seconds * 0.25)  # type: ignore
    )
    ctx.logger.info("100% complete")

    return {
        "completed": True,
        "duration_seconds": duration_seconds,
        "message": f"Task completed successfully after {duration_seconds} seconds"
    }


async def main() -> None:
    """Run the worker and register with coordinator."""
    from agnt5 import Worker

    worker = Worker(
        service_name="default-service",
        service_version="1.0.0",
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
