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
    stored = await ctx.get("input_data")
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


async def main() -> None:
    """Run examples."""
    print("=== Example 1: Basic Function ===")
    ctx1 = Context(run_id="example-1")
    result1 = await greet_user(ctx1, "Alice")
    print(f"Result: {result1}\n")

    print("=== Example 2: State Management ===")
    ctx2 = Context(run_id="example-2")
    result2 = await process_data(ctx2, "hello world")
    print(f"Result: {result2}\n")

    print("=== Example 3: Checkpointing ===")
    ctx3 = Context(run_id="example-3")
    result3 = await expensive_pipeline(ctx3, "dataset-123")
    print(f"Result: {result3}\n")


if __name__ == "__main__":
    asyncio.run(main())
