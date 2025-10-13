"""
Simple Test Functions

Basic functions for testing platform functionality without LM dependencies.
"""

import asyncio
from agnt5 import Context, function


@function
async def greet(ctx: Context, name: str) -> dict:
    """Simple greeting function for basic execution tests."""
    ctx.logger.info(f"Greeting {name}")
    return {"message": f"Hello, {name}!"}


@function
async def long_task(ctx: Context, duration: int) -> dict:
    """Simulates a long-running task."""
    ctx.logger.info(f"Starting long task ({duration}s)")
    await asyncio.sleep(duration)
    return {"status": "completed", "duration": duration}


@function(retries={"max_attempts": 5, "initial_interval_ms": 100})
async def flaky_function(ctx: Context, fail_count: int) -> dict:
    """
    Fails `fail_count` times, then succeeds.

    Used to test retry logic.
    """
    attempt = ctx.attempt if hasattr(ctx, 'attempt') else 0

    ctx.logger.info(f"Flaky function attempt {attempt + 1}")

    if attempt < fail_count:
        raise Exception(f"Simulated failure (attempt {attempt + 1})")

    return {"succeeded": True, "attempts": attempt + 1}


@function
async def failing_function(ctx: Context, error: str) -> dict:
    """Always fails with given error message."""
    ctx.logger.error(f"Failing with: {error}")
    raise Exception(error)


@function
async def generate_text(ctx: Context, prompt: str) -> str:
    """Simulates streaming text generation."""
    ctx.logger.info(f"Generating text for prompt: {prompt}")
    # For streaming tests - future implementation
    # For now, return simple response
    return f"Generated response for: {prompt}"


__all__ = [
    "greet",
    "long_task",
    "flaky_function",
    "failing_function",
    "generate_text",
]
