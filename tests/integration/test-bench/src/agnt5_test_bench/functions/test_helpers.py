"""Test Helper Functions - Backward compatibility for integration tests.

These functions provide compatibility with existing integration tests
that expect specific function names like 'greet', 'flaky_function', etc.
"""

from agnt5 import function, FunctionContext


@function(name="greet")
async def greet(ctx: FunctionContext, name: str) -> dict:
    """Simple greeting function for testing."""
    ctx.logger.info(f"Greeting: {name}")
    return {"message": f"Hello, {name}!", "name": name}


@function(name="long_task", retries={"max_attempts": 1})
async def long_task(ctx: FunctionContext, duration_seconds: int = 5) -> dict:
    """Simulated long-running task."""
    import asyncio
    ctx.logger.info(f"Starting long task ({duration_seconds}s)")
    await asyncio.sleep(duration_seconds)
    ctx.logger.info(f"Completed long task")
    return {"completed": True, "duration": duration_seconds}


@function(name="flaky_function", retries={"max_attempts": 3, "initial_interval_ms": 100})
async def flaky_function(ctx: FunctionContext, fail_count: int = 1) -> dict:
    """Function that fails N times before succeeding."""
    ctx.logger.info(f"Attempt {ctx.attempt + 1}, fail_count={fail_count}")

    if ctx.attempt < fail_count:
        raise RuntimeError(f"Simulated failure on attempt {ctx.attempt + 1}")

    ctx.logger.info(f"✓ Success on attempt {ctx.attempt + 1}")
    return {"success": True, "attempts": ctx.attempt + 1, "failed": fail_count}


@function(name="generate_text")
async def generate_text(ctx: FunctionContext, prompt: str, max_tokens: int = 100) -> dict:
    """Mock LLM text generation."""
    ctx.logger.info(f"Generating text for prompt: {prompt[:50]}...")
    return {
        "text": f"Generated response for: {prompt}",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "model": "mock-model",
    }


__all__ = [
    "greet",
    "long_task",
    "flaky_function",
    "generate_text",
]
