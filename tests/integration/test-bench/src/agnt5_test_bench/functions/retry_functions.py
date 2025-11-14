"""Retry Functions - Test retry behavior and exhaustion."""

from agnt5 import function, FunctionContext


@function(retries={"max_attempts": 3, "initial_interval_ms": 100})
async def fn_08_retry_succeeds_on_attempt_2(ctx: FunctionContext) -> dict:
    """Function that fails once, then succeeds."""
    ctx.logger.info(f"Attempt {ctx.attempt + 1}/3")

    if ctx.attempt < 1:  # Fail on first attempt (attempt 0)
        raise RuntimeError(f"Simulated failure on attempt {ctx.attempt + 1}")

    ctx.logger.info(f"✓ Success on attempt {ctx.attempt + 1}")
    return {"success": True, "attempts": ctx.attempt + 1}


@function(retries={"max_attempts": 2, "initial_interval_ms": 100})
async def fn_09_retry_exhausted(ctx: FunctionContext) -> dict:
    """Function that always fails, exhausting retries."""
    ctx.logger.info(f"Attempt {ctx.attempt + 1}/2 - will fail")
    raise RuntimeError(f"Always fails, attempt {ctx.attempt + 1}")


__all__ = ["fn_08_retry_succeeds_on_attempt_2", "fn_09_retry_exhausted"]
