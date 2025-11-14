"""Error Functions - Test error handling and propagation."""

from agnt5 import function, FunctionContext


@function
async def fn_05_raises_value_error(ctx: FunctionContext, value: int) -> dict:
    """Function that raises ValueError for negative values."""
    ctx.logger.info(f"Processing value: {value}")
    if value < 0:
        raise ValueError(f"Value must be non-negative, got {value}")
    return {"value": value, "squared": value**2}


@function
async def fn_06_raises_runtime_error(ctx: FunctionContext, message: str) -> dict:
    """Function that raises RuntimeError."""
    ctx.logger.info(f"About to raise RuntimeError: {message}")
    raise RuntimeError(message)


@function
async def fn_07_raises_custom_error(ctx: FunctionContext) -> dict:
    """Function that raises a custom exception."""
    ctx.logger.info("Raising custom error")

    class CustomError(Exception):
        pass

    raise CustomError("This is a custom error")


@function
async def intermittent_error(ctx: FunctionContext, success_rate: float) -> dict:
    """Randomly succeeds or fails based on success_rate (0.0 to 1.0)."""
    ctx.logger.info(f"Running with {success_rate*100}% success rate")
    import random

    if random.random() > success_rate:
        error_types = [
            "ConnectionError: Failed to connect to service",
            "TimeoutError: Operation timed out after 30s",
            "ValueError: Invalid input parameter",
            "RuntimeError: Unexpected state encountered",
        ]
        raise Exception(random.choice(error_types))

    return {"status": "success", "success_rate": success_rate}


__all__ = [
    "fn_05_raises_value_error",
    "fn_06_raises_runtime_error",
    "fn_07_raises_custom_error",
    "intermittent_error",
]
