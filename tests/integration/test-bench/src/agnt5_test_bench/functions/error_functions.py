import random
from agnt5 import FunctionContext, function


@function
async def intermittent_error(ctx: FunctionContext, success_rate: float) -> dict:
    """Randomly succeeds or fails based on success_rate (0.0 to 1.0)."""
    ctx.logger.info(f"Running with {success_rate*100}% success rate")

    if random.random() > success_rate:
        error_types = [
            "ConnectionError: Failed to connect to service",
            "TimeoutError: Operation timed out after 30s",
            "ValueError: Invalid input parameter",
            "RuntimeError: Unexpected state encountered",
        ]
        raise Exception(random.choice(error_types))

    return {"status": "success", "success_rate": success_rate}
