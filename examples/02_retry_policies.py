"""
Example 2: Retry Policies and Error Handling

This example demonstrates:
- Configuring retry policies
- Different backoff strategies
- Error handling with RetryError
- Custom retry configurations
"""

import asyncio

from agnt5 import BackoffPolicy, BackoffType, Context, RetryPolicy, function


# Simple function with default retry (3 attempts, exponential backoff)
@function
async def flaky_operation(ctx: Context, success_rate: float) -> str:
    """Simulates a flaky operation."""
    import random

    if random.random() < success_rate:
        return "Success!"
    else:
        raise Exception("Transient failure")


# Custom retry policy with 5 attempts
@function(
    retries=RetryPolicy(max_attempts=5, initial_interval_ms=500, max_interval_ms=5000)
)
async def api_call_with_retry(ctx: Context, endpoint: str) -> dict:
    """Simulate API call with custom retry."""
    ctx.logger.info(f"Calling endpoint: {endpoint} (attempt {ctx.attempt + 1})")

    # Simulate failure on first 2 attempts
    if ctx.attempt < 2:
        raise Exception(f"API error: Connection timeout")

    return {"endpoint": endpoint, "status": "success", "attempts": ctx.attempt + 1}


# Exponential backoff (default)
@function(
    retries=RetryPolicy(max_attempts=4, initial_interval_ms=1000),
    backoff=BackoffPolicy(type=BackoffType.EXPONENTIAL, multiplier=2.0),
)
async def exponential_backoff_example(ctx: Context, job_id: str) -> dict:
    """Demonstrate exponential backoff: 1s, 2s, 4s, 8s"""
    ctx.logger.info(f"Processing job {job_id} (attempt {ctx.attempt + 1})")

    # Fail on first 2 attempts
    if ctx.attempt < 2:
        raise Exception("Job processing failed")

    return {"job_id": job_id, "status": "completed"}


# Linear backoff
@function(
    retries=RetryPolicy(max_attempts=4, initial_interval_ms=1000),
    backoff=BackoffPolicy(type=BackoffType.LINEAR, multiplier=1.0),
)
async def linear_backoff_example(ctx: Context, task_id: str) -> dict:
    """Demonstrate linear backoff: 1s, 2s, 3s, 4s"""
    ctx.logger.info(f"Processing task {task_id} (attempt {ctx.attempt + 1})")

    if ctx.attempt < 2:
        raise Exception("Task processing failed")

    return {"task_id": task_id, "status": "completed"}


# Constant backoff
@function(
    retries=RetryPolicy(max_attempts=3, initial_interval_ms=1000),
    backoff=BackoffPolicy(type=BackoffType.CONSTANT),
)
async def constant_backoff_example(ctx: Context, request_id: str) -> dict:
    """Demonstrate constant backoff: 1s, 1s, 1s"""
    ctx.logger.info(f"Processing request {request_id} (attempt {ctx.attempt + 1})")

    if ctx.attempt < 1:
        raise Exception("Request failed")

    return {"request_id": request_id, "status": "completed"}


async def main() -> None:
    """Run examples."""
    import logging

    # Enable logging to see retry behavior
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=== Example 1: API Call with Retry ===")
    ctx1 = Context(run_id="retry-1")
    result1 = await api_call_with_retry(ctx1, "/api/users")
    print(f"Result: {result1}\n")

    print("=== Example 2: Exponential Backoff ===")
    ctx2 = Context(run_id="retry-2")
    result2 = await exponential_backoff_example(ctx2, "job-123")
    print(f"Result: {result2}\n")

    print("=== Example 3: Linear Backoff ===")
    ctx3 = Context(run_id="retry-3")
    result3 = await linear_backoff_example(ctx3, "task-456")
    print(f"Result: {result3}\n")

    print("=== Example 4: Constant Backoff ===")
    ctx4 = Context(run_id="retry-4")
    result4 = await constant_backoff_example(ctx4, "req-789")
    print(f"Result: {result4}\n")

    # Example: Handling RetryError
    print("=== Example 5: RetryError Handling ===")
    from agnt5 import RetryError

    @function(retries=RetryPolicy(max_attempts=2))
    async def always_fails(ctx: Context) -> str:
        raise Exception("Permanent failure")

    try:
        ctx5 = Context(run_id="retry-5")
        await always_fails(ctx5)
    except RetryError as e:
        print(f"Caught RetryError: {e}")
        print(f"  Attempts: {e.attempts}")
        print(f"  Last error: {e.last_error}\n")


if __name__ == "__main__":
    asyncio.run(main())
