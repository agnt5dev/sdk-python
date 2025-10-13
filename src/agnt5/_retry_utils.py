"""Retry and backoff utilities for durable execution.

This module provides utilities for parsing retry policies, calculating backoff delays,
and executing functions with retry logic.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Union

from .exceptions import RetryError
from .types import BackoffPolicy, BackoffType, HandlerFunc, RetryPolicy


def parse_retry_policy(retries: Optional[Union[int, Dict[str, Any], RetryPolicy]]) -> RetryPolicy:
    """Parse retry configuration from various forms.

    Args:
        retries: Can be:
            - int: max_attempts (e.g., 5)
            - dict: RetryPolicy parameters (e.g., {"max_attempts": 5, "initial_interval_ms": 1000})
            - RetryPolicy: pass through
            - None: use default

    Returns:
        RetryPolicy instance
    """
    if retries is None:
        return RetryPolicy()
    elif isinstance(retries, int):
        return RetryPolicy(max_attempts=retries)
    elif isinstance(retries, dict):
        return RetryPolicy(**retries)
    elif isinstance(retries, RetryPolicy):
        return retries
    else:
        raise TypeError(f"retries must be int, dict, or RetryPolicy, got {type(retries)}")


def parse_backoff_policy(backoff: Optional[Union[str, Dict[str, Any], BackoffPolicy]]) -> BackoffPolicy:
    """Parse backoff configuration from various forms.

    Args:
        backoff: Can be:
            - str: backoff type ("constant", "linear", "exponential")
            - dict: BackoffPolicy parameters (e.g., {"type": "exponential", "multiplier": 2.0})
            - BackoffPolicy: pass through
            - None: use default

    Returns:
        BackoffPolicy instance
    """
    if backoff is None:
        return BackoffPolicy()
    elif isinstance(backoff, str):
        backoff_type = BackoffType(backoff.lower())
        return BackoffPolicy(type=backoff_type)
    elif isinstance(backoff, dict):
        # Convert string type to enum if present
        if "type" in backoff and isinstance(backoff["type"], str):
            backoff = {**backoff, "type": BackoffType(backoff["type"].lower())}
        return BackoffPolicy(**backoff)
    elif isinstance(backoff, BackoffPolicy):
        return backoff
    else:
        raise TypeError(f"backoff must be str, dict, or BackoffPolicy, got {type(backoff)}")


def calculate_backoff_delay(
    attempt: int,
    retry_policy: RetryPolicy,
    backoff_policy: BackoffPolicy,
) -> float:
    """Calculate backoff delay in seconds based on attempt number.

    Args:
        attempt: Current attempt number (0-indexed)
        retry_policy: Retry configuration
        backoff_policy: Backoff configuration

    Returns:
        Delay in seconds
    """
    if backoff_policy.type == BackoffType.CONSTANT:
        delay_ms = retry_policy.initial_interval_ms
    elif backoff_policy.type == BackoffType.LINEAR:
        delay_ms = retry_policy.initial_interval_ms * (attempt + 1)
    else:  # EXPONENTIAL
        delay_ms = retry_policy.initial_interval_ms * (backoff_policy.multiplier**attempt)

    # Cap at max_interval_ms
    delay_ms = min(delay_ms, retry_policy.max_interval_ms)
    return delay_ms / 1000.0  # Convert to seconds


async def execute_with_retry(
    handler: HandlerFunc,
    ctx: Any,  # FunctionContext, but avoid circular import
    retry_policy: RetryPolicy,
    backoff_policy: BackoffPolicy,
    needs_context: bool,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute handler with retry logic.

    Args:
        handler: The function handler to execute
        ctx: Context for logging and attempt tracking (FunctionContext)
        retry_policy: Retry configuration
        backoff_policy: Backoff configuration
        needs_context: Whether handler accepts ctx parameter
        *args: Arguments to pass to handler (excluding ctx if needs_context=False)
        **kwargs: Keyword arguments to pass to handler

    Returns:
        Result of successful execution

    Raises:
        RetryError: If all retry attempts fail
    """
    # Import here to avoid circular dependency
    from .function import FunctionContext

    last_error: Optional[Exception] = None

    for attempt in range(retry_policy.max_attempts):
        try:
            # Create context for this attempt (FunctionContext is immutable)
            attempt_ctx = FunctionContext(
                run_id=ctx.run_id,
                attempt=attempt,
                retry_policy=retry_policy
            )

            # Execute handler (pass context only if needed)
            if needs_context:
                result = await handler(attempt_ctx, *args, **kwargs)
            else:
                result = await handler(*args, **kwargs)
            return result

        except Exception as e:
            last_error = e
            ctx.logger.warning(
                f"Function execution failed (attempt {attempt + 1}/{retry_policy.max_attempts}): {e}"
            )

            # If this was the last attempt, raise RetryError
            if attempt == retry_policy.max_attempts - 1:
                raise RetryError(
                    f"Function failed after {retry_policy.max_attempts} attempts",
                    attempts=retry_policy.max_attempts,
                    last_error=e,
                )

            # Calculate backoff delay
            delay = calculate_backoff_delay(attempt, retry_policy, backoff_policy)
            ctx.logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    assert last_error is not None
    raise RetryError(
        f"Function failed after {retry_policy.max_attempts} attempts",
        attempts=retry_policy.max_attempts,
        last_error=last_error,
    )
