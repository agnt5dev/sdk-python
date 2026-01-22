"""Retry and backoff utilities for durable execution.

This module provides utilities for parsing retry policies and calculating backoff delays.
Retry execution is handled by the platform (Execution Engine), not the SDK.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .types import BackoffPolicy, BackoffType, RetryPolicy


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
