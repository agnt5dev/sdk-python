"""Durable function registration and retry policy primitives.

This module exposes the `@durable.function` decorator that is used to
register Python callables as durable components within the AGNT5 runtime.
It keeps track of retry and backoff configuration so the runtime can
coordinate checkpoints and retries across restarts.
"""

from .policies import BackoffPolicy, RetryPolicy
from .registry import (
    DurableFunctionDefinition,
    DurableFunctionRegistry,
    function,
    get_registry,
)

__all__ = [
    "BackoffPolicy",
    "RetryPolicy",
    "DurableFunctionDefinition",
    "DurableFunctionRegistry",
    "function",
    "get_registry",
]
