"""Language-local execution of registered functions inside workflow steps."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from ._retry_utils import calculate_backoff_delay
from .types import BackoffPolicy


async def call_workflow_function(
    context: Any,
    config: Any,
    function_context: Any,
    step_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    policy = config.retries
    maximum = (
        policy.max_attempts if policy is not None and context._activation_client is None else 1
    )
    from .activation import current_activation

    for attempt in range(maximum):
        activation = current_activation()
        function_context._attempt = activation.attempt - 1 if activation else attempt
        call_kwargs = dict(kwargs)
        try:
            if not args and "input" in call_kwargs:
                value = call_kwargs.pop("input")
                result = config.handler(function_context, value, **call_kwargs)
            else:
                result = config.handler(function_context, *args, **call_kwargs)
            if inspect.isasyncgen(result):
                return await context._consume_streaming_result(result, step_name)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as error:
            if attempt + 1 >= maximum or type(error).__name__ in {
                "ActivationError",
                "WaitingForUserInputException",
                "DurableSleepSuspension",
                "SuspensionRequestedException",
            }:
                raise
            await asyncio.sleep(
                calculate_backoff_delay(attempt, policy, config.backoff or BackoffPolicy())
            )
