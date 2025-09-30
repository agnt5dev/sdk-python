"""Durable function registry and decorator implementation."""

from __future__ import annotations

import inspect
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union

from .policies import BackoffPolicy, RetryPolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")
AsyncCallable = Callable[..., Awaitable[T]]
DurableCallable = Union[Callable[..., Awaitable[T]], Callable[..., T]]


@dataclass
class DurableFunctionDefinition:
    """Metadata for a registered durable function."""

    name: str
    handler: DurableCallable[Any]
    retries: RetryPolicy
    backoff: BackoffPolicy
    module: str
    qualname: str
    signature: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "handler": self.qualname,
            "module": self.module,
            "signature": self.signature,
            "retry_policy": self.retries.to_dict(),
            "backoff_policy": self.backoff.to_dict(),
        }


class DurableFunctionRegistry:
    """Thread-safe registry holding durable function definitions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._functions: Dict[str, DurableFunctionDefinition] = {}

    def register(self, definition: DurableFunctionDefinition) -> None:
        with self._lock:
            if definition.name in self._functions:
                existing = self._functions[definition.name]
                raise ValueError(
                    f"Durable function '{definition.name}' already registered "
                    f"with handler '{existing.qualname}'."
                )
            self._functions[definition.name] = definition
            logger.debug(
                "Registered durable function '%s' from %s", definition.name, definition.module
            )

    def get(self, name: str) -> Optional[DurableFunctionDefinition]:
        with self._lock:
            return self._functions.get(name)

    def functions(self) -> Dict[str, DurableFunctionDefinition]:
        with self._lock:
            # Return a shallow copy to avoid accidental mutation.
            return dict(self._functions)

    def clear(self) -> None:
        with self._lock:
            self._functions.clear()


_REGISTRY = DurableFunctionRegistry()


def get_registry() -> DurableFunctionRegistry:
    return _REGISTRY


def _ensure_async_callable(func: DurableCallable[Any]) -> None:
    if inspect.iscoroutinefunction(func):
        return
    # Support sync functions but warn, as they cannot leverage async primitives fully.
    logger.warning(
        "Durable function '%s' is synchronous. Consider using an async function to "
        "take advantage of durable Context APIs.",
        func.__qualname__,
    )


def function(
    _func: Optional[DurableCallable[Any]] = None,
    *,
    name: Optional[str] = None,
    retries: Optional[RetryPolicy] = None,
    backoff: Optional[BackoffPolicy] = None,
) -> DurableCallable[Any]:
    """Decorator used to register a durable function handler.

    Supports being used with or without parameters:

    .. code-block:: python

        @durable.function
        async def handler(ctx: Context, value: int) -> int:
            ...

        @durable.function(name="custom", retries=RetryPolicy(max_attempts=5))
        async def other(ctx: Context):
            ...
    """

    def decorator(func: DurableCallable[Any]) -> DurableCallable[Any]:
        if not callable(func):
            raise TypeError("@durable.function can only decorate callables")

        effective_name = name or func.__name__
        if not effective_name:
            raise ValueError("Durable functions must have a non-empty name")

        _ensure_async_callable(func)

        effective_retry = (retries or RetryPolicy()).with_backoff(backoff)
        effective_backoff = effective_retry.backoff

        signature = str(inspect.signature(func))
        definition = DurableFunctionDefinition(
            name=effective_name,
            handler=func,
            retries=effective_retry,
            backoff=effective_backoff,
            module=func.__module__ or "<unknown>",
            qualname=func.__qualname__,
            signature=signature,
        )

        get_registry().register(definition)

        setattr(func, "_agnt5_durable", definition)
        setattr(func, "_agnt5_handler_name", effective_name)

        logger.debug(
            "Durable function '%s' registered with retry policy %s",
            effective_name,
            json.dumps(effective_retry.to_dict()),
        )

        return func

    if _func is not None:
        return decorator(_func)
    return decorator
