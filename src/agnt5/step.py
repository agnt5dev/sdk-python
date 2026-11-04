"""Declared workflow steps with one journal-authoritative activation."""

from __future__ import annotations

import asyncio
import functools
import inspect
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ._retry_utils import parse_backoff_policy, parse_retry_policy
from ._schema_utils import extract_function_schemas
from .context import Context, set_current_context
from .types import BackoffPolicy, RetryPolicy

if TYPE_CHECKING:
    from .activation import ActivationClient
    from .workflow import WorkflowContext


_declared_step_scope: ContextVar[bool] = ContextVar("agnt5_declared_step_scope", default=False)


def assert_workflow_wait_boundary() -> None:
    if _declared_step_scope.get():
        from .exceptions import ConfigurationError

        raise ConfigurationError("Managed waits must be called between workflow steps")


class StepContext(Context):
    """Context for a step admitted by the runtime, including its idempotency key."""

    _activation_client: ActivationClient | None


@dataclass(frozen=True)
class StepConfig:
    name: str
    handler: Callable[..., Any]
    needs_context: bool
    retries: RetryPolicy | None
    backoff: BackoffPolicy
    timeout_ms: int | None
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None


def step(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    retries: int | dict[str, Any] | RetryPolicy | None = None,
    backoff: str | dict[str, Any] | BackoffPolicy | None = None,
    timeout_ms: int | None = None,
) -> Callable[..., Any]:
    """Declare a durable operation invoked with ``await ctx.step(handler, ...)``.

    A step awaits durable admission before running its handler and durable
    completion before returning its result. It emits no separate function
    checkpoints and is not registered for standalone invocation. A first
    parameter named ``ctx`` receives a :class:`StepContext`; other parameters
    are supplied by the caller. Sync handlers run in an executor thread.

    Use an explicit ``key=`` in ``ctx.step`` for parallel or repeated work.
    Direct invocation and execution without durable activation support fail
    rather than silently executing without durability.
    """
    if timeout_ms is not None and (
        isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0
    ):
        raise ValueError("timeout_ms must be a positive integer")

    retry_policy = parse_retry_policy(retries)
    backoff_policy = parse_backoff_policy(backoff)

    def decorate(handler: Callable[..., Any]) -> Callable[..., Any]:
        step_name = name if name is not None else handler.__name__
        if not isinstance(step_name, str) or not step_name.strip():
            raise ValueError("step name must not be empty")
        parameters = list(inspect.signature(handler).parameters.values())
        input_schema, output_schema = extract_function_schemas(handler)
        config = StepConfig(
            name=step_name,
            handler=handler,
            needs_context=bool(parameters) and parameters[0].name == "ctx",
            retries=retry_policy,
            backoff=backoff_policy,
            timeout_ms=timeout_ms,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        @functools.wraps(handler)
        async def declared(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(f"Step '{step_name}' must be invoked with await ctx.step(...)")

        declared._agnt5_step_config = config  # type: ignore[attr-defined]
        return declared

    return decorate if _func is None else decorate(_func)


async def run_declared_step(
    context: WorkflowContext,
    config: StepConfig,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    key: str | None,
) -> Any:
    from .activation import stable_step_key
    from .exceptions import ActivationError, ActivationErrorCode
    from .workflow_activation import run_durable_step

    if context._activation_client is None:
        raise ActivationError(
            ActivationErrorCode.DURABILITY_UNAVAILABLE,
            "@step requires runtime durable activation support",
        )
    step_key = stable_step_key(config.name, context._next_step_ordinal(), key)
    saved_args, saved_kwargs = deepcopy((args, kwargs))
    memo_namespace = context.allocate_memo_child_scope("step", step_key)
    parent_cid = (
        context.activation.activation_id
        if context.activation is not None
        else context._correlation_id
    )

    async def execute(activation_cid: str) -> Any:
        active = context.activation
        step_context = StepContext(
            run_id=context.run_id,
            correlation_id=activation_cid,
            parent_correlation_id=parent_cid,
            attempt=active.attempt - 1 if active is not None else 0,
            runtime_context=context._runtime_context,
            session_id=context._session_id,
            is_streaming=context._is_streaming,
            worker=context._worker,
            trace_metadata=context._trace_metadata,
            memo_namespace=memo_namespace,
        )
        step_context._component_name = config.name
        step_context._activation_client = context._activation_client

        async def call() -> Any:
            # Every retry must execute the input that admission hashed, even
            # when an earlier attempt mutated its arguments before failing.
            attempt_args, attempt_kwargs = deepcopy((saved_args, saved_kwargs))
            call_args = (step_context, *attempt_args) if config.needs_context else attempt_args
            if inspect.iscoroutinefunction(config.handler) or inspect.isasyncgenfunction(
                config.handler
            ):
                result = config.handler(*call_args, **attempt_kwargs)
            else:
                result = await asyncio.to_thread(config.handler, *call_args, **attempt_kwargs)
            if inspect.isasyncgen(result):
                return await context._consume_streaming_result(result, config.name)
            if inspect.isawaitable(result):
                return await result
            return result

        token = set_current_context(step_context)
        scope_token = _declared_step_scope.set(True)
        try:
            if not inspect.iscoroutinefunction(config.handler) and not inspect.isasyncgenfunction(
                config.handler
            ):
                # Cancelling an await cannot stop an executor thread. Keep its
                # activation in flight until it settles before failing/retrying.
                execution = asyncio.create_task(call())
                try:
                    if config.timeout_ms is not None:
                        return await asyncio.wait_for(
                            asyncio.shield(execution), config.timeout_ms / 1000
                        )
                    return await asyncio.shield(execution)
                except BaseException:
                    while not execution.done():
                        try:
                            await asyncio.shield(execution)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not execution.cancelled():
                        execution.exception()
                    raise
            if config.timeout_ms is not None:
                return await asyncio.wait_for(call(), config.timeout_ms / 1000)
            return await call()
        finally:
            from .context import _current_context

            _declared_step_scope.reset(scope_token)
            _current_context.reset(token)

    return await run_durable_step(
        context,
        name=config.name,
        step_key=step_key,
        handler_name=config.name,
        input_value={"args": list(saved_args), "kwargs": saved_kwargs},
        execute=execute,
        step_config=config,
    )
