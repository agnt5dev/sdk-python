import pytest

from agnt5.context import (
    Context,
    ContextConfig,
    ContextNotReadyError,
    FunctionCall,
    FunctionStatus,
    create_context,
)
from agnt5.context.function_registry import FunctionRegistry


def test_context_config_helpers():
    registry = FunctionRegistry()
    cfg = (
        ContextConfig("tenant", "session", "run", "step", attempt=2)
        .with_invocation_id("invoke")
        .with_metadata("region", "us-west")
        .with_function_registry(registry)
    )

    assert cfg.tenant_id == "tenant"
    assert cfg.session_id == "session"
    assert cfg.metadata["region"] == "us-west"
    assert cfg.function_registry is registry


def test_context_factory():
    ctx = Context(ContextConfig("tenant", "session", "run", "step"))
    assert ctx.config.tenant_id == "tenant"

    created = create_context("tenant", "session", "run", "step", attempt=1)
    assert created.config.attempt == 1


@pytest.mark.asyncio
async def test_function_namespace_executes_registered_call():
    registry = FunctionRegistry()

    async def echo(call: FunctionCall, _ctx):
        return {"echo": call.payload}

    registry.register("svc", "handler", echo)
    cfg = ContextConfig("tenant", "session", "run", "step").with_function_registry(registry)
    ctx = Context(cfg)

    handle = await ctx.functions().call(FunctionCall("svc", "handler", {"value": 42}))
    assert handle.status() is FunctionStatus.SUCCEEDED

    result = await handle.result()
    assert result.status is FunctionStatus.SUCCEEDED
    assert result.output == {"echo": {"value": 42}}


@pytest.mark.asyncio
async def test_function_namespace_reports_handler_error():
    registry = FunctionRegistry()

    async def fail(_call: FunctionCall, _ctx):
        raise RuntimeError("boom")

    registry.register("svc", "fail", fail)
    cfg = ContextConfig("tenant", "session", "run", "step").with_function_registry(registry)
    ctx = Context(cfg)

    handle = await ctx.functions().call(FunctionCall("svc", "fail", {}))
    result = await handle.result()
    assert result.status is FunctionStatus.FAILED
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_function_namespace_missing_handler_raises():
    ctx = Context(ContextConfig("tenant", "session", "run", "step"))

    with pytest.raises(ValueError):
        await ctx.functions().call(FunctionCall("missing", "handler", {}))


@pytest.mark.asyncio
async def test_other_namespaces_still_placeholder():
    ctx = Context(ContextConfig("tenant", "session", "run", "step"))

    with pytest.raises(ContextNotReadyError):
        await ctx.signals().wait("signal")

    with pytest.raises(ContextNotReadyError):
        await ctx.timers().sleep(1)

    with pytest.raises(ContextNotReadyError):
        await ctx.language_model().generate({})
