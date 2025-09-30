import pytest

from agnt5.context import Context, ContextConfig, ContextNotReadyError, create_context


def test_context_config_helpers():
    cfg = (
        ContextConfig("tenant", "session", "run", "step", attempt=2)
        .with_invocation_id("invoke")
    )
    cfg = cfg.with_metadata("region", "us-west")

    assert cfg.tenant_id == "tenant"
    assert cfg.session_id == "session"
    assert cfg.metadata["region"] == "us-west"


def test_context_factory():
    ctx = Context(ContextConfig("tenant", "session", "run", "step"))
    assert ctx.config.tenant_id == "tenant"

    created = create_context("tenant", "session", "run", "step", attempt=1)
    assert created.config.attempt == 1


@pytest.mark.asyncio
async def test_context_namespaces_raise_placeholder_errors():
    ctx = Context(ContextConfig("tenant", "session", "run", "step"))

    with pytest.raises(ContextNotReadyError):
        await ctx.functions().call()

    with pytest.raises(ContextNotReadyError):
        await ctx.signals().wait("signal")

    with pytest.raises(ContextNotReadyError):
        await ctx.timers().sleep(1)

    with pytest.raises(ContextNotReadyError):
        await ctx.language_model().generate({})
