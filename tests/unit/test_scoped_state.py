import pytest

from agnt5._state_adapter import (
    StateAdapter,
    _entity_state_adapter_ctx,
    _state_request_route_ctx,
    create_state_context,
)
from agnt5.agent import AgentContext
from agnt5.workflow import WorkflowContext, WorkflowEntity


@pytest.fixture
def state_context():
    adapter, token = create_state_context()
    try:
        yield adapter
    finally:
        _entity_state_adapter_ctx.reset(token)
        adapter.clear_all()


@pytest.mark.asyncio
async def test_workflow_session_state_persists_across_turns(state_context):
    """Two workflow turns with the same session_id share session-scoped state."""
    turn_1 = WorkflowContext(
        workflow_entity=WorkflowEntity(run_id="run-1"),
        run_id="run-1",
        session_id="session-123",
    )
    await turn_1.session.state.set("conversation_goal", "compare state systems")

    turn_2 = WorkflowContext(
        workflow_entity=WorkflowEntity(run_id="run-2"),
        run_id="run-2",
        session_id="session-123",
    )

    assert await turn_2.session.state.get("conversation_goal") == "compare state systems"


@pytest.mark.asyncio
async def test_workflow_run_and_session_state_are_isolated(state_context):
    """Run-scoped workflow state does not collide with session-scoped state."""
    ctx = WorkflowContext(
        workflow_entity=WorkflowEntity(run_id="run-1"),
        run_id="run-1",
        session_id="session-123",
    )

    ctx.state.set("topic", "run-local")
    await ctx.session.state.set("topic", "session-shared")

    assert ctx.state.get("topic") == "run-local"
    assert await ctx.session.state.get("topic") == "session-shared"


@pytest.mark.asyncio
async def test_agent_scoped_state_isolated_by_scope(state_context):
    """The standalone state adapter keys by scope and scope_id."""
    ctx = AgentContext(
        run_id="run-1",
        agent_name="assistant",
        session_id="session-123",
        user_id="user-456",
    )

    await ctx.state.set("preference", "run")
    await ctx.session.state.set("preference", "session")
    assert ctx.user is not None
    await ctx.user.state.set("preference", "user")

    assert await ctx.state.get("preference") == "run"
    assert await ctx.session.state.get("preference") == "session"
    assert await ctx.user.state.get("preference") == "user"


@pytest.mark.asyncio
async def test_workflow_user_state_persists_across_sessions(state_context):
    """Workflow user-scoped state is shared across sessions for the same user."""
    first_session = WorkflowContext(
        workflow_entity=WorkflowEntity(run_id="run-1"),
        run_id="run-1",
        session_id="session-a",
        user_id="user-456",
    )
    assert first_session.user is not None
    await first_session.user.state.set("language", "en")

    second_session = WorkflowContext(
        workflow_entity=WorkflowEntity(run_id="run-2"),
        run_id="run-2",
        session_id="session-b",
        user_id="user-456",
    )

    assert second_session.user is not None
    assert await second_session.user.state.get("language") == "en"


@pytest.mark.asyncio
async def test_versioned_load_fails_closed_when_platform_is_unavailable():
    class FailingRustState:
        async def py_get_cached_or_load(self, *_args):
            raise OSError("engine unavailable")

    adapter = StateAdapter(FailingRustState())

    with pytest.raises(RuntimeError, match="Failed to load durable state"):
        await adapter.load_with_version(
            "WorkflowEntity",
            "ks_sequential",
            scope="run",
            scope_id="run-1",
        )


@pytest.mark.asyncio
async def test_state_adapter_passes_current_execution_route_to_rust():
    class CapturingRustState:
        def __init__(self):
            self.load_args = None
            self.save_args = None

        async def py_get_cached_or_load(self, *args):
            self.load_args = args
            return b"{}", 0

        async def py_save_state(self, *args):
            self.save_args = args
            return 1

    rust_state = CapturingRustState()
    adapter = StateAdapter(rust_state)
    token = _state_request_route_ctx.set("run-route-1")
    try:
        await adapter.load_with_version("WorkflowEntity", "run-1", "run", "run-1")
        await adapter.save_state("WorkflowEntity", "run-1", {}, 0, "run", "run-1")
    finally:
        _state_request_route_ctx.reset(token)

    assert rust_state.load_args[-1] == "run-route-1"
    assert rust_state.save_args[-1] == "run-route-1"


@pytest.mark.asyncio
async def test_successful_workflow_persistence_marks_changes_durable_without_clearing_audit():
    adapter = StateAdapter()
    token = _entity_state_adapter_ctx.set(adapter)
    entity = WorkflowEntity(run_id="run-state")
    entity.state.set("total_steps", 2)
    try:
        await entity._persist_state()
    finally:
        _entity_state_adapter_ctx.reset(token)

    assert entity.state.has_changes()
    assert not entity.state.has_unpersisted_changes()
