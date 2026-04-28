import pytest

from agnt5._state_adapter import _entity_state_adapter_ctx, create_state_context
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
