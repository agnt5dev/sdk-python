"""
Tests for Agent workflow state consolidation.

Tests cover:
- Storage mode detection (workflow vs standalone)
- Agent conversation storage in workflow state
- Backward compatibility with standalone agents
"""

import pytest

from agnt5.agent import AgentContext
from agnt5.lm import Message
from agnt5.workflow import WorkflowContext, WorkflowEntity


@pytest.mark.asyncio
async def test_agent_storage_mode_standalone():
    """Test that standalone agent uses standalone storage mode."""
    ctx = AgentContext(run_id="test-123", agent_name="TestAgent")

    assert ctx._storage_mode == "standalone"
    assert ctx._workflow_entity is None


@pytest.mark.asyncio
async def test_agent_storage_mode_workflow():
    """Test that agent with workflow parent uses workflow storage mode."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create agent context with workflow parent
    agent_ctx = AgentContext(
        run_id="test-456",
        agent_name="TestAgent",
        parent_context=wf_ctx
    )

    assert agent_ctx._storage_mode == "workflow"
    assert agent_ctx._workflow_entity is workflow_entity


def test_direct_workflow_agents_get_distinct_memo_namespaces():
    """Direct agent calls in one workflow run must not share lm.0/tool.0 keys."""
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    planner_ctx = AgentContext(
        run_id=wf_ctx.run_id,
        agent_name="planner",
        parent_context=wf_ctx,
    )
    investigator_ctx = AgentContext(
        run_id=wf_ctx.run_id,
        agent_name="investigator",
        parent_context=wf_ctx,
    )
    planner_again_ctx = AgentContext(
        run_id=wf_ctx.run_id,
        agent_name="planner",
        parent_context=wf_ctx,
    )

    assert planner_ctx._memo is not None
    assert investigator_ctx._memo is not None
    assert planner_again_ctx._memo is not None

    planner_key, _ = planner_ctx._memo.lm_call_key("mock", [Message.user("plan")], {})
    investigator_key, _ = investigator_ctx._memo.lm_call_key(
        "mock", [Message.user("investigate")], {}
    )
    planner_again_key, _ = planner_again_ctx._memo.lm_call_key(
        "mock", [Message.user("plan again")], {}
    )

    assert planner_key == "agent.planner.0.lm.0"
    assert investigator_key == "agent.investigator.0.lm.0"
    assert planner_again_key == "agent.planner.1.lm.0"


def test_agent_memo_namespace_can_nest_under_workflow_step_scope():
    """Agents inside ctx.step awaitables inherit the active step memo scope."""
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    with wf_ctx.memo_child_scope("step", "collect"):
        agent_ctx = AgentContext(
            run_id=wf_ctx.run_id,
            agent_name="investigator",
            parent_context=wf_ctx,
        )
        assert agent_ctx._memo is not None
        step_key, _ = agent_ctx._memo.lm_call_key("mock", [Message.user("look")], {})

    assert step_key == "step.collect.0.agent.investigator.0.lm.0"


@pytest.mark.asyncio
async def test_agent_conversation_in_workflow_state():
    """Test that agent stores conversation in workflow state."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create agent context with workflow parent
    agent_ctx = AgentContext(
        run_id="test-456",
        agent_name="TestAgent",
        parent_context=wf_ctx
    )

    # Save messages
    messages = [
        Message.user("Hello"),
        Message.assistant("Hi there!")
    ]
    await agent_ctx.save_conversation_history(messages)

    # Verify stored in workflow state
    agent_data = workflow_entity.state.get("agent.TestAgent")
    assert agent_data is not None
    assert len(agent_data["messages"]) == 2
    assert agent_data["messages"][0]["role"] == "user"
    assert agent_data["messages"][0]["content"] == "Hello"
    assert agent_data["messages"][1]["role"] == "assistant"
    assert agent_data["messages"][1]["content"] == "Hi there!"
    assert agent_data["agent_name"] == "TestAgent"
    assert agent_data["message_count"] == 2


@pytest.mark.asyncio
async def test_agent_conversation_retrieval_from_workflow():
    """Test that agent retrieves conversation from workflow state."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create agent context with workflow parent
    agent_ctx = AgentContext(
        run_id="test-456",
        agent_name="TestAgent",
        parent_context=wf_ctx
    )

    # Save messages
    messages = [
        Message.user("What is AI?"),
        Message.assistant("AI is artificial intelligence.")
    ]
    await agent_ctx.save_conversation_history(messages)

    # Retrieve messages
    retrieved = await agent_ctx.get_conversation_history()
    assert len(retrieved) == 2
    assert retrieved[0].content == "What is AI?"
    assert retrieved[1].content == "AI is artificial intelligence."


@pytest.mark.asyncio
async def test_multiple_agents_in_workflow():
    """Test that multiple agents store separately in workflow state."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create first agent
    agent1_ctx = AgentContext(
        run_id="test-456",
        agent_name="Agent1",
        parent_context=wf_ctx
    )

    # Create second agent
    agent2_ctx = AgentContext(
        run_id="test-789",
        agent_name="Agent2",
        parent_context=wf_ctx
    )

    # Save messages for agent1
    await agent1_ctx.save_conversation_history([
        Message.user("Agent 1 question")
    ])

    # Save messages for agent2
    await agent2_ctx.save_conversation_history([
        Message.user("Agent 2 question")
    ])

    # Verify both agents have separate data
    agent1_data = workflow_entity.state.get("agent.Agent1")
    agent2_data = workflow_entity.state.get("agent.Agent2")

    assert agent1_data is not None
    assert agent2_data is not None
    assert agent1_data["messages"][0]["content"] == "Agent 1 question"
    assert agent2_data["messages"][0]["content"] == "Agent 2 question"

    # Verify list_agents helper
    agents = workflow_entity.list_agents()
    assert "Agent1" in agents
    assert "Agent2" in agents


@pytest.mark.asyncio
async def test_agent_metadata_in_workflow():
    """Test that agent metadata is stored in workflow state."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create agent context with workflow parent
    agent_ctx = AgentContext(
        run_id="test-456",
        agent_name="TestAgent",
        parent_context=wf_ctx
    )

    # Update metadata
    agent_ctx.update_metadata(user_id="user-123", session_type="research")

    # Save messages (which persists metadata)
    await agent_ctx.save_conversation_history([Message.user("Test")])

    # Retrieve metadata
    metadata = await agent_ctx.get_metadata()
    assert metadata["custom"]["user_id"] == "user-123"
    assert metadata["custom"]["session_type"] == "research"
    assert metadata["message_count"] == 1


@pytest.mark.asyncio
async def test_workflow_entity_helper_methods():
    """Test WorkflowEntity helper methods for agent data access."""
    # Create workflow entity
    workflow_entity = WorkflowEntity(run_id="wf-123")

    # Manually add agent data (simulating what AgentContext would do)
    workflow_entity.state.set("agent.TestAgent", {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ],
        "agent_name": "TestAgent"
    })

    # Test get_agent_data
    agent_data = workflow_entity.get_agent_data("TestAgent")
    assert agent_data["agent_name"] == "TestAgent"
    assert len(agent_data["messages"]) == 2

    # Test get_agent_messages
    messages = workflow_entity.get_agent_messages("TestAgent")
    assert len(messages) == 2
    assert messages[0]["content"] == "Hello"

    # Test list_agents
    agents = workflow_entity.list_agents()
    assert "TestAgent" in agents


@pytest.mark.asyncio
async def test_standalone_agent_unchanged():
    """Test that standalone agents still work with entity storage."""
    # Create standalone agent context (no workflow parent)
    agent_ctx = AgentContext(
        run_id="test-123",
        agent_name="StandaloneAgent"
    )

    # Verify standalone mode
    assert agent_ctx._storage_mode == "standalone"

    # Save and retrieve should still work (using entity storage)
    messages = [Message.user("Test message")]
    await agent_ctx.save_conversation_history(messages)

    retrieved = await agent_ctx.get_conversation_history()
    assert len(retrieved) == 1
    assert retrieved[0].content == "Test message"


@pytest.mark.asyncio
async def test_empty_conversation_history_workflow():
    """Test retrieving empty conversation history from workflow state."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create agent context with workflow parent
    agent_ctx = AgentContext(
        run_id="test-456",
        agent_name="TestAgent",
        parent_context=wf_ctx
    )

    # Retrieve without saving anything
    retrieved = await agent_ctx.get_conversation_history()
    assert len(retrieved) == 0


@pytest.mark.asyncio
async def test_metadata_before_conversation_workflow():
    """Test getting metadata before any conversation exists in workflow."""
    # Create workflow entity and context
    workflow_entity = WorkflowEntity(run_id="wf-123")
    wf_ctx = WorkflowContext(workflow_entity=workflow_entity, run_id="wf-123")

    # Create agent context with workflow parent
    agent_ctx = AgentContext(
        run_id="test-456",
        agent_name="TestAgent",
        parent_context=wf_ctx
    )

    # Get metadata before any conversation
    metadata = await agent_ctx.get_metadata()
    assert metadata["created_at"] is None
    assert metadata["last_activity"] is None
    assert metadata["message_count"] == 0
    assert metadata["custom"] == {}
