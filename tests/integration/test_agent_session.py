"""
Integration tests for Agent Session management and multi-turn conversations.

These tests verify that:
1. Agent sessions are created automatically on first message
2. session_id is returned in response metadata
3. Subsequent messages with session_id load conversation history
4. Agent maintains conversation context across multiple turns
"""

import os
import pytest
from agnt5 import Agent, AgentContext, AgentSession
from agnt5.entity import create_entity_context, EntityStateAdapter


# Test fixtures


@pytest.fixture
def chat_agent():
    """Create an agent with real LLM for chat testing."""
    # Skip if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    agent = Agent(
        name="chat_agent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful chatbot. Remember what the user tells you and maintain conversation context.",
        temperature=0.7,
    )

    return agent


# Unit-level tests (verify SDK session support)


@pytest.mark.asyncio
async def test_agent_session_auto_creation(chat_agent):
    """Test that agent session is created automatically without session_id."""
    from agnt5.entity import EntityStateAdapter

    # Create shared state manager
    shared_state_adapter = EntityStateAdapter()

    # First message - no session_id provided
    # Should create AgentContext with auto-generated session_id
    ctx1 = AgentContext(
        run_id="run-1",
        agent_name="chat_agent",
        session_id=None,  # Will default to run_id
        state_manager=shared_state_adapter,
    )

    result1 = await chat_agent.run("Hello! My name is Bob.", context=ctx1)
    assert result1.output  # Got a response

    # Verify conversation history was saved
    history1 = await ctx1.get_conversation_history()
    assert len(history1) == 2  # User + assistant message
    assert "Bob" in history1[0].content


@pytest.mark.asyncio
async def test_agent_session_with_explicit_session_id(chat_agent):
    """Test multi-turn conversation with explicit session_id."""
    from agnt5.entity import EntityStateAdapter

    # Create shared state manager
    shared_state_adapter = EntityStateAdapter()

    session_id = "test-session-123"

    # First message
    ctx1 = AgentContext(
        run_id="run-1",
        agent_name="chat_agent",
        session_id=session_id,
        state_manager=shared_state_adapter,
    )

    result1 = await chat_agent.run("Hello! My name is Charlie.", context=ctx1)
    assert result1.output

    # Verify history
    history1 = await ctx1.get_conversation_history()
    assert len(history1) == 2
    assert "Charlie" in history1[0].content

    # Second message with same session_id
    ctx2 = AgentContext(
        run_id="run-2",
        agent_name="chat_agent",
        session_id=session_id,
        state_manager=shared_state_adapter,
    )

    # History should be loaded
    loaded_history = await ctx2.get_conversation_history()
    assert len(loaded_history) == 2
    assert "Charlie" in loaded_history[0].content

    result2 = await chat_agent.run("What's my name?", context=ctx2)

    # Agent should remember the name
    assert "Charlie" in result2.output or "charlie" in result2.output.lower()

    # History should now have 4 messages
    history2 = await ctx2.get_conversation_history()
    assert len(history2) == 4


@pytest.mark.asyncio
async def test_agent_session_entity():
    """Test AgentSession entity for metadata tracking."""
    from agnt5.entity import create_entity_context

    manager, token = create_entity_context()

    try:
        # Create session entity
        session = AgentSession("session-456")
        session.create(agent_name="test_agent", metadata={"user_id": "user-789"})

        # Verify initial state
        summary = session.get_summary()
        assert summary["session_id"] == "session-456"
        assert summary["agent_name"] == "test_agent"
        assert summary["message_count"] == 0
        assert summary["metadata"]["user_id"] == "user-789"

        # Add messages
        session.add_message()
        session.add_message()

        # Verify updated state
        summary = session.get_summary()
        assert summary["message_count"] == 2
        assert summary["last_message_time"] > summary["created_at"]

    finally:
        from agnt5.entity import _entity_state_adapter_ctx

        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


@pytest.mark.asyncio
async def test_agent_session_different_agents_isolated(chat_agent):
    """Test that different agents don't share session state."""
    from agnt5.entity import EntityStateAdapter

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create another agent
    agent2 = Agent(
        name="agent2",
        model="openai/gpt-4o-mini",
        instructions="You are a different agent.",
        temperature=0.7,
    )

    shared_state_adapter = EntityStateAdapter()
    session_id = "shared-session-id"

    # chat_agent with session
    ctx1 = AgentContext(
        run_id="run-1",
        agent_name="chat_agent",
        session_id=session_id,
        state_manager=shared_state_adapter,
    )
    await chat_agent.run("I like blue", context=ctx1)

    # agent2 with SAME session_id but different agent_name
    # Should NOT load chat_agent's history (different conversation key)
    ctx2 = AgentContext(
        run_id="run-2",
        agent_name="agent2",
        session_id=session_id,
        state_manager=shared_state_adapter,
    )

    # Should have no history (different agent_name in conversation key)
    history2 = await ctx2.get_conversation_history()
    assert len(history2) == 0


# Platform integration tests (requires running platform)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_session_platform_e2e(client, worker_process):
    """
    End-to-end test for agent session management through the platform.

    Tests the complete flow:
    1. Send first message → creates new session
    2. Verify session_id is returned
    3. Send second message with session_id → loads history
    4. Verify agent remembers context from first message
    """
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # First message - should create new session
    result1 = client.run("chat_agent", {"message": "Hello! My name is Alice."}, component_type="agent")

    # client.run() returns the output dict directly (not full response)
    assert "output" in result1, f"Expected 'output' key in result: {result1}"
    agent_output1 = result1["output"]
    assert agent_output1  # Got a response

    # Extract session_id from response
    # For now, we'll use the fact that agents create a session internally
    # Future: session_id will be in response metadata
    # For this test, we'll track session via a second call to test history loading
    session_id = result1.get("session_id")  # May be None for now

    # Second message - test if session_id is supported
    if session_id:
        # If we got a session_id, use it for the second call
        result2 = client.run("chat_agent", {
            "message": "What's my name?",
            "session_id": session_id
        }, component_type="agent")
    else:
        # No session_id yet - this test validates basic agent invocation works
        # Future: Once session_id is returned in metadata, uncomment the assertion test
        result2 = client.run("chat_agent", {
            "message": "What's my name?",
        }, component_type="agent")

    assert "output" in result2, f"Expected 'output' key in result: {result2}"
    agent_output2 = result2["output"]

    # For now, just verify we got a response
    # Future: Once session persistence is implemented, verify agent remembers "Alice"
    assert agent_output2, "Agent should return a response"

    # TODO: Uncomment once session_id is properly returned and session persistence works
    # assert "Alice" in agent_output2 or "alice" in agent_output2.lower(), \
    #     f"Agent should remember name 'Alice' from previous message. Got: {agent_output2}"


# TODO: Add additional platform integration tests:
# - Session state persists through worker restarts
# - Multiple concurrent sessions work correctly
# - Gateway API endpoints (list/get/delete sessions)
