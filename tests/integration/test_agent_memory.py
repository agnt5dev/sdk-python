"""
Integration tests for Agent memory and conversation persistence.

These tests verify that:
1. AgentContext properly persists conversation history
2. Multiple agent runs share conversation state via session_id
3. Agent state survives worker restarts
4. Memory works correctly in the platform execution context
"""

import os
import pytest
from agnt5 import Agent, AgentContext, tool, Context


# Test fixtures


@pytest.fixture
def memory_agent():
    """Create an agent with real LLM for memory testing."""
    # Skip if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    agent = Agent(
        name="memory_agent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful assistant. Be concise and remember conversation context.",
        temperature=0.7,
    )

    return agent


# Unit-level tests (don't require platform)


@pytest.mark.asyncio
async def test_agent_conversation_history_persistence(memory_agent):
    """Test that conversation history persists across multiple runs with same session_id."""
    from agnt5.entity import create_entity_context, EntityStateAdapter

    manager, token = create_entity_context()

    # Create a shared state manager for both contexts to persist state
    shared_state_adapter = EntityStateAdapter()

    try:
        # First conversation turn
        ctx1 = AgentContext(
            run_id="run-1",
            agent_name="memory_agent",
            session_id="session-123",
            state_manager=shared_state_adapter,  # Share state manager
        )

        result1 = await memory_agent.run("Hello! My name is Alice.", context=ctx1)
        assert result1.output  # Got a response

        # Verify history was saved
        history1 = await ctx1.get_conversation_history()
        assert len(history1) == 2  # User message + assistant response
        assert "Alice" in history1[0].content

        # Second conversation turn (same session, same state manager)
        ctx2 = AgentContext(
            run_id="run-2",
            agent_name="memory_agent",
            session_id="session-123",  # Same session
            state_manager=shared_state_adapter,  # Share state manager
        )

        # History should be loaded automatically
        loaded_history = await ctx2.get_conversation_history()
        assert len(loaded_history) == 2  # Previous conversation loaded
        assert "Alice" in loaded_history[0].content

        result2 = await memory_agent.run("What's my name?", context=ctx2)

        # After second run, should have 4 messages total
        history2 = await ctx2.get_conversation_history()
        assert len(history2) == 4  # Previous 2 + new user + new assistant

        # Agent should remember the name from previous conversation
        assert "Alice" in result2.output or "alice" in result2.output.lower()

    finally:
        from agnt5.entity import _entity_state_adapter_ctx

        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


@pytest.mark.asyncio
async def test_agent_different_sessions_isolated(memory_agent):
    """Test that different sessions don't share conversation history."""
    from agnt5.entity import create_entity_context, EntityStateAdapter

    manager, token = create_entity_context()

    # Create a shared state manager for all contexts
    shared_state_adapter = EntityStateAdapter()

    try:
        # First session
        ctx1 = AgentContext(
            run_id="run-1",
            agent_name="memory_agent",
            session_id="session-A",
            state_manager=shared_state_adapter,
        )
        await memory_agent.run("My favorite color is blue.", context=ctx1)

        # Second session (different session_id, same state manager)
        ctx2 = AgentContext(
            run_id="run-2",
            agent_name="memory_agent",
            session_id="session-B",
            state_manager=shared_state_adapter,
        )

        # Should NOT have session A's history
        history = await ctx2.get_conversation_history()
        assert len(history) == 0  # No history from session-A

        await memory_agent.run("My favorite color is red.", context=ctx2)

        # Verify session A history unchanged (reload with same session ID and state manager)
        ctx1_reload = AgentContext(
            run_id="run-3",
            agent_name="memory_agent",
            session_id="session-A",
            state_manager=shared_state_adapter,
        )
        history_a = await ctx1_reload.get_conversation_history()
        assert len(history_a) == 2  # Only session A's messages
        assert "blue" in history_a[0].content.lower()

    finally:
        from agnt5.entity import _entity_state_adapter_ctx

        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


# TODO: Add tests for custom state persistence using entities
# Note: AgentContext.state is an in-memory dict per context instance and does not
# persist across context instances. For persistent state, use Entity or workflow state.

# TODO: Add platform integration tests that verify:
# 1. Agent memory persists through worker restarts
# 2. Multiple concurrent agent sessions maintain separate state
# 3. Agent state is properly checkpointed and recovered
# 4. Long conversation histories don't cause memory issues
#
# These tests would follow the pattern in test_entity_definition_persistence.py
# but for agents instead of entities.
