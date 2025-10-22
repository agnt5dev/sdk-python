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
    from agnt5.entity import create_entity_context, EntityStateManager

    manager, token = create_entity_context()

    # Create a shared state manager for both contexts to persist state
    shared_state_manager = EntityStateManager()

    try:
        # First conversation turn
        ctx1 = AgentContext(
            run_id="run-1",
            agent_name="memory_agent",
            session_id="session-123",
            state_manager=shared_state_manager,  # Share state manager
        )

        result1 = await memory_agent.run("Hello! My name is Alice.", context=ctx1)
        assert result1.output  # Got a response

        # Verify history was saved
        history1 = ctx1.get_conversation_history()
        assert len(history1) == 2  # User message + assistant response
        assert "Alice" in history1[0].content

        # Second conversation turn (same session, same state manager)
        ctx2 = AgentContext(
            run_id="run-2",
            agent_name="memory_agent",
            session_id="session-123",  # Same session
            state_manager=shared_state_manager,  # Share state manager
        )

        # History should be loaded automatically
        loaded_history = ctx2.get_conversation_history()
        assert len(loaded_history) == 2  # Previous conversation loaded
        assert "Alice" in loaded_history[0].content

        result2 = await memory_agent.run("What's my name?", context=ctx2)

        # After second run, should have 4 messages total
        history2 = ctx2.get_conversation_history()
        assert len(history2) == 4  # Previous 2 + new user + new assistant

        # Agent should remember the name from previous conversation
        assert "Alice" in result2.output or "alice" in result2.output.lower()

    finally:
        from agnt5.entity import _entity_state_manager_ctx

        _entity_state_manager_ctx.reset(token)
        manager.clear_all()


@pytest.mark.asyncio
async def test_agent_different_sessions_isolated(memory_agent):
    """Test that different sessions don't share conversation history."""
    from agnt5.entity import create_entity_context, EntityStateManager

    manager, token = create_entity_context()

    # Create a shared state manager for all contexts
    shared_state_manager = EntityStateManager()

    try:
        # First session
        ctx1 = AgentContext(
            run_id="run-1",
            agent_name="memory_agent",
            session_id="session-A",
            state_manager=shared_state_manager,
        )
        await memory_agent.run("My favorite color is blue.", context=ctx1)

        # Second session (different session_id, same state manager)
        ctx2 = AgentContext(
            run_id="run-2",
            agent_name="memory_agent",
            session_id="session-B",
            state_manager=shared_state_manager,
        )

        # Should NOT have session A's history
        history = ctx2.get_conversation_history()
        assert len(history) == 0  # No history from session-A

        await memory_agent.run("My favorite color is red.", context=ctx2)

        # Verify session A history unchanged (reload with same session ID and state manager)
        ctx1_reload = AgentContext(
            run_id="run-3",
            agent_name="memory_agent",
            session_id="session-A",
            state_manager=shared_state_manager,
        )
        history_a = ctx1_reload.get_conversation_history()
        assert len(history_a) == 2  # Only session A's messages
        assert "blue" in history_a[0].content.lower()

    finally:
        from agnt5.entity import _entity_state_manager_ctx

        _entity_state_manager_ctx.reset(token)
        manager.clear_all()


@pytest.mark.asyncio
async def test_agent_context_state_persistence():
    """Test that AgentContext state persists custom data."""
    from agnt5.entity import create_entity_context, EntityStateManager

    manager, token = create_entity_context()

    # Create a shared state manager for both contexts
    shared_state_manager = EntityStateManager()

    try:
        ctx1 = AgentContext(
            run_id="run-1",
            agent_name="test_agent",
            session_id="session-123",
            state_manager=shared_state_manager,
        )

        # Store custom state
        ctx1.state.set("user_preferences", {"theme": "dark", "language": "en"})
        ctx1.state.set("interaction_count", 5)

        # Create new context with same session and state manager
        ctx2 = AgentContext(
            run_id="run-2",
            agent_name="test_agent",
            session_id="session-123",
            state_manager=shared_state_manager,
        )

        # State should persist
        prefs = ctx2.state.get("user_preferences")
        assert prefs == {"theme": "dark", "language": "en"}

        count = ctx2.state.get("interaction_count")
        assert count == 5

    finally:
        from agnt5.entity import _entity_state_manager_ctx

        _entity_state_manager_ctx.reset(token)
        manager.clear_all()


@pytest.mark.asyncio
async def test_agent_with_stateful_tool():
    """Test agent using tools that store state in AgentContext."""
    from agnt5.entity import create_entity_context, EntityStateManager

    # Skip if no API key available
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    manager, token = create_entity_context()

    # Create a shared state manager for both contexts
    shared_state_manager = EntityStateManager()

    try:

        @tool(auto_schema=True)
        async def remember_fact(ctx: Context, fact: str) -> str:
            """Remember a fact for later.

            Args:
                fact: The fact to remember
            """
            if isinstance(ctx, AgentContext):
                facts = ctx.state.get("remembered_facts", [])
                facts.append(fact)
                ctx.state.set("remembered_facts", facts)
                return f"Remembered: {fact}"
            return "Cannot remember (no agent context)"

        @tool(auto_schema=True)
        async def recall_facts(ctx: Context) -> str:
            """Recall all remembered facts.

            Returns:
                A string containing all remembered facts
            """
            if isinstance(ctx, AgentContext):
                facts = ctx.state.get("remembered_facts", [])
                if not facts:
                    return "No facts remembered yet."
                return "Remembered facts: " + ", ".join(facts)
            return "Cannot recall (no agent context)"

        agent = Agent(
            name="memory_agent",
            model="openai/gpt-4o-mini",
            instructions="Help users remember and recall facts. Use the remember_fact tool to store facts and recall_facts tool to retrieve them.",
            tools=[remember_fact, recall_facts],
            temperature=0.7,
        )

        # First interaction - remember a fact
        ctx1 = AgentContext(
            run_id="run-1",
            agent_name="memory_agent",
            session_id="session-123",
            state_manager=shared_state_manager,
        )
        result1 = await agent.run("Remember that Python is awesome", context=ctx1)

        # Verify the agent called the tool (check tool_calls in result)
        assert result1.tool_calls  # Agent used tools

        # Verify fact was stored in state
        facts = ctx1.state.get("remembered_facts", [])
        assert len(facts) > 0  # At least one fact stored
        assert any("Python" in fact and "awesome" in fact for fact in facts)

        # Second interaction - recall facts (same session, same state manager)
        ctx2 = AgentContext(
            run_id="run-2",
            agent_name="memory_agent",
            session_id="session-123",
            state_manager=shared_state_manager,
        )
        result2 = await agent.run("What facts do you remember?", context=ctx2)

        # Facts should persist across runs
        recalled_facts = ctx2.state.get("remembered_facts", [])
        assert len(recalled_facts) > 0  # Facts persisted
        assert any("Python" in fact and "awesome" in fact for fact in recalled_facts)

        # Agent's response should mention the fact
        assert "Python" in result2.output or "python" in result2.output.lower()

    finally:
        from agnt5.entity import _entity_state_manager_ctx
        from agnt5.tool import ToolRegistry

        _entity_state_manager_ctx.reset(token)
        manager.clear_all()
        ToolRegistry.clear()


# TODO: Add platform integration tests that verify:
# 1. Agent memory persists through worker restarts
# 2. Multiple concurrent agent sessions maintain separate state
# 3. Agent state is properly checkpointed and recovered
# 4. Long conversation histories don't cause memory issues
#
# These tests would follow the pattern in test_entity_definition_persistence.py
# but for agents instead of entities.
