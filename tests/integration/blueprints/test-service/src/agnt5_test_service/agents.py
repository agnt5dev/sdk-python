"""Agent components for integration testing."""

from agnt5 import Agent, agent


@agent
def chat_agent():
    """Chat agent for session testing.

    This agent is used in test_agent_session.py to verify:
    - Agent sessions are created automatically
    - session_id is returned in response metadata
    - Conversation history is maintained across multiple turns
    - Agent remembers context from previous messages
    """
    return Agent(
        name="chat_agent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful chatbot. Remember what the user tells you and maintain conversation context.",
        temperature=0.7,
    )
