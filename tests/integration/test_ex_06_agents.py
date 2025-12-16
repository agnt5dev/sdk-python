"""
Integration Tests for ex_06_agents.py

Tests basic AI agent features through the platform:
- simple_agent - Basic OpenAI agent
- claude_agent - Anthropic Claude agent
- code_assistant - Specialized coding agent
- creative_writer - Creative writing agent
- conversation_agent - Multi-turn conversation

Run with:
    # Local mode (requires running dev server with examples worker)
    # Requires OPENAI_API_KEY environment variable
    pytest tests/integration/test_ex_06_agents.py -v

    # Skip LLM tests if no API key
    pytest tests/integration/test_ex_06_agents.py -v -m "not llm"
"""

import os
import pytest

# Skip all agent tests if no API key is available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping agent tests"
    ),
]


# =============================================================================
# SIMPLE AGENT
# =============================================================================


def test_simple_agent_basic(client, worker_process):
    """Test basic agent response."""
    result = client.run("simple_agent", {
        "message": "What is 2 + 2?",
    })

    assert "response" in result
    assert "4" in result["response"]
    assert result["model"] == "openai/gpt-4o-mini"


def test_simple_agent_longer_question(client, worker_process):
    """Test agent with a longer question."""
    result = client.run("simple_agent", {
        "message": "Explain what an API is in one sentence.",
    })

    assert "response" in result
    assert len(result["response"]) > 10


# =============================================================================
# CODE ASSISTANT
# =============================================================================


def test_code_assistant_python(client, worker_process):
    """Test code assistant for Python."""
    result = client.run("code_assistant", {
        "question": "How do I read a file in Python?",
        "language": "python",
    })

    assert "response" in result
    assert result["language"] == "python"
    # Should mention file operations
    response_lower = result["response"].lower()
    assert "open" in response_lower or "read" in response_lower


def test_code_assistant_javascript(client, worker_process):
    """Test code assistant for JavaScript."""
    result = client.run("code_assistant", {
        "question": "How do I declare a variable?",
        "language": "javascript",
    })

    assert "response" in result
    assert result["language"] == "javascript"
    # Should mention let, const, or var
    response_lower = result["response"].lower()
    assert any(kw in response_lower for kw in ["let", "const", "var"])


# =============================================================================
# CREATIVE WRITER
# =============================================================================


def test_creative_writer_neutral(client, worker_process):
    """Test creative writer with neutral style."""
    result = client.run("creative_writer", {
        "prompt": "Write a one-sentence description of a sunset.",
        "style": "neutral",
    })

    assert "response" in result
    assert result["style"] == "neutral"
    assert len(result["response"]) > 0


def test_creative_writer_humorous(client, worker_process):
    """Test creative writer with humorous style."""
    result = client.run("creative_writer", {
        "prompt": "Write a short joke about programming.",
        "style": "humorous",
    })

    assert "response" in result
    assert result["style"] == "humorous"


def test_creative_writer_formal(client, worker_process):
    """Test creative writer with formal style."""
    result = client.run("creative_writer", {
        "prompt": "Write a brief announcement about a product launch.",
        "style": "formal",
    })

    assert "response" in result
    assert result["style"] == "formal"


# =============================================================================
# CONVERSATION AGENT
# =============================================================================


def test_conversation_agent_single_turn(client, worker_process):
    """Test single-turn conversation."""
    result = client.run("conversation_agent", {
        "message": "Hello, my name is Alice.",
    })

    assert "response" in result
    assert "history" in result
    assert result["turn_count"] == 1
    assert len(result["history"]) == 2  # user + assistant


def test_conversation_agent_multi_turn(client, worker_process):
    """Test multi-turn conversation with context."""
    # Turn 1
    result1 = client.run("conversation_agent", {
        "message": "My favorite color is blue.",
    })

    assert result1["turn_count"] == 1

    # Turn 2 - ask about what was said
    result2 = client.run("conversation_agent", {
        "message": "What is my favorite color?",
        "history": result1["history"],
    })

    assert result2["turn_count"] == 2
    # Agent should remember the color
    assert "blue" in result2["response"].lower()


def test_conversation_agent_empty_history(client, worker_process):
    """Test conversation with explicitly empty history."""
    result = client.run("conversation_agent", {
        "message": "Hi there!",
        "history": [],
    })

    assert result["turn_count"] == 1
    assert len(result["history"]) == 2


# =============================================================================
# CLAUDE AGENT (requires ANTHROPIC_API_KEY)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_claude_agent_basic(client, worker_process):
    """Test Claude agent response."""
    result = client.run("claude_agent", {
        "message": "What is 3 + 3?",
    })

    assert "response" in result
    assert "6" in result["response"]
    assert result["provider"] == "anthropic"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
def test_claude_agent_explanation(client, worker_process):
    """Test Claude agent with explanation task."""
    result = client.run("claude_agent", {
        "message": "Explain what recursion is in one sentence.",
    })

    assert "response" in result
    assert len(result["response"]) > 20
