"""
Integration Tests for ex_07_agents_with_tools.py

Tests agents with tool capabilities through the platform:
- calculator_agent - Agent with calculator tool
- weather_assistant - Agent with weather tool
- multi_tool_agent - Agent with multiple tools
- research_coordinator - Multi-agent with agents-as-tools
- support_router - Multi-agent with handoffs

Run with:
    # Local mode (requires running dev server with examples worker)
    # Requires OPENAI_API_KEY environment variable
    pytest tests/integration/test_ex_07_agents_with_tools.py -v

    # Skip LLM tests if no API key
    pytest tests/integration/test_ex_07_agents_with_tools.py -v -m "not llm"
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
# CALCULATOR AGENT
# =============================================================================


def test_calculator_agent_simple(client, worker_process):
    """Test calculator agent with simple math."""
    result = client.run("calculator_agent", {
        "question": "What is 15 times 23?",
    }, component_type="workflow")

    assert "response" in result
    assert "345" in result["response"]
    assert result["tool_calls"] >= 1


def test_calculator_agent_addition(client, worker_process):
    """Test calculator agent with addition."""
    result = client.run("calculator_agent", {
        "question": "What is 100 + 200 + 300?",
    }, component_type="workflow")

    assert "response" in result
    assert "600" in result["response"]


def test_calculator_agent_complex(client, worker_process):
    """Test calculator agent with multi-step calculation."""
    result = client.run("calculator_agent", {
        "question": "Calculate 10 * 5, then add 25",
    }, component_type="workflow")

    assert "response" in result
    assert "75" in result["response"]
    # Should use tool multiple times or do compound calculation
    assert result["tool_calls"] >= 1


# =============================================================================
# WEATHER ASSISTANT
# =============================================================================


def test_weather_assistant_single_city(client, worker_process):
    """Test weather assistant for one city."""
    result = client.run("weather_assistant", {
        "query": "What's the weather in Tokyo?",
    }, component_type="workflow")

    assert "response" in result
    assert result["tool_calls"] >= 1
    # Should mention Tokyo weather data
    response_lower = result["response"].lower()
    assert "tokyo" in response_lower or "68" in result["response"]


def test_weather_assistant_comparison(client, worker_process):
    """Test weather assistant comparing cities."""
    result = client.run("weather_assistant", {
        "query": "Compare weather in New York and London",
    }, component_type="workflow")

    assert "response" in result
    assert result["tool_calls"] >= 2  # Should call for both cities


def test_weather_assistant_unknown_city(client, worker_process):
    """Test weather assistant with unknown city."""
    result = client.run("weather_assistant", {
        "query": "What's the weather in Atlantis?",
    })

    assert "response" in result
    # Should handle gracefully and mention data not available


# =============================================================================
# MULTI-TOOL AGENT
# =============================================================================


def test_multi_tool_agent_calculation(client, worker_process):
    """Test multi-tool agent using calculator."""
    result = client.run("multi_tool_agent", {
        "task": "Calculate 500 * 3",
    })

    assert "response" in result
    assert "1500" in result["response"]
    assert "calculate" in result["tools_used"]


def test_multi_tool_agent_search(client, worker_process):
    """Test multi-tool agent using search."""
    result = client.run("multi_tool_agent", {
        "task": "Search for quarterly report information",
    })

    assert "response" in result
    assert "search_database" in result["tools_used"]


def test_multi_tool_agent_email(client, worker_process):
    """Test multi-tool agent sending email."""
    result = client.run("multi_tool_agent", {
        "task": "Send an email to test@example.com with subject 'Hello' and body 'Test message'",
    })

    assert "response" in result
    assert "send_email" in result["tools_used"]


def test_multi_tool_agent_multiple_tools(client, worker_process):
    """Test multi-tool agent using multiple tools."""
    result = client.run("multi_tool_agent", {
        "task": "Calculate 100 * 5 and search for project updates",
    })

    assert "response" in result
    assert len(result["tools_used"]) >= 2


# =============================================================================
# RESEARCH COORDINATOR (Multi-Agent: Agents-as-Tools)
# =============================================================================


@pytest.mark.integration
def test_research_coordinator_basic(client, worker_process):
    """Test research coordinator with simple topic."""
    result = client.run("research_coordinator", {
        "topic": "artificial intelligence",
    })

    assert "response" in result
    assert result["tool_calls"] >= 1  # Should invoke specialist agents


@pytest.mark.integration
def test_research_coordinator_analysis(client, worker_process):
    """Test research coordinator requiring analysis."""
    result = client.run("research_coordinator", {
        "topic": "renewable energy trends",
    })

    assert "response" in result
    # Response should be synthesized from specialists


# =============================================================================
# SUPPORT ROUTER (Multi-Agent: Handoffs)
# =============================================================================


def test_support_router_technical(client, worker_process):
    """Test support router with technical question."""
    result = client.run("support_router", {
        "query": "How do I reset my password?",
    })

    assert "response" in result
    assert result["routed"] is True
    assert result["handoff_to"] == "technical_support"


def test_support_router_billing(client, worker_process):
    """Test support router with billing question."""
    result = client.run("support_router", {
        "query": "I have a question about my invoice",
    })

    assert "response" in result
    assert result["routed"] is True
    assert result["handoff_to"] == "billing_support"


def test_support_router_general(client, worker_process):
    """Test support router with general question."""
    result = client.run("support_router", {
        "query": "What features does your product have?",
    })

    assert "response" in result
    assert result["routed"] is True
    assert result["handoff_to"] == "general_support"


# =============================================================================
# TOOL BEHAVIOR (P1.3 - Require tool_calls validation)
# =============================================================================


def test_calculator_agent_validates_tool_calls(client, worker_process):
    """Test that calculator agent MUST use tools for math (P1.3).

    This test validates that tool-using agents actually record their tool calls.
    A calculator agent MUST use the calculate tool for math operations.
    """
    result = client.run("calculator_agent", {
        "question": "What is 42 * 17?",  # Clear math question
    })

    # Response must exist
    assert "response" in result
    assert "714" in result["response"], (
        f"Expected 714 (42*17) in response, got: {result['response']}"
    )

    # Tool calls MUST be present and non-zero
    assert "tool_calls" in result, "Calculator agent must record tool_calls"
    assert isinstance(result["tool_calls"], int), "tool_calls should be an integer count"
    assert result["tool_calls"] >= 1, (
        f"Calculator agent MUST use tools for math. "
        f"Expected at least 1 tool call, got {result['tool_calls']}"
    )


def test_tool_returns_correct_format(client, worker_process):
    """Test that tool results are properly formatted."""
    result = client.run("calculator_agent", {
        "question": "What is 7 + 8?",
    })

    assert "question" in result
    assert "response" in result
    assert "tool_calls" in result
    assert isinstance(result["tool_calls"], int)


def test_agent_continues_after_tool_error(client, worker_process):
    """Test that agent handles tool errors gracefully."""
    result = client.run("calculator_agent", {
        "question": "What is undefined divided by zero?",
    })

    # Agent should handle the error and still provide a response
    assert "response" in result
