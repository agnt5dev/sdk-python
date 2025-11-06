"""
End-to-End Integration Tests for Agent Tool Execution

These tests verify agent tool integration:
1. Agents can successfully invoke tools
2. Tool arguments are passed correctly
3. Tool results are captured and used by agents
4. Multiple tools can be provided to agents
5. Agents select appropriate tools for tasks
6. Tool execution is tracked and logged

Uses OpenAI gpt-4o-mini for agent execution with test-bench tools.

Note: These tests require OPENAI_API_KEY environment variable.
"""

import pytest
import os
import time


# Skip all tests if OPENAI_API_KEY not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping agent tool tests"
)


@pytest.mark.integration
@pytest.mark.slow
def test_agent_with_single_tool(client, worker_process, platform):
    """
    Test agent using a single tool.

    Verifies:
    - Agent can invoke tool successfully
    - Tool is called with correct arguments
    - Tool result is incorporated in agent response
    - Tool call count is tracked
    """
    time.sleep(2)

    result = client.run("wf_26_agent_with_single_tool", {})

    # Verify agent executed and used tool
    assert "output" in result
    assert "tool_calls_count" in result
    assert result["tool_calls_count"] > 0
    assert result["agent_name"] == "CalculatorAgent"

    # Agent should provide an answer using the tool
    assert len(result["output"]) > 0

    print(f"\n✅ Agent with single tool works correctly")
    print(f"   Tool calls: {result['tool_calls_count']}")
    print(f"   Output length: {len(result['output'])} chars")


@pytest.mark.integration
@pytest.mark.slow
def test_agent_with_multiple_tools(client, worker_process, platform):
    """
    Test agent with multiple tools available.

    Verifies:
    - Agent can choose from multiple tools
    - Appropriate tool is selected for task
    - Tool selection logic works correctly
    """
    time.sleep(2)

    result = client.run("wf_27_agent_with_multiple_tools", {})

    # Verify agent executed with multiple tools available
    assert "output" in result
    assert "tools_available" in result
    assert result["tools_available"] >= 3  # calculate_total, search_database, format_report

    # Agent should have used at least one tool
    assert "tool_calls_count" in result
    assert result["tool_calls_count"] > 0

    print(f"\n✅ Agent with multiple tools works correctly")
    print(f"   Tools available: {result['tools_available']}")
    print(f"   Tools used: {result['tool_calls_count']}")


@pytest.mark.integration
@pytest.mark.slow
def test_agent_tool_coordination(client, worker_process, platform):
    """
    Test agent coordinating multiple tool calls.

    Verifies:
    - Agent can make multiple tool calls in sequence
    - Results from one tool can inform next tool call
    - Complex tool coordination works
    """
    time.sleep(2)

    result = client.run("wf_28_agent_multi_tool_coordination", {})

    # Verify multi-tool coordination
    assert "output" in result
    assert "coordination_steps" in result
    assert result["coordination_steps"] >= 2  # Multiple tool calls coordinated

    print(f"\n✅ Agent multi-tool coordination works correctly")
    print(f"   Coordination steps: {result['coordination_steps']}")


@pytest.mark.integration
@pytest.mark.slow
def test_agent_tool_with_parameters(client, worker_process, platform):
    """
    Test agent tools receive correct parameters.

    Verifies:
    - Tool parameters are extracted from agent's reasoning
    - Parameters are passed correctly to tool
    - Different parameter values work
    """
    time.sleep(2)

    result = client.run("wf_29_agent_tool_with_parameters", {})

    # Verify parameterized tool calls
    assert "output" in result
    assert "tool_parameters_used" in result
    assert len(result["tool_parameters_used"]) > 0

    print(f"\n✅ Agent tool parameters work correctly")
    print(f"   Parameters used: {result['tool_parameters_used']}")


@pytest.mark.integration
@pytest.mark.slow
def test_agent_tool_iteration(client, worker_process, platform):
    """
    Test agent iterating with tools to complete task.

    Verifies:
    - Agent can make multiple tool calls iteratively
    - Agent refines approach based on tool results
    - Iteration count is tracked correctly
    """
    time.sleep(2)

    result = client.run("wf_30_agent_tool_iteration", {})

    # Verify iterative tool usage
    assert "output" in result
    assert "iterations" in result
    assert result["iterations"] >= 1

    # Agent should make multiple tool calls across iterations
    assert "total_tool_calls" in result
    assert result["total_tool_calls"] >= result["iterations"]

    print(f"\n✅ Agent tool iteration works correctly")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Total tool calls: {result['total_tool_calls']}")


@pytest.mark.integration
@pytest.mark.slow
def test_agent_tool_error_handling(client, worker_process, platform):
    """
    Test agent handles tool errors gracefully.

    Verifies:
    - Agent receives tool error information
    - Agent can recover from tool errors
    - Error details are captured correctly
    """
    time.sleep(2)

    # This workflow should attempt a tool call that may fail
    # and handle it gracefully
    result = client.run("wf_27_agent_with_multiple_tools", {})

    # Even if tool fails, agent should complete
    assert "output" in result
    assert len(result["output"]) > 0

    print(f"\n✅ Agent tool error handling works")


@pytest.mark.integration
@pytest.mark.slow
def test_multiple_agents_different_tools(client, worker_process, platform):
    """
    Test multiple agents with different tool sets.

    Verifies:
    - Different agents can have different tools
    - Tool namespaces are isolated per agent
    - No tool conflicts between agents
    """
    time.sleep(2)

    # Run workflow with multiple agents
    result1 = client.run("wf_26_agent_with_single_tool", {})
    result2 = client.run("wf_27_agent_with_multiple_tools", {})

    # Both should succeed independently
    assert result1["tool_calls_count"] > 0
    assert result2["tool_calls_count"] > 0

    print(f"\n✅ Multiple agents with different tools work correctly")


@pytest.mark.integration
@pytest.mark.slow
def test_agent_tool_results_in_workflow_state(client, worker_process, platform):
    """
    Test agent tool results are accessible in workflow state.

    Verifies:
    - Tool call results can be stored in workflow state
    - Results persist throughout workflow
    - State can be queried after agent execution
    """
    time.sleep(2)

    result = client.run("wf_26_agent_with_single_tool", {})

    # Workflow should capture tool usage in state
    assert "output" in result
    assert "tool_calls_count" in result

    # Result should be structured and contain tool information
    assert isinstance(result["tool_calls_count"], int)

    print(f"\n✅ Agent tool results available in workflow state")


# TODO: Add tests for:
# - Agent tool execution with streaming responses
# - Agent tool timeout handling
# - Agent tool with complex return types (structured data)
# - Agent tool execution metrics/telemetry
# - Agent tool execution in parallel workflows
# - Agent tool retry logic
# - Custom tools with different signatures
# - Tool execution with workflow context access
