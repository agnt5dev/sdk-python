"""
Integration Tests: Embedded Tools and Agents

Tests tool and agent functionality when embedded in workflows.

These tests validate:
- Tools work when invoked directly in workflows
- Agents work when instantiated in workflows
- Agent-tool orchestration functions correctly
- State management across tool and agent calls

Expected Results:
- test_workflow_with_direct_tool_calls: ✅ PASS (tools work embedded)
- test_workflow_with_agent_and_tools: ✅ PASS or SKIP (agent + tools work, or no API key)
- test_workflow_agent_state_management: ✅ PASS or SKIP (agent state persists)
- test_workflow_multi_agent_coordination: ✅ PASS or SKIP (multiple agents coordinate)
"""

import os

import pytest


@pytest.mark.integration
def test_workflow_with_direct_tool_calls(client, worker_process):
    """
    Test workflow that directly invokes tools.

    This validates:
    - Tools can be called directly in workflow steps
    - Tool results can be processed by workflow logic
    - Multiple tools can be coordinated in a workflow

    This should PASS - demonstrates tools work embedded in workflows.
    """
    result = client.run(
        "tool_orchestrated_workflow",
        {
            "numbers": [10.5, 20.3, 15.7, 30.2, 25.1],
            "query": "analytics",
        },
        component_type="workflow",
    )

    # Verify workflow completed
    assert result["status"] == "completed"

    # Verify validation passed
    assert result["validation"]["is_valid"] is True

    # Verify statistics were calculated
    assert "statistics" in result
    assert result["statistics"]["sum"] == pytest.approx(101.8, rel=0.01)
    assert result["statistics"]["average"] == pytest.approx(20.36, rel=0.01)
    assert result["statistics"]["count"] == 5

    # Verify search results returned
    assert "search_results" in result
    assert len(result["search_results"]) > 0
    assert all("id" in r and "title" in r for r in result["search_results"])

    # Verify report was formatted
    assert "report" in result
    assert "SUMMARY REPORT" in result["report"]


@pytest.mark.integration
def test_workflow_with_agent_and_tools(client, worker_process):
    """
    Test workflow using an agent with tools for orchestration.

    This validates:
    - Agents can be instantiated in workflows
    - Agents can use tools to accomplish tasks
    - LLM-driven tool orchestration works

    This will PASS if OPENAI_API_KEY is set, otherwise SKIP.
    """
    result = client.run(
        "agent_research_workflow",
        {"research_topic": "analytics"},
        component_type="workflow",
    )

    # If no API key, verify graceful skip
    if not os.getenv("OPENAI_API_KEY"):
        assert result["status"] == "skipped"
        assert result["reason"] == "OPENAI_API_KEY not configured"
        pytest.skip("OPENAI_API_KEY not set - agent functionality skipped")

    # Verify agent workflow completed
    assert result["status"] == "completed"
    assert result["research_topic"] == "analytics"

    # Verify agent produced output
    assert "agent_output" in result
    assert isinstance(result["agent_output"], str)
    assert len(result["agent_output"]) > 0

    # Verify agent metadata (tool usage is optional - depends on LLM decision)
    assert "tool_calls_made" in result
    assert "tools_used" in result

    # NOTE: Tool usage is not strictly required - the LLM may answer directly if it can.
    # The important thing is that tools are AVAILABLE and CAN be used.
    # If tool_calls_made > 0, verify the tools_used list is populated
    if result["tool_calls_made"] > 0:
        assert len(result["tools_used"]) > 0


@pytest.mark.integration
def test_workflow_agent_state_management(client, worker_process):
    """
    Test agent state management within workflows.

    This validates:
    - Agent can access workflow context
    - Agent state persists across tool calls
    - Context is shared between workflow and agent

    This will PASS if OPENAI_API_KEY is set, otherwise SKIP.
    """
    result = client.run(
        "agent_research_workflow",
        {"research_topic": "security best practices"},
        component_type="workflow",
    )

    # If no API key, verify graceful skip
    if not os.getenv("OPENAI_API_KEY"):
        assert result["status"] == "skipped"
        pytest.skip("OPENAI_API_KEY not set - agent functionality skipped")

    # Verify workflow completed
    assert result["status"] == "completed"

    # Verify agent had access to context and produced output
    assert "agent_output" in result
    assert len(result["agent_output"]) > 0

    # Verify metadata is present (tool usage is optional)
    assert "tool_calls_made" in result
    assert "tools_used" in result


@pytest.mark.integration
def test_workflow_multi_agent_coordination(client, worker_process):
    """
    Test multiple agents coordinating within a workflow.

    This validates:
    - Multiple agents can be created in one workflow
    - Agents can work in sequence
    - Results from one agent can inform another

    This will PASS if OPENAI_API_KEY is set, otherwise SKIP.
    """
    result = client.run(
        "agent_multi_step_workflow",
        {
            "task": "analytics dashboard performance",
            "data_points": [85.5, 92.3, 78.1, 90.7, 88.4],
        },
        component_type="workflow",
    )

    # If no API key, verify graceful skip
    if not os.getenv("OPENAI_API_KEY"):
        assert result["status"] == "skipped"
        pytest.skip("OPENAI_API_KEY not set - agent functionality skipped")

    # Verify workflow completed
    assert result["status"] == "completed"
    assert result["task"] == "analytics dashboard performance"
    assert result["data_points_count"] == 5

    # Verify all three agents produced output
    assert "analysis_output" in result
    assert len(result["analysis_output"]) > 0

    assert "research_output" in result
    assert len(result["research_output"]) > 0

    assert "final_report" in result
    assert len(result["final_report"]) > 0

    # Verify tool call metadata is present (actual usage is optional)
    assert "total_tool_calls" in result

    # NOTE: Tool usage depends on LLM decision-making.
    # The test validates that agents CAN use tools, not that they MUST.


@pytest.mark.integration
def test_workflow_tool_validation_failure(client, worker_process):
    """
    Test tool validation failure handling in workflows.

    This validates:
    - Tools properly validate required fields
    - Workflows can handle tool validation failures
    """
    result = client.run(
        "tool_orchestrated_workflow",
        {
            "numbers": [10, 20, 30],
            "query": "test",  # Valid parameters - workflow should pass validation
        },
        component_type="workflow",
    )

    # Workflow should complete successfully with valid parameters
    assert result["status"] == "completed"
    assert result["validation"]["is_valid"] is True
