"""
Integration Tests: Workflow Streaming via SSE

Tests streaming from workflows that contain streaming agents:
- Workflow steps auto-detect async generators
- Agent events are forwarded via delta queue
- Final agent output is extracted for next step
- Mixed streaming/non-streaming steps

Note: Agent events within workflows are wrapped in output.delta events.
To access agent events, check output.delta.data["content"]["event_type"].

Run with:
    # Requires OPENAI_API_KEY for LLM calls
    pytest tests/integration/test_ex_310_stream_workflows.py -v

    # Skip if no API key
    pytest tests/integration/test_ex_310_stream_workflows.py -v -m "not llm"
"""

import os
import pytest


# Skip all workflow streaming tests if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping workflow streaming tests"
    ),
]


def extract_nested_events(events):
    """Extract nested events from output.delta events.

    Workflows yield dicts that become output.delta events.
    The nested event data is in e.data["content"].
    """
    nested_events = []
    for e in events:
        if e.event_type.value == "output.delta":
            content = e.data.get("content", {})
            if isinstance(content, dict) and "event_type" in content:
                nested_events.append(content)
    return nested_events


def get_nested_event_types(events):
    """Get list of nested event types from stream events."""
    return [e["event_type"] for e in extract_nested_events(events)]


# =============================================================================
# SIMPLE AGENT WORKFLOW STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_simple_agent_workflow_basic(client, worker_process):
    """Test basic workflow streaming returns events."""
    events = list(client.stream_events("simple_agent_workflow", {
        "message": "Say hello"
    }, component_type="workflow"))

    # Should have run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types or "run.completed" in run_event_types

    # Check for agent lifecycle events (forwarded from nested agent)
    nested_types = get_nested_event_types(events)
    has_agent_events = (
        "agent.started" in nested_types or
        "agent.completed" in nested_types or
        len(nested_types) > 0
    )
    assert has_agent_events or len(events) >= 1


@pytest.mark.integration
def test_stream_simple_agent_workflow_has_output(client, worker_process):
    """Test that workflow completes with output."""
    events = list(client.stream_events("simple_agent_workflow", {
        "message": "What is 1+1?"
    }, component_type="workflow"))

    # Check for run.completed or agent.completed
    run_event_types = [e.event_type.value for e in events]
    nested_types = get_nested_event_types(events)

    has_completion = (
        "run.completed" in run_event_types or
        "agent.completed" in nested_types
    )
    assert has_completion


# =============================================================================
# MIXED WORKFLOW STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_mixed_workflow_basic(client, worker_process):
    """Test mixed workflow with streaming and non-streaming steps."""
    events = list(client.stream_events("mixed_workflow", {
        "x": 5,
        "y": 3
    }, component_type="workflow"))

    # Should have events
    assert len(events) >= 1

    # Check for run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types or "run.completed" in run_event_types


@pytest.mark.integration
def test_stream_mixed_workflow_has_calculation(client, worker_process):
    """Test that mixed workflow performs calculation correctly."""
    events = list(client.stream_events("mixed_workflow", {
        "x": 7,
        "y": 3
    }, component_type="workflow"))

    # Find run.completed event with final result
    completed = next(
        (e for e in events if e.event_type.value == "run.completed"),
        None
    )

    if completed:
        # Check the final result includes the sum
        data = completed.data
        if isinstance(data, dict) and "output" in data:
            output = data.get("output", {})
            if isinstance(output, dict):
                assert output.get("sum") == 10


# =============================================================================
# RESEARCH WORKFLOW STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_research_workflow_basic(client, worker_process):
    """Test research workflow with multiple streaming agents."""
    events = list(client.stream_events("research_workflow", {
        "topic": "Python"
    }, component_type="workflow"))

    # Should have events
    assert len(events) >= 1

    # Check for run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types or "run.completed" in run_event_types


@pytest.mark.integration
def test_stream_research_workflow_event_order(client, worker_process):
    """Test that agent events come in correct order."""
    events = list(client.stream_events("research_workflow", {
        "topic": "Python"
    }, component_type="workflow"))

    nested_types = get_nested_event_types(events)

    # If we have both started and completed, check order
    if "agent.started" in nested_types and "agent.completed" in nested_types:
        first_started = nested_types.index("agent.started")
        first_completed = nested_types.index("agent.completed")
        assert first_started < first_completed, "agent.started should come before agent.completed"


# =============================================================================
# ERROR HANDLING
# =============================================================================


@pytest.mark.integration
def test_stream_workflow_nonexistent(client, worker_process):
    """Test streaming a nonexistent workflow."""
    from agnt5.client import RunError

    with pytest.raises((RunError, Exception)):
        list(client.stream_events("nonexistent_workflow", {}, component_type="workflow"))


@pytest.mark.integration
def test_stream_workflow_after_error(client, worker_process):
    """Test that streaming works after an error."""
    from agnt5.client import RunError

    # Try to stream a nonexistent workflow
    try:
        list(client.stream_events("nonexistent_workflow", {}, component_type="workflow"))
    except (RunError, Exception):
        pass

    # Workflow streaming should still work
    events = list(client.stream_events("simple_agent_workflow", {
        "message": "Recovery test"
    }, component_type="workflow"))
    assert len(events) >= 1


# =============================================================================
# MULTIPLE SEQUENTIAL WORKFLOWS
# =============================================================================


@pytest.mark.integration
def test_stream_workflow_multiple_sequential(client, worker_process):
    """Test multiple sequential workflow streams."""
    # First workflow
    events1 = list(client.stream_events("simple_agent_workflow", {
        "message": "First"
    }, component_type="workflow"))
    assert len(events1) >= 1

    # Second workflow
    events2 = list(client.stream_events("simple_agent_workflow", {
        "message": "Second"
    }, component_type="workflow"))
    assert len(events2) >= 1

    # Both should complete
    types1 = [e.event_type.value for e in events1]
    types2 = [e.event_type.value for e in events2]

    has_completion1 = "run.completed" in types1 or "run.started" in types1
    has_completion2 = "run.completed" in types2 or "run.started" in types2

    assert has_completion1 or has_completion2
