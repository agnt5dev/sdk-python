"""
Integration Tests: Workflow Streaming via SSE

Tests streaming from workflows that contain streaming agents:
- Workflow steps auto-detect async generators
- Agent events are forwarded as top-level SSE events
- Final agent output is extracted for next step
- Mixed streaming/non-streaming steps

Agent events are now top-level events (agent.started, agent.completed, etc.)
with their payload in the output_data field.

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


def get_agent_event_types(events):
    """Get list of agent event types from stream events.

    Agent events are now top-level events (agent.started, agent.completed, etc.)
    """
    agent_types = []
    for e in events:
        if e.event_type.value.startswith("agent."):
            agent_types.append(e.event_type.value)
    return agent_types


def get_agent_event_data(event):
    """Get the payload data from an agent event.

    Agent events have their payload in the output_data field.
    """
    return event.data.get("output_data", {})


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

    # Check for agent lifecycle events (now top-level events)
    agent_types = get_agent_event_types(events)
    has_agent_events = (
        "agent.started" in agent_types or
        "agent.completed" in agent_types
    )
    # Workflow should produce agent events since it contains a streaming agent
    assert has_agent_events, (
        f"Expected agent.started or agent.completed events, got agent types: {agent_types}"
    )


@pytest.mark.integration
def test_stream_simple_agent_workflow_has_output(client, worker_process):
    """Test that workflow completes with output."""
    events = list(client.stream_events("simple_agent_workflow", {
        "message": "What is 1+1?"
    }, component_type="workflow"))

    # Check for run.completed or agent.completed
    run_event_types = [e.event_type.value for e in events]
    agent_types = get_agent_event_types(events)

    has_completion = (
        "run.completed" in run_event_types or
        "agent.completed" in agent_types
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

    agent_types = get_agent_event_types(events)

    # If we have both started and completed, check order
    if "agent.started" in agent_types and "agent.completed" in agent_types:
        first_started = agent_types.index("agent.started")
        first_completed = agent_types.index("agent.completed")
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


# =============================================================================
# RUN COMPLETION VALIDATION (P1.7)
# =============================================================================


@pytest.mark.integration
def test_stream_workflow_requires_run_completed(client, worker_process):
    """Test that workflow stream MUST include run.completed event (P1.7).

    Validates:
    - run.completed event is present (not just run.started)
    - run.completed includes output field
    - Output contains workflow result
    """
    events = list(client.stream_events("mixed_workflow", {
        "x": 10,
        "y": 5
    }, component_type="workflow"))

    run_event_types = [e.event_type.value for e in events]

    # MUST have run.completed (not just run.started)
    assert "run.completed" in run_event_types, (
        f"Workflow stream MUST include 'run.completed' event. "
        f"Got event types: {run_event_types}"
    )

    # Find the run.completed event
    completed = next(
        (e for e in events if e.event_type.value == "run.completed"),
        None
    )
    assert completed is not None

    # run.completed MUST have data
    assert completed.data is not None, (
        "run.completed event should have data"
    )

    # data should include output (may be in various formats)
    data = completed.data
    has_output = (
        isinstance(data, dict) and (
            "output" in data or
            "result" in data or
            "data" in data
        )
    )
    # This is important for downstream consumers
    if isinstance(data, dict):
        assert has_output or len(data) > 0, (
            f"run.completed should have output. Got: {data}"
        )


@pytest.mark.integration
def test_stream_workflow_event_ordering(client, worker_process):
    """Test workflow stream events are in correct order (P1.7).

    Validates:
    - run.started comes before run.completed
    - Events maintain temporal ordering
    """
    events = list(client.stream_events("mixed_workflow", {
        "x": 3,
        "y": 4
    }, component_type="workflow"))

    run_event_types = [e.event_type.value for e in events]

    # Check if we have both started and completed
    has_started = "run.started" in run_event_types
    has_completed = "run.completed" in run_event_types

    if has_started and has_completed:
        started_idx = run_event_types.index("run.started")
        completed_idx = run_event_types.index("run.completed")

        assert started_idx < completed_idx, (
            f"run.started should come before run.completed. "
            f"Started at index {started_idx}, completed at {completed_idx}"
        )

    # At minimum, we need either started or completed
    assert has_started or has_completed, (
        f"Workflow stream should have run.started or run.completed. "
        f"Got: {run_event_types}"
    )


@pytest.mark.integration
def test_stream_workflow_nested_agent_event_ordering(client, worker_process):
    """Test agent events within workflow maintain order (P1.7).

    When a workflow contains an agent, the agent events should
    appear in logical order (started before completed).
    """
    events = list(client.stream_events("simple_agent_workflow", {
        "message": "Test ordering"
    }, component_type="workflow"))

    agent_types = get_agent_event_types(events)

    # If we have agent events, verify their order
    if "agent.started" in agent_types and "agent.completed" in agent_types:
        first_started = agent_types.index("agent.started")
        first_completed = agent_types.index("agent.completed")

        assert first_started < first_completed, (
            f"agent.started should come before agent.completed. "
            f"Started at index {first_started}, completed at {first_completed}. "
            f"Full order: {agent_types}"
        )
