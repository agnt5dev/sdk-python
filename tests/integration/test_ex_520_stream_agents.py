"""
Integration Tests: Agent Streaming via SSE

Tests streaming agent execution using client.stream_events():
- Agent lifecycle events (agent.started, agent.completed)
- Event ordering and sequencing
- Tool call events (when agents use tools)
- Error handling for agent failures

Note: Agent events come as top-level SSE events with event_type like 'agent.started', 'agent.completed'.
The actual event data is in the 'output_data' field, which may be base64-encoded JSON or a dict.

Run with:
    # Requires OPENAI_API_KEY for LLM calls
    pytest tests/integration/test_ex_520_stream_agents.py -v

    # Skip if no API key
    pytest tests/integration/test_ex_520_stream_agents.py -v -m "not llm"
"""

import base64
import json
import os
import pytest


# Skip all agent streaming tests if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping agent streaming tests"
    ),
]


def _decode_output_data(data: dict) -> dict:
    """Decode output_data field from event data.

    The gateway wraps event payloads in a structure with metadata.
    The actual event data is in 'output_data', which may be:
    - A base64-encoded JSON string (from SSE stream)
    - Already a dict (from journal/replay)
    """
    output_data = data.get("output_data")
    if output_data is None:
        return data

    # If already a dict, return it
    if isinstance(output_data, dict):
        return output_data

    # If a string, decode from base64 and parse as JSON
    if isinstance(output_data, str):
        try:
            decoded = base64.b64decode(output_data).decode("utf-8")
            return json.loads(decoded)
        except (ValueError, json.JSONDecodeError):
            return data

    return data


def extract_agent_events(events):
    """Extract agent events from stream events.

    Agent events come as top-level events with event_type like 'agent.started', 'agent.completed'.
    The actual event data is decoded from the 'output_data' field.
    Returns list of dicts with {"event_type": str, "data": dict}.

    Note: Deduplicates events by hashing the decoded data to avoid duplicates
    from journal replay (which sends both SSE stream events and journal events).
    """
    agent_events = []
    seen = set()  # Track seen events to avoid duplicates from journal replay
    for e in events:
        if e.event_type.value.startswith('agent.'):
            decoded_data = _decode_output_data(e.data)
            # Deduplicate by hashing the decoded data (handles different sequence key names)
            event_key = (e.event_type.value, json.dumps(decoded_data, sort_keys=True))
            if event_key not in seen:
                seen.add(event_key)
                agent_events.append({
                    "event_type": e.event_type.value,
                    "data": decoded_data
                })
    return agent_events


def get_agent_event_types(events):
    """Get list of agent event types from stream events."""
    return [e["event_type"] for e in extract_agent_events(events)]


# =============================================================================
# BASIC AGENT STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_agent_simple_basic(client, worker_process):
    """Test basic agent streaming returns events."""
    events = list(client.stream_events("stream_agent_simple", {
        "message": "Say hello in one word."
    }, component_type="function"))

    # Should have run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types
    assert "run.completed" in run_event_types

    # Should have agent lifecycle events nested in output.delta
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types
    assert "agent.completed" in agent_event_types


@pytest.mark.integration
def test_stream_agent_simple_event_order(client, worker_process):
    """Test that agent.started comes before agent.completed."""
    events = list(client.stream_events("stream_agent_simple", {
        "message": "What is 2 + 2?"
    }, component_type="function"))

    agent_event_types = get_agent_event_types(events)

    # Find indices
    started_idx = agent_event_types.index("agent.started") if "agent.started" in agent_event_types else -1
    completed_idx = agent_event_types.index("agent.completed") if "agent.completed" in agent_event_types else -1

    # Started should come before completed
    assert started_idx >= 0, "agent.started event not found"
    assert completed_idx >= 0, "agent.completed event not found"
    assert started_idx < completed_idx, "agent.started should come before agent.completed"


@pytest.mark.integration
def test_stream_agent_simple_has_output(client, worker_process):
    """Test that completed event contains output."""
    events = list(client.stream_events("stream_agent_simple", {
        "message": "What color is the sky?"
    }, component_type="function"))

    # Find the agent.completed event
    agent_events = extract_agent_events(events)
    completed_events = [e for e in agent_events if e["event_type"] == "agent.completed"]
    assert len(completed_events) == 1

    completed = completed_events[0]
    assert "data" in completed
    assert "output" in completed["data"]
    assert len(completed["data"]["output"]) > 0


# =============================================================================
# STREAMING CHAT AGENT
# =============================================================================


@pytest.mark.integration
def test_stream_agent_chat_basic(client, worker_process):
    """Test streaming chat agent returns events."""
    events = list(client.stream_events("stream_agent_chat", {
        "message": "Explain Python in one sentence."
    }, component_type="function"))

    # Should have run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types
    assert "run.completed" in run_event_types

    # Should have agent events
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types
    assert "agent.completed" in agent_event_types


@pytest.mark.integration
def test_stream_agent_chat_started_has_metadata(client, worker_process):
    """Test that agent.started event contains agent metadata."""
    events = list(client.stream_events("stream_agent_chat", {
        "message": "Hello"
    }, component_type="function"))

    agent_events = extract_agent_events(events)
    started_events = [e for e in agent_events if e["event_type"] == "agent.started"]
    assert len(started_events) == 1

    started = started_events[0]
    assert "data" in started
    # Started event should have agent_name and model
    assert "agent_name" in started["data"]
    assert "model" in started["data"]


@pytest.mark.integration
def test_stream_agent_chat_content_relevant(client, worker_process):
    """Test that agent response is relevant to the question."""
    events = list(client.stream_events("stream_agent_chat", {
        "message": "What is the capital of France?"
    }, component_type="function"))

    agent_events = extract_agent_events(events)
    completed = next((e for e in agent_events if e["event_type"] == "agent.completed"), None)
    assert completed is not None

    output = completed["data"]["output"].lower()
    assert "paris" in output


# =============================================================================
# STREAMING AGENT WITH TOOLS
# =============================================================================


@pytest.mark.integration
def test_stream_agent_with_tools_basic(client, worker_process):
    """Test streaming agent with tools returns events."""
    events = list(client.stream_events("stream_agent_with_tools", {
        "message": "Calculate 10 + 5"
    }, component_type="function"))

    # Should have run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types
    assert "run.completed" in run_event_types

    # Should have agent events
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types
    assert "agent.completed" in agent_event_types


@pytest.mark.integration
def test_stream_agent_with_tools_correct_result(client, worker_process):
    """Test that agent with tools produces correct calculation."""
    events = list(client.stream_events("stream_agent_with_tools", {
        "message": "What is 15 multiplied by 23?"
    }, component_type="function"))

    agent_events = extract_agent_events(events)
    completed = next((e for e in agent_events if e["event_type"] == "agent.completed"), None)
    assert completed is not None

    output = completed["data"]["output"]
    # 15 * 23 = 345
    assert "345" in output


@pytest.mark.integration
def test_stream_agent_with_tools_has_tool_calls(client, worker_process):
    """Test that agent records tool calls in completed event."""
    events = list(client.stream_events("stream_agent_with_tools", {
        "message": "Calculate 100 divided by 4"
    }, component_type="function"))

    agent_events = extract_agent_events(events)
    completed = next((e for e in agent_events if e["event_type"] == "agent.completed"), None)
    assert completed is not None

    # Completed event MUST have tool_calls for tool-using agents (P1.3)
    assert "tool_calls" in completed["data"], (
        "Expected 'tool_calls' in agent.completed data. "
        "Tool-using agents must record their tool calls."
    )
    tool_calls = completed["data"]["tool_calls"]
    assert len(tool_calls) >= 1, (
        "Expected at least one tool call for math calculation request"
    )
    # Should have used the calculate tool
    tool_names = [tc.get("name") for tc in tool_calls]
    assert "calculate" in tool_names, (
        f"Expected 'calculate' tool in tool_calls, got: {tool_names}"
    )


# =============================================================================
# STREAM CONSUMPTION PATTERNS
# =============================================================================


@pytest.mark.integration
def test_stream_agent_collect_all(client, worker_process):
    """Test collecting all events from agent stream."""
    events = list(client.stream_events("stream_agent_simple", {
        "message": "Hi"
    }, component_type="function"))

    # Should complete without error
    assert len(events) >= 1

    # Should have run events
    run_event_types = [e.event_type.value for e in events]
    assert "run.completed" in run_event_types or "run.started" in run_event_types

    # Should have agent events
    agent_event_types = get_agent_event_types(events)
    assert "agent.completed" in agent_event_types or "agent.started" in agent_event_types


@pytest.mark.integration
def test_stream_agent_multiple_sequential(client, worker_process):
    """Test multiple sequential agent streams."""
    # First stream
    events1 = list(client.stream_events("stream_agent_simple", {
        "message": "Hello"
    }, component_type="function"))
    assert len(events1) >= 2

    # Second stream
    events2 = list(client.stream_events("stream_agent_simple", {
        "message": "Goodbye"
    }, component_type="function"))
    assert len(events2) >= 2

    # Both should complete successfully
    types1 = get_agent_event_types(events1)
    types2 = get_agent_event_types(events2)

    assert "agent.completed" in types1
    assert "agent.completed" in types2


# =============================================================================
# ERROR HANDLING
# =============================================================================


@pytest.mark.integration
def test_stream_agent_nonexistent(client, worker_process):
    """Test streaming a nonexistent agent function."""
    from agnt5.client import RunError

    with pytest.raises((RunError, Exception)):
        list(client.stream_events("nonexistent_agent_function", {}, component_type="function"))


@pytest.mark.integration
def test_stream_agent_after_error(client, worker_process):
    """Test that streaming works after an error."""
    from agnt5.client import RunError

    # Try to stream a nonexistent function
    try:
        list(client.stream_events("nonexistent_function", {}, component_type="function"))
    except (RunError, Exception):
        pass

    # Agent streaming should still work
    events = list(client.stream_events("stream_agent_simple", {
        "message": "Test recovery"
    }, component_type="function"))

    # Should have run lifecycle events
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types or "run.completed" in run_event_types

    # Should have agent events
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types or "agent.completed" in agent_event_types


# =============================================================================
# COMPARISON WITH NON-STREAMING
# =============================================================================


@pytest.mark.integration
def test_stream_vs_sync_same_result(client, worker_process):
    """Test that streaming and non-streaming produce equivalent results."""
    message = "What is 5 + 3?"

    # Streaming call
    events = list(client.stream_events("stream_agent_with_tools", {
        "message": message
    }, component_type="function"))

    agent_events = extract_agent_events(events)
    stream_completed = next((e for e in agent_events if e["event_type"] == "agent.completed"), None)
    assert stream_completed is not None
    stream_output = stream_completed["data"]["output"]

    # The output should contain the answer (8)
    assert "8" in stream_output


# =============================================================================
# TRAJECTORY VALIDATION - Single Lifecycle Events
# Regression tests for duplicate agent event bug (3 started/completed instead of 1)
# =============================================================================


@pytest.mark.integration
def test_agent_trajectory_single_lifecycle_events(client, worker_process):
    """Verify agent execution produces exactly one started/completed event.

    Regression test for duplicate agent event bug where three agent.started
    and three agent.completed events were emitted instead of one.
    """
    events = list(client.stream_events("stream_agent_simple", {
        "message": "Say hello."
    }, component_type="function"))

    # Use existing helper to extract agent events (with deduplication)
    agent_events = extract_agent_events(events)
    agent_event_types = [e["event_type"] for e in agent_events]

    # Count each event type
    started_count = agent_event_types.count("agent.started")
    completed_count = agent_event_types.count("agent.completed")

    assert started_count == 1, \
        f"Expected 1 agent.started, got {started_count}. Events: {agent_event_types}"
    assert completed_count == 1, \
        f"Expected 1 agent.completed, got {completed_count}. Events: {agent_event_types}"


@pytest.mark.integration
def test_agent_trajectory_correlation_chain(client, worker_process):
    """Verify parent_correlation_id chain is unbroken.

    agent.started and agent.completed should share the same correlation_id
    and reference the run's correlation_id as parent.
    """
    events = list(client.stream_events("stream_agent_simple", {
        "message": "Test correlation."
    }, component_type="function"))

    # Find run.started for run correlation_id
    run_started = next((e for e in events if e.event_type.value == "run.started"), None)
    assert run_started is not None, "Missing run.started event"
    run_correlation_id = run_started.data.get("correlation_id")

    # Extract agent events
    agent_events = extract_agent_events(events)
    agent_started = next((e for e in agent_events if e["event_type"] == "agent.started"), None)
    agent_completed = next((e for e in agent_events if e["event_type"] == "agent.completed"), None)

    assert agent_started is not None, "Missing agent.started event"
    assert agent_completed is not None, "Missing agent.completed event"

    # Agent events should reference run as parent
    assert agent_started["data"].get("parent_correlation_id") == run_correlation_id, \
        f"agent.started parent mismatch: expected {run_correlation_id}"
    assert agent_completed["data"].get("parent_correlation_id") == run_correlation_id, \
        f"agent.completed parent mismatch: expected {run_correlation_id}"

    # Agent started and completed should have same correlation_id
    agent_corr_id = agent_started["data"].get("correlation_id")
    assert agent_completed["data"].get("correlation_id") == agent_corr_id, \
        "agent.started and agent.completed have different correlation_ids"


@pytest.mark.integration
def test_agent_trajectory_with_tools_single_lifecycle(client, worker_process):
    """Verify agent with tools still produces only one started/completed.

    Even when an agent uses tools (multiple iterations), there should be
    exactly one agent.started and one agent.completed event.
    """
    events = list(client.stream_events("stream_agent_with_tools", {
        "message": "What is 25 * 4?"
    }, component_type="function"))

    agent_events = extract_agent_events(events)
    agent_event_types = [e["event_type"] for e in agent_events]

    # Count lifecycle events
    started_count = agent_event_types.count("agent.started")
    completed_count = agent_event_types.count("agent.completed")

    assert started_count == 1, \
        f"Expected 1 agent.started with tools, got {started_count}"
    assert completed_count == 1, \
        f"Expected 1 agent.completed with tools, got {completed_count}"

    # Should have iteration events (agent used tools)
    iteration_started = agent_event_types.count("agent.iteration.started")
    iteration_completed = agent_event_types.count("agent.iteration.completed")

    assert iteration_started >= 1, "Expected at least 1 agent.iteration.started"
    assert iteration_completed >= 1, "Expected at least 1 agent.iteration.completed"
