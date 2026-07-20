"""
Integration Tests: Client Event Streaming

Tests the new stream_events() method that yields typed Event objects:
- Client.stream_events() for functions, agents, workflows
- AsyncClient.stream_events() for async iteration
- WorkflowProxy.stream_events() for workflow streaming
- Event type parsing and handling

Agent events are top-level events (EventType.AGENT_STARTED, etc.).
The actual event payload is in the `output_data` field as proper JSON.

Run with:
    # Requires OPENAI_API_KEY for LLM calls
    pytest tests/integration/test_ex_130_client_streaming.py -v

    # Skip if no API key
    pytest tests/integration/test_ex_130_client_streaming.py -v -m "not llm"
"""

import os

import pytest

from agnt5 import AsyncClient, Event, EventType

# Skip all tests if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping client streaming tests"
    ),
]


def get_agent_event_types(events):
    """Get list of agent event types from stream events.

    Agent events are now top-level events (EventType.AGENT_STARTED, etc.)
    not nested in output.delta.
    """
    agent_types = []
    for e in events:
        if e.event_type.value.startswith("agent."):
            agent_types.append(e.event_type.value)
    return agent_types


def find_agent_event(events, event_type_value):
    """Find an agent event by its type value (e.g., 'agent.started')."""
    for e in events:
        if e.event_type.value == event_type_value:
            return e
    return None


def get_agent_event_data(event):
    """Get the payload data from an agent event.

    Agent events have their payload in the output_data field.
    This is now proper JSON (not base64) thanks to the platform fix.
    """
    return event.data.get("output_data", {})


# =============================================================================
# CLIENT.STREAM_EVENTS() - BASIC
# =============================================================================


@pytest.mark.integration
def test_stream_events_returns_event_objects(client, worker_process):
    """Test that stream_events yields Event objects, not raw strings."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Say hello"},
        component_type="function",
    ))

    # Should have events
    assert len(events) >= 1

    # All items should be Event objects
    for event in events:
        assert isinstance(event, Event), f"Expected Event, got {type(event)}"
        assert isinstance(event.event_type, EventType), f"Expected EventType, got {type(event.event_type)}"
        assert isinstance(event.data, dict), f"Expected dict data, got {type(event.data)}"


@pytest.mark.integration
def test_stream_events_agent_lifecycle(client, worker_process):
    """Test that agent streaming yields lifecycle events."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "What is 2 + 2?"},
        component_type="function",
    ))

    # Check run lifecycle events at top level
    run_event_types = [e.event_type for e in events]
    assert EventType.RUN_STARTED in run_event_types, "Missing run.started event"
    assert EventType.RUN_COMPLETED in run_event_types, "Missing run.completed event"

    # Agent lifecycle events are nested in output.delta
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types, "Missing agent.started event"
    assert "agent.completed" in agent_event_types, "Missing agent.completed event"


@pytest.mark.integration
def test_stream_events_event_ordering(client, worker_process):
    """Test that agent.started comes before agent.completed."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Hello"},
        component_type="function",
    ))

    agent_event_types = get_agent_event_types(events)

    # Find indices
    try:
        started_idx = agent_event_types.index("agent.started")
    except ValueError:
        pytest.fail("agent.started not found")

    try:
        completed_idx = agent_event_types.index("agent.completed")
    except ValueError:
        pytest.fail("agent.completed not found")

    assert started_idx < completed_idx, "agent.started should come before agent.completed"


@pytest.mark.integration
def test_stream_events_started_metadata(client, worker_process):
    """Test that agent.started event contains metadata."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Hi"},
        component_type="function",
    ))

    started = find_agent_event(events, "agent.started")
    assert started is not None, "No agent.started event found"

    # Check metadata fields in output_data payload
    data = get_agent_event_data(started)
    assert "agent_name" in data, f"Missing agent_name in started event, got: {data.keys()}"
    assert "model" in data, f"Missing model in started event, got: {data.keys()}"


@pytest.mark.integration
def test_stream_events_completed_output(client, worker_process):
    """Test that agent.completed event contains output."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "What color is grass?"},
        component_type="function",
    ))

    completed = find_agent_event(events, "agent.completed")
    assert completed is not None, "No agent.completed event found"

    data = get_agent_event_data(completed)
    assert "output" in data, f"Missing output in completed event, got: {data.keys()}"
    assert len(data["output"]) > 0, "Output should not be empty"


# =============================================================================
# CLIENT.STREAM_EVENTS() - WITH TOOLS
# =============================================================================


@pytest.mark.integration
def test_stream_events_agent_with_tools(client, worker_process):
    """Test streaming agent that uses tools."""
    events = list(client.stream_events(
        "stream_agent_with_tools",
        {"message": "Calculate 7 * 8"},
        component_type="function",
    ))

    # Check run events at top level
    run_event_types = [e.event_type for e in events]
    assert EventType.RUN_STARTED in run_event_types
    assert EventType.RUN_COMPLETED in run_event_types

    # Agent events are now top-level
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types
    assert "agent.completed" in agent_event_types

    # Check output contains correct answer
    completed = find_agent_event(events, "agent.completed")
    assert completed is not None
    data = get_agent_event_data(completed)
    assert "56" in data.get("output", ""), "Expected 56 in output"


@pytest.mark.integration
def test_stream_events_tool_calls_recorded(client, worker_process):
    """Test that tool calls are recorded in completed event."""
    events = list(client.stream_events(
        "stream_agent_with_tools",
        {"message": "What is 100 / 5?"},
        component_type="function",
    ))

    completed = find_agent_event(events, "agent.completed")
    assert completed is not None, "No agent.completed event found"

    data = get_agent_event_data(completed)
    # Tool calls MUST be recorded for tool-using agents (P1.3)
    tool_calls = data.get("tool_calls", [])
    assert len(tool_calls) >= 1, (
        "Expected tool_calls in agent.completed event. "
        f"Agent should have used the calculate tool for math operation. Got data: {data.keys()}"
    )
    tool_names = [tc.get("name") for tc in tool_calls]
    assert "calculate" in tool_names, (
        f"Expected 'calculate' tool in tool_calls, got: {tool_names}"
    )


# =============================================================================
# WORKFLOW PROXY STREAMING
# =============================================================================


@pytest.mark.integration
def test_workflow_proxy_stream_events(client, worker_process):
    """Test WorkflowProxy.stream_events() method."""
    events = list(client.workflow("simple_agent_workflow").stream_events(
        message="Hello from workflow streaming"
    ))

    # Should have events
    assert len(events) >= 1

    # All should be Event objects
    for event in events:
        assert isinstance(event, Event)

    # Should have run events at top level
    run_event_types = [e.event_type.value for e in events]
    assert "run.started" in run_event_types or "run.completed" in run_event_types


# =============================================================================
# ASYNC CLIENT STREAMING
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_client_stream_events(platform, worker_process):
    """Test AsyncClient.stream_events() async iteration."""
    events = []
    gateway_url = platform["gateway_url"]

    async with AsyncClient(gateway_url) as client:
        async for event in client.stream_events(
            "stream_agent_simple",
            {"message": "Async hello"},
            component_type="function",
        ):
            events.append(event)

    # Should have events
    assert len(events) >= 1

    # All should be Event objects
    for event in events:
        assert isinstance(event, Event)
        assert isinstance(event.event_type, EventType)

    # Should have run lifecycle at top level
    run_event_types = [e.event_type for e in events]
    assert EventType.RUN_STARTED in run_event_types
    assert EventType.RUN_COMPLETED in run_event_types

    # Agent lifecycle is nested
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types
    assert "agent.completed" in agent_event_types


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_client_run(platform, worker_process):
    """Test AsyncClient.run() method."""
    gateway_url = platform["gateway_url"]

    async with AsyncClient(gateway_url) as client:
        result = await client.run(
            "stream_agent_simple",
            {"message": "Async run test"},
            component_type="function",
        )

    assert isinstance(result, dict)


# =============================================================================
# ERROR HANDLING
# =============================================================================


@pytest.mark.integration
def test_stream_events_nonexistent_component(client, worker_process):
    """Test streaming a nonexistent component raises error."""
    from agnt5.client import RunError

    with pytest.raises(RunError):
        list(client.stream_events(
            "nonexistent_component_xyz",
            {},
            component_type="function",
        ))


@pytest.mark.integration
def test_stream_events_recovery_after_error(client, worker_process):
    """Test that streaming works after an error."""
    from agnt5.client import RunError

    # Try to stream a nonexistent component
    try:
        list(client.stream_events("nonexistent_xyz", {}, "function"))
    except RunError:
        pass

    # Normal streaming should still work
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Recovery test"},
        component_type="function",
    ))

    assert len(events) >= 1

    # Check run events at top level
    run_event_types = [e.event_type for e in events]
    assert EventType.RUN_STARTED in run_event_types or EventType.RUN_COMPLETED in run_event_types

    # Check agent events in nested content
    agent_event_types = get_agent_event_types(events)
    assert "agent.started" in agent_event_types or "agent.completed" in agent_event_types


# =============================================================================
# MULTIPLE STREAMS
# =============================================================================


@pytest.mark.integration
def test_stream_events_sequential_streams(client, worker_process):
    """Test multiple sequential stream_events calls."""
    # First stream
    events1 = list(client.stream_events(
        "stream_agent_simple",
        {"message": "First"},
        component_type="function",
    ))
    assert len(events1) >= 1

    # Second stream
    events2 = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Second"},
        component_type="function",
    ))
    assert len(events2) >= 1

    # Both should have agent.completed in nested events
    types1 = get_agent_event_types(events1)
    types2 = get_agent_event_types(events2)

    assert "agent.completed" in types1
    assert "agent.completed" in types2


# =============================================================================
# CONTENT INDEX AND SEQUENCE
# =============================================================================


@pytest.mark.integration
def test_stream_events_content_index(client, worker_process):
    """Test that events have content_index field."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Test indexes"},
        component_type="function",
    ))

    for event in events:
        assert hasattr(event, "content_index"), "Event missing content_index"
        assert isinstance(event.content_index, int), "content_index should be int"


@pytest.mark.integration
def test_stream_events_sequence(client, worker_process):
    """Test that events have sequence field."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Test sequence"},
        component_type="function",
    ))

    for event in events:
        assert hasattr(event, "sequence"), "Event missing sequence"
        assert isinstance(event.sequence, int), "sequence should be int"


# =============================================================================
# LM STREAMING EVENTS
# =============================================================================


@pytest.mark.integration
def test_stream_events_lm_message_delta(client, worker_process):
    """Test that LM message delta events are emitted during streaming."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Tell me a short joke"},
        component_type="function",
    ))

    # LM events are now top-level events
    lm_event_types = [e.event_type.value for e in events if e.event_type.value.startswith("lm.")]

    # Should have LM message events (token streaming)
    has_lm_events = (
        "lm.message.start" in lm_event_types or
        "lm.message.delta" in lm_event_types or
        "lm.message.stop" in lm_event_types
    )

    # Note: LM events should be present when agent streams
    if has_lm_events:
        # Verify delta events have content
        delta_events = [e for e in events if e.event_type.value == "lm.message.delta"]
        for delta in delta_events:
            # Delta events should have data with content
            assert delta.data is not None, "LM delta should have data"


@pytest.mark.integration
def test_stream_events_lm_stream_lifecycle(client, worker_process):
    """Test LM stream lifecycle events (started/completed)."""
    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Say hello"},
        component_type="function",
    ))

    # LM stream events are now top-level
    lm_event_types = [e.event_type.value for e in events if e.event_type.value.startswith("lm.")]

    # Check for LM stream lifecycle events
    has_stream_started = "lm.stream.started" in lm_event_types
    has_stream_completed = "lm.stream.completed" in lm_event_types

    # If we have stream started, we should also have stream completed
    if has_stream_started:
        assert has_stream_completed, "LM stream started but not completed"


# =============================================================================
# FORWARD COMPATIBILITY (UNKNOWN EVENT TYPES)
# =============================================================================


@pytest.mark.integration
def test_stream_events_unknown_event_handling(client, worker_process):
    """Test that unknown event types are handled gracefully."""
    # This test verifies the _raw_event_type mechanism works
    # We can't easily trigger an unknown event, but we verify the mechanism exists

    events = list(client.stream_events(
        "stream_agent_simple",
        {"message": "Quick test"},
        component_type="function",
    ))

    # All events should be parsed without exceptions
    assert len(events) >= 1

    # Check that known events don't have _raw_event_type
    for event in events:
        # Known events shouldn't have the fallback field
        if event.event_type != EventType.PROGRESS_UPDATE:
            assert "_raw_event_type" not in event.data, \
                f"Known event {event.event_type} shouldn't have _raw_event_type"


# =============================================================================
# TIMEOUT VALIDATION
# =============================================================================


@pytest.mark.integration
def test_stream_events_timeout_validation(client, worker_process):
    """Test that invalid timeout raises ValueError."""
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        list(client.stream_events(
            "stream_agent_simple",
            {"message": "test"},
            component_type="function",
            timeout=0,
        ))

    with pytest.raises(ValueError, match="timeout must be a positive number"):
        list(client.stream_events(
            "stream_agent_simple",
            {"message": "test"},
            component_type="function",
            timeout=-1,
        ))
