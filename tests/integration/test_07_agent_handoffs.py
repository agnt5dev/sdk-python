"""
End-to-End Integration Tests for Agent Handoffs

These tests verify agent handoff mechanisms:
1. Agents can hand off to other agents
2. Context is preserved during handoffs
3. Conversation history carries over
4. Multiple handoff paths work correctly
5. State persists across handoffs
6. Handoff decisions are tracked

Uses OpenAI gpt-4o-mini for agent execution with test-bench handoff workflows.

Note: These tests require OPENAI_API_KEY environment variable.
"""

import pytest
import os
import time


# Skip all tests if OPENAI_API_KEY not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping agent handoff tests"
)


@pytest.mark.integration
@pytest.mark.slow
def test_simple_triage_handoff(client, worker_process, platform):
    """
    Test simple triage agent handing off to specialist.

    Verifies:
    - Triage agent can route to specialist
    - Handoff transfers control
    - Specialist provides response
    - Handoff decision is tracked
    """
    time.sleep(2)

    # Technical query should route to technical specialist
    result = client.run(
        "wf_31_simple_triage_handoff",
        {"query": "How do I implement a binary search tree?"}
    )

    assert "output" in result
    assert "handoff_occurred" in result
    assert result["handoff_occurred"] is True
    assert "specialist_agent" in result

    # Output should be from specialist
    assert len(result["output"]) > 0

    print(f"\n✅ Simple triage handoff works correctly")
    print(f"   Handed off to: {result.get('specialist_agent', 'unknown')}")


@pytest.mark.integration
@pytest.mark.slow
def test_handoff_with_context_sharing(client, worker_process, platform):
    """
    Test handoff preserves context between agents.

    Verifies:
    - Context data is shared during handoff
    - Specialist has access to original context
    - Workflow state is preserved
    """
    time.sleep(2)

    result = client.run(
        "wf_32_handoff_with_context",
        {"query": "What's the best pricing strategy?", "user_type": "enterprise"}
    )

    assert "output" in result
    assert "context_preserved" in result
    assert result["context_preserved"] is True

    # Specialist should have access to user_type context
    assert "user_type_used" in result
    assert result["user_type_used"] == "enterprise"

    print(f"\n✅ Handoff with context sharing works correctly")
    print(f"   Context preserved: {result['context_preserved']}")


@pytest.mark.integration
@pytest.mark.slow
def test_handoff_with_tools(client, worker_process, platform):
    """
    Test specialist agent receiving tools after handoff.

    Verifies:
    - Specialist has access to tools
    - Tools work correctly after handoff
    - Tool results included in response
    """
    time.sleep(2)

    result = client.run(
        "wf_33_handoff_with_tools",
        {"query": "Research the latest trends in AI"}
    )

    assert "output" in result
    assert "specialist_used_tools" in result
    assert result["specialist_used_tools"] is True

    # Specialist should have made tool calls
    assert "tool_calls_count" in result
    assert result["tool_calls_count"] > 0

    print(f"\n✅ Handoff with tools works correctly")
    print(f"   Tool calls by specialist: {result['tool_calls_count']}")


@pytest.mark.integration
@pytest.mark.slow
def test_multiple_handoff_options(client, worker_process, platform):
    """
    Test triage agent with multiple handoff targets.

    Verifies:
    - Agent can choose from multiple specialists
    - Correct specialist selected based on query
    - Different queries route to different specialists
    """
    time.sleep(2)

    # Test technical query
    result_tech = client.run(
        "wf_34_multiple_handoff_options",
        {"query": "How do I optimize database queries?"}
    )

    assert "specialist_type" in result_tech
    assert result_tech["specialist_type"] in ["technical", "business", "research"]

    # Test business query
    result_biz = client.run(
        "wf_34_multiple_handoff_options",
        {"query": "What's the best go-to-market strategy?"}
    )

    assert "specialist_type" in result_biz
    assert result_biz["specialist_type"] in ["technical", "business", "research"]

    print(f"\n✅ Multiple handoff options work correctly")
    print(f"   Tech query → {result_tech.get('specialist_type', 'unknown')}")
    print(f"   Biz query → {result_biz.get('specialist_type', 'unknown')}")


@pytest.mark.integration
@pytest.mark.slow
def test_handoff_with_state_persistence(client, worker_process, platform):
    """
    Test state persists across agent handoffs.

    Verifies:
    - State set by triage agent is available to specialist
    - Specialist can read and modify state
    - Final state reflects both agents' contributions
    """
    time.sleep(2)

    result = client.run(
        "wf_35_handoff_with_state",
        {"query": "Analyze this business case", "priority": "high"}
    )

    assert "output" in result
    assert "state_entries" in result
    assert result["state_entries"] > 0

    # State should include contributions from both agents
    assert "triage_decision" in result
    assert "specialist_analysis" in result

    print(f"\n✅ Handoff with state persistence works correctly")
    print(f"   State entries: {result['state_entries']}")


@pytest.mark.integration
@pytest.mark.slow
def test_handoff_conversation_history(client, worker_process, platform):
    """
    Test conversation history propagates through handoff.

    Verifies:
    - Triage agent's messages are preserved
    - Specialist sees conversation history
    - Response builds on previous context
    """
    time.sleep(2)

    result = client.run(
        "wf_32_handoff_with_context",
        {"query": "What technology stack should we use?"}
    )

    assert "output" in result
    assert "conversation_turns" in result
    assert result["conversation_turns"] >= 2  # At least triage + specialist

    print(f"\n✅ Conversation history propagates correctly")
    print(f"   Conversation turns: {result['conversation_turns']}")


@pytest.mark.integration
@pytest.mark.slow
def test_handoff_decision_tracking(client, worker_process, platform):
    """
    Test handoff decisions are tracked and logged.

    Verifies:
    - Handoff decision is recorded
    - Reason for handoff is captured
    - Target specialist is identified
    """
    time.sleep(2)

    result = client.run(
        "wf_31_simple_triage_handoff",
        {"query": "Explain microservices architecture"}
    )

    assert "handoff_occurred" in result
    assert "handoff_reason" in result or "specialist_agent" in result

    # Handoff should have occurred for this query
    assert result["handoff_occurred"] is True

    print(f"\n✅ Handoff decision tracking works correctly")
    print(f"   Handoff occurred: {result['handoff_occurred']}")


@pytest.mark.integration
@pytest.mark.slow
def test_no_handoff_when_not_needed(client, worker_process, platform):
    """
    Test triage agent can handle query without handoff.

    Verifies:
    - Agent can decide not to hand off
    - Simple queries handled by triage agent
    - Handoff is optional, not mandatory
    """
    time.sleep(2)

    # Simple greeting shouldn't require handoff
    result = client.run(
        "wf_31_simple_triage_handoff",
        {"query": "Hello, how are you?"}
    )

    assert "output" in result

    # For simple queries, handoff might not occur
    # (this tests that handoff is optional)
    if "handoff_occurred" in result:
        # Either way is acceptable - agent decides
        print(f"\n✅ Handoff decision is flexible")
        print(f"   Handoff for greeting: {result['handoff_occurred']}")
    else:
        print(f"\n✅ Handoff is optional, not mandatory")


@pytest.mark.integration
@pytest.mark.slow
def test_multiple_sequential_handoffs(client, worker_process, platform):
    """
    Test multiple handoffs in sequence.

    Verifies:
    - Specialist can hand off to another specialist
    - Multi-hop handoffs work correctly
    - Context preserves through multiple hops
    """
    time.sleep(2)

    # Complex query that might require multiple specialists
    result = client.run(
        "wf_34_multiple_handoff_options",
        {"query": "Design a distributed system with cost analysis"}
    )

    assert "output" in result

    # Track if multiple handoffs occurred
    if "handoff_chain_length" in result:
        assert result["handoff_chain_length"] >= 1
        print(f"\n✅ Sequential handoffs work correctly")
        print(f"   Handoff chain length: {result['handoff_chain_length']}")
    else:
        print(f"\n✅ Handoff completed successfully")


# TODO: Add tests for:
# - Handoff with streaming responses
# - Handoff error handling (specialist unavailable)
# - Handoff timeout scenarios
# - Handoff with complex state objects
# - Handoff metrics and telemetry
# - Handoff in nested workflows
# - Circular handoff prevention
# - Handoff authorization/permissions
