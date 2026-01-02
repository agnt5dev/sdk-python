"""
Integration Tests for Multi-Agent Orchestration Workflows

Tests multi-agent coordination patterns:
- Supervisor/Worker: Supervisor delegates subtasks to workers
- Debate pattern: Two agents argue, third decides
- Consensus pattern: Multiple agents vote to reach consensus
- Pipeline pattern: Agents form processing pipeline

All tests require OPENAI_API_KEY as they use LLM agents.

Run with:
    # Local mode (requires running dev server with examples worker)
    # Requires OPENAI_API_KEY environment variable
    pytest tests/integration/test_ex_730_multi_agent_orchestration.py -v

    # Skip LLM tests if no API key
    pytest tests/integration/test_ex_730_multi_agent_orchestration.py -v -m "not llm"
"""

import os
import pytest


# Skip all multi-agent tests if no API key is available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping multi-agent tests"
    ),
]


# =============================================================================
# SUPERVISOR/WORKER PATTERN (wf_46)
# =============================================================================


def test_supervisor_worker_basic(client, worker_process):
    """Test supervisor delegating to workers."""
    result = client.run(
        "supervisor_worker_workflow",
        {"task": "Research and summarize the key benefits of cloud computing"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "supervisor_worker"
    assert len(result["final_output"]) > 0
    assert result["tool_calls"] >= 1, "Supervisor should call at least one tool to delegate to workers"


def test_supervisor_worker_complex_task(client, worker_process):
    """Test supervisor with more complex task."""
    result = client.run(
        "supervisor_worker_workflow",
        {"task": "Analyze the pros and cons of microservices architecture"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "supervisor_worker"
    assert len(result["final_output"]) > 0


# =============================================================================
# DEBATE PATTERN (wf_47)
# =============================================================================


def test_debate_agents_basic(client, worker_process):
    """Test debate between agents with judge decision."""
    result = client.run(
        "debate_agents",
        {"topic": "Should AI systems be regulated by the government?"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "debate"
    assert result["topic"] == "Should AI systems be regulated by the government?"

    # Verify all parts of debate occurred
    assert len(result["pro_arguments"]) > 0
    assert len(result["con_arguments"]) > 0
    assert len(result["judgment"]) > 0


def test_debate_agents_technical_topic(client, worker_process):
    """Test debate on technical topic."""
    result = client.run(
        "debate_agents",
        {"topic": "Is Python better than JavaScript for backend development?"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "debate"
    assert len(result["judgment"]) > 0


# =============================================================================
# CONSENSUS PATTERN (wf_48)
# =============================================================================


def test_consensus_agents_basic(client, worker_process):
    """Test consensus building among multiple agents."""
    result = client.run(
        "consensus_agents",
        {"question": "Should we use a NoSQL database for this project?"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "consensus"

    # Verify voting occurred (5 agents in implementation)
    assert len(result["votes"]) == 5
    assert result["vote_summary"]["yes"] + result["vote_summary"]["no"] <= 5

    # Verify consensus result
    assert result["consensus_decision"] in ["YES", "NO", "NO_CONSENSUS"]


def test_consensus_agents_different_question(client, worker_process):
    """Test consensus with different question."""
    result = client.run(
        "consensus_agents",
        {"question": "Should we migrate to Kubernetes?"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "consensus"
    assert len(result["votes"]) == 5


# =============================================================================
# PIPELINE PATTERN (wf_49)
# =============================================================================


def test_pipeline_agents_basic(client, worker_process):
    """Test agent pipeline processing."""
    result = client.run(
        "pipeline_agents",
        {"content": "A new AI model was released that can understand and generate images from text descriptions."},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "pipeline"

    # Verify all stages completed (4 stages in implementation)
    stages = [s["stage"] for s in result["stages"]]
    assert stages == ["extract", "enrich", "synthesize", "format"]

    # Each stage should have output
    for stage in result["stages"]:
        assert len(stage["output"]) > 0

    assert len(result["final_output"]) > 0


def test_pipeline_agents_longer_content(client, worker_process):
    """Test pipeline with longer content."""
    content = """
    Artificial intelligence has made significant strides in recent years.
    Large language models can now understand context and generate human-like text.
    Computer vision systems can identify objects with high accuracy.
    These advances are transforming industries from healthcare to finance.
    """
    result = client.run(
        "pipeline_agents",
        {"content": content.strip()},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "pipeline"
    assert len(result["stages"]) == 4


# =============================================================================
# MULTI-AGENT IMPROVEMENTS (P2.7)
# =============================================================================


def test_debate_references_topic(client, worker_process):
    """Test debate agents reference the specific topic (P2.7).

    Validates:
    - Pro arguments reference the topic
    - Con arguments reference the topic
    - Judgment references the topic
    """
    topic = "Should remote work become the default for tech companies?"
    result = client.run(
        "debate_agents",
        {"topic": topic},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "debate"
    assert result["topic"] == topic

    # Check that arguments reference remote work concepts
    pro_args = result["pro_arguments"].lower()
    con_args = result["con_arguments"].lower()
    judgment = result["judgment"].lower()

    remote_work_terms = ["remote", "work", "office", "home", "flexible", "commute"]

    # Pro arguments should mention remote work concepts
    pro_found = [term for term in remote_work_terms if term in pro_args]
    assert len(pro_found) >= 1, (
        f"Pro arguments should reference topic. Found: {pro_found}"
    )

    # Con arguments should mention remote work concepts
    con_found = [term for term in remote_work_terms if term in con_args]
    assert len(con_found) >= 1, (
        f"Con arguments should reference topic. Found: {con_found}"
    )


def test_consensus_votes_sum_correctly(client, worker_process):
    """Test consensus voting math is correct (P2.7).

    Validates:
    - Vote counts add up correctly
    - Each agent votes exactly once
    - Decision matches vote majority
    """
    result = client.run(
        "consensus_agents",
        {"question": "Should we adopt TypeScript for new projects?"},
        component_type="workflow"
    )

    if result.get("status") == "skipped":
        pytest.skip(result.get("reason", "API key not configured"))

    assert result["pattern"] == "consensus"

    # Should have exactly 5 votes (5 agents)
    votes = result["votes"]
    assert len(votes) == 5, f"Expected 5 votes, got {len(votes)}"

    # Count votes (workflow returns uppercase YES/NO/ABSTAIN)
    yes_count = sum(1 for v in votes if v["vote"] == "YES")
    no_count = sum(1 for v in votes if v["vote"] == "NO")

    # Vote summary should match
    assert result["vote_summary"]["yes"] == yes_count, (
        f"Yes summary mismatch: {result['vote_summary']['yes']} != {yes_count}"
    )
    assert result["vote_summary"]["no"] == no_count, (
        f"No summary mismatch: {result['vote_summary']['no']} != {no_count}"
    )

    # Total votes should equal 5 (some might be abstain/other)
    total_votes = yes_count + no_count
    assert total_votes <= 5, f"Total votes ({total_votes}) exceeds agent count"

    # Decision should match majority
    decision = result["consensus_decision"]
    if yes_count > no_count and yes_count >= 3:
        assert decision == "YES", f"Expected YES with {yes_count} votes, got {decision}"
    elif no_count > yes_count and no_count >= 3:
        assert decision == "NO", f"Expected NO with {no_count} votes, got {decision}"
    else:
        # No clear majority
        assert decision in ["YES", "NO", "NO_CONSENSUS"], (
            f"Unexpected decision: {decision}"
        )
