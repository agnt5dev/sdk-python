"""
Integration Tests for Parallel Tool Execution Workflows

Tests tool execution patterns:
- Parallel tool calls: Multiple independent tools run concurrently
- Timing comparison: Performance of sequential vs parallel

Run with:
    # Local mode (requires running dev server with examples worker)
    pytest tests/integration/test_ex_710_parallel_tool_calls.py -v

    # Specific test
    pytest tests/integration/test_ex_710_parallel_tool_calls.py::test_parallel_tool_execution_basic -v
"""

import pytest


# =============================================================================
# PARALLEL TOOL EXECUTION (wf_44)
# =============================================================================


@pytest.mark.integration
def test_parallel_tool_execution_basic(client, worker_process):
    """Test parallel tool execution."""
    result = client.run(
        "parallel_tool_execution",
        {"sources": ["api1", "api2", "api3", "api4"]},
        component_type="workflow"
    )

    assert result["pattern"] == "parallel_execution"
    assert result["sources"] == ["api1", "api2", "api3", "api4"]
    assert result["parallel_fetches"] == 4
    assert result["parallel_transforms"] == 4
    assert result["execution_time_ms"] > 0


@pytest.mark.integration
def test_parallel_tool_execution_single_source(client, worker_process):
    """Test parallel execution with single source."""
    result = client.run(
        "parallel_tool_execution",
        {"sources": ["single"]},
        component_type="workflow"
    )

    assert result["parallel_fetches"] == 1
    assert result["parallel_transforms"] == 1


@pytest.mark.integration
def test_parallel_tool_execution_many_sources(client, worker_process):
    """Test parallel execution with many sources."""
    sources = [f"source_{i}" for i in range(10)]
    result = client.run(
        "parallel_tool_execution",
        {"sources": sources},
        component_type="workflow"
    )

    assert result["parallel_fetches"] == 10
    assert result["parallel_transforms"] == 10


# =============================================================================
# TIMING COMPARISON (wf_45)
# =============================================================================


@pytest.mark.integration
def test_timing_comparison_basic(client, worker_process):
    """Test timing comparison between sequential and parallel."""
    result = client.run(
        "timing_comparison",
        {"num_operations": 5},
        component_type="workflow"
    )

    assert result["num_operations"] == 5

    # Sequential should take longer than parallel
    assert result["sequential_time_ms"] > 0
    assert result["parallel_time_ms"] > 0

    # Parallel should be faster (speedup > 1)
    assert result["speedup_factor"] > 1.0


@pytest.mark.integration
def test_timing_comparison_more_operations(client, worker_process):
    """Test timing comparison with more operations."""
    result = client.run(
        "timing_comparison",
        {"num_operations": 10},
        component_type="workflow"
    )

    assert result["num_operations"] == 10

    # With more operations, speedup should be more pronounced
    assert result["speedup_factor"] > 1.0

    # Sequential time should be significantly longer
    assert result["sequential_time_ms"] > result["parallel_time_ms"]


@pytest.mark.integration
def test_timing_comparison_few_operations(client, worker_process):
    """Test timing comparison with few operations."""
    result = client.run(
        "timing_comparison",
        {"num_operations": 2},
        component_type="workflow"
    )

    assert result["num_operations"] == 2
    assert result["speedup_factor"] > 0  # Should still show some speedup


@pytest.mark.integration
def test_timing_comparison_default(client, worker_process):
    """Test timing comparison with default number of operations."""
    result = client.run(
        "timing_comparison",
        {},
        component_type="workflow"
    )

    # Default is 5 operations
    assert result["num_operations"] == 5
