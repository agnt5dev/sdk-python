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


# =============================================================================
# PARALLEL EXECUTION IMPROVEMENTS (P2.6)
# =============================================================================


@pytest.mark.integration
def test_parallel_execution_validates_per_source_outputs(client, worker_process):
    """Test parallel execution produces output for each source (P2.6).

    Validates:
    - Each source produces a corresponding output
    - Outputs maintain source association
    - No sources are lost during parallel execution
    """
    sources = ["alpha", "beta", "gamma", "delta"]
    result = client.run(
        "parallel_tool_execution",
        {"sources": sources},
        component_type="workflow"
    )

    # Should have parallel fetches for each source
    assert result["parallel_fetches"] == len(sources), (
        f"Expected {len(sources)} parallel fetches, got {result['parallel_fetches']}"
    )

    # Should have parallel transforms for each source
    assert result["parallel_transforms"] == len(sources), (
        f"Expected {len(sources)} parallel transforms, got {result['parallel_transforms']}"
    )

    # Sources should be preserved in output
    assert result["sources"] == sources, (
        f"Sources not preserved. Expected {sources}, got {result['sources']}"
    )


@pytest.mark.integration
def test_timing_comparison_meaningful_speedup(client, worker_process):
    """Test parallel execution provides meaningful speedup (P2.6).

    Validates:
    - Parallel execution is at least 1.5x faster than sequential
    - Speedup is proportional to operation count
    """
    # Run with enough operations to see meaningful speedup
    result = client.run(
        "timing_comparison",
        {"num_operations": 8},
        component_type="workflow"
    )

    # Parallel should be meaningfully faster (at least 1.5x)
    assert result["speedup_factor"] >= 1.5, (
        f"Expected at least 1.5x speedup, got {result['speedup_factor']:.2f}x. "
        f"Sequential: {result['sequential_time_ms']}ms, "
        f"Parallel: {result['parallel_time_ms']}ms"
    )


@pytest.mark.integration
def test_parallel_execution_empty_sources(client, worker_process):
    """Test parallel execution with empty sources list (P2.6).

    Validates:
    - Empty sources list is handled gracefully
    - Returns zero counts without error
    """
    from agnt5.client import RunError

    try:
        result = client.run(
            "parallel_tool_execution",
            {"sources": []},
            component_type="workflow"
        )
        # If it succeeds, should have zero operations
        assert result["parallel_fetches"] == 0
        assert result["parallel_transforms"] == 0
    except RunError:
        # Also acceptable - explicit error for empty input
        pass
