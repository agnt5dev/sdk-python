"""
Integration tests for batch_eval functionality.

Tests batch evaluation execution with scoring.
"""

import pytest

from agnt5 import BatchEvalItem


@pytest.mark.integration
def test_batch_eval_basic(client, worker_process, platform):
    """Test basic batch eval with exact_match scorer."""
    result = client.batch_eval(
        component="add",
        items=[
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ],
        expected=[3, 7, 11],
        scorers=["exact_match"],
    )

    assert result.status == "completed"
    assert result.stats.total_items == 3
    assert result.stats.completed_items == 3
    assert result.stats.failed_items == 0
    assert result.stats.passed_items == 3
    assert result.pass_rate == 1.0

    # Check individual results
    assert len(result.results) == 3
    for i, item_result in enumerate(result.results):
        assert item_result.index == i
        assert item_result.passed
        assert item_result.is_success
        assert len(item_result.scores) == 1
        assert item_result.scores[0].scorer == "exact_match"
        assert item_result.scores[0].passed


@pytest.mark.integration
def test_batch_eval_with_batch_eval_item(client, worker_process, platform):
    """Test batch eval using BatchEvalItem objects."""
    result = client.batch_eval(
        component="add",
        items=[
            BatchEvalItem(input={"a": 10, "b": 20}, expected=30, item_id="test-1"),
            BatchEvalItem(input={"a": 5, "b": 5}, expected=10, item_id="test-2"),
        ],
        scorers=["exact_match"],
    )

    assert result.status == "completed"
    assert result.stats.total_items == 2
    assert result.pass_rate == 1.0

    # Check item_ids are preserved
    assert result.results[0].item_id == "test-1"
    assert result.results[1].item_id == "test-2"


@pytest.mark.integration
def test_batch_eval_some_failing(client, worker_process, platform):
    """Test batch eval where some items don't match expected."""
    result = client.batch_eval(
        component="add",
        items=[
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ],
        expected=[3, 999],  # Second one is wrong
        scorers=["exact_match"],
    )

    assert result.status == "completed"
    assert result.stats.total_items == 2
    assert result.stats.completed_items == 2
    assert result.stats.passed_items == 1

    # First should pass, second should fail
    assert result.results[0].passed
    assert not result.results[1].passed

    # Check failing_items helper
    failing = result.failing_items()
    assert len(failing) == 1
    assert failing[0].index == 1


@pytest.mark.integration
def test_batch_eval_empty_items(client, worker_process, platform):
    """Test batch eval with empty items list."""
    result = client.batch_eval(
        component="add",
        items=[],
        scorers=["exact_match"],
    )

    assert result.status == "completed"
    assert result.stats.total_items == 0
    assert result.results == []


@pytest.mark.integration
def test_batch_eval_outputs(client, worker_process, platform):
    """Test that outputs are correctly sorted by index."""
    result = client.batch_eval(
        component="add",
        items=[
            {"a": 1, "b": 1},
            {"a": 2, "b": 2},
            {"a": 3, "b": 3},
        ],
        scorers=[],  # No scorers needed for this test
    )

    assert result.outputs == [2, 4, 6]


@pytest.mark.integration
def test_batch_eval_concurrency(client, worker_process, platform):
    """Test batch eval with explicit concurrency limit."""
    # Use a larger batch to test concurrency
    items = [{"a": i, "b": i} for i in range(10)]
    expected = [i * 2 for i in range(10)]

    result = client.batch_eval(
        component="add",
        items=items,
        expected=expected,
        scorers=["exact_match"],
        max_concurrency=3,  # Limit concurrency
    )

    assert result.status == "completed"
    assert result.stats.total_items == 10
    assert result.stats.passed_items == 10
    assert result.pass_rate == 1.0


@pytest.mark.integration
def test_batch_eval_dict_with_input_key(client, worker_process, platform):
    """Test batch eval with dict items containing 'input' key."""
    result = client.batch_eval(
        component="add",
        items=[
            {"input": {"a": 1, "b": 2}, "expected": 3},
            {"input": {"a": 10, "b": 20}, "expected": 30},
        ],
        scorers=["exact_match"],
    )

    assert result.status == "completed"
    assert result.stats.total_items == 2
    assert result.pass_rate == 1.0


@pytest.mark.integration
def test_batch_eval_get_score(client, worker_process, platform):
    """Test getting specific scorer results."""
    result = client.batch_eval(
        component="add",
        items=[{"a": 1, "b": 2}],
        expected=[3],
        scorers=["exact_match"],
    )

    item_result = result.results[0]
    exact_score = item_result.get_score("exact_match")
    assert exact_score is not None
    assert exact_score.passed
    assert exact_score.score == 1.0

    # Non-existent scorer
    missing = item_result.get_score("nonexistent")
    assert missing is None
