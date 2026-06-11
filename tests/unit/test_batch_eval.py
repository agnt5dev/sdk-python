"""Unit tests for batch_eval module."""

import pytest

from agnt5.batch_eval import (
    BatchEvalItem,
    BatchEvalItemResult,
    BatchEvalResult,
    BatchEvalStats,
    normalize_batch_eval_items,
)
from agnt5.responses import ScorerResultSummary


class TestBatchEvalItem:
    """Tests for BatchEvalItem dataclass."""

    def test_basic_item(self):
        """Test creating a basic batch eval item."""
        item = BatchEvalItem(input={"name": "Alice"})
        assert item.input == {"name": "Alice"}
        assert item.expected is None
        assert item.item_id is None
        assert item.index is None

    def test_item_with_all_fields(self):
        """Test creating an item with all fields."""
        item = BatchEvalItem(
            input={"name": "Alice"},
            expected="Hello, Alice!",
            item_id="test-1",
            index=0,
        )
        assert item.input == {"name": "Alice"}
        assert item.expected == "Hello, Alice!"
        assert item.item_id == "test-1"
        assert item.index == 0

    @pytest.mark.parametrize("input_value", ["Alabama", ["a", "b"], 42])
    def test_item_rejects_non_dict_input(self, input_value):
        """Test BatchEvalItem validates input at runtime."""
        with pytest.raises(TypeError, match="BatchEvalItem.input must be a dict"):
            BatchEvalItem(input=input_value)

    def test_to_dict_basic(self):
        """Test to_dict with only required fields."""
        item = BatchEvalItem(input={"name": "Alice"})
        d = item.to_dict()
        assert d == {"input": {"name": "Alice"}}

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        item = BatchEvalItem(
            input={"name": "Alice"},
            expected="Hello, Alice!",
            item_id="test-1",
            index=0,
        )
        d = item.to_dict()
        assert d == {
            "input": {"name": "Alice"},
            "expected": "Hello, Alice!",
            "item_id": "test-1",
            "index": 0,
        }


class TestBatchEvalItemResult:
    """Tests for BatchEvalItemResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic result."""
        result = BatchEvalItemResult(
            index=0,
            run_id="run-123",
            output={"message": "Hello!"},
            scores=[
                ScorerResultSummary(
                    scorer="exact_match",
                    score=1.0,
                    passed=True,
                )
            ],
            passed=True,
            duration_ms=100,
        )
        assert result.index == 0
        assert result.run_id == "run-123"
        assert result.output == {"message": "Hello!"}
        assert result.passed
        assert result.is_success
        assert not result.is_failed
        assert len(result.scores) == 1

    def test_failed_result(self):
        """Test creating a failed result."""
        result = BatchEvalItemResult(
            index=0,
            run_id="",
            output=None,
            scores=[],
            passed=False,
            duration_ms=0,
            error="Connection failed",
        )
        assert result.is_failed
        assert not result.is_success
        assert result.error == "Connection failed"

    def test_get_score(self):
        """Test getting a specific scorer result."""
        result = BatchEvalItemResult(
            index=0,
            run_id="run-123",
            output="Hello",
            scores=[
                ScorerResultSummary(scorer="exact_match", score=1.0, passed=True),
                ScorerResultSummary(scorer="contains", score=0.5, passed=False),
            ],
            passed=False,
            duration_ms=100,
        )

        exact = result.get_score("exact_match")
        assert exact is not None
        assert exact.score == 1.0

        contains = result.get_score("contains")
        assert contains is not None
        assert contains.score == 0.5

        missing = result.get_score("missing")
        assert missing is None

    def test_from_exception(self):
        """Test creating result from exception."""
        exc = ValueError("Test error")
        result = BatchEvalItemResult.from_exception(exc, index=5, item_id="item-5")

        assert result.index == 5
        assert result.item_id == "item-5"
        assert result.run_id == ""
        assert result.output is None
        assert result.scores == []
        assert result.passed is False
        assert result.is_failed
        assert "Test error" in result.error


class TestBatchEvalStats:
    """Tests for BatchEvalStats dataclass."""

    def test_empty_stats(self):
        """Test empty stats."""
        stats = BatchEvalStats()
        assert stats.total_items == 0
        assert stats.completed_items == 0
        assert stats.failed_items == 0
        assert stats.passed_items == 0
        assert stats.avg_duration_ms == 0
        assert stats.duration_ms == 0

    def test_from_results_all_passed(self):
        """Test stats calculation when all items pass."""
        results = [
            BatchEvalItemResult(
                index=i,
                run_id=f"run-{i}",
                output=f"output-{i}",
                scores=[ScorerResultSummary(scorer="test", score=1.0, passed=True)],
                passed=True,
                duration_ms=100,
            )
            for i in range(3)
        ]
        stats = BatchEvalStats.from_results(results, total_duration_ms=500)

        assert stats.total_items == 3
        assert stats.completed_items == 3
        assert stats.failed_items == 0
        assert stats.passed_items == 3
        assert stats.avg_duration_ms == 100
        assert stats.duration_ms == 500

    def test_from_results_some_failed(self):
        """Test stats calculation when some items fail."""
        results = [
            BatchEvalItemResult(
                index=0,
                run_id="run-0",
                output="output-0",
                scores=[ScorerResultSummary(scorer="test", score=1.0, passed=True)],
                passed=True,
                duration_ms=100,
            ),
            BatchEvalItemResult(
                index=1,
                run_id="",
                output=None,
                scores=[],
                passed=False,
                duration_ms=0,
                error="Failed",
            ),
            BatchEvalItemResult(
                index=2,
                run_id="run-2",
                output="output-2",
                scores=[ScorerResultSummary(scorer="test", score=0.5, passed=False)],
                passed=False,
                duration_ms=200,
            ),
        ]
        stats = BatchEvalStats.from_results(results, total_duration_ms=300)

        assert stats.total_items == 3
        assert stats.completed_items == 2  # 2 without errors
        assert stats.failed_items == 1  # 1 with error
        assert stats.passed_items == 1  # 1 passed scoring
        assert stats.avg_duration_ms == 150  # (100 + 200) / 2


class TestBatchEvalResult:
    """Tests for BatchEvalResult dataclass."""

    def test_empty_result(self):
        """Test empty batch result."""
        result = BatchEvalResult(
            batch_id="batch-123",
            status="completed",
            results=[],
            stats=BatchEvalStats(),
        )
        assert result.batch_id == "batch-123"
        assert result.status == "completed"
        assert result.pass_rate == 0.0
        assert result.is_success
        assert not result.is_partial_failure

    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        results = [
            BatchEvalItemResult(
                index=i,
                run_id=f"run-{i}",
                output=f"output-{i}",
                scores=[],
                passed=i < 2,  # First 2 pass
                duration_ms=100,
            )
            for i in range(4)
        ]
        result = BatchEvalResult(
            batch_id="batch-123",
            status="completed",
            results=results,
            stats=BatchEvalStats.from_results(results, 400),
        )

        assert result.pass_rate == 0.5  # 2/4

    def test_failed_items(self):
        """Test getting failed items."""
        results = [
            BatchEvalItemResult(
                index=0,
                run_id="run-0",
                output="output-0",
                scores=[],
                passed=True,
                duration_ms=100,
            ),
            BatchEvalItemResult(
                index=1,
                run_id="",
                output=None,
                scores=[],
                passed=False,
                duration_ms=0,
                error="Failed",
            ),
        ]
        result = BatchEvalResult(
            batch_id="batch-123",
            status="partial_failure",
            results=results,
        )

        failed = result.failed_items()
        assert len(failed) == 1
        assert failed[0].index == 1

    def test_passing_and_failing_items(self):
        """Test getting passing and failing items."""
        results = [
            BatchEvalItemResult(
                index=0,
                run_id="run-0",
                output="output-0",
                scores=[ScorerResultSummary(scorer="test", score=1.0, passed=True)],
                passed=True,
                duration_ms=100,
            ),
            BatchEvalItemResult(
                index=1,
                run_id="run-1",
                output="output-1",
                scores=[ScorerResultSummary(scorer="test", score=0.5, passed=False)],
                passed=False,
                duration_ms=100,
            ),
        ]
        result = BatchEvalResult(
            batch_id="batch-123",
            status="completed",
            results=results,
        )

        passing = result.passing_items()
        assert len(passing) == 1
        assert passing[0].index == 0

        failing = result.failing_items()
        assert len(failing) == 1
        assert failing[0].index == 1

    def test_outputs_sorted_by_index(self):
        """Test that outputs are sorted by index."""
        results = [
            BatchEvalItemResult(
                index=2,
                run_id="run-2",
                output="third",
                scores=[],
                passed=True,
                duration_ms=100,
            ),
            BatchEvalItemResult(
                index=0,
                run_id="run-0",
                output="first",
                scores=[],
                passed=True,
                duration_ms=100,
            ),
            BatchEvalItemResult(
                index=1,
                run_id="run-1",
                output="second",
                scores=[],
                passed=True,
                duration_ms=100,
            ),
        ]
        result = BatchEvalResult(
            batch_id="batch-123",
            status="completed",
            results=results,
        )

        assert result.outputs == ["first", "second", "third"]

    def test_is_partial_failure(self):
        """Test partial failure detection."""
        result = BatchEvalResult(
            batch_id="batch-123",
            status="partial_failure",
            results=[],
        )
        assert result.is_partial_failure
        assert not result.is_success


class TestNormalizeBatchEvalItems:
    """Tests for normalize_batch_eval_items function."""

    def test_normalize_batch_eval_items_list(self):
        """Test normalizing a list of BatchEvalItem objects."""
        items = [
            BatchEvalItem(input={"name": "Alice"}, expected="Hello, Alice!"),
            BatchEvalItem(input={"name": "Bob"}, expected="Hello, Bob!"),
        ]
        normalized = normalize_batch_eval_items(items)

        assert len(normalized) == 2
        assert normalized[0].index == 0
        assert normalized[1].index == 1
        assert normalized[0].input == {"name": "Alice"}
        assert normalized[0].expected == "Hello, Alice!"

    def test_normalize_dict_with_input_key(self):
        """Test normalizing dicts with 'input' key."""
        items = [
            {"input": {"name": "Alice"}, "expected": "Hello, Alice!"},
            {"input": {"name": "Bob"}, "item_id": "test-bob"},
        ]
        normalized = normalize_batch_eval_items(items)

        assert len(normalized) == 2
        assert normalized[0].input == {"name": "Alice"}
        assert normalized[0].expected == "Hello, Alice!"
        assert normalized[1].input == {"name": "Bob"}
        assert normalized[1].item_id == "test-bob"

    def test_normalize_plain_dicts_with_expected_list(self):
        """Test normalizing plain input dicts with separate expected list."""
        items = [
            {"name": "Alice"},
            {"name": "Bob"},
        ]
        expected = ["Hello, Alice!", "Hello, Bob!"]
        normalized = normalize_batch_eval_items(items, expected)

        assert len(normalized) == 2
        assert normalized[0].input == {"name": "Alice"}
        assert normalized[0].expected == "Hello, Alice!"
        assert normalized[1].input == {"name": "Bob"}
        assert normalized[1].expected == "Hello, Bob!"

    def test_normalize_preserves_existing_index(self):
        """Test that existing index is preserved."""
        items = [
            BatchEvalItem(input={"name": "Alice"}, index=5),
        ]
        normalized = normalize_batch_eval_items(items)

        assert normalized[0].index == 5

    def test_normalize_invalid_item_type(self):
        """Test that invalid item types raise TypeError."""
        items = ["not a dict"]
        with pytest.raises(TypeError) as exc_info:
            normalize_batch_eval_items(items)
        assert "Expected BatchEvalItem or dict" in str(exc_info.value)

    def test_normalize_empty_list(self):
        """Test normalizing an empty list."""
        normalized = normalize_batch_eval_items([])
        assert normalized == []

    def test_normalize_mixed_formats(self):
        """Test normalizing a mix of formats."""
        items = [
            BatchEvalItem(input={"a": 1}, expected="exp1"),
            {"input": {"b": 2}, "expected": "exp2"},
            {"c": 3},  # Plain input dict
        ]
        expected = [None, None, "exp3"]  # Only third item uses this
        normalized = normalize_batch_eval_items(items, expected)

        assert len(normalized) == 3
        assert normalized[0].expected == "exp1"
        assert normalized[1].expected == "exp2"
        assert normalized[2].input == {"c": 3}
        assert normalized[2].expected == "exp3"
