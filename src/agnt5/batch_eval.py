"""Batch evaluation types and utilities for AGNT5 SDK.

This module provides data classes for batch evaluation requests and responses,
enabling efficient bulk evaluation of functions and workflows with scoring.

Example:
    ```python
    from agnt5 import Client, BatchEvalItem

    client = Client("http://localhost:34181")

    # Evaluate a function in batch
    result = client.batch_eval(
        component="greet",
        items=[
            BatchEvalItem(input={"name": "Alice"}, expected="Hello, Alice!"),
            BatchEvalItem(input={"name": "Bob"}, expected="Hello, Bob!"),
        ],
        scorers=["exact_match"],
    )

    print(f"Pass rate: {result.pass_rate:.0%}")
    for item in result.results:
        print(f"Item {item.index}: passed={item.passed}, score={item.scores}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .responses import ScorerResultSummary


@dataclass
class BatchEvalItem:
    """Input item for batch evaluation.

    Attributes:
        input: Input data for the component
        expected: Expected output for comparison scorers (optional)
        item_id: Optional item identifier for tracking
        index: Position in the batch (auto-assigned if not provided)
    """
    input: Dict[str, Any]
    expected: Optional[Any] = None
    item_id: Optional[str] = None
    index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: Dict[str, Any] = {"input": self.input}
        if self.expected is not None:
            result["expected"] = self.expected
        if self.item_id is not None:
            result["item_id"] = self.item_id
        if self.index is not None:
            result["index"] = self.index
        return result


@dataclass
class BatchEvalItemResult:
    """Result of a single batch eval item.

    Attributes:
        index: The item's position in the batch
        run_id: The run ID for this item's execution
        output: The component's output
        scores: List of scorer results
        passed: Whether all scorers passed
        duration_ms: Execution duration in milliseconds
        item_id: The item's identifier (if provided)
        trace_id: OpenTelemetry trace ID for observability
        error: Error message if the eval failed
    """
    index: int
    run_id: str
    output: Any
    scores: List[ScorerResultSummary]
    passed: bool
    duration_ms: int
    item_id: Optional[str] = None
    trace_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if this item was evaluated without errors."""
        return self.error is None

    @property
    def is_failed(self) -> bool:
        """Check if this item failed to evaluate."""
        return self.error is not None

    def get_score(self, scorer_name: str) -> Optional[ScorerResultSummary]:
        """Get a specific scorer result by name.

        Args:
            scorer_name: Name of the scorer to retrieve.

        Returns:
            ScorerResultSummary or None if not found.
        """
        for score in self.scores:
            if score.scorer == scorer_name:
                return score
        return None

    @classmethod
    def from_eval_response(
        cls,
        eval_response: Any,  # EvalResponse
        index: int,
        item_id: Optional[str] = None,
    ) -> "BatchEvalItemResult":
        """Create from an EvalResponse object.

        Args:
            eval_response: The EvalResponse from client.eval()
            index: The item's position in the batch
            item_id: The item's identifier (if provided)

        Returns:
            BatchEvalItemResult
        """
        error = None
        if eval_response.error:
            error = eval_response.error.message

        return cls(
            index=index,
            run_id=eval_response.run_id,
            output=eval_response.output,
            scores=eval_response.scores,
            passed=eval_response.passed,
            duration_ms=eval_response.duration_ms or 0,
            item_id=item_id,
            trace_id=eval_response.trace_id,
            error=error,
        )

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        index: int,
        item_id: Optional[str] = None,
    ) -> "BatchEvalItemResult":
        """Create from an exception.

        Args:
            exception: The exception that occurred
            index: The item's position in the batch
            item_id: The item's identifier (if provided)

        Returns:
            BatchEvalItemResult with error set
        """
        return cls(
            index=index,
            run_id="",
            output=None,
            scores=[],
            passed=False,
            duration_ms=0,
            item_id=item_id,
            trace_id=None,
            error=str(exception),
        )


@dataclass
class BatchEvalStats:
    """Statistics for batch eval execution.

    Attributes:
        total_items: Total number of items in the batch
        completed_items: Number of successfully evaluated items (no errors)
        failed_items: Number of items that failed to evaluate (had errors)
        passed_items: Number of items where passed=True (all scorers passed)
        avg_duration_ms: Average duration per completed item
        duration_ms: Total batch duration in milliseconds
    """
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    passed_items: int = 0
    avg_duration_ms: int = 0
    duration_ms: int = 0

    @classmethod
    def from_results(
        cls,
        results: List[BatchEvalItemResult],
        total_duration_ms: int,
    ) -> "BatchEvalStats":
        """Calculate stats from a list of results.

        Args:
            results: List of BatchEvalItemResult
            total_duration_ms: Total batch duration in milliseconds

        Returns:
            BatchEvalStats
        """
        total_items = len(results)
        completed_items = sum(1 for r in results if r.is_success)
        failed_items = sum(1 for r in results if r.is_failed)
        passed_items = sum(1 for r in results if r.passed and r.is_success)

        # Calculate average duration from completed items
        durations = [r.duration_ms for r in results if r.is_success and r.duration_ms]
        avg_duration_ms = sum(durations) // len(durations) if durations else 0

        return cls(
            total_items=total_items,
            completed_items=completed_items,
            failed_items=failed_items,
            passed_items=passed_items,
            avg_duration_ms=avg_duration_ms,
            duration_ms=total_duration_ms,
        )


@dataclass
class BatchEvalResult:
    """Result of a batch evaluation.

    Attributes:
        batch_id: Unique identifier for this batch evaluation
        status: Overall status (completed, partial_failure, failed)
        results: List of individual item results
        stats: Batch statistics
    """
    batch_id: str
    status: str
    results: List[BatchEvalItemResult] = field(default_factory=list)
    stats: Optional[BatchEvalStats] = None

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate (passed items / total items).

        Returns:
            Float between 0.0 and 1.0 representing the pass rate.
            Returns 0.0 if there are no results.
        """
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passed and r.is_success)
        return passed / len(self.results)

    @property
    def is_success(self) -> bool:
        """Check if all items were evaluated successfully (no errors)."""
        return self.status == "completed"

    @property
    def is_partial_failure(self) -> bool:
        """Check if some items failed but others succeeded."""
        return self.status == "partial_failure"

    def failed_items(self) -> List[BatchEvalItemResult]:
        """Get items that failed to evaluate (had errors).

        Returns:
            List of failed item results.
        """
        return [r for r in self.results if r.is_failed]

    def failing_items(self) -> List[BatchEvalItemResult]:
        """Get items where scorers did not pass (passed=False).

        Returns:
            List of items that did not pass scoring.
        """
        return [r for r in self.results if not r.passed]

    def passing_items(self) -> List[BatchEvalItemResult]:
        """Get items where all scorers passed (passed=True).

        Returns:
            List of items that passed scoring.
        """
        return [r for r in self.results if r.passed and r.is_success]

    @property
    def outputs(self) -> List[Any]:
        """Get outputs from all items, sorted by index.

        Returns:
            List of outputs in order of item indices. Failed items return None.
        """
        sorted_results = sorted(self.results, key=lambda r: r.index)
        return [r.output for r in sorted_results]


def normalize_batch_eval_items(
    items: List[Union[Dict[str, Any], BatchEvalItem]],
    expected: Optional[List[Any]] = None,
) -> List[BatchEvalItem]:
    """Normalize various input formats to BatchEvalItem list.

    Supports the following input formats:
    1. List of BatchEvalItem objects
    2. List of dicts with {"input": {...}, "expected": ...} format
    3. List of plain input dicts (expected provided as separate list)

    Args:
        items: List of input dicts or BatchEvalItem objects
        expected: Optional list of expected outputs (parallel to items)

    Returns:
        List of BatchEvalItem objects with indices assigned
    """
    normalized = []
    for i, item in enumerate(items):
        if isinstance(item, BatchEvalItem):
            # Already a BatchEvalItem, just ensure index is set
            if item.index is None:
                item = BatchEvalItem(
                    input=item.input,
                    expected=item.expected,
                    item_id=item.item_id,
                    index=i,
                )
            normalized.append(item)
        elif isinstance(item, dict):
            if "input" in item:
                # Format: {"input": {...}, "expected": ...}
                normalized.append(BatchEvalItem(
                    input=item["input"],
                    expected=item.get("expected"),
                    item_id=item.get("item_id"),
                    index=i,
                ))
            else:
                # Format: plain input dict
                exp = expected[i] if expected and i < len(expected) else None
                normalized.append(BatchEvalItem(
                    input=item,
                    expected=exp,
                    index=i,
                ))
        else:
            raise TypeError(
                f"Expected BatchEvalItem or dict, got {type(item).__name__} at index {i}"
            )
    return normalized
