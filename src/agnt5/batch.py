"""Batch execution types and utilities for AGNT5 SDK.

This module provides data classes for batch execution requests and responses,
enabling efficient bulk execution of functions and workflows.

Example:
    ```python
    from agnt5 import Client

    client = Client("http://localhost:34181")

    # Execute a function in batch
    result = await client.batch(
        "process_item",
        items=[{"id": 1}, {"id": 2}, {"id": 3}],
        max_concurrency=10,
    )

    # Access results
    for item in result.results:
        if item.status == "completed":
            print(f"Item {item.index}: {item.output}")
        else:
            print(f"Item {item.index} failed: {item.error}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BatchConfig:
    """Configuration for batch execution.

    Attributes:
        max_concurrency: Maximum number of items to execute in parallel (default: 10)
        continue_on_failure: Continue processing other items if one fails (default: True)
        batch_timeout_ms: Overall batch timeout in milliseconds (default: 3600000 = 1 hour)
        default_item_timeout_ms: Default timeout per item in milliseconds (default: 30000 = 30 sec)
    """
    max_concurrency: int = 10
    continue_on_failure: bool = True
    batch_timeout_ms: int = 3600000  # 1 hour
    default_item_timeout_ms: int = 30000  # 30 seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        return {
            "max_concurrency": self.max_concurrency,
            "continue_on_failure": self.continue_on_failure,
            "batch_timeout_ms": self.batch_timeout_ms,
            "default_item_timeout_ms": self.default_item_timeout_ms,
        }


@dataclass
class BatchItemInput:
    """Input for a single batch item.

    Attributes:
        input: The input data for this item
        index: Optional index (auto-assigned if not provided)
        item_id: Optional item identifier for tracking
        metadata: Optional metadata for this item
        timeout_ms: Optional per-item timeout override
    """
    input: Dict[str, Any]
    index: Optional[int] = None
    item_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    timeout_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        result: Dict[str, Any] = {"input": self.input}
        if self.index is not None:
            result["index"] = self.index
        if self.item_id is not None:
            result["item_id"] = self.item_id
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.timeout_ms is not None:
            result["timeout_ms"] = self.timeout_ms
        return result


@dataclass
class BatchItemError:
    """Error information for a failed batch item.

    Attributes:
        code: Error code
        message: Error message
    """
    code: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["BatchItemError"]:
        """Create from dictionary."""
        if data is None:
            return None
        return cls(
            code=data.get("code"),
            message=data.get("message"),
        )


@dataclass
class BatchItemResult:
    """Result of a single batch item execution.

    Attributes:
        index: The item's position in the batch
        item_id: The item's identifier (if provided)
        run_id: The run ID for this item's execution
        status: Execution status (completed, failed, cancelled)
        output: The output data if successful
        error: Error information if failed
        duration_ms: Execution duration in milliseconds
        started_at: ISO timestamp when execution started
        completed_at: ISO timestamp when execution completed
    """
    index: int
    run_id: str
    status: str
    item_id: Optional[str] = None
    output: Optional[Any] = None
    error: Optional[BatchItemError] = None
    duration_ms: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if this item completed successfully."""
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        """Check if this item failed."""
        return self.status == "failed"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchItemResult":
        """Create from dictionary."""
        return cls(
            index=data.get("index", 0),
            item_id=data.get("item_id"),
            run_id=data.get("run_id", ""),
            status=data.get("status", "unknown"),
            output=data.get("output"),
            error=BatchItemError.from_dict(data.get("error")),
            duration_ms=data.get("duration_ms"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class BatchStats:
    """Statistics for a batch execution.

    Attributes:
        total_items: Total number of items in the batch
        completed_items: Number of successfully completed items
        failed_items: Number of failed items
        cancelled_items: Number of cancelled items
        pending_items: Number of pending items (for async batches)
        duration_ms: Total batch duration in milliseconds
        avg_item_duration_ms: Average duration per completed item
    """
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    cancelled_items: int = 0
    pending_items: int = 0
    duration_ms: int = 0
    avg_item_duration_ms: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BatchStats":
        """Create from dictionary."""
        if data is None:
            return cls()
        return cls(
            total_items=data.get("total_items", 0),
            completed_items=data.get("completed_items", 0),
            failed_items=data.get("failed_items", 0),
            cancelled_items=data.get("cancelled_items", 0),
            pending_items=data.get("pending_items", 0),
            duration_ms=data.get("duration_ms", 0),
            avg_item_duration_ms=data.get("avg_item_duration_ms", 0),
        )


@dataclass
class BatchResult:
    """Result of a batch execution.

    Attributes:
        batch_id: Unique identifier for this batch
        status: Overall batch status (completed, partial_failure, failed, cancelled)
        results: List of individual item results
        stats: Batch statistics
        trace_id: Trace ID for observability
        created_at: ISO timestamp when batch was created
    """
    batch_id: str
    status: str
    results: List[BatchItemResult] = field(default_factory=list)
    stats: Optional[BatchStats] = None
    trace_id: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if all items completed successfully."""
        return self.status == "completed"

    @property
    def is_partial_failure(self) -> bool:
        """Check if some items failed but others succeeded."""
        return self.status == "partial_failure"

    @property
    def outputs(self) -> List[Any]:
        """Get outputs from all items, sorted by index.

        Returns:
            List of outputs in order of item indices. Failed items return None.
        """
        sorted_results = sorted(self.results, key=lambda r: r.index)
        return [r.output for r in sorted_results]

    def successful_outputs(self) -> List[Any]:
        """Get outputs only from successfully completed items.

        Returns:
            List of outputs from successful items, sorted by index.
        """
        successful = [r for r in self.results if r.is_success]
        sorted_results = sorted(successful, key=lambda r: r.index)
        return [r.output for r in sorted_results]

    def failed_items(self) -> List[BatchItemResult]:
        """Get items that failed.

        Returns:
            List of failed item results.
        """
        return [r for r in self.results if r.is_failed]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchResult":
        """Create from dictionary (API response)."""
        return cls(
            batch_id=data.get("batch_id", ""),
            status=data.get("status", "unknown"),
            results=[BatchItemResult.from_dict(r) for r in data.get("results", [])],
            stats=BatchStats.from_dict(data.get("stats")),
            trace_id=data.get("trace_id"),
            created_at=data.get("created_at"),
        )


@dataclass
class BatchStatusResult:
    """Result of a batch status query.

    Attributes:
        batch_id: Unique identifier for this batch
        status: Current batch status
        results: List of individual item results (if included)
        stats: Batch statistics
        trace_id: Trace ID for observability
        submitted_at: ISO timestamp when batch was submitted
        started_at: ISO timestamp when batch started execution
        completed_at: ISO timestamp when batch completed
    """
    batch_id: str
    status: str
    results: List[BatchItemResult] = field(default_factory=list)
    stats: Optional[BatchStats] = None
    trace_id: Optional[str] = None
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def is_running(self) -> bool:
        """Check if batch is still running."""
        return self.status in ("pending", "queued", "running")

    @property
    def is_completed(self) -> bool:
        """Check if batch has completed (successfully or with failures)."""
        return self.status in ("completed", "partial_failure", "failed", "cancelled")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchStatusResult":
        """Create from dictionary (API response)."""
        return cls(
            batch_id=data.get("batch_id", ""),
            status=data.get("status", "unknown"),
            results=[BatchItemResult.from_dict(r) for r in data.get("results", [])],
            stats=BatchStats.from_dict(data.get("stats")),
            trace_id=data.get("trace_id"),
            submitted_at=data.get("submitted_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class CancelBatchResult:
    """Result of cancelling a batch.

    Attributes:
        batch_id: Unique identifier for the cancelled batch
        status: Final batch status (cancelled)
        cancelled_items: Number of items that were cancelled
        completed_items: Number of items that had already completed
    """
    batch_id: str
    status: str
    cancelled_items: int = 0
    completed_items: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CancelBatchResult":
        """Create from dictionary (API response)."""
        return cls(
            batch_id=data.get("batch_id", ""),
            status=data.get("status", "unknown"),
            cancelled_items=data.get("cancelled_items", 0),
            completed_items=data.get("completed_items", 0),
        )


class BatchError(Exception):
    """Exception raised when a batch execution fails.

    Attributes:
        message: Error message
        batch_result: The batch result (if available)
    """
    def __init__(self, message: str, batch_result: Optional[BatchResult] = None):
        super().__init__(message)
        self.message = message
        self.batch_result = batch_result

    def __str__(self) -> str:
        if self.batch_result:
            return f"{self.message} (batch_id={self.batch_result.batch_id}, status={self.batch_result.status})"
        return self.message
