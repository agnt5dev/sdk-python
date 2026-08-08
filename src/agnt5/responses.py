"""Response classes for AGNT5 SDK client.

This module provides type-safe response wrappers for platform API responses,
following httpx-style patterns with properties like is_success, raise_for_status(),
and elapsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class RunStatus(str, Enum):
    """Run execution status values."""

    ENQUEUED = "enqueued"
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_USER_INPUT = "awaiting_user_input"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunErrorDetail:
    """Structured error information from a failed run.

    Attributes:
        code: Error code (e.g., 'FUNCTION_ERROR', 'TIMEOUT', 'EXECUTION_FAILED').
        message: Human-readable error message.
        details: Additional error details (stack trace, context, etc.).
    """

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.details:
            return f"{self.code}: {self.message} (details: {self.details})"
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class SubmitLinks:
    """Hypermedia links for async responses.

    Attributes:
        self_url: URL to check this run's status.
    """

    self_url: str


@dataclass
class RunResponse(Generic[T]):
    """Response from run(), get_result(), wait_for_result().

    Contains the execution result along with metadata about the run.
    Follows httpx-style patterns for status checking.

    Example:
        ```python
        response = client.run("greet", {"name": "Alice"})

        if response.is_success:
            print(f"Result: {response.output}")
            print(f"Duration: {response.elapsed}")
        else:
            print(f"Failed: {response.error}")
            response.raise_for_status()  # Raises RunError
        ```

    Attributes:
        run_id: Unique identifier for this execution run.
        status_code: HTTP-style status code: 200=completed, 202=in-progress, 500=failed.
        status: Granular execution status.
        output: The function/workflow output (None if not completed or failed).
        error: Error details if the run failed (None if successful).
        duration_ms: Execution duration in milliseconds.
        trace_id: OpenTelemetry trace ID for distributed tracing.
        component: Component name that was executed (function, workflow, etc.).
        created_at: When the run was created/submitted.
        started_at: When execution actually started.
        completed_at: When execution completed (for successful runs).
        failed_at: When execution failed (for failed runs).
        session_id: Session ID if this was part of a multi-turn conversation.
        metadata: Additional metadata from the platform (retry info, etc.).
    """

    run_id: str
    status_code: int
    status: RunStatus
    output: Optional[T] = None
    error: Optional[RunErrorDetail] = None
    duration_ms: Optional[int] = None
    trace_id: Optional[str] = None
    component: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_success(self) -> bool:
        """True if the run completed successfully (status_code 200)."""
        return self.status_code == 200 and self.status == RunStatus.COMPLETED

    @property
    def is_pending(self) -> bool:
        """True if the run is still in progress (status_code 202)."""
        return self.status_code == 202

    @property
    def is_error(self) -> bool:
        """True if the run failed (status_code 500 or status is failed/cancelled/timeout)."""
        return self.status_code == 500 or self.status in (
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMEOUT,
        )

    @property
    def elapsed(self) -> Optional[timedelta]:
        """Execution duration as a timedelta (like httpx.Response.elapsed)."""
        if self.duration_ms is not None:
            return timedelta(milliseconds=self.duration_ms)
        return None

    def raise_for_status(self) -> None:
        """Raise RunError if the run failed (like httpx.Response.raise_for_status())."""
        if self.is_error:
            from .client import RunError

            if self.error:
                raise RunError(
                    self.error.message,
                    run_id=self.run_id,
                    error_code=self.error.code,
                    metadata=self.error.details,
                )
            else:
                raise RunError(
                    f"Run failed with status: {self.status.value}",
                    run_id=self.run_id,
                )


@dataclass
class SubmitResponse:
    """Response from submit().

    Contains the run ID and metadata for an async submission.

    Example:
        ```python
        response = client.submit("long_task", {"data": "..."})
        print(f"Run ID: {response.run_id}")
        print(f"Status URL: {response.status_url}")
        ```

    Attributes:
        run_id: Unique identifier for this execution run.
        status_code: HTTP-style status code (202 for accepted).
        status: Execution status (typically 'enqueued').
        trace_id: OpenTelemetry trace ID for distributed tracing.
        component: Component name that was submitted.
        created_at: When the run was created/submitted.
        links: Hypermedia links for async operations.
    """

    run_id: str
    status_code: int
    status: RunStatus
    trace_id: Optional[str] = None
    component: Optional[str] = None
    created_at: Optional[datetime] = None
    links: Optional[SubmitLinks] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def status_url(self) -> Optional[str]:
        """Convenience property to get the status URL."""
        return self.links.self_url if self.links else None


@dataclass
class StatusResponse:
    """Response from get_status().

    Contains the current execution status of a run.

    Example:
        ```python
        status = client.get_status(run_id)
        if status.status == RunStatus.COMPLETED:
            result = client.get_result(run_id)
        ```

    Attributes:
        run_id: Unique identifier for this execution run.
        status_code: HTTP-style status code.
        status: Current execution status.
        trace_id: OpenTelemetry trace ID for distributed tracing.
        component: Component name being executed.
        created_at: When the run was created/submitted.
        started_at: When execution started.
        completed_at: When execution completed (if completed).
    """

    run_id: str
    status_code: int
    status: RunStatus
    trace_id: Optional[str] = None
    component: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_complete(self) -> bool:
        """True if the run has finished (completed, failed, or cancelled)."""
        return self.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMEOUT,
        )

    @property
    def is_running(self) -> bool:
        """True if the run is currently executing."""
        return self.status in (RunStatus.RUNNING, RunStatus.STARTED)


@dataclass
class Event:
    """A single event from the run's event stream.

    Events capture the full lifecycle of a run including all
    streaming deltas, tool calls, and state changes.

    Attributes:
        id: Unique identifier for this event.
        event_type: Type of event (e.g., 'run.started', 'lm.content_block.delta').
        run_id: The run this event belongs to.
        step_key: Workflow step identifier if applicable.
        input_data: Input data for this event.
        output_data: Output data for this event.
        parent_event_id: Parent event ID if this is a child event.
        metadata: Additional metadata for the event.
        trace_id: OpenTelemetry trace ID for observability.
        timestamp_ns: Event timestamp in nanoseconds.
        created_at: When the event was recorded.
    """

    id: str
    event_type: str
    run_id: str
    step_key: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    parent_event_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    timestamp_ns: Optional[int] = None
    created_at: Optional[datetime] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class EventsResponse:
    """Response from get_events() containing a list of events.

    Example:
        ```python
        events = client.get_events(run_id)
        for event in events.items:
            print(f"{event.event_type}: {event.output_data}")
        print(f"Total events: {events.count}")
        ```

    Attributes:
        items: List of events for the run.
        count: Total number of events returned.
    """

    items: List[Event]
    count: int
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __iter__(self):
        """Allow iteration over events directly."""
        return iter(self.items)

    def __len__(self):
        """Return the number of events."""
        return self.count


@dataclass
class ScorerResultSummary:
    """Summary of a single scorer result.

    Attributes:
        scorer: Name of the scorer that was run.
        score: Score between 0.0 and 1.0.
        passed: Whether the scorer passed.
        explanation: Human-readable explanation from the scorer.
        label: Categorical label (optional).
        metadata: Additional metadata from the scorer.
    """

    scorer: str
    score: float
    passed: bool
    explanation: Optional[str] = None
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvalResponse(Generic[T]):
    """Response from client.eval() containing component output and scorer results.

    Example:
        ```python
        result = client.eval(
            component="my_agent",
            component_type="agent",
            input={"task": "summarize this"},
            expected="A summary of...",
            scorers=["exact_match", "format_checker"]
        )

        print(f"Output: {result.output}")
        print(f"Passed: {result.passed}")
        for score in result.scores:
            print(f"  {score.scorer}: {score.score} ({score.explanation})")
        ```

    Attributes:
        output: The component's output.
        scores: List of scorer results.
        passed: Overall pass/fail (True if all scorers passed).
        run_id: Unique identifier for this execution run.
        trace_id: OpenTelemetry trace ID for distributed tracing.
        duration_ms: Execution duration in milliseconds.
        error: Error details if the run failed.
    """

    output: Optional[T]
    scores: List[ScorerResultSummary]
    passed: bool
    run_id: str = ""
    trace_id: Optional[str] = None
    duration_ms: Optional[int] = None
    error: Optional[RunErrorDetail] = None
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_success(self) -> bool:
        """True if the component ran successfully (may still have failing scores)."""
        return self.error is None

    @property
    def is_error(self) -> bool:
        """True if the component run failed."""
        return self.error is not None

    @property
    def elapsed(self) -> Optional[timedelta]:
        """Execution duration as a timedelta."""
        if self.duration_ms is not None:
            return timedelta(milliseconds=self.duration_ms)
        return None

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

    def raise_for_status(self) -> None:
        """Raise RunError if the component run failed."""
        if self.is_error and self.error:
            from .client import RunError

            raise RunError(
                self.error.message,
                run_id=self.run_id,
                error_code=self.error.code,
                metadata=self.error.details,
            )


# -----------------------------------------------------------------------------
# Parser Functions
# -----------------------------------------------------------------------------


def _parse_datetime(value) -> Optional[datetime]:
    """Parse datetime from ISO string or Unix timestamp (ms or ns)."""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            # Handle ISO format with 'Z' suffix for UTC
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        elif isinstance(value, (int, float)):
            # Handle Unix timestamp - detect if ms or ns based on magnitude
            # Timestamps after year 2001 in seconds are > 1_000_000_000
            # Timestamps in ms are > 1_000_000_000_000
            # Timestamps in ns are > 1_000_000_000_000_000_000
            if value > 1_000_000_000_000_000:
                # Nanoseconds
                return datetime.fromtimestamp(value / 1_000_000_000)
            elif value > 1_000_000_000_000:
                # Milliseconds
                return datetime.fromtimestamp(value / 1000)
            else:
                # Seconds
                return datetime.fromtimestamp(value)
        return None
    except (ValueError, OSError):
        return None


def _parse_status(value: str) -> RunStatus:
    """Parse status string to enum."""
    try:
        return RunStatus(value)
    except ValueError:
        return RunStatus.UNKNOWN


def parse_run_response(data: Dict[str, Any]) -> RunResponse[Any]:
    """Parse platform JSON response into RunResponse.

    Args:
        data: Raw JSON response from the platform.

    Returns:
        Typed RunResponse with all fields populated.
    """
    # Parse error if present
    error = None
    if data.get("error"):
        err_data = data["error"]
        if isinstance(err_data, dict):
            error = RunErrorDetail(
                code=err_data.get("code", "UNKNOWN"),
                message=err_data.get("message", "Unknown error"),
                details=err_data.get("details"),
            )
        else:
            # error is a plain string — check for top-level error_code field
            error = RunErrorDetail(
                code=data.get("error_code", "UNKNOWN"),
                message=str(err_data),
            )

    # Extract output - handle various response formats
    output = data.get("output")
    if output is None:
        # Try nested structure from older response format
        result = data.get("result", {})
        if isinstance(result, dict):
            output_obj = result.get("output", {})
            if isinstance(output_obj, dict) and "output_data" in output_obj:
                output = output_obj["output_data"]
            elif output_obj:
                output = output_obj

    # Determine status code
    status_code = data.get("status_code")
    if status_code is None:
        status_str = data.get("status", "unknown")
        if status_str == "completed":
            status_code = 200
        elif status_str in ("enqueued", "queued", "running", "started"):
            status_code = 202
        else:
            status_code = 500

    return RunResponse(
        run_id=data.get("run_id", ""),
        status_code=status_code,
        status=_parse_status(data.get("status", "unknown")),
        output=output,
        error=error,
        duration_ms=data.get("duration_ms"),
        trace_id=data.get("trace_id"),
        component=data.get("function") or data.get("component"),
        created_at=_parse_datetime(data.get("created_at")),
        started_at=_parse_datetime(data.get("started_at")),
        completed_at=_parse_datetime(data.get("completed_at")),
        failed_at=_parse_datetime(data.get("failed_at")),
        session_id=data.get("session_id"),
        metadata=data.get("metadata"),
        _raw=data,
    )


def parse_submit_response(data: Dict[str, Any]) -> SubmitResponse:
    """Parse platform JSON response into SubmitResponse.

    Args:
        data: Raw JSON response from the platform.

    Returns:
        Typed SubmitResponse with all fields populated.
    """
    links = None
    if data.get("links"):
        links = SubmitLinks(self_url=data["links"].get("self", ""))

    return SubmitResponse(
        run_id=data.get("run_id", ""),
        status_code=data.get("status_code", 202),
        status=_parse_status(data.get("status", "enqueued")),
        trace_id=data.get("trace_id"),
        component=data.get("function") or data.get("component"),
        created_at=_parse_datetime(data.get("created_at")),
        links=links,
        _raw=data,
    )


def parse_status_response(data: Dict[str, Any]) -> StatusResponse:
    """Parse platform JSON response into StatusResponse.

    Args:
        data: Raw JSON response from the platform.

    Returns:
        Typed StatusResponse with all fields populated.
    """
    # Handle both camelCase and snake_case field names
    run_id = data.get("runId") or data.get("run_id", "")
    created_at = data.get("submittedAt") or data.get("created_at")
    started_at = data.get("startedAt") or data.get("started_at")
    completed_at = data.get("completedAt") or data.get("completed_at")

    # Determine status code from status if not present
    status_code = data.get("status_code")
    if status_code is None:
        status_str = data.get("status", "unknown")
        if status_str == "completed":
            status_code = 200
        elif status_str == "failed":
            status_code = 500
        else:
            status_code = 202

    return StatusResponse(
        run_id=run_id,
        status_code=status_code,
        status=_parse_status(data.get("status", "unknown")),
        trace_id=data.get("trace_id"),
        component=data.get("function") or data.get("component"),
        created_at=_parse_datetime(created_at),
        started_at=_parse_datetime(started_at),
        completed_at=_parse_datetime(completed_at),
        _raw=data,
    )


def parse_event(data: Dict[str, Any]) -> Event:
    """Parse a single event from platform JSON.

    Args:
        data: Raw JSON for a single event.

    Returns:
        Typed Event with all fields populated.
    """
    return Event(
        id=data.get("id", ""),
        event_type=data.get("event_type", ""),
        run_id=data.get("run_id", ""),
        step_key=data.get("step_key"),
        input_data=data.get("input_data"),
        output_data=data.get("output_data"),
        parent_event_id=data.get("parent_event_id"),
        metadata=data.get("metadata"),
        trace_id=data.get("trace_id"),
        timestamp_ns=data.get("timestamp_ns"),
        created_at=_parse_datetime(data.get("created_at")),
        _raw=data,
    )


def parse_events_response(data: Dict[str, Any]) -> EventsResponse:
    """Parse platform JSON response into EventsResponse.

    Args:
        data: Raw JSON response from the platform containing events list.

    Returns:
        Typed EventsResponse with all events parsed.
    """
    items_data = data.get("items", [])
    items = [parse_event(item) for item in items_data]

    return EventsResponse(
        items=items,
        count=data.get("count", len(items)),
        _raw=data,
    )


def parse_eval_response(data: Dict[str, Any]) -> EvalResponse[Any]:
    """Parse platform JSON response into EvalResponse.

    Args:
        data: Raw JSON response from the platform containing eval results.

    Returns:
        Typed EvalResponse with all fields populated.
    """
    # Parse error if present
    error = None
    if data.get("error"):
        err_data = data["error"]
        if isinstance(err_data, dict):
            error = RunErrorDetail(
                code=err_data.get("code", "UNKNOWN"),
                message=err_data.get("message", "Unknown error"),
                details=err_data.get("details"),
            )
        else:
            error = RunErrorDetail(code="UNKNOWN", message=str(err_data))

    # Parse scorer results
    scores_data = data.get("scores", [])
    scores = []
    for score_data in scores_data:
        scores.append(
            ScorerResultSummary(
                scorer=score_data.get("scorer", ""),
                score=score_data.get("score", 0.0),
                passed=score_data.get("passed", False),
                explanation=score_data.get("explanation"),
                label=score_data.get("label"),
                metadata=score_data.get("metadata"),
            )
        )

    # Determine overall passed status
    passed = data.get("passed")
    if passed is None:
        # Default: passed if all scorers passed
        passed = all(s.passed for s in scores) if scores else True

    return EvalResponse(
        output=data.get("output"),
        scores=scores,
        passed=passed,
        run_id=data.get("run_id", ""),
        trace_id=data.get("trace_id"),
        duration_ms=data.get("duration_ms"),
        error=error,
        _raw=data,
    )
