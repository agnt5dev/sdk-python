"""Python types for evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    """Event from execution trace (for glassbox testing).

    Attributes:
        event_type: Event type (e.g., "run.started", "lm.call.completed")
        event_id: Unique event identifier
        correlation_id: Correlation ID linking related events
        parent_correlation_id: Parent correlation ID (for hierarchical events)
        timestamp_ns: Timestamp in nanoseconds
        data: Event-specific data
        name: Optional name (e.g., step name, function name)
    """

    event_type: str
    event_id: str
    correlation_id: str
    parent_correlation_id: Optional[str] = None
    timestamp_ns: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Rust FFI."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "parent_correlation_id": self.parent_correlation_id,
            "timestamp_ns": self.timestamp_ns,
            "data": self.data,
            "name": self.name,
        }


@dataclass
class EvalContext:
    """Context provided to custom scorers.

    Attributes:
        input: Original input to the component
        output: Output from the component
        expected: Expected output (if available)
        run_id: Run identifier (if available)
        trace_id: Trace identifier (if available)
        events: Execution trace events (if available)
    """

    input: Any
    output: Any
    expected: Optional[Any] = None
    run_id: Optional[str] = None
    trace_id: Optional[str] = None
    events: Optional[List[TraceEvent]] = None

    def get_events_by_type(self, event_type: str) -> List[TraceEvent]:
        """Filter events by type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of matching events
        """
        if not self.events:
            return []
        return [e for e in self.events if e.event_type == event_type]

    def get_lm_calls(self) -> List[TraceEvent]:
        """Get all LLM call completion events.

        Returns:
            List of lm.call.completed events
        """
        return self.get_events_by_type("lm.call.completed")

    def get_total_tokens(self) -> int:
        """Calculate total token usage from trace.

        Returns:
            Sum of total_tokens from all LLM calls
        """
        return sum(e.data.get("total_tokens", 0) for e in self.get_lm_calls())

    def get_step_events(self, step_name: str) -> List[TraceEvent]:
        """Get events for a specific workflow step.

        Args:
            step_name: Name of the step

        Returns:
            List of events for the step
        """
        if not self.events:
            return []
        return [e for e in self.events if e.name == step_name]


@dataclass
class ScorerRequest:
    """Request passed to a scorer.

    Attributes:
        output: Component output to evaluate (required)
        expected: Expected output for comparison (optional)
        input: Original input to the component (optional, for context)
        trace: Execution trace events (optional, for glassbox testing)
        config: Scorer-specific configuration (optional)
    """

    output: Any
    expected: Optional[Any] = None
    input: Optional[Any] = None
    trace: Optional[List[TraceEvent]] = None
    config: Optional[Dict[str, Any]] = None

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value with default.

        Args:
            key: Config key to look up
            default: Default value if key not found

        Returns:
            Config value or default
        """
        if self.config is None:
            return default
        return self.config.get(key, default)

    def get_trace_events(self, event_type: str) -> List[TraceEvent]:
        """Filter trace events by type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of matching events
        """
        if not self.trace:
            return []
        return [e for e in self.trace if e.event_type == event_type]

    def get_total_tokens(self) -> int:
        """Calculate total token usage from trace.

        Returns:
            Sum of total_tokens from all LLM calls
        """
        lm_calls = self.get_trace_events("lm.call.completed")
        return sum(e.data.get("total_tokens", 0) for e in lm_calls)


@dataclass
class ScorerResult:
    """Result from a scorer.

    Attributes:
        score: Score between 0.0 and 1.0
        passed: Whether the score passes a threshold (optional)
        label: Categorical label (optional)
        explanation: Human-readable explanation (optional)
        metadata: Additional metadata (optional)
    """

    score: float
    passed: Optional[bool] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate and normalize score."""
        self.score = max(0.0, min(1.0, self.score))
        if self.passed is None:
            self.passed = self.score >= 0.5

    @classmethod
    def pass_result(cls, explanation: Optional[str] = None) -> "ScorerResult":
        """Create a passing result.

        Args:
            explanation: Optional explanation

        Returns:
            ScorerResult with score=1.0 and passed=True
        """
        return cls(score=1.0, passed=True, label="pass", explanation=explanation)

    @classmethod
    def fail_result(cls, explanation: Optional[str] = None) -> "ScorerResult":
        """Create a failing result.

        Args:
            explanation: Optional explanation

        Returns:
            ScorerResult with score=0.0 and passed=False
        """
        return cls(score=0.0, passed=False, label="fail", explanation=explanation)


# Alias for backwards compatibility
ScorerResultPy = ScorerResult
