"""Python types for evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """Tool call extracted from journal events or normalized session payloads."""

    name: str
    arguments: Optional[Any] = None
    call_id: Optional[str] = None
    span_id: Optional[str] = None
    timestamp_ns: Optional[int] = None
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def get_tool_calls(self) -> List[ToolCall]:
        """Extract typed tool calls from available journal events."""
        return extract_tool_calls(self.events or [])

    def get_tool_call_names(self) -> List[str]:
        """Return tool names in observed call order."""
        return tool_call_names(self.get_tool_calls())

    def tool_trajectory_matches(self, expected: List[str], mode: str = "exact") -> bool:
        """Compare observed tool-call order to an expected trajectory."""
        return tool_trajectory_matches(self.get_tool_call_names(), expected, mode=mode)


@dataclass
class ScorerRequest:
    """Request passed to a scorer.

    Attributes:
        output: Component output to evaluate (required)
        expected: Expected output for comparison (optional)
        input: Original input to the component (optional, for context)
        trace: Execution trace events (optional, for glassbox testing)
        config: Scorer-specific configuration (optional)
        peer_scores: Scores already produced for this item by earlier scorers
        trace_eval_context: Normalized trace-eval context artifact, when available
    """

    output: Any
    expected: Optional[Any] = None
    input: Optional[Any] = None
    trace: Optional[List[TraceEvent]] = None
    config: Optional[Dict[str, Any]] = None
    peer_scores: Optional[List[Dict[str, Any]]] = None
    trace_eval_context: Optional[Dict[str, Any]] = None

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

    def get_tool_calls(self) -> List[ToolCall]:
        """Extract typed tool calls from request journal events."""
        return extract_tool_calls(self.trace or [])

    def get_tool_call_names(self) -> List[str]:
        """Return tool names in observed call order."""
        return tool_call_names(self.get_tool_calls())

    def tool_trajectory_matches(self, expected: List[str], mode: str = "exact") -> bool:
        """Compare observed tool-call order to an expected trajectory."""
        return tool_trajectory_matches(self.get_tool_call_names(), expected, mode=mode)


def extract_tool_calls(events: List[TraceEvent]) -> List[ToolCall]:
    """Extract tool calls from normalized session data and legacy journal events."""
    calls: List[ToolCall] = []
    by_key: Dict[str, int] = {}

    def add(call: Optional[ToolCall], fallback_key: str) -> None:
        if call is None or not call.name:
            return
        key = call.call_id or call.span_id or fallback_key
        if key in by_key:
            calls[by_key[key]] = _merge_tool_calls(calls[by_key[key]], call)
            return
        by_key[key] = len(calls)
        calls.append(call)

    for event in events:
        data = event.data if isinstance(event.data, dict) else {}
        for index, payload in enumerate(_iter_tool_call_payloads(data)):
            add(_tool_call_from_mapping(payload, event, index), f"{event.event_id}:payload:{index}")
        if "tool" in event.event_type:
            add(_tool_call_from_event(event), event.event_id)
    return calls


def tool_call_names(calls: List[ToolCall]) -> List[str]:
    """Return non-empty tool-call names in call order."""
    return [call.name for call in calls if call.name]


def tool_trajectory_exact(actual: List[str], expected: List[str]) -> bool:
    """Return true when the observed trajectory exactly matches expected."""
    return actual == expected


def tool_trajectory_in_order(actual: List[str], expected: List[str]) -> bool:
    """Return true when expected appears as an ordered subsequence."""
    if not expected:
        return True
    index = 0
    for name in actual:
        if name == expected[index]:
            index += 1
            if index == len(expected):
                return True
    return False


def tool_trajectory_any_order(actual: List[str], expected: List[str]) -> bool:
    """Return true when actual contains the expected names with matching counts."""
    remaining: Dict[str, int] = {}
    for name in actual:
        remaining[name] = remaining.get(name, 0) + 1
    for name in expected:
        count = remaining.get(name, 0)
        if count <= 0:
            return False
        remaining[name] = count - 1
    return True


def tool_trajectory_matches(actual: List[str], expected: List[str], mode: str = "exact") -> bool:
    """Compare a tool trajectory using exact, in_order, or any_order semantics."""
    if mode == "exact":
        return tool_trajectory_exact(actual, expected)
    if mode == "in_order":
        return tool_trajectory_in_order(actual, expected)
    if mode == "any_order":
        return tool_trajectory_any_order(actual, expected)
    raise ValueError("mode must be one of: exact, in_order, any_order")


def _iter_tool_call_payloads(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []

    def extend_from(value: Any) -> None:
        if isinstance(value, list):
            payloads.extend(item for item in value if isinstance(item, dict))

    extend_from(data.get("tool_calls"))
    extend_from(data.get("toolCalls"))
    for key in ("normalized_session", "session", "trace_session", "journal_session"):
        nested = data.get(key)
        if isinstance(nested, dict):
            extend_from(nested.get("tool_calls"))
            extend_from(nested.get("toolCalls"))
    for key in ("response", "output", "message"):
        nested = data.get(key)
        if isinstance(nested, dict):
            extend_from(nested.get("tool_calls"))
            extend_from(nested.get("toolCalls"))
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                extend_from(message.get("tool_calls"))
                extend_from(message.get("toolCalls"))
    return payloads


def _tool_call_from_event(event: TraceEvent) -> Optional[ToolCall]:
    data = event.data if isinstance(event.data, dict) else {}
    return _tool_call_from_mapping(data, event, 0)


def _tool_call_from_mapping(
    payload: Dict[str, Any],
    event: TraceEvent,
    index: int,
) -> Optional[ToolCall]:
    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
    name = _string_or_none(
        _first_present(
            payload.get("name"),
            payload.get("tool_name"),
            function.get("name"),
            event.name if "tool" in event.event_type else None,
        )
    )
    if not name:
        return None
    call_id = _string_or_none(
        _first_present(
            payload.get("call_id"),
            payload.get("tool_call_id"),
            payload.get("id"),
            event.correlation_id if "tool" in event.event_type else None,
        )
    )
    arguments = _first_present(
        payload.get("arguments"), payload.get("args"), function.get("arguments")
    )
    return ToolCall(
        name=name,
        arguments=_decode_arguments(arguments),
        call_id=call_id,
        span_id=_string_or_none(payload.get("span_id")) or event.correlation_id,
        timestamp_ns=_int_or_none(payload.get("timestamp_ns")) or event.timestamp_ns,
        started_at=_int_or_none(payload.get("started_at")),
        ended_at=_int_or_none(payload.get("ended_at")),
        status=_string_or_none(payload.get("status")) or _status_from_event_type(event.event_type),
        metadata=_tool_call_metadata(payload, event, index),
    )


def _tool_call_metadata(payload: Dict[str, Any], event: TraceEvent, index: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source_event_id": event.event_id,
        "source_event_type": event.event_type,
        "source_index": index,
    }
    for key in (
        "arguments_ref",
        "args_ref",
        "arguments_hash",
        "args_hash",
        "result_ref",
        "result_hash",
        "output_ref",
        "output_hash",
        "duration_ms",
        "error_code",
        "error_message_sanitized",
    ):
        if payload.get(key) is not None:
            metadata[key] = payload[key]
    attributes = payload.get("attributes_safe")
    if isinstance(attributes, dict):
        metadata["attributes_safe"] = attributes
    return metadata


def _merge_tool_calls(existing: ToolCall, incoming: ToolCall) -> ToolCall:
    metadata = dict(existing.metadata)
    metadata.update(incoming.metadata)
    return ToolCall(
        name=incoming.name or existing.name,
        arguments=incoming.arguments if incoming.arguments is not None else existing.arguments,
        call_id=incoming.call_id or existing.call_id,
        span_id=incoming.span_id or existing.span_id,
        timestamp_ns=existing.timestamp_ns or incoming.timestamp_ns,
        started_at=existing.started_at or incoming.started_at,
        ended_at=incoming.ended_at or existing.ended_at,
        status=incoming.status or existing.status,
        metadata=metadata,
    )


def _decode_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        import json

        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _status_from_event_type(event_type: str) -> Optional[str]:
    if event_type.endswith(".started"):
        return "started"
    if event_type.endswith(".completed"):
        return "completed"
    if event_type.endswith(".failed"):
        return "failed"
    return None


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
