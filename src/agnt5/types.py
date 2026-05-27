"""Type definitions for AGNT5 SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

# Type aliases
JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
HandlerFunc = Callable[..., Awaitable[Any]]

T = TypeVar("T")


class BackoffType(str, Enum):
    """Backoff strategy for retry policies."""

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class RetryPolicy:
    """Configuration for function retry behavior."""

    max_attempts: int = 3
    initial_interval_ms: int = 1000
    max_interval_ms: int = 60000

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_interval_ms < 0:
            raise ValueError("initial_interval_ms must be non-negative")
        if self.max_interval_ms < self.initial_interval_ms:
            raise ValueError("max_interval_ms must be >= initial_interval_ms")


@dataclass
class BackoffPolicy:
    """Configuration for retry backoff strategy."""

    type: BackoffType = BackoffType.EXPONENTIAL
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")


@dataclass
class TriggerSpec:
    """Typed trigger declaration attached to a component registration."""

    trigger_type: str
    event_name: str = ""
    trigger_id: str = ""
    filter_expression: str = ""
    input_mapping: str = ""
    batch_window_ms: int = 0
    delay_expression: str = ""


def event(
    name: str,
    *,
    trigger_id: str = "",
    filter_expression: str = "",
    input_mapping: str = "",
    batch_window_ms: int = 0,
    delay_expression: str = "",
) -> TriggerSpec:
    """Declare an event trigger for a workflow."""

    event_name = name.strip()
    if not event_name:
        raise ValueError("event trigger name is required")
    return TriggerSpec(
        trigger_id=trigger_id,
        trigger_type="event",
        event_name=event_name,
        filter_expression=filter_expression,
        input_mapping=input_mapping,
        batch_window_ms=batch_window_ms,
        delay_expression=delay_expression,
    )


def webhook(
    source: str,
    *,
    event: str,
    trigger_id: str = "",
    filter_expression: str = "",
    input_mapping: str = "",
    batch_window_ms: int = 0,
    delay_expression: str = "",
) -> TriggerSpec:
    """Declare a webhook-event trigger for a workflow.

    Webhook deliveries arriving at
    ``POST /v1/webhooks/{source}/{integration_id}`` are dispatched as
    events with name ``{source}.{event}``. The runtime gateway emits
    that exact event name in ``run.queued.metadata.event_type``, so
    declaring ``webhook("sentry", event="issue.created")`` subscribes
    the workflow to ``sentry.issue.created``.

    Args:
        source: Webhook source. One of ``standard``, ``sentry``,
            ``stripe``, ``github``, ``slack`` (lowercased).
        event: Event identifier within the source. For Sentry use
            ``"<resource>.<action>"`` (e.g. ``"issue.created"``); for
            GitHub use ``"<event>.<action>"`` (e.g.
            ``"issues.opened"``); for Stripe use the body ``type``
            (e.g. ``"payment_intent.succeeded"``); for Slack use the
            event-callback type (e.g. ``"app_mention"``); for Standard
            Webhooks publishers, use the ``webhook-event`` header
            value.
    """

    src = source.strip().lower()
    evt = event.strip()
    if not src:
        raise ValueError("webhook source is required")
    if not evt:
        raise ValueError("webhook event is required")
    return TriggerSpec(
        trigger_id=trigger_id,
        trigger_type="event",
        event_name=f"{src}.{evt}",
        filter_expression=filter_expression,
        input_mapping=input_mapping,
        batch_window_ms=batch_window_ms,
        delay_expression=delay_expression,
    )


@dataclass
class FunctionConfig:
    """Configuration for a function handler."""

    name: str
    handler: HandlerFunc
    retries: Optional[RetryPolicy] = None
    backoff: Optional[BackoffPolicy] = None
    timeout_ms: Optional[int] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class WorkflowConfig:
    """Configuration for a workflow handler."""

    name: str
    handler: HandlerFunc
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, str]] = None
    triggers: Optional[List[TriggerSpec]] = None
