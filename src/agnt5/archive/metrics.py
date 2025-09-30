"""Metrics and tracing utilities."""

from __future__ import annotations

from typing import Any, Dict


class MetricsClient:
    """Simple metrics facade that integrates with OpenTelemetry when available."""

    def __init__(self, attributes: Dict[str, Any]):
        self._attributes = attributes
        try:
            from opentelemetry.metrics import get_meter  # type: ignore

            self._meter = get_meter("agnt5.sdk")
        except Exception:  # pragma: no cover - only triggered without OTEL installed
            self._meter = None
        self._counters: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}

    def increment(self, name: str, value: float = 1.0, **labels: Any) -> None:
        attributes = {**self._attributes, **labels}
        if self._meter:
            if name not in self._counters:
                self._counters[name] = self._meter.create_counter(name)
            self._counters[name].add(value, attributes=attributes)
        else:
            logger.debug("Metric increment %s=%s %s", name, value, attributes)

    def observe(self, name: str, value: float, **labels: Any) -> None:
        attributes = {**self._attributes, **labels}
        if self._meter:
            if name not in self._histograms:
                self._histograms[name] = self._meter.create_histogram(name)
            self._histograms[name].record(value, attributes=attributes)
        else:
            logger.debug("Metric observe %s=%s %s", name, value, attributes)


class SpanContext:
    """Tracing helper exposing a context manager for child spans."""

    def __init__(self, attributes: Dict[str, Any]):
        self._attributes = attributes
        try:
            from opentelemetry import trace  # type: ignore

            self._tracer = trace.get_tracer("agnt5.sdk")
        except Exception:  # pragma: no cover - only triggered without OTEL installed
            self._tracer = None

    @contextmanager
    def start(self, name: str, **attributes: Any):
        merged = {**self._attributes, **attributes}
        if self._tracer:
            from opentelemetry.trace import Status, StatusCode  # type: ignore

            with self._tracer.start_as_current_span(name, attributes=merged) as span:
                try:
                    yield span
                except Exception as exc:  # pragma: no cover - user exception path
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
        else:
            yield None
