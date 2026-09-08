import asyncio
from types import SimpleNamespace

import pytest

from agnt5 import _core_metrics as metrics


@pytest.mark.asyncio
async def test_business_timing_records_failure_without_replacing_it(monkeypatch):
    observations = []
    backend = SimpleNamespace(core_metric_time_ms=lambda: 100.0,
                              record_core_business_timing=lambda *args: observations.append(args))
    monkeypatch.setattr(metrics, "_backend", lambda: backend)
    with pytest.raises(ValueError, match="user failure"):
        with metrics.business_timing("run"):
            raise ValueError("user failure")
    assert observations == [("run", 100.0, "error")]


def test_telemetry_failure_cannot_fail_a_successful_body(monkeypatch):
    def fail(*args):
        raise RuntimeError("telemetry unavailable")
    monkeypatch.setattr(metrics, "_backend", lambda: SimpleNamespace(
        core_metric_time_ms=lambda: 10.0, record_core_business_timing=fail))
    with metrics.business_timing("run"):
        result = 42
    assert result == 42


def test_cancellation_is_recorded_and_propagated(monkeypatch):
    observations = []
    monkeypatch.setattr(metrics, "_backend", lambda: SimpleNamespace(
        core_metric_time_ms=lambda: 10.0,
        record_core_business_timing=lambda *args: observations.append(args)))
    with pytest.raises(asyncio.CancelledError):
        with metrics.business_timing("run"):
            raise asyncio.CancelledError()
    assert observations[0][-1] == "cancelled"
