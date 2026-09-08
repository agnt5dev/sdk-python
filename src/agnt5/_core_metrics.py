"""Best-effort measurements using the worker's shared native monotonic clock."""
import asyncio
from contextlib import contextmanager


def _backend():
    try:
        from . import _core
        if hasattr(_core, "core_metric_time_ms") and hasattr(_core, "record_core_business_timing"):
            return _core
    except ImportError:
        pass
    return None


@contextmanager
def business_timing(run_id):
    backend, started = None, None
    try:
        backend = _backend()
        if backend is not None:
            started = backend.core_metric_time_ms()
    except Exception:
        pass
    outcome = "success"
    try:
        yield
    except BaseException as error:
        outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
        raise
    finally:
        if backend is not None and started is not None:
            try:
                backend.record_core_business_timing(run_id, started, outcome)
            except Exception:
                # Missing or failed telemetry cannot alter an execution decision.
                pass
