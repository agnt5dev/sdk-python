"""Run-scoped log correlation (AGNT5-1070).

Only ``ctx.logger`` records used to carry a run id, so every other application
log line -- the worker dispatcher, the executors, third-party libraries hitting
the root handler -- reached the control plane unattributed and ``get_run_logs``
could not return them. ``run_scope`` binds the id for the whole invocation and
the OTLP handler falls back to it.
"""

import asyncio
import importlib.util
import logging
from pathlib import Path

# Import _telemetry directly, without triggering the full agnt5 package
# (mirrors tests/unit/test_context_logger.py).
_telemetry_path = Path(__file__).parent.parent.parent / "src" / "agnt5" / "_telemetry.py"
spec = importlib.util.spec_from_file_location("_telemetry", _telemetry_path)
_telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_telemetry)

run_scope = _telemetry.run_scope
get_current_run_id = _telemetry.get_current_run_id
OpenTelemetryHandler = _telemetry.OpenTelemetryHandler

RUN_ID = "01a05cae-d48c-72a2-be86-3154a8979ca7"


def _record(name="agnt5.worker._core", **attrs):
    record = logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname="/app/agnt5/worker/_core.py",
        lineno=804,
        msg="Handling function request: ks_analyze_text, input size: 58 bytes",
        args=(),
        exc_info=None,
    )
    for key, value in attrs.items():
        setattr(record, key, value)
    return record


def _handler_capturing(calls):
    handler = OpenTelemetryHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._log_from_python = lambda **kwargs: calls.append(kwargs)
    return handler


class TestRunScope:
    def test_binds_and_resets(self):
        assert get_current_run_id() is None
        with run_scope(RUN_ID):
            assert get_current_run_id() == RUN_ID
        assert get_current_run_id() is None

    def test_falsy_run_id_is_a_noop(self):
        for empty in (None, ""):
            with run_scope(empty):
                assert get_current_run_id() is None

    def test_nested_scope_restores_outer(self):
        inner = "01a05c0f-1848-71a0-8549-95c9eb50f560"
        with run_scope(RUN_ID):
            with run_scope(inner):
                assert get_current_run_id() == inner
            assert get_current_run_id() == RUN_ID

    def test_reset_even_when_body_raises(self):
        try:
            with run_scope(RUN_ID):
                raise ValueError("boom")
        except ValueError:
            pass
        assert get_current_run_id() is None


class TestHandlerAttribution:
    def test_record_without_run_id_inherits_the_scope(self):
        """The bug: this line reached the control plane with no run id."""
        calls = []
        handler = _handler_capturing(calls)

        with run_scope(RUN_ID):
            handler.emit(_record())

        assert len(calls) == 1
        assert calls[0]["run_id"] == RUN_ID

    def test_record_run_id_wins_over_the_scope(self):
        """ctx.logger sets run_id on the record; it must not be overridden."""
        calls = []
        handler = _handler_capturing(calls)
        explicit = "01a05c0f-1848-71a0-8549-95c9eb50f560"

        with run_scope(RUN_ID):
            handler.emit(_record(name="agnt5.execution", run_id=explicit))

        assert calls[0]["run_id"] == explicit

    def test_outside_any_run_stays_unattributed(self):
        """Worker startup logs belong to no run and must not borrow one."""
        calls = []
        handler = _handler_capturing(calls)

        handler.emit(_record())

        assert calls[0]["run_id"] is None


class TestConcurrency:
    def test_concurrent_runs_do_not_leak_ids(self):
        """Two runs in one worker must not see each other's id."""
        other = "01a05c11-ab9d-7010-96af-e0c145c90a41"
        seen = {}

        async def run_one(run_id, delay):
            with run_scope(run_id):
                await asyncio.sleep(delay)
                seen[run_id] = get_current_run_id()

        async def main():
            await asyncio.gather(run_one(RUN_ID, 0.02), run_one(other, 0.01))

        asyncio.run(main())

        assert seen == {RUN_ID: RUN_ID, other: other}
