"""Internal diagnostic verbosity must be applied before telemetry is generated."""
import importlib.util
import logging
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "_internal_log_telemetry",
    Path(__file__).parents[2] / "src" / "agnt5" / "_telemetry.py",
)
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)


@pytest.fixture
def forwarded(monkeypatch):
    calls = []
    monkeypatch.delenv("AGNT5_DEBUG", raising=False)
    monkeypatch.setattr(telemetry, "_default_log_level", None)
    monkeypatch.setattr(telemetry.OpenTelemetryHandler, "emit", lambda _self, record: calls.append(record))
    yield calls
    for name in ("agnt5.test.internal", "agnt5.test.execution"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)


def test_default_internal_debug_never_reaches_telemetry(forwarded):
    logger = telemetry.setup_module_logger("agnt5.test.internal")
    logger.debug("Recorded step completion: %s", "step-1")
    assert forwarded == [], "hidden internal DEBUG records still create telemetry/stream work"
    logger.info("worker ready")
    assert [record.getMessage() for record in forwarded] == ["worker ready"]


def test_internal_debug_can_be_enabled_explicitly(forwarded, monkeypatch):
    monkeypatch.setenv("AGNT5_DEBUG", "true")
    logger = telemetry.setup_module_logger("agnt5.test.internal")
    logger.debug("diagnostic")
    assert [record.getMessage() for record in forwarded] == ["diagnostic"]


def test_explicit_module_log_level_controls_telemetry(forwarded):
    logger = telemetry.setup_module_logger("agnt5.test.internal", logging.WARNING)
    logger.info("hidden")
    logger.warning("visible")
    assert [record.getMessage() for record in forwarded] == ["visible"]


def test_application_context_logging_keeps_existing_telemetry_behavior(forwarded):
    logger = logging.getLogger("agnt5.test.execution")
    telemetry.setup_context_logger(logger, logging.INFO)
    logger.debug("application diagnostic")
    assert [record.getMessage() for record in forwarded] == ["application diagnostic"]


def test_log_configuration_updates_previously_imported_modules(forwarded):
    logger = telemetry.setup_module_logger("agnt5.test.internal")
    telemetry.set_log_level("DEBUG")
    logger.debug("enabled")
    telemetry.set_log_level("INFO")
    logger.debug("disabled")
    assert [record.getMessage() for record in forwarded] == ["enabled"]


def test_explicit_module_override_survives_global_configuration(forwarded):
    logger = telemetry.setup_module_logger("agnt5.test.internal", logging.WARNING)
    telemetry.set_log_level("DEBUG")
    logger.info("hidden")
    logger.warning("visible")
    assert [record.getMessage() for record in forwarded] == ["visible"]
