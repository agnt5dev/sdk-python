"""
Root conftest.py for pytest configuration.

Defines custom pytest options and markers for test suite.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-live-lm-tests",
        action="store_true",
        default=False,
        help="Run integration tests that make real LM API calls (requires API keys)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "lm_live: marks tests that make real LM API calls and require API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--run-live-lm-tests"):
        skip_live = pytest.mark.skip(reason="need --run-live-lm-tests option to run")
        for item in items:
            if "lm_live" in item.keywords:
                item.add_marker(skip_live)
