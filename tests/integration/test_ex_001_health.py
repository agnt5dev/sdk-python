"""
Integration Tests: Basic Health Checks

Simple smoke tests to verify the platform stack is working:
- Platform is accessible (either local dev server or Docker container)
- Platform health endpoint responds
- Gateway is accessible
- Basic connectivity works

These tests should ALWAYS pass if the infrastructure is set up correctly.

Run with:
    pytest tests/integration/test_ex_001_health.py -v
"""

import pytest
import requests


# Valid configurations per mode
VALID_MODES = ["local", "subprocess", "embedded", "postgres", "managed"]
VALID_DB_TYPES = ["sqlite", "postgres", "cockroach"]
VALID_JOURNAL_BACKENDS = ["embedded", "redpanda"]
VALID_ORCHESTRATION_BACKENDS = ["sqlite", "postgres", "cockroach"]


@pytest.mark.integration
def test_platform_starts_successfully(platform):
    """
    Test that platform fixture starts successfully.

    This validates:
    - Platform is accessible (local or container)
    - Ports are exposed correctly
    - Platform configuration is returned
    """
    assert platform is not None
    assert platform["mode"] in VALID_MODES
    assert "gateway_url" in platform
    assert "coordinator_url" in platform
    assert "db_url" in platform

    print(f"\n✅ Platform started in {platform['mode']} mode")
    print(f"   Gateway: {platform['gateway_url']}")
    print(f"   Coordinator: {platform['coordinator_url']}")
    print(f"   Database: {platform['db_url']}")


@pytest.mark.integration
def test_gateway_health_check(platform):
    """
    Test gateway health endpoint responds.

    This validates:
    - Gateway HTTP service is running
    - Health endpoint is accessible
    - Basic HTTP connectivity works
    """
    gateway_url = platform["gateway_url"]

    response = requests.get(f"{gateway_url}/v1/health", timeout=5)

    assert response.status_code == 200, (
        f"Expected 200 OK from health endpoint, got {response.status_code}"
    )

    print(f"\n✅ Gateway health check passed")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:100]}")


@pytest.mark.integration
def test_database_accessible(platform):
    """
    Test database configuration is correct.

    This validates:
    - Database URL/path is configured
    - Platform is using a valid backend type
    """
    db_url = platform["db_url"]
    db_type = platform["db_type"]

    assert db_type in VALID_DB_TYPES, f"Invalid db_type: {db_type}"
    assert db_url is not None and len(db_url) > 0, "db_url should not be empty"

    print(f"\n✅ Database configuration correct")
    print(f"   Type: {db_type}")
    print(f"   URL: {db_url}")


@pytest.mark.integration
def test_client_creation(client):
    """
    Test Client instance can be created.

    This validates:
    - Client fixture works
    - Client can connect to platform
    - Basic client setup succeeds
    """
    assert client is not None

    print(f"\n✅ Client created successfully")


@pytest.mark.integration
def test_platform_mode_configuration(platform):
    """
    Test platform is configured correctly for the current mode.

    This validates:
    - Mode is valid
    - Backend configurations are consistent with mode
    """
    mode = platform["mode"]
    journal_backend = platform["journal_backend"]
    orchestration_backend = platform["orchestration_backend"]
    db_type = platform["db_type"]

    assert mode in VALID_MODES
    assert journal_backend in VALID_JOURNAL_BACKENDS
    assert orchestration_backend in VALID_ORCHESTRATION_BACKENDS
    assert db_type in VALID_DB_TYPES

    # Validate mode-specific configurations
    if mode in ["local", "subprocess", "embedded"]:
        assert db_type == "sqlite"
        assert orchestration_backend == "sqlite"
        assert journal_backend == "embedded"
    elif mode == "postgres":
        assert db_type == "postgres"
        assert orchestration_backend == "postgres"
    elif mode == "managed":
        assert db_type == "cockroach"
        assert orchestration_backend == "cockroach"
        assert journal_backend == "redpanda"

    print(f"\n✅ Platform configuration correct")
    print(f"   Mode: {mode}")
    print(f"   Journal: {journal_backend}")
    print(f"   Orchestration: {orchestration_backend}")
    print(f"   Database: {db_type}")
