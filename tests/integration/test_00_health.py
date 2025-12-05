"""
Integration Tests: Basic Health Checks

Simple smoke tests to verify the platform stack is working:
- Platform is accessible (either local dev server or Docker container)
- Platform health endpoint responds
- Gateway is accessible
- Basic connectivity works

These tests should ALWAYS pass if the infrastructure is set up correctly.
"""

import pytest
import requests


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
    assert platform["mode"] in ["local", "embedded", "postgres", "managed"]
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
    - SQLite database path is configured
    - Platform is using correct backend
    """
    db_url = platform["db_url"]
    db_type = platform["db_type"]

    assert db_type == "sqlite", f"Expected sqlite, got: {db_type}"
    # In local mode, db is at /tmp/agnt5/...
    # In embedded mode, db is at /data/orchestration.db (inside container)
    assert "agnt5" in db_url or "orchestration" in db_url, f"Unexpected db_url: {db_url}"

    print(f"\n✅ Database configuration correct")
    print(f"   Type: {db_type}")
    print(f"   Path: {db_url}")


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
    Test platform is configured correctly.

    This validates:
    - Correct backend configuration
    - Journal backend is embedded
    - Orchestration backend is SQLite
    """
    assert platform["mode"] in ["local", "embedded", "postgres", "managed"]
    assert platform["journal_backend"] == "embedded"
    assert platform["orchestration_backend"] == "sqlite"
    assert platform["db_type"] == "sqlite"

    print(f"\n✅ Platform configuration correct")
    print(f"   Mode: {platform['mode']}")
    print(f"   Journal: {platform['journal_backend']}")
    print(f"   Orchestration: {platform['orchestration_backend']}")
