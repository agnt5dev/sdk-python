"""
Pytest configuration for AGNT5 Python SDK integration tests.

Supports two test modes:
1. EMBEDDED (default) - Starts dev-server binary as subprocess
2. HOSTED - Connects to external platform via AGNT5_PLATFORM_URL

Usage:
    # Embedded mode (default)
    pytest tests/integration/

    # Hosted mode (external platform)
    AGNT5_PLATFORM_URL=http://localhost:34181 pytest tests/integration/ --hosted
"""

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import pytest
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--hosted",
        action="store_true",
        default=False,
        help="Connect to external platform via AGNT5_PLATFORM_URL environment variable",
    )


def pytest_generate_tests(metafunc):
    """Configure test parametrization based on CLI options."""
    if "test_mode" in metafunc.fixturenames:
        # Determine mode from CLI flag
        mode = "hosted" if metafunc.config.getoption("--hosted") else "embedded"
        metafunc.parametrize("test_mode", [mode], scope="session")


# ============================================================================
# Setup Functions
# ============================================================================


def find_standalone_binary() -> str:
    """
    Find standalone dev-server binary in repo.

    Search order:
    1. AGNT5_STANDALONE_BINARY environment variable
    2. ../../../platform/bin/standalone (relative to tests/integration/)
    3. Raise error if not found

    Returns:
        Path to standalone binary

    Raises:
        FileNotFoundError: If binary not found
    """
    # Check environment variable
    binary = os.getenv("AGNT5_STANDALONE_BINARY")
    if binary and os.path.exists(binary):
        logger.info(f"📦 Using standalone binary from env: {binary}")
        return binary

    # Check relative path from tests/integration/
    # __file__ is .../agnt5/sdk/sdk-python/tests/integration/conftest.py
    # repo_root is .../agnt5/
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    binary_path = repo_root / "platform" / "bin" / "standalone"

    if binary_path.exists():
        logger.info(f"📦 Using standalone binary from repo: {binary_path}")
        return str(binary_path)

    raise FileNotFoundError(
        "❌ Standalone binary not found!\n\n"
        "Build it first:\n"
        "  cd platform && go build -o bin/standalone cmd/standalone/main.go\n\n"
        "Or set environment variable:\n"
        "  export AGNT5_STANDALONE_BINARY=/path/to/binary"
    )


def wait_for_health(url: str, timeout: int = 30) -> None:
    """
    Wait for platform health endpoint to respond.

    Args:
        url: Base URL (e.g., http://localhost:34181)
        timeout: Maximum seconds to wait

    Raises:
        TimeoutError: If health check fails within timeout
    """
    health_url = f"{url}/v1/health"
    start = time.time()

    logger.info(f"📡 Waiting for platform at {url}...")

    while time.time() - start < timeout:
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ Platform is healthy")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"Platform at {url} did not become healthy within {timeout}s")


def setup_embedded_mode(data_dir: str) -> Dict:
    """
    Start dev-server binary as subprocess (embedded mode).

    Args:
        data_dir: Temporary directory for SQLite database

    Returns:
        dict: {
            "mode": "embedded",
            "gateway_url": "http://localhost:34181",
            "coordinator_url": "http://localhost:34186",
            "process": subprocess.Popen,
        }

    Raises:
        FileNotFoundError: If binary not found
        TimeoutError: If platform doesn't start
    """
    logger.info("\n🚀 Setting up EMBEDDED mode (subprocess)")

    # 1. Find binary
    binary_path = find_standalone_binary()

    logger.info(f"📁 Data dir: {data_dir}")

    # 2. Start subprocess (no --project-path since we use --disable-worker
    # and register components via the test worker instead)
    env = os.environ.copy()
    process = subprocess.Popen(
        [
            binary_path,
            "--data-dir",
            data_dir,
            "--disable-worker",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    logger.info(f"   Platform PID: {process.pid}")

    # 4. Wait for health
    try:
        wait_for_health("http://localhost:34181")
    except TimeoutError:
        # Kill process and show logs
        process.terminate()
        stdout, _ = process.communicate(timeout=5)
        logger.error(f"Platform logs:\n{stdout}")
        raise

    logger.info("✅ Subprocess mode ready")
    logger.info("   Gateway: http://localhost:34181")
    logger.info("   Coordinator: http://localhost:34186")

    # 5. Return metadata
    return {
        "mode": "embedded",
        "gateway_url": "http://localhost:34181",
        "coordinator_url": "http://localhost:34186",
        "process": process,
    }


def setup_hosted_mode() -> Dict:
    """
    Connect to external platform via environment variables (hosted mode).

    Requires:
        AGNT5_PLATFORM_URL: Gateway URL (e.g., http://localhost:34181)
        AGNT5_COORDINATOR_URL: Optional, defaults to gateway with port 34186

    Returns:
        dict: {
            "mode": "hosted",
            "gateway_url": "http://...",
            "coordinator_url": "http://...",
        }

    Raises:
        ValueError: If AGNT5_PLATFORM_URL not set
        TimeoutError: If platform not reachable
    """
    logger.info("\n🌐 Setting up HOSTED mode (external platform)")

    # 1. Get gateway URL from environment
    gateway_url = os.getenv("AGNT5_PLATFORM_URL")
    if not gateway_url:
        raise ValueError(
            "❌ AGNT5_PLATFORM_URL environment variable required for hosted mode!\n\n"
            "Example:\n"
            "  export AGNT5_PLATFORM_URL=http://localhost:34181\n"
            "  pytest tests/integration/ --hosted"
        )

    # 2. Derive coordinator URL if not provided
    coordinator_url = os.getenv("AGNT5_COORDINATOR_URL")
    if not coordinator_url:
        # Replace gateway port (34181) with coordinator port (34186)
        coordinator_url = gateway_url.replace(":34181", ":34186")
        # If no port specified, append coordinator port
        if ":34181" not in gateway_url and "://" in gateway_url:
            coordinator_url = f"{gateway_url.rstrip('/')}:34186"

    logger.info(f"📡 Gateway: {gateway_url}")
    logger.info(f"📡 Coordinator: {coordinator_url}")

    # 3. Verify platform is reachable
    wait_for_health(gateway_url)

    logger.info("✅ Hosted mode ready")
    logger.info("   Connected to external platform")

    # 4. Return metadata
    return {
        "mode": "hosted",
        "gateway_url": gateway_url,
        "coordinator_url": coordinator_url,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def persistent_data_dir(request):
    """
    Create persistent data directory for test session.

    Only used in embedded mode. In hosted mode, returns None.

    Yields:
        Optional[str]: Path to temp directory or None
    """
    # Skip in hosted mode
    if request.config.getoption("--hosted"):
        yield None
        return

    # Create temp directory for embedded mode
    temp_dir = tempfile.mkdtemp(prefix="agnt5_data")
    logger.info(f"\n📁 Created persistent data directory: {temp_dir}")

    yield temp_dir

    # Cleanup handled by tempfile module


@pytest.fixture(scope="session")
def test_mode(request):
    """
    Get test mode from pytest configuration.

    Returns:
        str: "embedded" or "hosted"
    """
    return "hosted" if request.config.getoption("--hosted") else "embedded"


@pytest.fixture(scope="session")
def platform(request, test_mode, persistent_data_dir):
    """
    Setup platform for testing (embedded or hosted mode).

    Args:
        test_mode: "embedded" or "hosted"
        persistent_data_dir: Temp directory (only used in embedded)

    Yields:
        dict: Platform metadata with URLs
            {
                "mode": str,
                "gateway_url": str,
                "coordinator_url": str,
                "process": Optional[subprocess.Popen],
            }
    """
    # Setup based on mode
    if test_mode == "embedded":
        if persistent_data_dir is None:
            raise ValueError("persistent_data_dir required for embedded mode")
        setup = setup_embedded_mode(persistent_data_dir)
    else:  # hosted
        setup = setup_hosted_mode()

    yield setup

    # Cleanup
    if "process" in setup and setup["process"]:
        logger.info("\n🧹 Stopping platform subprocess...")
        process = setup["process"]
        process.terminate()
        try:
            process.wait(timeout=5)
            logger.info("✅ Platform stopped")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Platform didn't stop gracefully, killing...")
            process.kill()
            process.wait()


@pytest.fixture(scope="session")
def worker_process(platform):
    """
    Start worker subprocess that registers test fixtures.

    Works for both modes:
    - Embedded: Worker connects to local subprocess
    - Hosted: Worker connects to external platform

    Args:
        platform: Platform fixture with connection details

    Yields:
        subprocess.Popen: Worker process
    """
    coordinator_url = platform["coordinator_url"]

    logger.info(f"\n🔌 Worker connecting to coordinator: {coordinator_url}")

    # Find worker script
    worker_script = Path(__file__).parent / "test_worker_app.py"
    logger.info(f"📁 Worker service path: {worker_script.parent}")
    logger.info(f"📄 Worker script: {worker_script.name}")

    # Start worker subprocess
    env = os.environ.copy()
    env["AGNT5_COORDINATOR_ENDPOINT"] = coordinator_url

    process = subprocess.Popen(
        ["uv", "run", "python", str(worker_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=worker_script.parent,
    )

    logger.info(f"   Worker PID: {process.pid}")

    # Wait for worker registration
    max_wait = 5
    start = time.time()

    logger.info(f"⏳ Waiting for worker registration (max {max_wait}s)...")

    while time.time() - start < max_wait:
        # Check if process died
        if process.poll() is not None:
            stdout, _ = process.communicate()
            logger.error(f"Worker process died:\n{stdout}")
            raise RuntimeError("Worker process died during startup")

        # Check if worker registered via components endpoint
        # Wait for the 'add' function specifically (from test fixtures)
        try:
            response = requests.get(
                f"{platform['gateway_url']}/v1/components",
                timeout=1,
            )
            if response.status_code == 200:
                components = response.json()
                component_names = [c.get("name") for c in components]
                if "add" in component_names:
                    logger.info(f"✅ Worker registered ({len(components)} components: {component_names})")
                    break
        except Exception:
            pass

        time.sleep(0.5)
    else:
        # Timeout - log warning but continue
        logger.warning(f"⚠️  Worker process running but not registered after {max_wait}s")

    logger.info(f"✅ Worker started (PID: {process.pid})")

    yield process

    # Cleanup
    logger.info("\n🧹 Stopping worker (PID: {})...".format(process.pid))
    process.terminate()
    try:
        stdout, _ = process.communicate(timeout=5)
        # Only show logs if there's an error or verbose mode
        if process.returncode != 0 and process.returncode != -15:  # -15 is SIGTERM
            logger.info(f"\n📋 Worker logs (all):\n{stdout}")
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Worker didn't stop gracefully, killing...")
        process.kill()
        process.wait()


@pytest.fixture(scope="function")
def client(platform):
    """
    Create AGNT5 client for testing.

    Args:
        platform: Platform fixture with connection details

    Returns:
        agnt5.Client: Client instance connected to platform
    """
    from agnt5 import Client

    gateway_url = platform["gateway_url"]
    client = Client(gateway_url=gateway_url, timeout=60)

    logger.info(f"✅ Client created: {gateway_url} (timeout=60s)")

    return client


# ============================================================================
# Helper Functions
# ============================================================================


def wait_for_worker_registration(platform: Dict, timeout: int = 10) -> None:
    """
    Wait for worker to register with platform.

    Args:
        platform: Platform metadata dict
        timeout: Maximum seconds to wait

    Raises:
        TimeoutError: If worker doesn't register within timeout
    """
    start = time.time()
    gateway_url = platform["gateway_url"]

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{gateway_url}/v1/components", timeout=1)
            if response.status_code == 200:
                components = response.json()
                if len(components) > 0:
                    return
        except Exception:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"Worker did not register within {timeout}s")


def restart_worker(worker_process, platform) -> subprocess.Popen:
    """
    Restart worker process (for testing recovery scenarios).

    Args:
        worker_process: Current worker process
        platform: Platform metadata dict

    Returns:
        subprocess.Popen: New worker process
    """
    # Kill existing worker
    worker_process.terminate()
    worker_process.wait(timeout=5)

    # Start new worker
    coordinator_url = platform["coordinator_url"]
    worker_script = Path(__file__).parent / "test_worker_app.py"

    env = os.environ.copy()
    env["AGNT5_COORDINATOR_ENDPOINT"] = coordinator_url

    process = subprocess.Popen(
        ["uv", "run", "python", str(worker_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=worker_script.parent,
    )

    # Wait for registration
    wait_for_worker_registration(platform)

    return process
