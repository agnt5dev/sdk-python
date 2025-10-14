"""
Integration Test Fixtures

Provides Testcontainers-based platform infrastructure for E2E testing
across three runtime modes:

1. Embedded - Dev server with SQLite + embedded journal (fastest)
2. Postgres - Community edition with PostgreSQL backend
3. Managed - Production mode with Redpanda + CockroachDB

Fixtures:
- runtime_mode: Parametrized fixture for testing across all modes
- platform: Mode-aware platform fixture
- worker_process: Start Python worker subprocess
- client: Create agnt5.Client instance for testing
"""

import os
import subprocess
import time
from typing import Dict, Generator

import pytest
import requests
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


# ==================== Docker Configuration ====================


@pytest.fixture(scope="session", autouse=True)
def configure_docker():
    """
    Configure Docker environment for testcontainers.

    On macOS, Docker Desktop uses ~/.docker/run/docker.sock instead of
    the default /var/run/docker.sock. This fixture automatically detects
    and configures the correct Docker socket path.
    """
    # Check if DOCKER_HOST is already set
    if "DOCKER_HOST" in os.environ:
        print(f"\n🐳 Using existing DOCKER_HOST: {os.environ['DOCKER_HOST']}")
        return

    # Detect Docker socket location
    docker_socket_paths = [
        "/var/run/docker.sock",  # Default Linux/Docker daemon
        os.path.expanduser("~/.docker/run/docker.sock"),  # Docker Desktop for Mac
        os.path.expanduser("~/.colima/default/docker.sock"),  # Colima
    ]

    for socket_path in docker_socket_paths:
        if os.path.exists(socket_path):
            docker_host = f"unix://{socket_path}"
            os.environ["DOCKER_HOST"] = docker_host
            print(f"\n🐳 Configured DOCKER_HOST: {docker_host}")
            return

    # If no socket found, let testcontainers use its default detection
    print("\n🐳 No Docker socket detected, using testcontainers default detection")


# ==================== Mode Configuration ====================


@pytest.fixture(scope="session")
def runtime_mode(request) -> str:
    """
    Runtime mode for integration tests.

    Currently: embedded mode only (SQLite + embedded journal)

    TODO: Add parametrization for postgres and managed modes:
        params=["embedded", "postgres", "managed"]
    """
    return "embedded"


@pytest.fixture(scope="session")
def persistent_data_dir(tmp_path_factory):
    """
    Create persistent directory for SQLite databases.

    This directory persists for the entire test session, allowing:
    - Database state to survive worker restarts
    - Direct database inspection with MCP tools or sqlite3 CLI
    - State verification across multiple test operations

    The directory is automatically cleaned up after the test session.
    """
    data_dir = tmp_path_factory.mktemp("agnt5_data")
    print(f"\n📁 Created persistent data directory: {data_dir}")
    return str(data_dir)


# ==================== Mode-Specific Setup Functions ====================


def setup_embedded_mode(data_dir: str = None) -> Dict[str, any]:
    """
    Set up Embedded mode (dev-server with SQLite + embedded journal).

    This is the fastest mode - single container, no external dependencies.

    Architecture:
    - Journal: Embedded (in-memory event log)
    - State: SQLite (local file)
    - Containers: 1 (dev-server only)

    Args:
        data_dir: Optional host directory to mount as /data in container.
                  If provided, SQLite databases will be persisted to this location
                  and accessible from the host for inspection.
    """
    print("\n🔧 Setting up EMBEDDED mode (SQLite + embedded journal)")

    # Start dev-server container
    dev_server = DockerContainer("agnt5/dev-server:latest")
    dev_server.with_exposed_ports(34181, 34182, 34186, 4317, 34180)  # HTTP, gRPC, Coordinator, OTLP, MCP

    # Configure for embedded mode (SQLite + embedded journal)
    dev_server.with_env("AGNT5_DATA_DIR", "/data")
    dev_server.with_env("AGNT5_JOURNAL_BACKEND", "embedded")
    dev_server.with_env("AGNT5_ORCHESTRATION_BACKEND", "sqlite")
    dev_server.with_env("AGNT5_DISABLE_WORKER", "true")  # Disable worker for testing
    dev_server.with_env("AGNT5_VERBOSE", "true")  # Enable verbose logging for troubleshooting

    # Mount host directory to /data for persistent SQLite databases
    if data_dir:
        # Ensure directory exists
        os.makedirs(data_dir, exist_ok=True)
        # Mount with read-write access
        dev_server.with_volume_mapping(data_dir, "/data", "rw")
        print(f"   📂 Mounting host directory: {data_dir} → /data")

    dev_server.start()

    # Give services time to fully initialize
    time.sleep(3)

    # Get container connection details
    gateway_host = dev_server.get_container_host_ip()
    gateway_http_port = dev_server.get_exposed_port(34181)
    gateway_grpc_port = dev_server.get_exposed_port(34182)
    coordinator_port = dev_server.get_exposed_port(34186)
    otlp_port = dev_server.get_exposed_port(4317)
    mcp_port = dev_server.get_exposed_port(34180)

    gateway_url = f"http://{gateway_host}:{gateway_http_port}"
    otlp_endpoint = f"http://{gateway_host}:{otlp_port}"
    mcp_endpoint = f"http://{gateway_host}:{mcp_port}"

    print(f"✅ Dev-server container started")
    print(f"   Gateway HTTP: {gateway_url}")
    print(f"   Gateway gRPC: {gateway_host}:{gateway_grpc_port}")
    print(f"   Coordinator: {gateway_host}:{coordinator_port}")
    print(f"   OTLP Endpoint: {otlp_endpoint}")
    print(f"   MCP Endpoint: {mcp_endpoint}")

    # Wait for platform health with container reference for logging
    _wait_for_platform_health(gateway_url, container=dev_server)

    # SQLite database paths
    container_db_path = "/data/orchestration.db"
    container_observability_path = "/data/observability.db"

    print(f"✅ Embedded mode ready")
    if data_dir:
        host_db_path = os.path.join(data_dir, "orchestration.db")
        host_observability_path = os.path.join(data_dir, "observability.db")
        print(f"   Orchestration DB (host): {host_db_path}")
        print(f"   Observability DB (host): {host_observability_path}")
        print(f"   💡 Access with: sqlite3 {host_db_path}")
    else:
        print(f"   Orchestration DB: {container_db_path} (inside container only)")
        print(f"   Observability DB: {container_observability_path} (inside container only)")

    return {
        "mode": "embedded",
        "gateway_url": gateway_url,
        "coordinator_url": f"http://{gateway_host}:{coordinator_port}",
        "otlp_endpoint": otlp_endpoint,
        "mcp_endpoint": mcp_endpoint,
        "gateway_http_port": int(gateway_http_port),
        "gateway_grpc_port": int(gateway_grpc_port),
        "coordinator_port": int(coordinator_port),
        "otlp_port": int(otlp_port),
        "mcp_port": int(mcp_port),
        "db_url": container_db_path,
        "observability_db_url": container_observability_path,
        "db_type": "sqlite",
        "journal_backend": "embedded",
        "orchestration_backend": "sqlite",
        "host_data_dir": data_dir,  # Host path for database inspection
        "host_db_path": os.path.join(data_dir, "orchestration.db") if data_dir else None,
        "host_observability_path": os.path.join(data_dir, "observability.db") if data_dir else None,
        "containers": {
            "dev-server": dev_server,
        }
    }


def setup_postgres_mode() -> Dict[str, any]:
    """
    Set up Postgres mode (community edition with PostgreSQL backend).

    Architecture:
    - Journal: Embedded (or could use PostgreSQL)
    - State: PostgreSQL
    - Containers: 2 (dev-server + PostgreSQL)
    """
    print("\n🔧 Setting up POSTGRES mode (PostgreSQL backend)")

    # Start PostgreSQL container
    postgres = DockerContainer("postgres:16-alpine")
    postgres.with_exposed_ports(5432)
    postgres.with_env("POSTGRES_USER", "agnt5")
    postgres.with_env("POSTGRES_PASSWORD", "agnt5")
    postgres.with_env("POSTGRES_DB", "orchestration")
    postgres.start()

    # Wait for PostgreSQL to be ready
    wait_for_logs(postgres, "database system is ready to accept connections", timeout=30)
    time.sleep(2)

    postgres_host = postgres.get_container_host_ip()
    postgres_port = postgres.get_exposed_port(5432)
    postgres_url = f"postgresql://agnt5:agnt5@{postgres_host}:{postgres_port}/orchestration?sslmode=disable"

    print(f"✅ PostgreSQL started: {postgres_url}")

    # Start dev-server container configured for PostgreSQL
    dev_server = DockerContainer("agnt5/dev-server:latest")
    dev_server.with_exposed_ports(34181, 34182, 34186)

    # Configure for postgres mode
    dev_server.with_env("AGNT5_JOURNAL_BACKEND", "embedded")
    dev_server.with_env("AGNT5_ORCHESTRATION_BACKEND", "postgres")
    dev_server.with_env("AGNT5_ORCHESTRATION_DB_URL", postgres_url)

    dev_server.start()

    # Get container connection details
    gateway_host = dev_server.get_container_host_ip()
    gateway_http_port = dev_server.get_exposed_port(34181)
    gateway_grpc_port = dev_server.get_exposed_port(34182)
    coordinator_port = dev_server.get_exposed_port(34186)

    gateway_url = f"http://{gateway_host}:{gateway_http_port}"

    print(f"✅ Dev-server container started")
    print(f"   Gateway HTTP: {gateway_url}")

    _wait_for_platform_health(gateway_url)

    print(f"✅ Postgres mode ready")

    return {
        "mode": "postgres",
        "gateway_url": gateway_url,
        "coordinator_url": f"http://{gateway_host}:{coordinator_port}",
        "gateway_http_port": int(gateway_http_port),
        "gateway_grpc_port": int(gateway_grpc_port),
        "coordinator_port": int(coordinator_port),
        "db_url": postgres_url,
        "db_type": "postgres",
        "journal_backend": "embedded",
        "orchestration_backend": "postgres",
        "containers": {
            "postgres": postgres,
            "dev-server": dev_server,
        }
    }


def setup_managed_mode() -> Dict[str, any]:
    """
    Set up Managed mode (Redpanda + CockroachDB production setup).

    Architecture:
    - Journal: Redpanda (distributed event log)
    - State: CockroachDB (distributed SQL)
    - Containers: 3 (dev-server + Redpanda + CockroachDB)
    """
    print("\n🔧 Setting up MANAGED mode (Redpanda + CockroachDB)")

    # Start CockroachDB
    cockroach = DockerContainer("cockroachdb/cockroach:latest-v24.3")
    cockroach.with_exposed_ports(26257, 8080)
    cockroach.with_command("start-single-node --insecure --store=type=mem,size=0.25")
    cockroach.start()

    wait_for_logs(cockroach, "nodeID", timeout=30)
    time.sleep(2)

    cockroach_host = cockroach.get_container_host_ip()
    cockroach_port = cockroach.get_exposed_port(26257)
    cockroach_url = f"postgresql://root@{cockroach_host}:{cockroach_port}/defaultdb?sslmode=disable"

    print(f"✅ CockroachDB started: {cockroach_url}")

    # Start Redpanda
    redpanda = DockerContainer("docker.redpanda.com/vectorized/redpanda:v24.3.1")
    redpanda.with_exposed_ports(9092, 9644)
    redpanda.with_command(
        "redpanda start --smp 1 --memory 1G --reserve-memory 0M "
        "--overprovisioned --node-id 0 "
        "--kafka-addr PLAINTEXT://0.0.0.0:29092,OUTSIDE://0.0.0.0:9092 "
        "--advertise-kafka-addr PLAINTEXT://redpanda:29092,OUTSIDE://localhost:9092"
    )
    redpanda.start()

    wait_for_logs(redpanda, "Successfully started Redpanda", timeout=30)
    time.sleep(2)

    redpanda_host = redpanda.get_container_host_ip()
    redpanda_port = redpanda.get_exposed_port(9092)
    redpanda_broker = f"{redpanda_host}:{redpanda_port}"

    print(f"✅ Redpanda started: {redpanda_broker}")

    # Start dev-server container configured for managed mode
    dev_server = DockerContainer("agnt5/dev-server:latest")
    dev_server.with_exposed_ports(34181, 34182, 34186)

    # Configure for managed mode (Redpanda + CockroachDB)
    dev_server.with_env("AGNT5_JOURNAL_BACKEND", "redpanda")
    dev_server.with_env("AGNT5_ORCHESTRATION_BACKEND", "cockroach")
    dev_server.with_env("AGNT5_ORCHESTRATION_DB_URL", cockroach_url)
    dev_server.with_env("AGNT5_REDPANDA_BROKERS", redpanda_broker)

    dev_server.start()

    # Get container connection details
    gateway_host = dev_server.get_container_host_ip()
    gateway_http_port = dev_server.get_exposed_port(34181)
    gateway_grpc_port = dev_server.get_exposed_port(34182)
    coordinator_port = dev_server.get_exposed_port(34186)

    gateway_url = f"http://{gateway_host}:{gateway_http_port}"

    print(f"✅ Dev-server container started")
    print(f"   Gateway HTTP: {gateway_url}")

    _wait_for_platform_health(gateway_url)

    print(f"✅ Managed mode ready")

    return {
        "mode": "managed",
        "gateway_url": gateway_url,
        "coordinator_url": f"http://{gateway_host}:{coordinator_port}",
        "gateway_http_port": int(gateway_http_port),
        "gateway_grpc_port": int(gateway_grpc_port),
        "coordinator_port": int(coordinator_port),
        "db_url": cockroach_url,
        "db_type": "cockroach",
        "journal_backend": "redpanda",
        "orchestration_backend": "cockroach",
        "redpanda_broker": redpanda_broker,
        "containers": {
            "cockroach": cockroach,
            "redpanda": redpanda,
            "dev-server": dev_server,
        }
    }


def _wait_for_platform_health(gateway_url: str, timeout: int = 60, container=None):
    """Wait for platform to become healthy."""
    print(f"📡 Waiting for platform at {gateway_url}...")

    for i in range(timeout):
        try:
            response = requests.get(f"{gateway_url}/api/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Platform is healthy")
                return
        except requests.exceptions.RequestException as e:
            if i == timeout - 1:
                error_msg = f"Platform failed to become healthy after {timeout}s\n"
                error_msg += f"Last error: {str(e)}\n"

                # Get container logs for debugging
                if container:
                    try:
                        logs = container.get_logs()
                        error_msg += f"\nContainer logs (last 50 lines):\n"
                        error_msg += "\n".join(logs[0].decode('utf-8').split('\n')[-50:])
                    except Exception as log_error:
                        error_msg += f"\nCould not retrieve container logs: {log_error}"

                raise Exception(error_msg)

            # Print progress every 5 seconds
            if i % 5 == 0 and i > 0:
                print(f"   Still waiting... ({i}/{timeout}s) - {type(e).__name__}")

            time.sleep(1)


@pytest.fixture(scope="function")
def platform(runtime_mode, persistent_data_dir) -> Generator[Dict[str, any], None, None]:
    """
    Start AGNT5 platform in embedded mode.

    Uses function scope so each test gets a clean platform state.
    However, the SQLite database is mounted to a persistent directory
    (persistent_data_dir), allowing the database to survive worker restarts
    and be inspected from the host.

    Currently supports:
    - embedded: SQLite + embedded journal (dev-server container)

    TODO: Add postgres and managed modes
    """
    # Setup embedded mode with persistent data directory
    platform_config = setup_embedded_mode(data_dir=persistent_data_dir)

    yield platform_config

    # Cleanup containers
    containers = platform_config.get("containers", {})
    if containers:
        print(f"\n🧹 Stopping {runtime_mode} mode containers...")
        for name, container in containers.items():
            try:
                container.stop()
            except Exception as e:
                print(f"⚠️  Failed to stop {name}: {e}")


@pytest.fixture
def worker_process(platform) -> Generator[subprocess.Popen, None, None]:
    """
    Start Python worker process connected to platform.

    Worker runs the test service blueprint and connects to Worker Coordinator.
    """
    service_path = os.path.join(
        os.path.dirname(__file__),
        "blueprints",
        "test-service"
    )

    env = {
        **os.environ,
        "AGNT5_COORDINATOR_ENDPOINT": f"http://localhost:{platform['coordinator_port']}",
        "AGNT5_SERVICE_NAME": "test-service",
        "AGNT5_TENANT_ID": "test-tenant-001",
        "AGNT5_DEPLOYMENT_ID": "test-deployment-001",
        "OTEL_EXPORTER_OTLP_ENDPOINT": platform['otlp_endpoint'],
    }

    print(f"🔌 Worker connecting to coordinator: {env['AGNT5_COORDINATOR_ENDPOINT']}")
    print(f"📊 Worker telemetry exporting to: {env['OTEL_EXPORTER_OTLP_ENDPOINT']}")

    worker = subprocess.Popen(
        ["uv", "run", "python", "app.py"],
        cwd=service_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for worker to register and become available
    print(f"⏳ Waiting for worker to register...")
    max_wait = 30  # seconds
    start_time = time.time()
    worker_registered = False

    while (time.time() - start_time) < max_wait:
        # Check if worker is still running
        if worker.poll() is not None:
            stdout, stderr = worker.communicate()
            raise Exception(
                f"Worker failed to start:\n"
                f"STDOUT: {stdout.decode()}\n"
                f"STDERR: {stderr.decode()}"
            )

        # Check if worker is registered
        try:
            # Query /v1/workers with default UUIDs that coordinator uses
            # Coordinator converts non-UUID values to these defaults (see grpc_handlers.go:499-511)
            response = requests.get(
                f"{platform['gateway_url']}/v1/workers",
                params={
                    "tenant_id": "00000000-0000-0000-0000-000000000001",
                    "deployment_id": "00000000-0000-0000-0000-000000000002"
                },
                timeout=2
            )
            print(f"   Worker check: status={response.status_code}")
            if response.status_code == 200:
                workers = response.json()
                if workers and len(workers) > 0:
                    print(f"✅ Worker registered: {len(workers)} worker(s) found\n")

                    # Print detailed worker information
                    for idx, worker_info in enumerate(workers, 1):
                        print(f"   Worker #{idx}:")
                        print(f"     • ID: {worker_info.get('worker_id', 'N/A')}")
                        print(f"     • Service: {worker_info.get('service_name', 'N/A')}")
                        print(f"     • Health: {worker_info.get('health_status', 'N/A')}")

                        # Print components grouped by type
                        components = worker_info.get('components', {})
                        if components:
                            print(f"     • Components:")
                            for comp_type, comp_list in components.items():
                                if comp_list:
                                    print(f"       - {comp_type}: {', '.join(comp_list)}")
                        print()

                    worker_registered = True
                    break
                else:
                    print(f"   Empty workers list")
            else:
                print(f"   Worker check failed: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"   Worker check error: {type(e).__name__}: {e}")
            pass  # Keep waiting

        time.sleep(1)

    if not worker_registered:
        print(f"⚠️  Worker process running but not registered after {max_wait}s")

    print(f"✅ Worker started (PID: {worker.pid})")

    yield worker

    # Cleanup
    print(f"\n🧹 Stopping worker (PID: {worker.pid})...")
    worker.terminate()
    try:
        stdout, stderr = worker.communicate(timeout=5)
        # Print worker logs for debugging (last 100 lines)
        stderr_lines = stderr.decode('utf-8', errors='ignore').split('\n')
        if len(stderr_lines) > 100:
            print(f"\n📋 Worker logs (last 100 lines):")
            print('\n'.join(stderr_lines[-100:]))
        else:
            print(f"\n📋 Worker logs (all):")
            print('\n'.join(stderr_lines))
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()


@pytest.fixture
def client(platform):
    """
    Create agnt5.Client instance for testing.

    This is the primary test interface - all tests use this to interact
    with the platform, just like end users do.
    """
    from agnt5 import Client

    client = Client(platform["gateway_url"])
    print(f"✅ Client created: {platform['gateway_url']}")

    return client


# Helper utilities for tests

def wait_for_worker_registration(platform: Dict[str, any], timeout: int = 10) -> bool:
    """
    Wait for worker to register with platform.

    Returns True if worker registered, False if timeout.
    """
    # TODO: Add health check endpoint to verify worker registration
    # For now, use simple delay
    time.sleep(2)
    return True


def restart_worker(worker_process: subprocess.Popen, platform: Dict[str, any]) -> subprocess.Popen:
    """
    Restart worker process (simulate crash recovery).

    Returns new worker process.
    """
    # Terminate old worker
    worker_process.terminate()
    try:
        worker_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker_process.kill()
        worker_process.wait()

    # Start new worker
    service_path = os.path.join(
        os.path.dirname(__file__),
        "blueprints",
        "test-service"
    )

    env = {
        **os.environ,
        "AGNT5_COORDINATOR_ENDPOINT": f"http://localhost:{platform['coordinator_port']}",
        "AGNT5_SERVICE_NAME": "test-service",
        "AGNT5_TENANT_ID": "test-tenant-001",
        "AGNT5_DEPLOYMENT_ID": "test-deployment-001",
        "OTEL_EXPORTER_OTLP_ENDPOINT": platform['otlp_endpoint'],
    }

    new_worker = subprocess.Popen(
        ["uv", "run", "python", "app.py"],
        cwd=service_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for registration (increased from 3s to 8s for reliable connection)
    time.sleep(8)

    return new_worker
