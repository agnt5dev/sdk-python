"""
Integration Test Fixtures

Provides platform infrastructure for E2E testing across multiple modes:

1. Local - Use already-running dev server (fastest, recommended for development)
2. Embedded - Dev server in Docker with SQLite + embedded journal
3. Postgres - Community edition with PostgreSQL backend
4. Managed - Production mode with Redpanda + CockroachDB

Fixtures:
- runtime_mode: Parametrized fixture for testing across all modes
- platform: Mode-aware platform fixture
- worker_process: Start Python worker subprocess
- client: Create agnt5.Client instance for testing

Command-line options:
- --use-local-server: Use already-running local dev server (default: True)
- --runtime-mode: Specify runtime mode (embedded|postgres|managed|all) - only when not using local server
"""

import os
import subprocess
import time
import uuid
from typing import Dict, Generator

import pytest
import requests

# Testcontainers imports - only needed for Docker-based modes
try:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    DockerContainer = None
    wait_for_logs = None


# ==================== Pytest Configuration ====================


def pytest_addoption(parser):
    """Add command-line options for runtime mode selection."""
    parser.addoption(
        "--use-local-server",
        action="store_true",
        default=True,
        help="Use already-running local dev server instead of Docker containers (default: True)",
    )
    parser.addoption(
        "--use-docker",
        action="store_true",
        default=False,
        help="Use Docker containers instead of local dev server",
    )
    parser.addoption(
        "--use-subprocess",
        action="store_true",
        default=False,
        help="Start platform binary as subprocess (for CI, no Docker needed)",
    )
    parser.addoption(
        "--runtime-mode",
        action="store",
        default="embedded",
        choices=["embedded", "postgres", "managed", "all"],
        help="Runtime mode for Docker-based tests (embedded|postgres|managed|all)",
    )


def pytest_generate_tests(metafunc):
    """Generate test parametrization based on command-line options."""
    if "runtime_mode" in metafunc.fixturenames:
        use_docker = metafunc.config.getoption("--use-docker")
        use_subprocess = metafunc.config.getoption("--use-subprocess")

        if use_subprocess:
            # Subprocess mode: start platform binary directly (for CI)
            modes = ["subprocess"]
        elif use_docker:
            # Docker mode: parametrize across runtime modes
            mode = metafunc.config.getoption("--runtime-mode")
            if mode == "all":
                modes = ["embedded", "postgres", "managed"]
            else:
                modes = [mode]
        else:
            # Local server mode: single "local" mode
            modes = ["local"]

        metafunc.parametrize("runtime_mode", modes, scope="session", indirect=True)


# ==================== Docker Configuration ====================


@pytest.fixture(scope="session", autouse=True)
def configure_docker(request):
    """
    Configure Docker environment for testcontainers.

    Only runs when --use-docker is specified. On macOS, Docker Desktop uses
    ~/.docker/run/docker.sock instead of the default /var/run/docker.sock.
    This fixture automatically detects and configures the correct Docker socket path.
    """
    use_docker = request.config.getoption("--use-docker")
    if not use_docker:
        print("\n🏠 Using local dev server (Docker configuration skipped)")
        return

    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers package not installed. Install with: pip install testcontainers")

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

    Parametrized across runtime modes via pytest_generate_tests:
    - local: Use already-running local dev server (fastest, default)
    - subprocess: Start platform binary directly (for CI, no Docker)
    - embedded: SQLite + embedded journal in Docker container
    - postgres: PostgreSQL + embedded journal (community edition)
    - managed: Redpanda + CockroachDB (production-like, slowest)

    Default: local mode for fastest test execution.

    Examples:
        # Local development (requires running dev server)
        pytest tests/integration/

        # CI mode (starts binary subprocess)
        pytest tests/integration/ --use-subprocess

        # Docker mode
        pytest tests/integration/ --use-docker
        pytest tests/integration/ --use-docker --runtime-mode=postgres
    """
    return request.param


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


def setup_local_mode() -> Dict[str, any]:
    """
    Set up Local mode (use already-running dev server).

    This is the fastest mode - no container startup, just connect to localhost.
    Requires dev server to be running via: just platform standalone python

    Architecture:
    - Journal: Embedded (in-memory event log)
    - State: SQLite (local file at /tmp/agnt5/<service-name>/agnt5.db)
    - Process: PM2-managed standalone binary

    Ports (hardcoded in standalone server):
    - HTTP Gateway: 34181
    - gRPC Gateway: 34182
    - Worker Coordinator: 34186
    - OTLP: 4317
    """
    print("\n🏠 Setting up LOCAL mode (using already-running dev server)")

    gateway_url = "http://localhost:34181"
    coordinator_url = "http://localhost:34186"
    otlp_endpoint = "http://localhost:4317"

    # Wait for platform health
    _wait_for_platform_health(gateway_url, timeout=10)

    # Determine data directory from environment or default
    # The standalone server uses: /tmp/agnt5/<service-name>/agnt5.db
    data_dir = os.environ.get("AGNT5_DATA_DIR", "/tmp/agnt5")

    print(f"✅ Local mode ready")
    print(f"   Gateway HTTP: {gateway_url}")
    print(f"   Coordinator: {coordinator_url}")
    print(f"   OTLP Endpoint: {otlp_endpoint}")
    print(f"   Data directory: {data_dir}")

    return {
        "mode": "local",
        "gateway_url": gateway_url,
        "coordinator_url": coordinator_url,
        "otlp_endpoint": otlp_endpoint,
        "gateway_http_port": 34181,
        "gateway_grpc_port": 34182,
        "coordinator_port": 34186,
        "otlp_port": 4317,
        "db_url": f"{data_dir}/agnt5.db",
        "db_type": "sqlite",
        "journal_backend": "embedded",
        "orchestration_backend": "sqlite",
        "host_data_dir": data_dir,
        "containers": {},  # No containers in local mode
    }


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

    # Pull latest image to ensure we have the most recent version
    import docker

    image_name = "ghcr.io/agnt5dev/agnt5-dev-server-py:latest"
    # image_name = "agnt5/dev-server:latest"  # Use Docker Hub mirror for CI speed
    print(f"📥 Pulling latest image: {image_name}")
    try:
        docker_client = docker.from_env()
        docker_client.images.pull(image_name)
        print(f"✅ Image pulled successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not pull image: {e}")
        print(f"   Proceeding with local image if available")

    # Start dev-server container
    dev_server = DockerContainer(image_name)
    dev_server.with_exposed_ports(
        34181, 34182, 34186, 4317, 34180
    )  # HTTP, gRPC, Coordinator, OTLP, MCP

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
        },
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
    postgres_url = (
        f"postgresql://agnt5:agnt5@{postgres_host}:{postgres_port}/orchestration?sslmode=disable"
    )

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
        },
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
        },
    }


def setup_subprocess_mode(data_dir: str) -> Dict[str, any]:
    """
    Set up Subprocess mode (start platform binary directly).

    This is the recommended mode for CI:
    - No Docker required
    - Fast startup (~5s)
    - Clean isolation (processes die with tests)
    - Easy debugging (stdout/stderr captured)

    Architecture:
    - Platform: Standalone binary started as subprocess
    - Worker: Python subprocess from examples/
    - State: SQLite in temp directory
    """
    print("\n🚀 Setting up SUBPROCESS mode (direct binary execution)")

    # Find the standalone binary
    # Look in common locations relative to the test file
    test_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(test_dir, "../../../../"))

    binary_paths = [
        os.path.join(repo_root, "platform/bin/standalone"),
        os.path.join(repo_root, "platform/standalone"),
        # CI might build to different location
        os.path.join(os.environ.get("GITHUB_WORKSPACE", ""), "platform/bin/standalone"),
    ]

    binary_path = None
    for path in binary_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            binary_path = path
            break

    if not binary_path:
        raise Exception(
            f"Standalone binary not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in binary_paths)
            + "\n\nBuild it with: cd platform && go build -o bin/standalone ./cmd/standalone"
        )

    print(f"   Binary: {binary_path}")

    # Examples directory for the worker
    examples_dir = os.path.abspath(os.path.join(test_dir, "../../examples"))
    print(f"   Examples: {examples_dir}")
    print(f"   Data dir: {data_dir}")

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Start the standalone binary
    platform_process = subprocess.Popen(
        [
            binary_path,
            "--project-path", examples_dir,
            "--data-dir", data_dir,
            "--service-name", "agnt5-examples",
            "--disable-worker",  # We'll start worker separately for better control
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repo_root,
    )

    print(f"   Platform PID: {platform_process.pid}")

    # Give platform time to start
    time.sleep(3)

    # Check if process is still running
    if platform_process.poll() is not None:
        stdout, stderr = platform_process.communicate()
        raise Exception(
            f"Platform process died immediately:\n"
            f"STDOUT: {stdout.decode()}\n"
            f"STDERR: {stderr.decode()}"
        )

    gateway_url = "http://localhost:34181"
    coordinator_url = "http://localhost:34186"

    # Wait for platform health
    _wait_for_platform_health(gateway_url, timeout=30)

    print(f"✅ Subprocess mode ready")
    print(f"   Gateway: {gateway_url}")
    print(f"   Coordinator: {coordinator_url}")

    return {
        "mode": "subprocess",
        "gateway_url": gateway_url,
        "coordinator_url": coordinator_url,
        "gateway_http_port": 34181,
        "gateway_grpc_port": 34182,
        "coordinator_port": 34186,
        "otlp_port": 4317,
        "otlp_endpoint": "http://localhost:4317",
        "db_url": os.path.join(data_dir, "orchestration.db"),
        "db_type": "sqlite",
        "journal_backend": "embedded",
        "orchestration_backend": "sqlite",
        "host_data_dir": data_dir,
        "examples_dir": examples_dir,
        "processes": {
            "platform": platform_process,
        },
        "containers": {},  # No containers in subprocess mode
    }


def _wait_for_platform_health(gateway_url: str, timeout: int = 60, container=None):
    """Wait for platform to become healthy."""
    print(f"📡 Waiting for platform at {gateway_url}...")

    for i in range(timeout):
        try:
            response = requests.get(f"{gateway_url}/v1/health", timeout=2)
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
                        error_msg += "\n".join(logs[0].decode("utf-8").split("\n")[-50:])
                    except Exception as log_error:
                        error_msg += f"\nCould not retrieve container logs: {log_error}"

                raise Exception(error_msg)

            # Print progress every 5 seconds
            if i % 5 == 0 and i > 0:
                print(f"   Still waiting... ({i}/{timeout}s) - {type(e).__name__}")

            time.sleep(1)


@pytest.fixture(scope="session")
def platform(runtime_mode, persistent_data_dir) -> Generator[Dict[str, any], None, None]:
    """
    Start AGNT5 platform in the specified runtime mode.

    Uses session scope so the platform runs once for all tests.
    This significantly improves test performance by avoiding startup overhead.

    Supports four runtime modes:
    - local: Use already-running dev server (fastest, no containers)
    - embedded: SQLite + embedded journal (dev-server container)
    - postgres: PostgreSQL + embedded journal (dev-server + postgres containers)
    - managed: Redpanda + CockroachDB (dev-server + redpanda + cockroach containers)

    Note: Tests should use unique entity keys and run IDs to avoid conflicts,
    as the platform state persists across all tests in the session.
    """
    # Setup platform based on runtime mode
    if runtime_mode == "local":
        platform_config = setup_local_mode()
    elif runtime_mode == "subprocess":
        platform_config = setup_subprocess_mode(data_dir=persistent_data_dir)
    elif runtime_mode == "embedded":
        platform_config = setup_embedded_mode(data_dir=persistent_data_dir)
    elif runtime_mode == "postgres":
        platform_config = setup_postgres_mode()
    elif runtime_mode == "managed":
        platform_config = setup_managed_mode()
    else:
        raise ValueError(f"Unknown runtime mode: {runtime_mode}")

    yield platform_config

    # Cleanup processes (subprocess mode)
    processes = platform_config.get("processes", {})
    if processes:
        print(f"\n🧹 Stopping {runtime_mode} mode processes...")
        for name, process in processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"   Stopped {name} (PID: {process.pid})")
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print(f"   Killed {name} (PID: {process.pid})")
            except Exception as e:
                print(f"⚠️  Failed to stop {name}: {e}")

    # Cleanup containers (Docker modes)
    containers = platform_config.get("containers", {})
    if containers:
        print(f"\n🧹 Stopping {runtime_mode} mode containers...")
        for name, container in containers.items():
            try:
                container.stop()
            except Exception as e:
                print(f"⚠️  Failed to stop {name}: {e}")


def _wait_for_worker_registration(platform: Dict[str, any], max_wait: int = 30) -> bool:
    """
    Wait for worker to register with platform.

    Returns True if worker registered, raises Exception if timeout.
    """
    print(f"⏳ Waiting for worker to register...")
    start_time = time.time()

    # In local/subprocess mode, the worker uses different tenant/deployment IDs
    # Use the /v1/components endpoint to verify worker is ready instead
    if platform["mode"] in ("local", "subprocess"):
        endpoint = f"{platform['gateway_url']}/v1/components"
        params = {}
    else:
        endpoint = f"{platform['gateway_url']}/v1/workers"
        params = {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "deployment_id": "00000000-0000-0000-0000-000000000002",
        }

    while (time.time() - start_time) < max_wait:
        try:
            response = requests.get(endpoint, params=params, timeout=2)
            if response.status_code == 200:
                data = response.json()

                # For components endpoint, check if components exist
                if platform["mode"] in ("local", "subprocess"):
                    components = data.get("components", [])
                    if components and len(components) > 0:
                        print(f"✅ Worker registered: {len(components)} components found\n")

                        # Group components by type
                        by_type = {}
                        for comp in components:
                            comp_type = comp.get("component_type", "unknown")
                            if comp_type not in by_type:
                                by_type[comp_type] = []
                            by_type[comp_type].append(comp.get("component_name", "unknown"))

                        for comp_type, names in by_type.items():
                            print(f"   • {comp_type}: {len(names)} components")

                        return True
                else:
                    # For workers endpoint
                    workers = data.get("workers", data) if isinstance(data, dict) else data
                    if workers and len(workers) > 0:
                        print(f"✅ Worker registered: {len(workers)} worker(s) found\n")

                        for idx, worker_info in enumerate(workers, 1):
                            print(f"   Worker #{idx}:")
                            print(f"     • ID: {worker_info.get('worker_id', 'N/A')}")
                            print(f"     • Service: {worker_info.get('service_name', 'N/A')}")
                            print(f"     • Health: {worker_info.get('health_status', 'N/A')}")

                            components = worker_info.get("components", {})
                            if components:
                                print(f"     • Components:")
                                for comp_type, comp_list in components.items():
                                    if comp_list:
                                        print(f"       - {comp_type}: {', '.join(comp_list)}")
                            print()

                        return True
        except Exception as e:
            if (time.time() - start_time) % 5 < 1:
                print(f"   Worker check error: {type(e).__name__}: {e}")

        time.sleep(1)

    raise Exception(f"Worker not registered after {max_wait}s")


@pytest.fixture(scope="session")
def worker_process(platform) -> Generator[subprocess.Popen | None, None, None]:
    """
    Start Python worker process connected to platform.

    In local mode, the worker is already running (started by PM2 with the dev server).
    In subprocess/Docker modes, we start a separate worker subprocess.

    Uses session scope so the worker runs once for all tests, avoiding
    the overhead of starting/stopping the worker for each test.
    """
    # In local mode, worker is already running via PM2
    if platform["mode"] == "local":
        print("\n🏠 Local mode: Worker already running via PM2")
        _wait_for_worker_registration(platform)
        yield None  # No subprocess to manage
        return

    # Determine service path and script based on mode
    if platform["mode"] == "subprocess":
        # Subprocess mode: use examples directory
        service_path = platform.get("examples_dir") or os.path.join(
            os.path.dirname(__file__), "../../examples"
        )
        worker_script = "app.py"
        service_name = "agnt5-examples"
    else:
        # Docker modes: use test-bench directory
        service_path = os.path.join(os.path.dirname(__file__), "test-bench")
        worker_script = "app.py"
        service_name = "test-bench"

    env = {
        **os.environ,
        "AGNT5_COORDINATOR_ENDPOINT": f"http://localhost:{platform['coordinator_port']}",
        "AGNT5_SERVICE_NAME": service_name,
        "AGNT5_TENANT_ID": "00000000-0000-0000-0000-000000000001",
        "AGNT5_DEPLOYMENT_ID": "00000000-0000-0000-0000-000000000002",
        "OTEL_EXPORTER_OTLP_ENDPOINT": platform.get("otlp_endpoint", "http://localhost:4317"),
    }

    print(f"🔌 Worker connecting to coordinator: {env['AGNT5_COORDINATOR_ENDPOINT']}")
    print(f"📁 Worker service path: {service_path}")
    print(f"📄 Worker script: {worker_script}")

    worker = subprocess.Popen(
        ["uv", "run", "python", worker_script],
        cwd=service_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for worker to register
    max_wait = 30
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

        try:
            response = requests.get(
                f"{platform['gateway_url']}/v1/workers",
                params={
                    "tenant_id": "00000000-0000-0000-0000-000000000001",
                    "deployment_id": "00000000-0000-0000-0000-000000000002",
                },
                timeout=2,
            )
            print(f"   Worker check: status={response.status_code}")
            if response.status_code == 200:
                workers = response.json()
                if workers and len(workers) > 0:
                    print(f"✅ Worker registered: {len(workers)} worker(s) found\n")

                    # Print worker details (best effort - API format may vary)
                    try:
                        for idx, worker_info in enumerate(workers, 1):
                            print(f"   Worker #{idx}:")
                            if isinstance(worker_info, dict):
                                print(f"     • ID: {worker_info.get('worker_id', 'N/A')}")
                                print(f"     • Service: {worker_info.get('service_name', 'N/A')}")
                                print(f"     • Health: {worker_info.get('health_status', 'N/A')}")

                                components = worker_info.get("components", {})
                                if components:
                                    print(f"     • Components:")
                                    for comp_type, comp_list in components.items():
                                        if comp_list:
                                            print(f"       - {comp_type}: {', '.join(comp_list)}")
                            else:
                                print(f"     • {worker_info}")
                            print()
                    except Exception as e:
                        print(f"   (Could not print worker details: {e})")

                    worker_registered = True
                    break
                else:
                    print(f"   Empty workers list")
            else:
                print(f"   Worker check failed: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"   Worker check error: {type(e).__name__}: {e}")

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
        stderr_lines = stderr.decode("utf-8", errors="ignore").split("\n")
        if len(stderr_lines) > 100:
            print(f"\n📋 Worker logs (last 100 lines):")
            print("\n".join(stderr_lines[-100:]))
        else:
            print(f"\n📋 Worker logs (all):")
            print("\n".join(stderr_lines))
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
    service_path = os.path.join(os.path.dirname(__file__), "test-bench")

    env = {
        **os.environ,
        "AGNT5_COORDINATOR_ENDPOINT": f"http://localhost:{platform['coordinator_port']}",
        "AGNT5_SERVICE_NAME": "test-bench",
        "AGNT5_TENANT_ID": "00000000-0000-0000-0000-000000000001",  # Fixed UUID for test tenant
        "AGNT5_DEPLOYMENT_ID": "00000000-0000-0000-0000-000000000002",  # Fixed UUID for test deployment
        "OTEL_EXPORTER_OTLP_ENDPOINT": platform["otlp_endpoint"],
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
