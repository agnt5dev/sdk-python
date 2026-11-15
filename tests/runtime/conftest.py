"""Fixtures that run tests against a real agnt5-runtime.

The runtime is the single-process `agnt5-oss-runtime` binary, started once per
test session on a temporary SQLite file and free loopback ports. Workers run as
separate processes so a test can kill one and start another, the way a crash
happens in production.

Point `AGNT5_RUNTIME_BIN` at the binary. It ships in the
`ghcr.io/agnt5dev/agnt5-runtime` image, pinned by commit; copy it out with
`docker create` and `docker cp` from `/usr/local/bin/agnt5-oss-runtime`. Without it these tests are skipped,
unless `AGNT5_REQUIRE_RUNTIME=1` is set, which turns the skip into a
failure so CI can't pass by quietly skipping them.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import pytest

APPS_DIR = Path(__file__).parent / "apps"
PROJECT_ID = "sdk-tests"
DEPLOYMENT_ID = "local"
READY_TIMEOUT_S = 30.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _runtime_binary() -> str | None:
    configured = os.environ.get("AGNT5_RUNTIME_BIN")
    if configured:
        return configured
    return shutil.which("agnt5-oss-runtime")


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    # A macOS-only ARCHFLAGS in a developer's shell makes native builds and
    # some tools target the wrong platform; it has no meaning here.
    env.pop("ARCHFLAGS", None)
    # Never let a test send telemetry anywhere real.
    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:9"
    return env


@dataclass
class Agnt5Runtime:
    """A running agnt5-runtime and the settings a worker or client needs."""

    http_url: str
    grpc_url: str
    log_path: Path
    process: subprocess.Popen[bytes]

    def worker_env(self, **extra: str) -> dict[str, str]:
        env = _base_env()
        env.update(
            {
                "AGNT5_WORKER_MODE": "pull",
                "AGNT5_COORDINATOR_ENDPOINT": self.grpc_url,
                "AGNT5_ENGINE_URL": self.grpc_url,
                "AGNT5_PROJECT_ID": PROJECT_ID,
                "AGNT5_DEPLOYMENT_ID": DEPLOYMENT_ID,
            }
        )
        # Workers must not inherit a fingerprint from the developer's shell:
        # the point of these tests is what a self-hosted worker does without one.
        env.pop("AGNT5_ACTIVATION_ARTIFACT_SHA256", None)
        env.update(extra)
        return env

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "<no runtime log>"


@pytest.fixture(scope="session")
def agnt5_runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Agnt5Runtime]:
    binary = _runtime_binary()
    if not binary:
        message = "set AGNT5_RUNTIME_BIN to an agnt5-oss-runtime binary to run these tests"
        if os.environ.get("AGNT5_REQUIRE_RUNTIME") == "1":
            pytest.fail(message)
        pytest.skip(message)

    workdir = tmp_path_factory.mktemp("agnt5-runtime")
    http_port, grpc_port = _free_port(), _free_port()
    env = _base_env()
    env.update(
        {
            "AGNT5_DATABASE_BACKEND": "sqlite",
            "AGNT5_SQLITE_PATH": str(workdir / "runtime.db"),
            "AGNT5_HTTP_ADDR": f"127.0.0.1:{http_port}",
            "AGNT5_GRPC_ADDR": f"127.0.0.1:{grpc_port}",
            "AGNT5_RUNTIME_NAMESPACE": f"sdk-tests-{uuid.uuid4().hex[:8]}",
            "AGNT5_PROJECT_ID": PROJECT_ID,
            "AGNT5_DEPLOYMENT_ID": DEPLOYMENT_ID,
        }
    )
    for name in ("AGNT5_API_KEY", "AGNT5_JWT_SECRET"):
        env.pop(name, None)

    log_path = workdir / "runtime.log"
    log = log_path.open("wb")
    process = subprocess.Popen([binary], env=env, stdout=log, stderr=subprocess.STDOUT)
    runtime = Agnt5Runtime(
        http_url=f"http://127.0.0.1:{http_port}",
        grpc_url=f"http://127.0.0.1:{grpc_port}",
        log_path=log_path,
        process=process,
    )
    try:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while True:
            if process.poll() is not None:
                pytest.fail(f"agnt5-runtime exited early:\n{runtime.log_tail()}")
            try:
                if httpx.get(f"{runtime.http_url}/readyz", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"agnt5-runtime not ready after {READY_TIMEOUT_S}s:\n{runtime.log_tail()}")
            time.sleep(0.1)
        yield runtime
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


@dataclass
class WorkerProcess:
    """One worker process serving the test app."""

    process: subprocess.Popen[bytes]
    log_path: Path
    health_dir: Path

    def kill(self) -> None:
        """Kill without any chance to clean up, as a crash would."""
        self.process.kill()
        self.process.wait()

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()

    def log_tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return "<no worker log>"


@dataclass
class WorkerFactory:
    runtime: Agnt5Runtime
    workdir: Path
    effects_path: Path
    started: list[WorkerProcess] = field(default_factory=list)

    def start(self, app: str = "durable_app.py") -> WorkerProcess:
        index = len(self.started)
        health_dir = self.workdir / f"health-{index}"
        log_path = self.workdir / f"worker-{index}.log"
        env = self.runtime.worker_env(
            AGNT5_HEALTH_DIR=str(health_dir),
            SDK_TEST_EFFECTS_FILE=str(self.effects_path),
        )
        log = log_path.open("wb")
        process = subprocess.Popen(
            [sys.executable, str(APPS_DIR / app)], env=env, stdout=log, stderr=subprocess.STDOUT
        )
        worker = WorkerProcess(process=process, log_path=log_path, health_dir=health_dir)
        self.started.append(worker)
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not any(health_dir.glob("worker_*.txt")):
            if process.poll() is not None:
                pytest.fail(f"worker exited early:\n{worker.log_tail()}")
            if time.monotonic() > deadline:
                pytest.fail(f"worker not ready after {READY_TIMEOUT_S}s:\n{worker.log_tail()}")
            time.sleep(0.1)
        return worker

    def effects(self) -> list[str]:
        """External side effects the app recorded, one line per effect."""
        if not self.effects_path.exists():
            return []
        return self.effects_path.read_text().splitlines()


@pytest.fixture
def workers(agnt5_runtime: Agnt5Runtime, tmp_path: Path) -> Iterator[WorkerFactory]:
    factory = WorkerFactory(
        runtime=agnt5_runtime, workdir=tmp_path, effects_path=tmp_path / "effects.txt"
    )
    try:
        yield factory
    finally:
        for worker in factory.started:
            worker.stop()
