import sys
from types import ModuleType

import pytest

import agnt5
from agnt5.worker import _core as worker_core
from agnt5.worker._core import Worker


class FakePyWorkerConfig:
    def __init__(
        self,
        service_name: str,
        service_version: str,
        service_type: str,
        max_concurrency: int | None = None,
    ) -> None:
        self.service_name = service_name
        self.service_version = service_version
        self.service_type = service_type
        self.max_concurrency = max_concurrency


class FakePyWorker:
    def __init__(self, config: FakePyWorkerConfig) -> None:
        self.config = config


class FakePyActivationClient:
    def __init__(self, endpoint: str | None = None, worker: FakePyWorker | None = None) -> None:
        self.endpoint = endpoint
        self.worker = worker


class FakeEntityStateManager:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id


class FakeComponentInfo:
    pass


class FakeTriggerSpec:
    pass


@pytest.fixture
def fake_native_core(monkeypatch):
    fake_module = ModuleType("agnt5._core")
    fake_module.PyWorkerConfig = FakePyWorkerConfig
    fake_module.PyWorker = FakePyWorker
    fake_module.PyActivationClient = FakePyActivationClient
    fake_module.PyComponentInfo = FakeComponentInfo
    fake_module.PyTriggerSpec = FakeTriggerSpec
    fake_module.EntityStateManager = FakeEntityStateManager
    fake_module.log_from_python = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "agnt5._core", fake_module)
    monkeypatch.setattr(agnt5, "_core", fake_module)
    monkeypatch.setattr(worker_core, "init_sdk_telemetry", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker_core._sentry, "initialize_sentry", lambda **_kwargs: False)


@pytest.fixture(autouse=True)
def clear_worker_env(monkeypatch):
    for key in (
        "AGNT5_COORDINATOR_ENDPOINT",
        "AGNT5_PROJECT_ID",
        "AGNT5_DEPLOYMENT_ID",
        "AGNT5_WORKER_MODE",
        "AGNT5_MIN_SLOTS",
        "AGNT5_MAX_SLOTS",
        "AGNT5_CLAIM_TIMEOUT_MS",
        "AGNT5_ACTIVATION_ARTIFACT_SHA256",
    ):
        monkeypatch.delenv(key, raising=False)


def test_worker_pull_options_configure_sdk_core_environment(fake_native_core):
    worker = Worker(
        service_name="py-worker",
        coordinator_endpoint="http://localhost:34186",
        project_id="project-py",
        deployment_id="deployment-py",
        worker_mode="pull",
        min_slots=2,
        max_slots=10,
        claim_timeout_ms=120_000,
        max_concurrency=10,
    )

    assert worker.metadata["project_id"] == "project-py"
    assert worker.metadata["deployment_id"] == "deployment-py"
    assert worker._rust_config.max_concurrency == 10
    assert worker._activation_client._transport._native_client.worker is worker._rust_worker
    assert worker_core.os.environ["AGNT5_COORDINATOR_ENDPOINT"] == "http://localhost:34186"
    assert worker_core.os.environ["AGNT5_PROJECT_ID"] == "project-py"
    assert worker_core.os.environ["AGNT5_DEPLOYMENT_ID"] == "deployment-py"
    assert worker_core.os.environ["AGNT5_WORKER_MODE"] == "pull"
    assert worker_core.os.environ["AGNT5_MIN_SLOTS"] == "2"
    assert worker_core.os.environ["AGNT5_MAX_SLOTS"] == "10"
    assert worker_core.os.environ["AGNT5_CLAIM_TIMEOUT_MS"] == "120000"


def test_worker_configures_durable_activation_artifact_identity(fake_native_core):
    digest = "61" * 32
    worker = Worker(
        service_name="py-worker",
        activation_artifact_sha256=digest,
    )

    assert worker.metadata["activation_artifact_sha256"] == digest
    assert worker_core.os.environ["AGNT5_ACTIVATION_ARTIFACT_SHA256"] == digest


def test_parked_polling_implies_pull_mode(fake_native_core):
    Worker(service_name="py-worker", parked_polling=True)

    assert worker_core.os.environ["AGNT5_WORKER_MODE"] == "pull"


def test_worker_mode_pull_rejects_long_poll_disable(fake_native_core):
    with pytest.raises(ValueError, match="pull workers always use long polling"):
        Worker(service_name="py-worker", worker_mode="pull", parked_polling=False)


@pytest.mark.parametrize("kwargs", [{"min_slots": 0}, {"max_slots": -1}, {"claim_timeout_ms": True}])
def test_pull_slot_options_must_be_positive_integers(fake_native_core, kwargs):
    with pytest.raises(ValueError, match="must be a positive integer"):
        Worker(service_name="py-worker", **kwargs)


def test_worker_mode_must_be_push_or_pull(fake_native_core):
    with pytest.raises(ValueError, match="worker_mode must be 'push' or 'pull'"):
        Worker(service_name="py-worker", worker_mode="sideways")
