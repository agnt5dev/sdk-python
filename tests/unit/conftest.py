"""Fixtures shared by the unit tests that construct a real ``Worker``."""

import sys
from types import ModuleType

import pytest

import agnt5
from agnt5.worker import _core as worker_core


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
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class FakeTriggerSpec:
    pass


class FakeExecuteComponentResponse:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_native_core(monkeypatch):
    fake_module = ModuleType("agnt5._core")
    fake_module.PyWorkerConfig = FakePyWorkerConfig
    fake_module.PyWorker = FakePyWorker
    fake_module.PyActivationClient = FakePyActivationClient
    fake_module.PyComponentInfo = FakeComponentInfo
    fake_module.PyTriggerSpec = FakeTriggerSpec
    fake_module.PyExecuteComponentResponse = FakeExecuteComponentResponse
    fake_module.EntityStateManager = FakeEntityStateManager
    fake_module.log_from_python = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "agnt5._core", fake_module)
    monkeypatch.setattr(agnt5, "_core", fake_module)
    monkeypatch.setattr(worker_core, "init_sdk_telemetry", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(worker_core._sentry, "initialize_sentry", lambda **_kwargs: False)
