"""Worker memory snapshot helper tests."""

from agnt5.worker._memory import capture_worker_memory, memory_metrics_enabled


def test_worker_memory_metrics_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGNT5_WORKER_MEMORY_METRICS", raising=False)
    monkeypatch.delenv("AGNT5_WORKER_MEMORY_LOG", raising=False)

    assert memory_metrics_enabled() is False
    assert capture_worker_memory() is None


def test_worker_memory_metrics_captures_process_snapshot(monkeypatch):
    monkeypatch.setenv("AGNT5_WORKER_MEMORY_METRICS", "1")

    snapshot = capture_worker_memory()

    assert snapshot is not None
    assert snapshot["rss_bytes"] > 0


def test_worker_memory_log_env_still_enables_metrics(monkeypatch):
    monkeypatch.delenv("AGNT5_WORKER_MEMORY_METRICS", raising=False)
    monkeypatch.setenv("AGNT5_WORKER_MEMORY_LOG", "1")

    assert memory_metrics_enabled() is True
