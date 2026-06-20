"""Opt-in worker process memory snapshots."""

from __future__ import annotations

import gc
import os
from pathlib import Path

try:  # pragma: no cover - unavailable on some non-Unix platforms
    import resource as _resource
except Exception:  # pragma: no cover
    _resource = None  # type: ignore[assignment]


_TRUTHY = {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def memory_metrics_enabled() -> bool:
    return _env_truthy("AGNT5_WORKER_MEMORY_METRICS") or _env_truthy(
        "AGNT5_WORKER_MEMORY_LOG"
    )


def memory_logging_enabled() -> bool:
    """Backward-compatible alias for the old env gate name."""
    return memory_metrics_enabled()


def _read_int(path: str) -> int | None:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "" or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _proc_status_kb(name: str) -> int | None:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{name}:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _rss_bytes() -> int | None:
    rss_kb = _proc_status_kb("VmRSS")
    if rss_kb is not None:
        return rss_kb * 1024
    if _resource is None:
        return None
    try:
        # Linux reports KiB, macOS reports bytes. This is a fallback only; the
        # Linux /proc path above is the production path in worker containers.
        value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if value <= 0:
        return None
    return value if value > 1024 * 1024 * 1024 else value * 1024


def _tracemalloc_snapshot() -> tuple[int | None, int | None]:
    if not _env_truthy("AGNT5_WORKER_MEMORY_TRACEMALLOC"):
        return None, None
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            frames = int(os.getenv("AGNT5_WORKER_MEMORY_TRACEMALLOC_FRAMES", "10"))
            tracemalloc.start(max(frames, 1))
        current, peak = tracemalloc.get_traced_memory()
        return int(current), int(peak)
    except Exception:
        return None, None


def capture_worker_memory() -> dict[str, int] | None:
    """Return a memory snapshot when worker memory metrics are enabled."""
    if not memory_metrics_enabled():
        return None

    if _env_truthy("AGNT5_WORKER_MEMORY_GC"):
        gc.collect()

    py_heap_current, py_heap_peak = _tracemalloc_snapshot()
    snapshot: dict[str, int] = {}

    values: dict[str, int | None] = {
        "rss_bytes": _rss_bytes(),
        "vm_hwm_bytes": (
            hwm_kb * 1024 if (hwm_kb := _proc_status_kb("VmHWM")) is not None else None
        ),
        "cgroup_current_bytes": _read_int("/sys/fs/cgroup/memory.current")
        or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "cgroup_limit_bytes": _read_int("/sys/fs/cgroup/memory.max")
        or _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "py_heap_current_bytes": py_heap_current,
        "py_heap_peak_bytes": py_heap_peak,
    }
    for key, value in values.items():
        if value is not None:
            snapshot[key] = value
    return snapshot


def record_worker_memory(
    *,
    phase: str,
    component_type: str,
    component_name: str,
) -> None:
    snapshot = capture_worker_memory()
    if not snapshot:
        return

    try:
        from .._core import record_worker_memory_metrics
    except Exception:
        return

    try:
        record_worker_memory_metrics(
            "python",
            phase,
            component_name,
            component_type,
            snapshot,
        )
    except Exception:
        return
