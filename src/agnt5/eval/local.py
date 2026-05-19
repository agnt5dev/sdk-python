"""Local eval file, cache, and report helpers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ..batch_eval import BatchEvalItem, BatchEvalItemResult, BatchEvalStats
from ..responses import ScorerResultSummary

LOCAL_EXPERIMENT_SCHEMA = "agnt5.local_experiment.v1"
LOCAL_CACHE_SCHEMA = "agnt5.local_result_cache.v1"
LOCAL_CI_SUMMARY_SCHEMA = "agnt5.local_ci_summary.v1"
_CACHE_MISS = object()

ScorerSpec = Union[str, Dict[str, Any]]
TaskFunction = Callable[[Dict[str, Any]], Any]


@dataclass
class LocalExperiment:
    """Serializable local experiment definition."""

    name: str
    cases: List[BatchEvalItem]
    scorers: List[ScorerSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = LOCAL_EXPERIMENT_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "cases": [case.to_dict() for case in self.cases],
            "scorers": list(self.scorers),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalExperiment":
        if data.get("schema_version", LOCAL_EXPERIMENT_SCHEMA) != LOCAL_EXPERIMENT_SCHEMA:
            raise ValueError(f"Unsupported local experiment schema: {data.get('schema_version')}")
        cases = [
            BatchEvalItem(
                input=case["input"],
                expected=case.get("expected"),
                item_id=case.get("item_id"),
                index=case.get("index"),
            )
            for case in data.get("cases", [])
        ]
        return cls(
            name=data["name"],
            cases=cases,
            scorers=list(data.get("scorers") or []),
            metadata=dict(data.get("metadata") or {}),
            schema_version=data.get("schema_version", LOCAL_EXPERIMENT_SCHEMA),
        )

    def save_json(self, path: Union[str, Path]) -> None:
        save_local_experiment(self, path)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "LocalExperiment":
        return load_local_experiment(path)


class LocalResultStore(Protocol):
    """Protocol for local result stores such as file, S3, or database backends."""

    def load(self, case_key: str) -> Optional[Dict[str, Any]]:
        """Load a cached result payload for a case key."""
        ...

    def save(self, case_key: str, payload: Dict[str, Any]) -> None:
        """Save a cached result payload for a case key."""
        ...


class LocalFileResultStore:
    """JSON-file-backed result store, one file per case."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, case_key: str) -> Optional[Dict[str, Any]]:
        path = self._path(case_key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return None
        return data

    def save(self, case_key: str, payload: Dict[str, Any]) -> None:
        path = self._path(case_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

    def _path(self, case_key: str) -> Path:
        safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in case_key)
        return self.root / f"{safe_key}.json"


class LocalResultCache:
    """Small result cache that validates task fingerprints before reuse."""

    def __init__(self, store: LocalResultStore) -> None:
        self.store = store

    def load(self, case_key: str, fingerprint: str) -> Any:
        payload = self.store.load(case_key)
        if not payload or payload.get("fingerprint") != fingerprint:
            return _CACHE_MISS
        if "output" not in payload:
            return _CACHE_MISS
        return payload.get("output")

    def save(self, case_key: str, fingerprint: str, output: Any) -> None:
        self.store.save(
            case_key,
            {
                "schema_version": LOCAL_CACHE_SCHEMA,
                "case_key": case_key,
                "fingerprint": fingerprint,
                "output": output,
                "saved_at_ms": int(time.time() * 1000),
            },
        )


@dataclass
class LocalEvalReport:
    """Local eval report with terminal and export helpers."""

    experiment_name: str
    results: List[BatchEvalItemResult]
    stats: BatchEvalStats
    cache_hits: int = 0
    cache_misses: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for result in self.results if result.passed and result.is_success)
        return passed / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "stats": {
                "total_items": self.stats.total_items,
                "completed_items": self.stats.completed_items,
                "failed_items": self.stats.failed_items,
                "passed_items": self.stats.passed_items,
                "avg_duration_ms": self.stats.avg_duration_ms,
                "duration_ms": self.stats.duration_ms,
                "pass_rate": self.pass_rate,
            },
            "cache": {"hits": self.cache_hits, "misses": self.cache_misses},
            "metadata": dict(self.metadata),
            "results": [batch_result_to_dict(result) for result in self.results],
        }

    def flatten(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for result in self.results:
            base = {
                "experiment_name": self.experiment_name,
                "index": result.index,
                "item_id": result.item_id,
                "run_id": result.run_id,
                "passed": result.passed,
                "error": result.error,
                "duration_ms": result.duration_ms,
            }
            if not result.scores:
                rows.append({**base, "scorer": None, "score": None, "label": None})
                continue
            for score in result.scores:
                rows.append(
                    {
                        **base,
                        "scorer": score.scorer,
                        "score": score.score,
                        "scorer_passed": score.passed,
                        "label": score.label,
                        "explanation": score.explanation,
                        "metadata": score.metadata,
                    }
                )
        return rows

    def render_terminal(self) -> str:
        lines = [
            f"Local eval: {self.experiment_name}",
            (
                f"Items: {self.stats.total_items} | Passed: {self.stats.passed_items} | "
                f"Failed: {self.stats.failed_items} | Pass rate: {self.pass_rate:.0%}"
            ),
            f"Cache: {self.cache_hits} hits, {self.cache_misses} misses",
            "",
            "idx  status  scores",
        ]
        for result in sorted(self.results, key=lambda item: item.index):
            status = "PASS" if result.passed and result.is_success else "FAIL"
            scores = ", ".join(
                f"{score.scorer}={score.score:.2f}{'' if score.passed else '!'}"
                for score in result.scores
            )
            if result.error:
                scores = result.error
            lines.append(f"{result.index:<4} {status:<6} {scores}")
        return "\n".join(lines)

    def export_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")

    def export_jsonl(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in self.flatten():
                handle.write(json.dumps(row, sort_keys=True, default=str))
                handle.write("\n")


def save_local_experiment(experiment: LocalExperiment, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(experiment.to_dict(), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def load_local_experiment(path: Union[str, Path]) -> LocalExperiment:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Local experiment file must contain a JSON object")
    return LocalExperiment.from_dict(data)


def run_local_experiment(
    experiment: LocalExperiment,
    task: TaskFunction,
    *,
    result_store: Optional[LocalResultStore] = None,
    task_version: str = "default",
) -> LocalEvalReport:
    cache = LocalResultCache(result_store) if result_store is not None else None
    results: List[BatchEvalItemResult] = []
    cache_hits = 0
    cache_misses = 0
    started = time.time()

    for index, case in enumerate(experiment.cases):
        case_index = case.index if case.index is not None else index
        case_key = case.item_id or str(case_index)
        fingerprint = stable_fingerprint(
            {
                "task_version": task_version,
                "input": case.input,
                "expected": case.expected,
            }
        )
        item_started = time.time()
        error: Optional[str] = None
        output: Any = None
        cache_hit = False
        if cache is not None:
            cached = cache.load(case_key, fingerprint)
            if cached is not _CACHE_MISS:
                output = cached
                cache_hits += 1
                cache_hit = True
            else:
                cache_misses += 1
        try:
            if not cache_hit:
                output = task(case.input)
                if cache is not None:
                    cache.save(case_key, fingerprint, output)
            scores = score_local_output(output, case.expected, experiment.scorers)
            passed = all(score.passed for score in scores) if scores else error is None
        except Exception as exc:
            error = str(exc)
            scores = []
            passed = False
        duration_ms = int((time.time() - item_started) * 1000)
        results.append(
            BatchEvalItemResult(
                index=case_index,
                run_id=f"local-{case_key}",
                output=output,
                scores=scores,
                passed=passed,
                duration_ms=duration_ms,
                item_id=case.item_id,
                error=error,
            )
        )

    total_duration_ms = int((time.time() - started) * 1000)
    return LocalEvalReport(
        experiment_name=experiment.name,
        results=results,
        stats=BatchEvalStats.from_results(results, total_duration_ms),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        metadata=dict(experiment.metadata),
    )


def score_local_output(
    output: Any,
    expected: Any,
    scorers: List[ScorerSpec],
) -> List[ScorerResultSummary]:
    return [_score_local_scorer(output, expected, scorer) for scorer in scorers]


def build_local_ci_payload(
    report: LocalEvalReport,
    *,
    include_items: bool = False,
    include_failures: bool = True,
) -> Dict[str, Any]:
    payload = {
        "schema_version": LOCAL_CI_SUMMARY_SCHEMA,
        "experiment_name": report.experiment_name,
        "stats": report.to_dict()["stats"],
        "cache": {"hits": report.cache_hits, "misses": report.cache_misses},
        "metadata": dict(report.metadata),
    }
    if include_items:
        payload["items"] = report.to_dict()["results"]
    elif include_failures:
        payload["failures"] = [
            result
            for result in report.to_dict()["results"]
            if result.get("error") or not result.get("passed")
        ]
    return payload


def upload_local_ci_summary(
    client: Any,
    report: LocalEvalReport,
    *,
    include_items: bool = False,
    include_failures: bool = True,
) -> Any:
    payload = build_local_ci_payload(
        report,
        include_items=include_items,
        include_failures=include_failures,
    )
    uploader = getattr(client, "upload_eval_summary", None)
    if callable(uploader):
        return uploader(payload)
    raise AttributeError("client must provide upload_eval_summary(payload)")


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def batch_result_to_dict(result: BatchEvalItemResult) -> Dict[str, Any]:
    return {
        "index": result.index,
        "run_id": result.run_id,
        "output": result.output,
        "scores": [score_result_to_dict(score) for score in result.scores],
        "passed": result.passed,
        "duration_ms": result.duration_ms,
        "item_id": result.item_id,
        "trace_id": result.trace_id,
        "error": result.error,
    }


def score_result_to_dict(score: ScorerResultSummary) -> Dict[str, Any]:
    return {
        "scorer": score.scorer,
        "score": score.score,
        "passed": score.passed,
        "explanation": score.explanation,
        "label": score.label,
        "metadata": score.metadata,
    }


def _score_local_scorer(output: Any, expected: Any, scorer: ScorerSpec) -> ScorerResultSummary:
    name, config = _scorer_name_config(scorer)
    if name == "exact_match":
        passed = output == expected
        return ScorerResultSummary(
            scorer=name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation="Exact match" if passed else "Output did not exactly match expected",
        )
    if name == "contains":
        pattern = str(config.get("pattern", expected if expected is not None else ""))
        case_sensitive = bool(config.get("case_sensitive", True))
        haystack = str(output)
        needle = pattern
        if not case_sensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        passed = needle in haystack
        return ScorerResultSummary(
            scorer=name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation=f"Output contains {pattern!r}" if passed else f"Output missing {pattern!r}",
        )
    if name == "json_valid":
        try:
            if isinstance(output, str):
                json.loads(output)
            passed = True
        except ValueError:
            passed = False
        return ScorerResultSummary(
            scorer=name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation="Valid JSON" if passed else "Invalid JSON",
        )
    raise ValueError(f"Unsupported local scorer: {name}")


def _scorer_name_config(scorer: ScorerSpec) -> tuple[str, Dict[str, Any]]:
    if isinstance(scorer, str):
        return scorer, {}
    return str(scorer.get("name")), dict(scorer.get("config") or {})
