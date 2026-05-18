"""Tests for local eval file, cache, and report helpers."""

from __future__ import annotations

import json

from agnt5.batch_eval import BatchEvalItem
from agnt5.eval.local import (
    LOCAL_CI_SUMMARY_SCHEMA,
    LOCAL_EXPERIMENT_SCHEMA,
    LocalExperiment,
    LocalFileResultStore,
    build_local_ci_payload,
    load_local_experiment,
    run_local_experiment,
    save_local_experiment,
    upload_local_ci_summary,
)


def test_local_experiment_save_load_json(tmp_path):
    experiment = LocalExperiment(
        name="greeting-smoke",
        cases=[
            BatchEvalItem(input={"name": "Ada"}, expected="Hello Ada", item_id="case-1"),
            BatchEvalItem(input={"name": "Grace"}, expected="Hello Grace", item_id="case-2"),
        ],
        scorers=["exact_match", {"name": "contains", "config": {"pattern": "Hello"}}],
        metadata={"owner": "sdk"},
    )

    path = tmp_path / "experiment.json"
    save_local_experiment(experiment, path)
    loaded = load_local_experiment(path)

    assert loaded.schema_version == LOCAL_EXPERIMENT_SCHEMA
    assert loaded.name == "greeting-smoke"
    assert loaded.cases[0].item_id == "case-1"
    assert loaded.cases[1].expected == "Hello Grace"
    assert loaded.scorers[1]["config"]["pattern"] == "Hello"
    assert loaded.metadata == {"owner": "sdk"}


def test_run_local_experiment_reuses_cached_task_outputs(tmp_path):
    calls = {"count": 0}
    experiment = LocalExperiment(
        name="cache-smoke",
        cases=[
            BatchEvalItem(input={"name": "Ada"}, expected="Hello Ada", item_id="ada"),
            BatchEvalItem(input={"name": "Grace"}, expected="Hello Grace", item_id="grace"),
        ],
        scorers=["exact_match"],
    )
    store = LocalFileResultStore(tmp_path / "cache")

    def task(input_data):
        calls["count"] += 1
        return f"Hello {input_data['name']}"

    first = run_local_experiment(experiment, task, result_store=store, task_version="v1")
    second = run_local_experiment(experiment, task, result_store=store, task_version="v1")

    assert calls["count"] == 2
    assert first.cache_hits == 0
    assert first.cache_misses == 2
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    assert second.stats.passed_items == 2
    assert second.results[0].scores[0].scorer == "exact_match"


def test_custom_result_store_protocol_is_supported():
    class MemoryStore:
        def __init__(self):
            self.data = {}

        def load(self, case_key):
            return self.data.get(case_key)

        def save(self, case_key, payload):
            self.data[case_key] = payload

    calls = {"count": 0}
    experiment = LocalExperiment(
        name="custom-store",
        cases=[BatchEvalItem(input={"value": 1}, expected=2, item_id="one")],
        scorers=["exact_match"],
    )
    store = MemoryStore()

    def task(input_data):
        calls["count"] += 1
        return input_data["value"] + 1

    run_local_experiment(experiment, task, result_store=store, task_version="v1")
    report = run_local_experiment(experiment, task, result_store=store, task_version="v1")

    assert calls["count"] == 1
    assert report.cache_hits == 1
    assert report.results[0].passed is True


def test_local_report_render_and_export(tmp_path):
    experiment = LocalExperiment(
        name="report-smoke",
        cases=[BatchEvalItem(input={"name": "Ada"}, expected="Hello Ada", item_id="ada")],
        scorers=["exact_match"],
    )

    report = run_local_experiment(
        experiment,
        lambda input_data: f"Hello {input_data['name']}",
    )
    json_path = tmp_path / "report.json"
    jsonl_path = tmp_path / "report.jsonl"
    report.export_json(json_path)
    report.export_jsonl(jsonl_path)

    rendered = report.render_terminal()
    exported = json.loads(json_path.read_text())
    jsonl_lines = jsonl_path.read_text().strip().splitlines()

    assert "Local eval: report-smoke" in rendered
    assert "Pass rate: 100%" in rendered
    assert exported["stats"]["passed_items"] == 1
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["scorer"] == "exact_match"


def test_local_ci_payload_and_upload_helper():
    experiment = LocalExperiment(
        name="ci-smoke",
        cases=[BatchEvalItem(input={"name": "Ada"}, expected="Hello Ada", item_id="ada")],
        scorers=["exact_match"],
    )
    report = run_local_experiment(experiment, lambda input_data: "Wrong")
    payload = build_local_ci_payload(report)

    assert payload["schema_version"] == LOCAL_CI_SUMMARY_SCHEMA
    assert payload["stats"]["passed_items"] == 0
    assert len(payload["failures"]) == 1

    class FakeClient:
        def upload_eval_summary(self, upload_payload):
            self.payload = upload_payload
            return {"ok": True}

    client = FakeClient()
    result = upload_local_ci_summary(client, report, include_items=True)

    assert result == {"ok": True}
    assert client.payload["items"][0]["item_id"] == "ada"
