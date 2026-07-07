"""AGNT5 control-plane backed self-improvement blocks."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx

from . import (
    BehaviorTopic,
    EvaluationResult,
    ImprovementLoopRequest,
    ImprovementProposal,
    PromotionAction,
    PromotionDecision,
    QualityCase,
    RepresentativeRun,
)

AGNT5_CONTROL_PLANE_URL_ENV = "AGNT5_CONTROL_PLANE_URL"
AGNT5_API_BASE_URL_ENV = "AGNT5_API_BASE_URL"
AGNT5_CONTROL_PLANE_TOKEN_ENV = "AGNT5_CONTROL_PLANE_TOKEN"
AGNT5_ACCESS_TOKEN_ENV = "AGNT5_ACCESS_TOKEN"
AGNT5_PROJECT_ID_ENV = "AGNT5_PROJECT_ID"


class ImprovementControlPlaneError(RuntimeError):
    """Raised when the control-plane improvement API returns an error."""

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class ImprovementControlPlaneClient:
    """Small client for the control-plane Quality/Self-Improvement API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        project_id: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        raw_base_url = (
            base_url
            or os.environ.get(AGNT5_CONTROL_PLANE_URL_ENV)
            or os.environ.get(AGNT5_API_BASE_URL_ENV)
        )
        if not raw_base_url:
            raise ValueError(
                "control-plane base URL is required; pass base_url or set "
                f"{AGNT5_CONTROL_PLANE_URL_ENV}"
            )
        self.base_url = _normalize_base_url(raw_base_url)
        self.project_id = project_id or os.environ.get(AGNT5_PROJECT_ID_ENV)
        if not self.project_id:
            raise ValueError(f"project_id is required; pass project_id or set {AGNT5_PROJECT_ID_ENV}")
        self.token = (
            token
            or os.environ.get(AGNT5_CONTROL_PLANE_TOKEN_ENV)
            or os.environ.get(AGNT5_ACCESS_TOKEN_ENV)
        )
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def list_topics(self, **params: Any) -> list[dict[str, Any]]:
        return self._request("GET", "/quality/topics", params=_clean_dict(params))

    def list_topic_runs(self, topic_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/quality/topics/{topic_id}/runs")

    def create_case_from_topic(
        self,
        topic_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", f"/quality/topics/{topic_id}/create-case", json=payload or {})

    def get_case(self, case_id: str) -> dict[str, Any]:
        return self._request("GET", f"/quality/cases/{case_id}")

    def list_case_links(self, case_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/quality/cases/{case_id}/links")

    def add_case_link(self, case_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/quality/cases/{case_id}/links", json=payload)

    def create_case_regression_dataset(
        self,
        case_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/quality/cases/{case_id}/regression-dataset",
            json=payload or {},
        )

    def create_proposal(
        self,
        case_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", f"/quality/cases/{case_id}/proposals", json=payload or {})

    def create_attempt_from_proposal(
        self,
        proposal_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/quality/proposals/{proposal_id}/create-attempt",
            json=payload or {},
        )

    def update_attempt(self, attempt_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/quality/attempts/{attempt_id}", json=payload)

    def approve_attempt(
        self,
        attempt_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", f"/quality/attempts/{attempt_id}/approve", json=payload or {})

    def reject_attempt(
        self,
        attempt_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", f"/quality/attempts/{attempt_id}/reject", json=payload or {})

    def create_experiment(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/experiments", json=payload)

    def create_experiment_run(
        self,
        experiment_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request("POST", f"/experiments/{experiment_id}/runs", json=payload or {})

    def get_experiment_run(self, experiment_id: str, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/experiments/{experiment_id}/runs/{run_id}")

    def add_case_event(
        self,
        case_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._request("POST", f"/quality/cases/{case_id}/events", json=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        url = self._url(path, params=params)
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = self._client.request(method, url, json=json, headers=headers)
        if response.status_code >= 400:
            raise _control_plane_error(response)
        if response.status_code == 204:
            return None
        if not response.content:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            raise ImprovementControlPlaneError(
                f"control-plane response was not JSON: {response.text[:200]}",
                response.status_code,
            ) from exc
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    def _url(self, path: str, *, params: Mapping[str, Any] | None = None) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}/projects/{self.project_id}{clean_path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        return url


class AGNT5ImprovementBlocks:
    """Improvement blocks backed by AGNT5 Quality/Self-Improvement APIs.

    This adapter creates and links control-plane evidence. It does not own the
    loop policy and it does not mutate production unless the caller's loop policy
    explicitly reaches a promotion/approval decision.
    """

    def __init__(
        self,
        client: ImprovementControlPlaneClient,
        *,
        topic_status: str = "new",
        page_size: int = 50,
        auto_create_regression_dataset: bool = True,
        auto_run_eval: bool = True,
    ) -> None:
        self.client = client
        self.topic_status = topic_status
        self.page_size = page_size
        self.auto_create_regression_dataset = auto_create_regression_dataset
        self.auto_run_eval = auto_run_eval

    def select_topic(self, request: ImprovementLoopRequest) -> BehaviorTopic | None:
        params: dict[str, Any] = {"page": 1, "page_size": self.page_size}
        if self.topic_status:
            params["status"] = request.metadata.get("topic_status", self.topic_status)
        if request.component_name:
            params["component_name"] = request.component_name
        topics = self.client.list_topics(**params)
        if request.topic_id:
            topic = next((item for item in topics if item.get("id") == request.topic_id), None)
            if topic is None and params.get("status"):
                fallback = dict(params)
                fallback.pop("status", None)
                topic = next(
                    (
                        item
                        for item in self.client.list_topics(**fallback)
                        if item.get("id") == request.topic_id
                    ),
                    None,
                )
            if topic is None:
                return None
        else:
            topic = _rank_topics(topics)[0] if topics else None
            if topic is None:
                return None

        runs = self.client.list_topic_runs(str(topic["id"]))
        return _behavior_topic_from_api(topic, runs)

    def create_case(self, topic: BehaviorTopic, request: ImprovementLoopRequest) -> QualityCase:
        if request.case_id:
            return _quality_case_from_api(self.client.get_case(request.case_id))

        payload = {
            "metadata": _merge_dicts(
                {
                    "source": "agnt5_sdk_self_improvement_loop",
                    "behavior_topic_id": topic.topic_id,
                    "component_name": topic.component_name,
                    "signal": topic.signal,
                },
                _mapping_or_none(request.metadata.get("case_metadata")),
            )
        }
        if title := request.metadata.get("case_title"):
            payload["title"] = title
        if severity := request.metadata.get("case_severity"):
            payload["severity"] = severity
        result = self.client.create_case_from_topic(topic.topic_id, _clean_dict(payload))
        return _quality_case_from_api(result["quality_case"])

    def propose_change(
        self,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> ImprovementProposal:
        candidate_artifact_type = str(
            request.metadata.get("candidate_artifact_type") or "candidate_change"
        )
        payload = {
            "candidate_artifact_type": candidate_artifact_type,
            "candidate_artifact": _merge_dicts(
                _candidate_change_artifact(case, request, candidate_artifact_type),
                _mapping_or_none(request.metadata.get("candidate_artifact")),
            ),
            "evidence": _merge_dicts(
                {
                    "source": "agnt5_sdk_self_improvement_loop",
                    "quality_case_id": case.case_id,
                    "behavior_topic_id": case.topic_id,
                    "representative_run_ids": list(case.representative_run_ids),
                },
                _mapping_or_none(request.metadata.get("proposal_evidence")),
            ),
            "metadata": _merge_dicts(
                {
                    "source": "agnt5_sdk_self_improvement_loop",
                    "quality_case_id": case.case_id,
                    "behavior_topic_id": case.topic_id,
                },
                _mapping_or_none(request.metadata.get("proposal_metadata")),
            ),
        }
        if title := request.metadata.get("proposal_title"):
            payload["title"] = title
        if summary := request.metadata.get("proposal_summary"):
            payload["summary"] = summary
        if recommendation := request.metadata.get("proposal_recommendation"):
            payload["recommendation"] = recommendation

        proposal = self.client.create_proposal(case.case_id, _clean_dict(payload))
        return _proposal_from_api(proposal)

    def evaluate_change(
        self,
        proposal: ImprovementProposal,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> EvaluationResult:
        if _bool_meta(
            request,
            "auto_create_regression_dataset",
            self.auto_create_regression_dataset,
        ):
            self._ensure_regression_dataset(case, request)

        candidate_deployment_id = request.metadata.get("candidate_deployment_id")
        payload = {
            "decision_notes": "Created from agnt5 SDK self-improvement loop",
        }
        if isinstance(candidate_deployment_id, str) and candidate_deployment_id:
            payload["candidate_deployment_id"] = candidate_deployment_id

        result = self.client.create_attempt_from_proposal(proposal.proposal_id, payload)
        attempt = result["attempt"]
        if _bool_meta(request, "run_eval", self.auto_run_eval):
            attempt = self._run_attempt_eval(case, attempt, request)
        status = str(attempt.get("status") or "")
        pass_rate = _float_or_zero(attempt.get("pass_rate"))
        failed_items = _int_or_zero(attempt.get("failed_items"))
        passed = status in {"passed", "approved"} and failed_items == 0
        pending = status in {"draft", "running", ""}

        return EvaluationResult(
            passed=passed,
            pass_rate=pass_rate,
            failed_items=failed_items,
            experiment_id=attempt.get("experiment_id"),
            experiment_run_id=attempt.get("latest_experiment_run_id"),
            summary=_attempt_summary(attempt),
            pending=pending,
            metadata={
                "source": "agnt5_control_plane",
                "quality_case_id": case.case_id,
                "proposal_id": proposal.proposal_id,
                "attempt_id": attempt.get("id"),
                "attempt_status": status,
                "attempt": attempt,
            },
        )

    def _ensure_regression_dataset(
        self,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> dict[str, str]:
        links = self.client.list_case_links(case.case_id)
        dataset_id = _case_link_target(links, "dataset", "quality_case_regression_dataset")
        dataset_version_id = _case_link_target(
            links,
            "dataset_version",
            "quality_case_regression_dataset",
        )
        if dataset_id and dataset_version_id:
            return {"dataset_id": dataset_id, "dataset_version_id": dataset_version_id}

        payload = _mapping_or_none(request.metadata.get("regression_dataset")) or {}
        result = self.client.create_case_regression_dataset(case.case_id, payload)
        dataset = _mapping_or_none(result.get("regression_dataset")) or {}
        version = _mapping_or_none(result.get("regression_dataset_version")) or {}
        return {
            "dataset_id": str(dataset.get("id") or ""),
            "dataset_version_id": str(version.get("id") or ""),
        }

    def _run_attempt_eval(
        self,
        case: QualityCase,
        attempt: Mapping[str, Any],
        request: ImprovementLoopRequest,
    ) -> dict[str, Any]:
        raw_case = _raw_mapping(case.metadata)
        case_metadata = _mapping_or_none(raw_case.get("metadata")) or {}
        dataset_id = _string_or_none(attempt.get("dataset_id")) or _string_or_none(
            request.metadata.get("dataset_id")
        )
        dataset_version_id = _string_or_none(
            attempt.get("dataset_version_id")
        ) or _string_or_none(request.metadata.get("dataset_version_id"))
        baseline_deployment_id = _string_or_none(
            attempt.get("baseline_deployment_id")
        ) or _string_or_none(case_metadata.get("deployment_id"))
        candidate_deployment_id = (
            _string_or_none(request.metadata.get("candidate_deployment_id"))
            or _string_or_none(attempt.get("candidate_deployment_id"))
            or baseline_deployment_id
        )
        component_name = case.component_name or _string_or_none(
            case_metadata.get("component_name")
        )
        component_type = _string_or_none(case_metadata.get("component_type")) or "function"

        if not dataset_id or not dataset_version_id:
            return dict(attempt)
        if not candidate_deployment_id or not component_name:
            return dict(attempt)

        experiment = self.client.create_experiment(
            {
                "name": f"{case.title} attempt eval",
                "description": f"Improvement attempt {attempt.get('id')} for quality case {case.case_id}.",
                "type": "evaluation",
                "dataset_id": dataset_id,
                "dataset_version_id": dataset_version_id,
                "target_type": "component",
                "deployment_id": candidate_deployment_id,
                "execution_mode": request.metadata.get("execution_mode", "dev_worker"),
                "component_name": component_name,
                "component_type": component_type,
                "repetitions": int(request.metadata.get("repetitions", 1)),
                "config": _clean_dict(
                    {
                        "source": "agnt5_sdk_self_improvement_loop",
                        "quality_case_id": case.case_id,
                        "improvement_attempt_id": attempt.get("id"),
                        "source_run_id": raw_case.get("source_run_id"),
                        "behavior_topic_id": attempt.get("behavior_topic_id") or case.topic_id,
                        "baseline_deployment_id": baseline_deployment_id,
                        "candidate_deployment_id": candidate_deployment_id,
                        "eval_lane": attempt.get("eval_lane"),
                    }
                ),
                "gate_config": {
                    "min_pass_rate": request.metadata.get("min_pass_rate", 1),
                    "max_failed_items": request.metadata.get("max_failed_items", 0),
                },
            }
        )
        experiment_id = str(experiment.get("id") or "")
        run = self.client.create_experiment_run(
            experiment_id,
            {
                "name": f"{case.title} attempt check",
                "deployment_id": candidate_deployment_id,
            },
        )
        run_id = str(run.get("id") or "")
        link_metadata = {
            "source": "agnt5_sdk_self_improvement_loop",
            "quality_case_id": case.case_id,
            "improvement_attempt_id": attempt.get("id"),
            "dataset_id": dataset_id,
            "dataset_version_id": dataset_version_id,
            "baseline_deployment_id": baseline_deployment_id,
            "candidate_deployment_id": candidate_deployment_id,
        }
        self.client.add_case_link(
            case.case_id,
            {
                "link_type": "experiment",
                "target_id": experiment_id,
                "metadata": link_metadata,
            },
        )
        self.client.add_case_link(
            case.case_id,
            {
                "link_type": "experiment_run",
                "target_id": run_id,
                "metadata": _merge_dicts(link_metadata, {"experiment_id": experiment_id}),
            },
        )
        updated = self.client.update_attempt(
            str(attempt.get("id") or ""),
            {
                "status": "running",
                "experiment_id": experiment_id,
                "latest_experiment_run_id": run_id,
                "metadata": _merge_dicts(
                    _mapping_or_none(attempt.get("metadata")) or {},
                    {
                        "latest_experiment_id": experiment_id,
                        "latest_experiment_run_id": run_id,
                    },
                ),
            },
        )
        if not _bool_meta(request, "wait_for_eval", False):
            return updated
        return self._wait_for_attempt_eval(updated, experiment_id, run_id, request)

    def _wait_for_attempt_eval(
        self,
        attempt: Mapping[str, Any],
        experiment_id: str,
        run_id: str,
        request: ImprovementLoopRequest,
    ) -> dict[str, Any]:
        max_polls = int(request.metadata.get("eval_poll_max", 30))
        interval_seconds = float(request.metadata.get("eval_poll_interval_seconds", 2.0))
        run = self.client.get_experiment_run(experiment_id, run_id)
        for _ in range(max(max_polls - 1, 0)):
            if _run_terminal(run):
                break
            time.sleep(interval_seconds)
            run = self.client.get_experiment_run(experiment_id, run_id)
        if not _run_terminal(run):
            return dict(attempt)

        passed = run.get("passed")
        failed_items = _int_or_zero(run.get("failed_items"))
        pass_rate = _float_or_zero(run.get("pass_rate"))
        total_items = _int_or_zero(run.get("total_items") or run.get("total_run_items"))
        status = "passed" if passed is True or (pass_rate >= 1.0 and failed_items == 0) else "failed"
        return self.client.update_attempt(
            str(attempt.get("id") or ""),
            {
                "status": status,
                "pass_rate": pass_rate,
                "failed_items": failed_items,
                "total_items": total_items,
                "metadata": _merge_dicts(
                    _mapping_or_none(attempt.get("metadata")) or {},
                    {
                        "latest_experiment_run_status": run.get("status"),
                        "latest_experiment_run_passed": passed,
                    },
                ),
            },
        )

    def record_decision(
        self,
        decision: PromotionDecision,
        proposal: ImprovementProposal,
        evaluation: EvaluationResult,
        request: ImprovementLoopRequest,
    ) -> dict[str, Any]:
        attempt_id = _string_or_none(evaluation.metadata.get("attempt_id"))
        case_id = _string_or_none(evaluation.metadata.get("quality_case_id") or proposal.case_id)
        record: dict[str, Any] = {
            "action": decision.action.value,
            "reason": decision.reason,
            "proposal_id": proposal.proposal_id,
            "attempt_id": attempt_id,
        }
        if attempt_id and decision.action == PromotionAction.PROMOTE:
            updated = self.client.approve_attempt(
                attempt_id,
                {"decision_notes": decision.reason},
            )
            record["attempt"] = updated
        elif attempt_id and decision.action == PromotionAction.BLOCK and not evaluation.pending:
            updated = self.client.reject_attempt(
                attempt_id,
                {"decision_notes": decision.reason},
            )
            record["attempt"] = updated
        elif case_id:
            event = self.client.add_case_event(
                case_id,
                {
                    "event_type": "self_improvement_loop_decision",
                    "actor_kind": "system",
                    "body": decision.reason,
                    "metadata": {
                        "source": "agnt5_sdk_self_improvement_loop",
                        "action": decision.action.value,
                        "proposal_id": proposal.proposal_id,
                        "attempt_id": attempt_id,
                    },
                },
            )
            record["event"] = event
        return record


def _normalize_base_url(value: str) -> str:
    base = value.rstrip("/")
    if not base.endswith("/api/v1"):
        base = f"{base}/api/v1"
    return base


def _control_plane_error(response: httpx.Response) -> ImprovementControlPlaneError:
    message = f"HTTP {response.status_code}: {response.reason_phrase}"
    code = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = str(payload.get("message") or payload.get("error") or message)
        code_value = payload.get("code")
        code = str(code_value) if code_value is not None else None
    return ImprovementControlPlaneError(message, response.status_code, code)


def _behavior_topic_from_api(
    topic: Mapping[str, Any],
    assignments: list[Mapping[str, Any]],
) -> BehaviorTopic:
    runs = tuple(_representative_run_from_assignment(item) for item in assignments)
    return BehaviorTopic(
        topic_id=str(topic.get("id") or ""),
        title=str(topic.get("title") or "Behavior topic"),
        component_name=str(topic.get("component_name") or ""),
        signal=str(topic.get("signal") or ""),
        suspicion_score=_float_or_zero(topic.get("suspicion_score")),
        run_count=_int_or_zero(topic.get("run_count")),
        representative_runs=runs,
        metadata={
            "source": "agnt5_control_plane",
            "raw": dict(topic),
        },
    )


def _representative_run_from_assignment(item: Mapping[str, Any]) -> RepresentativeRun:
    evidence = _mapping_or_none(item.get("evidence")) or {}
    output = evidence.get("output") if isinstance(evidence.get("output"), dict) else None
    error = evidence.get("error") or evidence.get("error_code") or evidence.get("message")
    return RepresentativeRun(
        run_id=str(item.get("run_id") or ""),
        input=_mapping_or_none(evidence.get("input")) or {},
        output=output,
        error=str(error) if error else None,
        score=_float_or_none(item.get("suspicion_score")),
        metadata={
            "source": "agnt5_control_plane",
            "assignment_id": item.get("id"),
            "evidence": evidence,
        },
    )


def _quality_case_from_api(item: Mapping[str, Any]) -> QualityCase:
    metadata = _mapping_or_none(item.get("metadata")) or {}
    topic_id = str(metadata.get("behavior_topic_id") or "")
    component_name = str(
        metadata.get("component_name")
        or metadata.get("component")
        or metadata.get("target_component")
        or ""
    )
    source_run_id = item.get("source_run_id")
    representative_run_ids = (str(source_run_id),) if source_run_id else ()
    return QualityCase(
        case_id=str(item.get("id") or ""),
        title=str(item.get("title") or "Quality case"),
        observed=str(item.get("observed_behavior") or ""),
        expected=str(item.get("expected_behavior") or ""),
        topic_id=topic_id,
        component_name=component_name,
        representative_run_ids=representative_run_ids,
        metadata={
            "source": "agnt5_control_plane",
            "raw": dict(item),
        },
    )


def _candidate_change_artifact(
    case: QualityCase,
    request: ImprovementLoopRequest,
    artifact_type: str,
) -> dict[str, Any]:
    expected = case.expected or f"{case.component_name} should satisfy the quality case."
    observed = case.observed or "Observed behavior not recorded."
    return {
        "schema_version": "agnt5.candidate_change.v1",
        "type": artifact_type or "candidate_change",
        "source": "agnt5_sdk_self_improvement_loop",
        "strategy": "candidate_change",
        "component_name": case.component_name,
        "objective": request.objective,
        "hypothesis": (
            f"Change {case.component_name} so the candidate behavior matches the "
            "expected quality case outcome."
        ),
        "before": {
            "label": "Observed production behavior",
            "observed_behavior": observed,
        },
        "proposed_changes": [
            {
                "title": f"Align {case.component_name} with expected behavior",
                "target": case.component_name,
                "artifact": artifact_type or "candidate_change",
                "operation": "prepare_candidate_change",
                "before": observed,
                "after": expected,
                "rationale": request.objective,
            }
        ],
        "after": {
            "label": "Expected candidate behavior",
            "expected_behavior": expected,
        },
        "evaluation_plan": {
            "offline_lane": "isolated_eval_lane",
            "regression_dataset_required": True,
            "online_comparison_recommended": True,
            "gate_required": True,
        },
        "risk_policy": {
            "human_gate_required": True,
            "side_effect_lane": "journal_replay_or_sandbox",
        },
    }


def _proposal_from_api(item: Mapping[str, Any]) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=str(item.get("id") or ""),
        title=str(item.get("title") or "Improvement proposal"),
        artifact_type=str(item.get("candidate_artifact_type") or "deployment"),
        component_name=str(
            (_mapping_or_none(item.get("candidate_artifact")) or {}).get("component_name") or ""
        ),
        summary=str(item.get("summary") or ""),
        candidate=_mapping_or_none(item.get("candidate_artifact")) or {},
        case_id=str(item.get("quality_case_id") or "") or None,
        metadata={
            "source": "agnt5_control_plane",
            "raw": dict(item),
        },
    )


def _rank_topics(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        topics,
        key=lambda topic: (
            _float_or_zero(topic.get("impact_score")),
            _float_or_zero(topic.get("suspicion_score")),
            _int_or_zero(topic.get("run_count")),
        ),
        reverse=True,
    )


def _attempt_summary(attempt: Mapping[str, Any]) -> str:
    status = str(attempt.get("status") or "draft")
    if status in {"draft", "running"}:
        return "candidate attempt created; run or attach evaluation evidence before approval"
    return f"candidate attempt status is {status}"


def _case_link_target(
    links: list[Mapping[str, Any]],
    link_type: str,
    source: str,
) -> str:
    for link in links:
        metadata = _mapping_or_none(link.get("metadata")) or {}
        if link.get("link_type") == link_type and metadata.get("source") == source:
            return str(link.get("target_id") or "")
    return ""


def _bool_meta(request: ImprovementLoopRequest, key: str, default: bool) -> bool:
    value = request.metadata.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return default


def _raw_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping_or_none(metadata.get("raw")) or {}


def _run_terminal(run: Mapping[str, Any]) -> bool:
    return str(run.get("status") or "").lower() in {"completed", "failed", "cancelled"}


def _clean_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _merge_dicts(
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(base)
    if override:
        result.update(override)
    return result


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_zero(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
