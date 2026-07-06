import httpx
import pytest

from agnt5.improvement import (
    AGNT5ImprovementBlocks,
    EvaluationResult,
    FixtureImprovementBlocks,
    ImprovementControlPlaneClient,
    ImprovementLoopPolicy,
    ImprovementLoopRequest,
    LoopStatus,
    PromotionAction,
    SelfImprovementLoop,
)


@pytest.mark.asyncio
async def test_fixture_loop_can_auto_promote():
    blocks = FixtureImprovementBlocks()
    loop = SelfImprovementLoop(
        blocks,
        ImprovementLoopPolicy(require_human_approval=False, allow_auto_promote=True),
    )

    result = await loop.run(None, ImprovementLoopRequest(component_name="ks_quality_error"))

    assert result.status == LoopStatus.PROMOTION_READY
    assert result.topic is not None
    assert result.case is not None
    assert result.proposal is not None
    assert result.evaluation is not None
    assert result.evaluation.passed is True
    assert result.decision is not None
    assert result.decision.action == PromotionAction.PROMOTE
    assert blocks.decisions == [result.decision]


@pytest.mark.asyncio
async def test_fixture_loop_requests_approval_by_default():
    loop = SelfImprovementLoop(FixtureImprovementBlocks())

    result = await loop.run(None)

    assert result.status == LoopStatus.NEEDS_APPROVAL
    assert result.decision is not None
    assert result.decision.action == PromotionAction.REQUEST_APPROVAL
    assert result.decision.requires_human is True


@pytest.mark.asyncio
async def test_loop_returns_no_topic_when_filter_does_not_match():
    loop = SelfImprovementLoop(FixtureImprovementBlocks())

    result = await loop.run(None, ImprovementLoopRequest(component_name="missing"))

    assert result.status == LoopStatus.NO_TOPIC
    assert result.topic is None
    assert result.case is None


@pytest.mark.asyncio
async def test_failed_candidate_blocks_promotion():
    blocks = FixtureImprovementBlocks(candidate={"AGNT5_KS_QUALITY_ERROR_FIXED": "0"})
    loop = SelfImprovementLoop(
        blocks,
        ImprovementLoopPolicy(require_human_approval=False, allow_auto_promote=True),
    )

    result = await loop.run(None)

    assert result.status == LoopStatus.EVALUATION_FAILED
    assert result.evaluation == EvaluationResult(
        passed=False,
        pass_rate=0.0,
        failed_items=1,
        experiment_id="fixture-exp-case-fixture-topic-ks-quality-error",
        experiment_run_id="fixture-run-case-fixture-topic-ks-quality-error",
        summary="fixture candidate still fails regression",
        metadata={"source": "fixture_blocks"},
    )
    assert result.decision is not None
    assert result.decision.action == PromotionAction.BLOCK


@pytest.mark.asyncio
async def test_control_plane_blocks_create_evidence_and_wait_for_eval():
    requests: list[tuple[str, str]] = []
    topic = {
        "id": "topic-1",
        "title": "Runtime error in invoice_agent",
        "component_name": "invoice_agent",
        "signal": "runtime_error",
        "suspicion_score": 0.91,
        "impact_score": 0.8,
        "run_count": 3,
    }
    case = {
        "id": "case-1",
        "title": "Runtime error in invoice_agent quality case",
        "observed_behavior": "Runtime error observed across 3 production runs.",
        "expected_behavior": "invoice_agent should complete without runtime errors.",
        "source_run_id": "run-1",
        "metadata": {
            "behavior_topic_id": "topic-1",
            "component_name": "invoice_agent",
        },
    }
    proposal = {
        "id": "proposal-1",
        "quality_case_id": "case-1",
        "title": "Improve invoice_agent",
        "summary": "Prepare a candidate deployment.",
        "candidate_artifact_type": "deployment",
        "candidate_artifact": {"component_name": "invoice_agent"},
    }
    attempt = {
        "id": "attempt-1",
        "quality_case_id": "case-1",
        "status": "draft",
        "pass_rate": None,
        "failed_items": None,
        "latest_experiment_run_id": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        path = request.url.path
        if request.method == "GET" and path.endswith("/projects/project-1/quality/topics"):
            assert request.url.params["component_name"] == "invoice_agent"
            return _api_response([topic])
        if request.method == "GET" and path.endswith("/quality/topics/topic-1/runs"):
            return _api_response([
                {
                    "id": "assignment-1",
                    "run_id": "run-1",
                    "suspicion_score": 0.91,
                    "evidence": {
                        "input": {"invoice_id": "inv-1"},
                        "error": "RuntimeError",
                    },
                }
            ])
        if request.method == "POST" and path.endswith("/quality/topics/topic-1/create-case"):
            return _api_response({"topic": topic, "quality_case": case, "links": []}, status_code=201)
        if request.method == "POST" and path.endswith("/quality/cases/case-1/proposals"):
            return _api_response(proposal, status_code=201)
        if request.method == "POST" and path.endswith("/quality/proposals/proposal-1/create-attempt"):
            return _api_response({"proposal": proposal, "attempt": attempt}, status_code=201)
        if request.method == "POST" and path.endswith("/quality/cases/case-1/events"):
            return _api_response({"id": "event-1", "quality_case_id": "case-1"}, status_code=201)
        return httpx.Response(404, json={"error": f"unexpected {request.method} {path}"})

    client = ImprovementControlPlaneClient(
        "http://control-plane.test/api/v1",
        project_id="project-1",
        token="token",
        transport=httpx.MockTransport(handler),
    )
    blocks = AGNT5ImprovementBlocks(
        client,
        auto_create_regression_dataset=False,
        auto_run_eval=False,
    )
    loop = SelfImprovementLoop(blocks)

    result = await loop.run(None, ImprovementLoopRequest(component_name="invoice_agent"))

    assert result.status == LoopStatus.EVALUATION_PENDING
    assert result.topic is not None
    assert result.topic.representative_runs[0].run_id == "run-1"
    assert result.case is not None
    assert result.case.case_id == "case-1"
    assert result.proposal is not None
    assert result.proposal.proposal_id == "proposal-1"
    assert result.evaluation is not None
    assert result.evaluation.pending is True
    assert result.evaluation.metadata["attempt_id"] == "attempt-1"
    assert result.decision is not None
    assert result.decision.action == PromotionAction.WAIT
    assert ("POST", "/api/v1/projects/project-1/quality/cases/case-1/events") in requests


@pytest.mark.asyncio
async def test_control_plane_blocks_run_eval_and_return_passed_attempt():
    requests: list[tuple[str, str]] = []
    topic = {
        "id": "topic-1",
        "title": "Runtime error in invoice_agent",
        "component_name": "invoice_agent",
        "signal": "runtime_error",
        "suspicion_score": 0.91,
        "impact_score": 0.8,
        "run_count": 3,
    }
    case = {
        "id": "case-1",
        "title": "Runtime error in invoice_agent quality case",
        "observed_behavior": "Runtime error observed across 3 production runs.",
        "expected_behavior": "invoice_agent should complete without runtime errors.",
        "source_run_id": "run-1",
        "metadata": {
            "behavior_topic_id": "topic-1",
            "component_name": "invoice_agent",
            "component_type": "function",
            "deployment_id": "baseline-deployment",
        },
    }
    proposal = {
        "id": "proposal-1",
        "quality_case_id": "case-1",
        "title": "Improve invoice_agent",
        "summary": "Prepare a candidate deployment.",
        "candidate_artifact_type": "deployment",
        "candidate_artifact": {"component_name": "invoice_agent"},
    }
    attempt = {
        "id": "attempt-1",
        "quality_case_id": "case-1",
        "behavior_topic_id": "topic-1",
        "dataset_id": "dataset-1",
        "dataset_version_id": "version-1",
        "baseline_deployment_id": "baseline-deployment",
        "candidate_deployment_id": "candidate-deployment",
        "eval_lane": "isolated_eval_lane",
        "status": "draft",
        "metadata": {},
    }
    experiment = {"id": "experiment-1"}
    run = {
        "id": "run-eval-1",
        "status": "completed",
        "passed": True,
        "pass_rate": 1.0,
        "failed_items": 0,
        "total_items": 1,
    }
    running_attempt = {
        **attempt,
        "status": "running",
        "experiment_id": "experiment-1",
        "latest_experiment_run_id": "run-eval-1",
    }
    passed_attempt = {
        **running_attempt,
        "status": "passed",
        "pass_rate": 1.0,
        "failed_items": 0,
        "total_items": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        path = request.url.path
        if request.method == "GET" and path.endswith("/projects/project-1/quality/topics"):
            return _api_response([topic])
        if request.method == "GET" and path.endswith("/quality/topics/topic-1/runs"):
            return _api_response([
                {
                    "id": "assignment-1",
                    "run_id": "run-1",
                    "suspicion_score": 0.91,
                    "evidence": {"input": {"invoice_id": "inv-1"}},
                }
            ])
        if request.method == "POST" and path.endswith("/quality/topics/topic-1/create-case"):
            return _api_response({"topic": topic, "quality_case": case, "links": []}, status_code=201)
        if request.method == "POST" and path.endswith("/quality/cases/case-1/proposals"):
            return _api_response(proposal, status_code=201)
        if request.method == "GET" and path.endswith("/quality/cases/case-1/links"):
            return _api_response([])
        if request.method == "POST" and path.endswith("/quality/cases/case-1/regression-dataset"):
            return _api_response(
                {
                    "quality_case": case,
                    "source_run_id": "run-1",
                    "regression_dataset": {"id": "dataset-1"},
                    "regression_dataset_version": {"id": "version-1"},
                    "dataset_item": {"id": "item-1"},
                },
                status_code=201,
            )
        if request.method == "POST" and path.endswith("/quality/proposals/proposal-1/create-attempt"):
            return _api_response({"proposal": proposal, "attempt": attempt}, status_code=201)
        if request.method == "POST" and path.endswith("/projects/project-1/experiments"):
            body = request_json(request)
            assert body["dataset_id"] == "dataset-1"
            assert body["dataset_version_id"] == "version-1"
            assert body["deployment_id"] == "candidate-deployment"
            return _api_response(experiment, status_code=201)
        if request.method == "POST" and path.endswith("/experiments/experiment-1/runs"):
            return _api_response({"id": "run-eval-1", "status": "running"}, status_code=201)
        if request.method == "POST" and path.endswith("/quality/cases/case-1/links"):
            return _api_response({"id": "link-1", "quality_case_id": "case-1"}, status_code=201)
        if request.method == "PATCH" and path.endswith("/quality/attempts/attempt-1"):
            body = request_json(request)
            if body.get("status") == "running":
                return _api_response(running_attempt)
            if body.get("status") == "passed":
                return _api_response(passed_attempt)
        if request.method == "GET" and path.endswith("/experiments/experiment-1/runs/run-eval-1"):
            return _api_response(run)
        if request.method == "POST" and path.endswith("/quality/cases/case-1/events"):
            return _api_response({"id": "event-1", "quality_case_id": "case-1"}, status_code=201)
        return httpx.Response(404, json={"error": f"unexpected {request.method} {path}"})

    client = ImprovementControlPlaneClient(
        "http://control-plane.test/api/v1",
        project_id="project-1",
        token="token",
        transport=httpx.MockTransport(handler),
    )
    loop = SelfImprovementLoop(AGNT5ImprovementBlocks(client))

    result = await loop.run(
        None,
        ImprovementLoopRequest(
            component_name="invoice_agent",
            metadata={
                "candidate_deployment_id": "candidate-deployment",
                "wait_for_eval": True,
                "eval_poll_max": 1,
            },
        ),
    )

    assert result.status == LoopStatus.NEEDS_APPROVAL
    assert result.evaluation is not None
    assert result.evaluation.passed is True
    assert result.evaluation.pending is False
    assert result.evaluation.experiment_id == "experiment-1"
    assert result.evaluation.experiment_run_id == "run-eval-1"
    assert result.decision is not None
    assert result.decision.action == PromotionAction.REQUEST_APPROVAL
    assert ("POST", "/api/v1/projects/project-1/experiments") in requests
    assert ("GET", "/api/v1/projects/project-1/experiments/experiment-1/runs/run-eval-1") in requests


def request_json(request: httpx.Request):
    return __import__("json").loads(request.content.decode("utf-8"))


def _api_response(data, status_code: int = 200):
    return httpx.Response(
        status_code,
        json={"success": True, "message": "ok", "data": data},
    )
