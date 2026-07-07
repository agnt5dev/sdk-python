"""Composable self-improvement loop primitives for AGNT5 workflows.

The loop is intentionally an SDK-level blueprint. AGNT5 supplies the durable
workflow runtime, observability, eval, and gate primitives; applications supply
agent-specific policies and blocks.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Mapping, Protocol, Sequence

JSONValue = Any
JSONObject = dict[str, JSONValue]


class LoopStatus(str, Enum):
    """Terminal status for one self-improvement loop pass."""

    NO_TOPIC = "no_topic"
    EVALUATION_PENDING = "evaluation_pending"
    EVALUATION_FAILED = "evaluation_failed"
    NEEDS_APPROVAL = "needs_approval"
    PROMOTION_READY = "promotion_ready"


class PromotionAction(str, Enum):
    """Action selected after candidate evaluation."""

    WAIT = "wait"
    BLOCK = "block"
    REQUEST_APPROVAL = "request_approval"
    PROMOTE = "promote"


@dataclass(frozen=True)
class RepresentativeRun:
    """One production run that explains a behavior topic."""

    run_id: str
    input: JSONObject = field(default_factory=dict)
    output: JSONObject | None = None
    error: str | None = None
    score: float | None = None
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorTopic:
    """Recurring behavior pattern discovered from production runs."""

    topic_id: str
    title: str
    component_name: str
    signal: str
    suspicion_score: float
    run_count: int
    representative_runs: tuple[RepresentativeRun, ...] = ()
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class QualityCase:
    """A human-reviewable issue case derived from a behavior topic."""

    case_id: str
    title: str
    observed: str
    expected: str
    topic_id: str
    component_name: str
    representative_run_ids: tuple[str, ...] = ()
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class ImprovementProposal:
    """A proposed agent change with enough context to evaluate it."""

    proposal_id: str
    title: str
    artifact_type: str
    component_name: str
    summary: str
    candidate: JSONObject = field(default_factory=dict)
    case_id: str | None = None
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Offline evaluation result for an improvement proposal."""

    passed: bool
    pass_rate: float
    failed_items: int
    experiment_id: str | None = None
    experiment_run_id: str | None = None
    summary: str = ""
    pending: bool = False
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionDecision:
    """Promotion decision produced by the loop policy."""

    action: PromotionAction
    reason: str
    requires_human: bool = True
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class ImprovementLoopRequest:
    """Inputs that tune one self-improvement loop pass."""

    objective: str = "Improve recurring production behavior."
    topic_id: str | None = None
    case_id: str | None = None
    component_name: str | None = None
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class ImprovementLoopPolicy:
    """Promotion policy for the default loop blueprint."""

    min_pass_rate: float = 1.0
    max_failed_items: int = 0
    require_human_approval: bool = True
    allow_auto_promote: bool = False


@dataclass(frozen=True)
class ImprovementLoopResult:
    """Structured result returned by the loop blueprint."""

    status: LoopStatus
    topic: BehaviorTopic | None = None
    case: QualityCase | None = None
    proposal: ImprovementProposal | None = None
    evaluation: EvaluationResult | None = None
    decision: PromotionDecision | None = None
    metadata: JSONObject = field(default_factory=dict)


class ImprovementBlocks(Protocol):
    """Application-supplied blocks that do agent-specific heavy lifting."""

    def select_topic(
        self,
        request: ImprovementLoopRequest,
    ) -> BehaviorTopic | None | Awaitable[BehaviorTopic | None]:
        """Choose a production behavior topic to improve."""

    def create_case(
        self,
        topic: BehaviorTopic,
        request: ImprovementLoopRequest,
    ) -> QualityCase | Awaitable[QualityCase]:
        """Turn a topic and representative runs into a quality case."""

    def propose_change(
        self,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> ImprovementProposal | Awaitable[ImprovementProposal]:
        """Draft a candidate agent change for the case."""

    def evaluate_change(
        self,
        proposal: ImprovementProposal,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> EvaluationResult | Awaitable[EvaluationResult]:
        """Evaluate the candidate against the case's regression evidence."""

    def record_decision(
        self,
        decision: PromotionDecision,
        proposal: ImprovementProposal,
        evaluation: EvaluationResult,
        request: ImprovementLoopRequest,
    ) -> JSONObject | None | Awaitable[JSONObject | None]:
        """Persist or emit the final decision."""


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class SelfImprovementLoop:
    """Durable workflow blueprint for one self-improvement pass.

    The blueprint owns only orchestration. Developers own the blocks, policy,
    candidate representation, and approval strategy.
    """

    def __init__(
        self,
        blocks: ImprovementBlocks,
        policy: ImprovementLoopPolicy | None = None,
    ) -> None:
        self.blocks = blocks
        self.policy = policy or ImprovementLoopPolicy()

    async def run(
        self,
        ctx: Any,
        request: ImprovementLoopRequest | None = None,
    ) -> ImprovementLoopResult:
        request = request or ImprovementLoopRequest()

        topic = await self._step(
            ctx,
            "improvement.select_topic",
            self.blocks.select_topic,
            request,
        )
        if topic is None:
            return ImprovementLoopResult(status=LoopStatus.NO_TOPIC)

        case = await self._step(
            ctx,
            "improvement.create_case",
            self.blocks.create_case,
            topic,
            request,
        )
        proposal = await self._step(
            ctx,
            "improvement.propose_change",
            self.blocks.propose_change,
            case,
            request,
        )
        evaluation = await self._step(
            ctx,
            "improvement.evaluate_change",
            self.blocks.evaluate_change,
            proposal,
            case,
            request,
        )
        decision = self.decide(evaluation)
        record = await self._step(
            ctx,
            "improvement.record_decision",
            self.blocks.record_decision,
            decision,
            proposal,
            evaluation,
            request,
        )

        if decision.action == PromotionAction.WAIT:
            status = LoopStatus.EVALUATION_PENDING
        elif decision.action == PromotionAction.BLOCK:
            status = LoopStatus.EVALUATION_FAILED
        elif decision.action == PromotionAction.REQUEST_APPROVAL:
            status = LoopStatus.NEEDS_APPROVAL
        else:
            status = LoopStatus.PROMOTION_READY

        metadata: JSONObject = {}
        if isinstance(record, dict):
            metadata["record"] = record

        return ImprovementLoopResult(
            status=status,
            topic=topic,
            case=case,
            proposal=proposal,
            evaluation=evaluation,
            decision=decision,
            metadata=metadata,
        )

    def decide(self, evaluation: EvaluationResult) -> PromotionDecision:
        if evaluation.pending:
            return PromotionDecision(
                action=PromotionAction.WAIT,
                reason="candidate evaluation is pending",
                requires_human=False,
            )
        if not evaluation.passed:
            return PromotionDecision(
                action=PromotionAction.BLOCK,
                reason="candidate failed evaluation",
                requires_human=False,
            )
        if evaluation.pass_rate < self.policy.min_pass_rate:
            return PromotionDecision(
                action=PromotionAction.BLOCK,
                reason=(
                    f"pass rate {evaluation.pass_rate:.2f} below "
                    f"{self.policy.min_pass_rate:.2f}"
                ),
                requires_human=False,
            )
        if evaluation.failed_items > self.policy.max_failed_items:
            return PromotionDecision(
                action=PromotionAction.BLOCK,
                reason=(
                    f"{evaluation.failed_items} failed items above "
                    f"{self.policy.max_failed_items}"
                ),
                requires_human=False,
            )
        if self.policy.require_human_approval or not self.policy.allow_auto_promote:
            return PromotionDecision(
                action=PromotionAction.REQUEST_APPROVAL,
                reason="candidate passed evaluation and needs approval",
                requires_human=True,
            )
        return PromotionDecision(
            action=PromotionAction.PROMOTE,
            reason="candidate passed evaluation and policy allows promotion",
            requires_human=False,
        )

    async def _step(
        self,
        ctx: Any,
        name: str,
        func: Any,
        *args: Any,
    ) -> Any:
        if ctx is not None and hasattr(ctx, "step"):
            return await ctx.step(name, func, *args)
        return await _maybe_await(func(*args))


class FixtureImprovementBlocks:
    """Deterministic blocks for demos, tests, and starter templates."""

    def __init__(
        self,
        topics: Sequence[BehaviorTopic] | None = None,
        candidate: Mapping[str, JSONValue] | None = None,
    ) -> None:
        self.topics = list(topics) if topics is not None else [default_fixture_topic()]
        self.candidate = dict(candidate or {"AGNT5_KS_QUALITY_ERROR_FIXED": "1"})
        self.decisions: list[PromotionDecision] = []

    def select_topic(self, request: ImprovementLoopRequest) -> BehaviorTopic | None:
        if request.topic_id:
            return next((topic for topic in self.topics if topic.topic_id == request.topic_id), None)
        if request.component_name:
            return next(
                (topic for topic in self.topics if topic.component_name == request.component_name),
                None,
            )
        return self.topics[0] if self.topics else None

    def create_case(
        self,
        topic: BehaviorTopic,
        request: ImprovementLoopRequest,
    ) -> QualityCase:
        run_ids = tuple(run.run_id for run in topic.representative_runs)
        return QualityCase(
            case_id=request.case_id or f"case-{topic.topic_id}",
            title=f"{topic.title} quality case",
            observed=f"{topic.title} observed across {topic.run_count} production run(s).",
            expected=(
                f"{topic.component_name} should complete production runs without "
                f"recurring {topic.signal} signals."
            ),
            topic_id=topic.topic_id,
            component_name=topic.component_name,
            representative_run_ids=run_ids,
            metadata={"source": "fixture_blocks"},
        )

    def propose_change(
        self,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> ImprovementProposal:
        candidate = {
            "schema_version": "agnt5.candidate_change.v1",
            "type": "configuration_patch",
            "source": "fixture_blocks",
            "strategy": "candidate_change",
            "component_name": case.component_name,
            "objective": request.objective,
            "hypothesis": (
                f"Enable the fixture recovery path for {case.component_name} "
                "and prove it with the regression dataset."
            ),
            "before": {
                "label": "Observed production behavior",
                "observed_behavior": case.observed,
            },
            "proposed_changes": [
                {
                    "title": "Enable fixture recovery flag",
                    "target": case.component_name,
                    "artifact": "configuration_patch",
                    "operation": "set_env_flag",
                    "before": case.observed,
                    "after": case.expected,
                    "rationale": request.objective,
                }
            ],
            "after": {
                "label": "Expected candidate behavior",
                "expected_behavior": case.expected,
            },
            "evaluation_plan": {
                "offline_lane": "isolated_eval_lane",
                "regression_dataset_required": True,
                "online_comparison_recommended": True,
                "gate_required": True,
            },
            **self.candidate,
        }
        return ImprovementProposal(
            proposal_id=f"proposal-{case.case_id}",
            title=f"Fix {case.component_name} recurring behavior",
            artifact_type="configuration_patch",
            component_name=case.component_name,
            summary="Set the fixture recovery flag before running regression evaluation.",
            candidate=candidate,
            case_id=case.case_id,
            metadata={"objective": request.objective},
        )

    def evaluate_change(
        self,
        proposal: ImprovementProposal,
        case: QualityCase,
        request: ImprovementLoopRequest,
    ) -> EvaluationResult:
        recovered = proposal.candidate.get("AGNT5_KS_QUALITY_ERROR_FIXED") == "1"
        return EvaluationResult(
            passed=recovered,
            pass_rate=1.0 if recovered else 0.0,
            failed_items=0 if recovered else 1,
            experiment_id=f"fixture-exp-{case.case_id}",
            experiment_run_id=f"fixture-run-{case.case_id}",
            summary=(
                "fixture candidate passes regression"
                if recovered
                else "fixture candidate still fails regression"
            ),
            metadata={"source": "fixture_blocks"},
        )

    def record_decision(
        self,
        decision: PromotionDecision,
        proposal: ImprovementProposal,
        evaluation: EvaluationResult,
        request: ImprovementLoopRequest,
    ) -> JSONObject:
        self.decisions.append(decision)
        return {
            "proposal_id": proposal.proposal_id,
            "action": decision.action.value,
            "pass_rate": evaluation.pass_rate,
            "failed_items": evaluation.failed_items,
        }


def default_fixture_topic() -> BehaviorTopic:
    run = RepresentativeRun(
        run_id="fixture-run-ks-quality-error",
        input={"message": "behavior topic fixture error"},
        error="RuntimeError: behavior topic fixture error",
        score=0.85,
    )
    return BehaviorTopic(
        topic_id="fixture-topic-ks-quality-error",
        title="Runtime error in ks_quality_error",
        component_name="ks_quality_error",
        signal="runtime_error",
        suspicion_score=0.85,
        run_count=1,
        representative_runs=(run,),
        metadata={"source": "fixture_blocks"},
    )


__all__ = [
    "BehaviorTopic",
    "EvaluationResult",
    "AGNT5ImprovementBlocks",
    "ImprovementControlPlaneClient",
    "ImprovementControlPlaneError",
    "FixtureImprovementBlocks",
    "ImprovementBlocks",
    "ImprovementLoopPolicy",
    "ImprovementLoopRequest",
    "ImprovementLoopResult",
    "ImprovementProposal",
    "LoopStatus",
    "PromotionAction",
    "PromotionDecision",
    "QualityCase",
    "RepresentativeRun",
    "SelfImprovementLoop",
    "default_fixture_topic",
]

from .control_plane import (  # noqa: E402
    AGNT5ImprovementBlocks,
    ImprovementControlPlaneClient,
    ImprovementControlPlaneError,
)
