"""Scorer component implementation for AGNT5 SDK.

Scorers are a component type for evaluating outputs from other components.
They receive a standardized ScorerRequest and return a ScorerResult.

Example:
    from agnt5 import scorer, ScorerContext, ScorerRequest, ScorerResult

    @scorer(name="format_checker")
    async def format_checker(ctx: ScorerContext, request: ScorerRequest) -> ScorerResult:
        output = str(request.output)
        valid = output.startswith("Result:")
        return ScorerResult(
            score=1.0 if valid else 0.0,
            passed=valid,
            explanation="Output has correct format" if valid else "Missing prefix"
        )
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .context import Context
from .eval.types import ScorerRequest, ScorerResult

T = TypeVar("T")

# Type alias for scorer handler functions
ScorerHandler = Callable[..., ScorerResult]

# Global scorer registry
_SCORER_REGISTRY: Dict[str, "ScorerConfig"] = {}
_BUILTIN_JUDGE_SCORER_REGISTRY: Dict[str, "ScorerConfig"] = {}
BUILTIN_DETERMINISTIC_SCORER_NAMES = (
    "exact_match",
    "contains",
    "regex_match",
    "json_valid",
    "json_schema",
    "numeric_range",
    "levenshtein",
    "tool_called",
    "tool_not_called",
    "tool_sequence",
    "tool_sequence_in_order",
    "tool_sequence_exact",
    "tool_sequence_any_order",
    "tool_trajectory",
    "tool_params_match",
    "max_tool_calls",
    "max_llm_calls",
    "max_tokens",
    "duration_under",
    "no_errors",
    "tool_failure_recovered",
    "step_efficiency",
    "plan_quality",
    "plan_adherence",
    "state_equals",
)
BUILTIN_JUDGE_SCORER_NAMES = (
    "llm_judge",
    "correctness",
    "faithfulness",
    "goal_success",
    "agent_judge",
)
RESERVED_BUILTIN_SCORER_NAMES = (
    BUILTIN_DETERMINISTIC_SCORER_NAMES + BUILTIN_JUDGE_SCORER_NAMES
)


@dataclass
class ScorerConfig:
    """Configuration for a registered scorer.

    Attributes:
        name: Unique scorer name
        handler: The scorer function
        description: Human-readable description
        scope: Evaluation scope: item, run, trace, span, session, or fleet_run
        is_async: Whether the handler is async
        input_schema: JSON schema for config (optional)
    """

    name: str
    handler: ScorerHandler
    description: str = ""
    scope: str = "item"
    is_async: bool = False
    depends_on: Optional[List[str]] = None
    input_schema: Optional[Dict[str, Any]] = None


class ScorerContext(Context):
    """Context for scorer execution.

    Provides access to run metadata and utilities for scorers.
    Scorers are stateless - they receive all data via ScorerRequest.
    """

    def __init__(
        self,
        run_id: str,
        correlation_id: str,
        parent_correlation_id: str,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        is_streaming: bool = False,
        worker: Optional[Any] = None,
        trace_metadata: Optional[dict[str, str]] = None,
        peer_scores: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            run_id,
            correlation_id,
            parent_correlation_id,
            attempt,
            runtime_context,
            is_streaming=is_streaming,
            worker=worker,
            trace_metadata=trace_metadata,
        )
        self._peer_scores = list(peer_scores or [])

    def log(self, message: str, **extra: Any) -> None:
        """Log with structured data: ctx.log("msg", key=value)"""
        self._logger.info(message, extra=extra)

    def peer_scores(self, scorer: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return scores already produced by earlier scorers for this item."""
        if scorer is None:
            return list(self._peer_scores)
        return [
            score
            for score in self._peer_scores
            if score.get("scorer") == scorer
            or score.get("scorer_name") == scorer
            or score.get("name") == scorer
        ]


class ScorerRegistry:
    """Registry for user-defined custom scorer handlers."""

    @staticmethod
    def register(config: ScorerConfig) -> None:
        """Register a scorer handler.

        Args:
            config: Scorer configuration to register

        Raises:
            ValueError: If a scorer with the same name is already registered
        """
        if config.name in RESERVED_BUILTIN_SCORER_NAMES:
            raise ValueError(
                f"Scorer name collision: '{config.name}' is an AGNT5 built-in scorer. "
                "Built-in scorers are not user components; use a different custom scorer name."
            )

        if config.name in _SCORER_REGISTRY:
            existing_config = _SCORER_REGISTRY[config.name]
            existing_module = existing_config.handler.__module__
            new_module = config.handler.__module__

            raise ValueError(
                f"Scorer name collision: '{config.name}' is already registered.\n"
                f"  Existing: {existing_module}.{existing_config.handler.__name__}\n"
                f"  New:      {new_module}.{config.handler.__name__}\n"
                f"Please use a different scorer name or use name= parameter."
            )

        _SCORER_REGISTRY[config.name] = config

    @staticmethod
    def get(name: str) -> Optional[ScorerConfig]:
        """Get scorer configuration by name."""
        return _SCORER_REGISTRY.get(name)

    @staticmethod
    def all() -> Dict[str, ScorerConfig]:
        """Get all registered scorers."""
        return _SCORER_REGISTRY.copy()

    @staticmethod
    def clear() -> None:
        """Clear all registered scorers."""
        _SCORER_REGISTRY.clear()

    @staticmethod
    def list_names() -> List[str]:
        """Get list of all registered scorer names."""
        return list(_SCORER_REGISTRY.keys())


_builtin_handlers_registered = False


def get_builtin_judge_scorer_config(name: str) -> Optional[ScorerConfig]:
    """Get an AGNT5-owned built-in judge scorer by name."""
    register_builtin_scorer_handlers()
    return _BUILTIN_JUDGE_SCORER_REGISTRY.get(name)


def all_builtin_judge_scorers() -> Dict[str, ScorerConfig]:
    """Get all AGNT5-owned built-in judge scorers."""
    register_builtin_scorer_handlers()
    return _BUILTIN_JUDGE_SCORER_REGISTRY.copy()

CORRECTNESS_JUDGE_CRITERIA = (
    "Evaluate whether the output correctly answers the input and matches the expected "
    "output. Score 1.0 for fully correct answers, 0.5 for partially correct answers, "
    "and 0.0 for incorrect or unsupported answers."
)

FAITHFULNESS_JUDGE_CRITERIA = (
    "Evaluate whether the output is faithful to the provided context. Penalize claims "
    "that are unsupported, contradicted by context, or omit critical context needed for "
    "the answer."
)

GOAL_SUCCESS_JUDGE_CRITERIA = (
    "Evaluate whether the overall session achieved the user's goal. Use available "
    "trace-eval context, journal events, session state, input, output, and expected "
    "result when provided. Penalize incomplete task completion, missing required "
    "actions, tool failures that affected the outcome, and unsupported success claims."
)

AGENT_JUDGE_DEFAULT_CRITERIA = (
    "Investigate the provided evidence before scoring. Check factual correctness, "
    "grounding in the trace and tool evidence, appropriate tool usage, and whether the "
    "final output is supported by the observed execution. Penalize unsupported claims, "
    "missing evidence, tool misuse, and reasoning that conflicts with the trace."
)

AGENT_JUDGE_SYSTEM_PROMPT = (
    "You are an AGNT5 agent-as-a-judge evaluator. Inspect the structured evidence, "
    "trace-eval context, tool-call trajectory, peer scores, and task input before "
    "returning a verdict. Do not assume facts that are not present in the provided "
    "evidence. If evidence is missing or inconclusive, lower the score and explain the "
    "gap. Return only the requested JSON verdict."
)


def _judge_model(config: Dict[str, Any]) -> str:
    provider = str(config.get("provider") or "openai")
    model = str(config.get("model") or "gpt-4o-mini")
    if "/" in model and "provider" not in config:
        return model
    return f"{provider}/{model}"


def _judge_temperature(config: Dict[str, Any]) -> float:
    value = config.get("temperature", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _judge_include_input(config: Dict[str, Any], default: bool = False) -> bool:
    value = config.get("include_input")
    if value is None:
        return default
    return bool(value)


def _judge_choice_scores(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    raw = config.get("choice_scores")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    scores: Dict[str, float] = {}
    for label, score in raw.items():
        if not isinstance(label, str) or not isinstance(score, (int, float)):
            return None
        scores[label] = float(score)
    return scores


def _selector_value(request: Any, selector: str) -> Any:
    root, sep, rest = selector.strip().partition(".")
    if sep == "" or not rest:
        raise KeyError(selector)
    roots = {
        "input": request.input,
        "output": request.output,
        "expected": request.expected,
    }
    if root not in roots:
        raise KeyError(selector)
    value = roots[root]
    for part in rest.split("."):
        if isinstance(value, dict):
            if part not in value:
                raise KeyError(selector)
            value = value[part]
            continue
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError):
                raise KeyError(selector) from None
            continue
        raise KeyError(selector)
    return value


def _optional_selector_value(request: Any, config: Dict[str, Any], key: str, fallback: Any) -> Any:
    selector = config.get(key)
    if not selector:
        return fallback
    if not isinstance(selector, str):
        raise KeyError(key)
    return _selector_value(request, selector)


def _faithfulness_context_fields(config: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    context_field = config.get("context_field")
    if isinstance(context_field, str) and context_field.strip():
        fields.append(context_field.strip())
    context_fields = config.get("context_fields")
    if isinstance(context_fields, list):
        fields.extend(
            field.strip() for field in context_fields if isinstance(field, str) and field.strip()
        )
    return fields


def _config_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key)
    return value if isinstance(value, bool) else default


def _config_string_list(config: Dict[str, Any], *keys: str) -> List[str]:
    values: List[str] = []
    for key in keys:
        raw = config.get(key)
        if isinstance(raw, str) and raw.strip():
            values.append(raw.strip())
        elif isinstance(raw, list):
            values.extend(item.strip() for item in raw if isinstance(item, str) and item.strip())
    return values


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _truncate_agent_judge_evidence(evidence: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    try:
        encoded = json.dumps(evidence, default=str, sort_keys=True)
    except TypeError:
        evidence = _to_jsonable(evidence)
        encoded = json.dumps(evidence, default=str, sort_keys=True)
    if len(encoded) <= max_chars:
        return evidence
    return {
        "truncated": True,
        "max_chars": max_chars,
        "evidence_excerpt": encoded[:max_chars],
    }


def _goal_success_evidence(
    request: ScorerRequest, config: Dict[str, Any]
) -> tuple[Dict[str, Any], List[str]]:
    evidence: Dict[str, Any] = {}
    provided_context = (
        config["context_data"]
        if "context_data" in config and config["context_data"] is not None
        else config.get("context")
    )
    if provided_context is not None:
        evidence["provided_context"] = _to_jsonable(provided_context)
    trace_eval_context = (
        request.trace_eval_context
        if request.trace_eval_context is not None
        else config.get("trace_eval_context")
    )
    if trace_eval_context is not None:
        evidence["trace_eval_context"] = _to_jsonable(trace_eval_context)
    if _config_bool(config, "include_trace", True) and request.trace:
        evidence["trace"] = [_to_jsonable(event) for event in request.trace]
    if request.peer_scores:
        evidence["peer_scores"] = _to_jsonable(request.peer_scores)
    for key in ("session_fields", "journal_event_fields"):
        values = _config_string_list(config, key)
        if values:
            evidence[key] = values
    sources = sorted(evidence.keys())
    return evidence, sources


def _agent_judge_evidence(request: ScorerRequest, config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    evidence: Dict[str, Any] = {}

    provided_context = (
        config["context_data"]
        if "context_data" in config and config["context_data"] is not None
        else config.get("context")
    )
    if provided_context is not None:
        evidence["provided_context"] = _to_jsonable(provided_context)

    if _config_bool(config, "include_trace_eval_context", True):
        trace_eval_context = (
            request.trace_eval_context
            if request.trace_eval_context is not None
            else config.get("trace_eval_context")
        )
        if trace_eval_context is not None:
            evidence["trace_eval_context"] = _to_jsonable(trace_eval_context)

    if _config_bool(config, "include_trace", False) and request.trace:
        evidence["trace"] = _to_jsonable(request.trace)

    if _config_bool(config, "include_tool_calls", True):
        tool_calls = request.get_tool_calls()
        if tool_calls:
            evidence["tool_calls"] = _to_jsonable(tool_calls)

    if _config_bool(config, "include_peer_scores", True) and request.peer_scores:
        evidence["peer_scores"] = _to_jsonable(request.peer_scores)

    allowed_tools = _config_string_list(config, "allowed_tools", "tools")
    if allowed_tools:
        evidence["allowed_tools"] = allowed_tools

    raw_max_chars = config.get("max_evidence_chars", 20000)
    max_chars = int(raw_max_chars) if isinstance(raw_max_chars, (int, float)) else 20000
    max_chars = max(1000, min(max_chars, 200000))

    sources = sorted(evidence.keys())
    return _truncate_agent_judge_evidence(evidence, max_chars), sources


def _config_error(message: str) -> "ScorerResult":
    from .eval.types import ScorerResult

    return ScorerResult(score=0.0, passed=False, label="config_error", explanation=message)


def _bound_field_value(value: Any, selector: str, root: str) -> Any:
    selector = selector.strip()
    prefix = f"{root}."
    if selector == root:
        return value
    if selector.startswith(prefix):
        selector = selector[len(prefix) :]
    if not selector:
        return value
    current = value
    for part in selector.split("."):
        if not part:
            raise KeyError(selector)
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(selector)
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise KeyError(selector) from None
            continue
        raise KeyError(selector)
    return current


def _value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _value_type_matches(value: Any, expected_type: str) -> bool:
    expected_type = expected_type.strip().lower()
    return (
        (expected_type == "null" and value is None)
        or (expected_type in {"boolean", "bool"} and isinstance(value, bool))
        or (
            expected_type == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        or (expected_type == "string" and isinstance(value, str))
        or (expected_type == "array" and isinstance(value, list))
        or (expected_type == "object" and isinstance(value, dict))
    )


def _field_binding_expected_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    expected_type = value.strip().lower()
    if expected_type in {"score", "classification", "json"}:
        return None
    return expected_type if expected_type else None


def _bind_request_field(
    config: Dict[str, Any],
    root: str,
    field_key: str,
    type_key: str,
    value: Any,
    metadata: Dict[str, Any],
) -> Any:
    selected = value
    selector = config.get(field_key)
    if isinstance(selector, str) and selector.strip():
        try:
            selected = _bound_field_value(value, selector, root)
        except KeyError:
            raise KeyError(f"{field_key} {selector!r} was not found") from None
        metadata[field_key] = selector.strip()
    expected_type = _field_binding_expected_type(config.get(type_key))
    if expected_type:
        if not _value_type_matches(selected, expected_type):
            actual = _value_type_name(selected)
            raise TypeError(f"{field_key} selected {actual}; expected {expected_type}")
        metadata[type_key] = expected_type
    return selected


def _has_field_binding(config: Dict[str, Any], field_key: str, type_key: str) -> bool:
    return config.get(field_key) is not None or config.get(type_key) is not None


def _apply_scorer_field_bindings(request: ScorerRequest) -> tuple[ScorerRequest, Dict[str, Any]]:
    config = request.config or {}
    metadata: Dict[str, Any] = {}
    output = _bind_request_field(
        config, "output", "output_field", "output_type", request.output, metadata
    )
    expected = (
        _bind_request_field(
            config,
            "expected",
            "expected_field",
            "expected_type",
            request.expected,
            metadata,
        )
        if request.expected is not None
        or _has_field_binding(config, "expected_field", "expected_type")
        else request.expected
    )
    input_value = (
        _bind_request_field(config, "input", "input_field", "input_type", request.input, metadata)
        if request.input is not None or _has_field_binding(config, "input_field", "input_type")
        else request.input
    )
    return (
        ScorerRequest(
            output=output,
            expected=expected,
            input=input_value,
            trace=request.trace,
            config=request.config,
            peer_scores=request.peer_scores,
            trace_eval_context=request.trace_eval_context,
        ),
        metadata,
    )


def _judge_result_to_scorer_result(result: Any, metadata: Dict[str, Any]) -> Any:
    from .eval.types import ScorerResult

    merged_metadata = dict(result.metadata or {})
    merged_metadata.update(metadata)
    return ScorerResult(
        score=result.score,
        passed=result.passed,
        label=result.label,
        explanation=result.explanation,
        metadata=merged_metadata,
    )


def register_builtin_scorer_handlers() -> None:
    """Register AGNT5-owned built-in judge scorers.

    These are not custom scorer components and do not live in ScorerRegistry.
    Worker dispatch resolves them before custom scorer lookup.
    """
    global _builtin_handlers_registered
    if _builtin_handlers_registered:
        return
    _builtin_handlers_registered = True

    if "llm_judge" not in _BUILTIN_JUDGE_SCORER_REGISTRY:

        async def _llm_judge_handler(ctx: "ScorerContext", request: Any) -> Any:
            from .eval.llm_judge import LLMJudgeConfig, llm_judge
            from .eval.types import ScorerResult

            config = request.config or {}
            provider = config.get("provider", "openai")
            model = config.get("model", "gpt-4o-mini")

            llm_config = LLMJudgeConfig(
                criteria=config.get("criteria", ""),
                model=f"{provider}/{model}",
                prompt_template=config.get("prompt_template"),
                system_prompt=config.get("system_prompt"),
                temperature=config.get("temperature", 0.0),
                include_input=config.get("include_input", False),
                choice_scores=_judge_choice_scores(config),
                use_cot=bool(config.get("use_cot", False)),
                output_schema=config.get("output_schema")
                if isinstance(config.get("output_schema"), dict)
                else None,
                metadata=config.get("metadata") if isinstance(config.get("metadata"), dict) else None,
                tags=config.get("tags") if isinstance(config.get("tags"), list) else None,
            )

            result = await llm_judge(
                output=request.output,
                config=llm_config,
                expected=request.expected,
                input_data=request.input,
                context_data=config.get("context_data") or config.get("context"),
            )

            return ScorerResult(
                score=result.score,
                passed=result.passed,
                label=result.label,
                explanation=result.explanation,
                metadata=result.metadata,
            )

        _BUILTIN_JUDGE_SCORER_REGISTRY["llm_judge"] = ScorerConfig(
            name="llm_judge",
            handler=_llm_judge_handler,
            description="LLM-as-judge scorer for semantic evaluation",
            scope="item",
            is_async=True,
        )

    if "correctness" not in _BUILTIN_JUDGE_SCORER_REGISTRY:

        async def _correctness_handler(ctx: "ScorerContext", request: Any) -> Any:
            from .eval.llm_judge import LLMJudgeConfig, llm_judge

            config = request.config or {}
            try:
                output = _optional_selector_value(request, config, "answer_field", request.output)
                expected = _optional_selector_value(
                    request, config, "reference_field", request.expected
                )
            except KeyError as e:
                return _config_error(f"correctness field selector not found: {e.args[0]}")
            result = await llm_judge(
                output=output,
                config=LLMJudgeConfig(
                    criteria=CORRECTNESS_JUDGE_CRITERIA,
                    model=_judge_model(config),
                    temperature=_judge_temperature(config),
                    include_input=_judge_include_input(config, True),
                ),
                expected=expected,
                input_data=request.input,
            )
            return _judge_result_to_scorer_result(result, {"judge_preset": "correctness"})

        _BUILTIN_JUDGE_SCORER_REGISTRY["correctness"] = ScorerConfig(
            name="correctness",
            handler=_correctness_handler,
            description="Managed LLM judge preset for answer correctness",
            scope="item",
            is_async=True,
        )

    if "faithfulness" not in _BUILTIN_JUDGE_SCORER_REGISTRY:

        async def _faithfulness_handler(ctx: "ScorerContext", request: Any) -> Any:
            from .eval.llm_judge import LLMJudgeConfig, llm_judge

            config = request.config or {}
            fields = _faithfulness_context_fields(config)
            if not fields:
                return _config_error(
                    "faithfulness requires config.context_fields or config.context_field"
                )
            context_values: Dict[str, Any] = {}
            try:
                output = _optional_selector_value(request, config, "answer_field", request.output)
                for field in fields:
                    context_values[field] = _selector_value(request, field)
            except KeyError as e:
                return _config_error(f"faithfulness field selector not found: {e.args[0]}")

            result = await llm_judge(
                output=output,
                config=LLMJudgeConfig(
                    criteria=FAITHFULNESS_JUDGE_CRITERIA,
                    model=_judge_model(config),
                    temperature=_judge_temperature(config),
                    include_input=_judge_include_input(config, False),
                ),
                input_data=request.input,
                context_data=context_values,
            )
            return _judge_result_to_scorer_result(
                result,
                {"judge_preset": "faithfulness", "context_fields": fields},
            )

        _BUILTIN_JUDGE_SCORER_REGISTRY["faithfulness"] = ScorerConfig(
            name="faithfulness",
            handler=_faithfulness_handler,
            description="Managed LLM judge preset for faithfulness to configured context",
            scope="item",
            is_async=True,
        )

    if "goal_success" not in _BUILTIN_JUDGE_SCORER_REGISTRY:

        async def _goal_success_handler(ctx: "ScorerContext", request: Any) -> Any:
            from .eval.llm_judge import LLMJudgeConfig, llm_judge

            config = request.config or {}
            evidence, evidence_sources = _goal_success_evidence(request, config)
            result = await llm_judge(
                output=request.output,
                config=LLMJudgeConfig(
                    criteria=GOAL_SUCCESS_JUDGE_CRITERIA,
                    model=_judge_model(config),
                    temperature=_judge_temperature(config),
                    include_input=_judge_include_input(config, True),
                ),
                expected=request.expected,
                input_data=request.input,
                context_data={"goal_success_evidence": evidence} if evidence else None,
            )
            return _judge_result_to_scorer_result(
                result,
                {
                    "judge_preset": "goal_success",
                    "evidence_sources": evidence_sources,
                },
            )

        _BUILTIN_JUDGE_SCORER_REGISTRY["goal_success"] = ScorerConfig(
            name="goal_success",
            handler=_goal_success_handler,
            description="Managed LLM judge preset for session goal success",
            scope="session",
            is_async=True,
        )

    if "agent_judge" not in _BUILTIN_JUDGE_SCORER_REGISTRY:

        async def _agent_judge_handler(ctx: "ScorerContext", request: Any) -> Any:
            from .eval.llm_judge import LLMJudgeConfig, llm_judge

            config = request.config or {}
            evidence, evidence_sources = _agent_judge_evidence(request, config)
            criteria = config.get("criteria")
            if not isinstance(criteria, str) or not criteria.strip():
                criteria = AGENT_JUDGE_DEFAULT_CRITERIA

            result = await llm_judge(
                output=request.output,
                config=LLMJudgeConfig(
                    criteria=criteria,
                    model=_judge_model(config),
                    prompt_template=config.get("prompt_template"),
                    system_prompt=config.get("system_prompt") or AGENT_JUDGE_SYSTEM_PROMPT,
                    temperature=_judge_temperature(config),
                    include_input=_judge_include_input(config, True),
                    choice_scores=_judge_choice_scores(config),
                    use_cot=bool(config.get("use_cot", False)),
                    output_schema=config.get("output_schema")
                    if isinstance(config.get("output_schema"), dict)
                    else None,
                    metadata=config.get("metadata")
                    if isinstance(config.get("metadata"), dict)
                    else None,
                    tags=config.get("tags") if isinstance(config.get("tags"), list) else None,
                ),
                expected=request.expected,
                input_data=request.input,
                context_data={"agent_judge_evidence": evidence},
            )
            return _judge_result_to_scorer_result(
                result,
                {
                    "judge_preset": "agent_judge",
                    "judge_mode": "evidence_inspection",
                    "agent_judge_version": "evidence_inspection_v1",
                    "evidence_sources": evidence_sources,
                },
            )

        _BUILTIN_JUDGE_SCORER_REGISTRY["agent_judge"] = ScorerConfig(
            name="agent_judge",
            handler=_agent_judge_handler,
            description="Managed agent-as-a-judge scorer over trace and tool evidence",
            scope="item",
            is_async=True,
        )


def scorer(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    scope: str = "item",
    depends_on: Optional[List[str]] = None,
) -> Callable[..., Any]:
    """Decorator to register a function as an AGNT5 scorer.

    Scorers evaluate component outputs and return a ScorerResult.
    They receive a ScorerContext and ScorerRequest as arguments.

    Args:
        name: Custom scorer name (default: function's __name__)
        description: Human-readable description (default: function's docstring)
        scope: Evaluation scope. Defaults to item.
        depends_on: Scorer names that should run before this scorer.

    Example:
        @scorer
        async def my_scorer(ctx: ScorerContext, request: ScorerRequest) -> ScorerResult:
            return ScorerResult(score=1.0, passed=True)

        @scorer(name="custom_name", description="Checks output format")
        async def format_check(ctx: ScorerContext, request: ScorerRequest) -> ScorerResult:
            valid = request.output.startswith("OK:")
            return ScorerResult(score=1.0 if valid else 0.0, passed=valid)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        scorer_name = name or func.__name__
        scorer_description = description or func.__doc__ or ""

        # Determine if async
        is_async = inspect.iscoroutinefunction(func)

        # Wrap sync functions to be async
        if is_async:
            handler_func = func
        else:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

            handler_func = async_wrapper

        # Create config and register
        config = ScorerConfig(
            name=scorer_name,
            handler=handler_func,
            description=scorer_description.strip() if scorer_description else "",
            scope=scope,
            is_async=True,  # Always async after wrapping
            depends_on=list(depends_on or []),
        )

        ScorerRegistry.register(config)

        # Add metadata to the function
        handler_func._scorer_name = scorer_name  # type: ignore
        handler_func._scorer_scope = scope  # type: ignore
        handler_func._scorer_depends_on = list(depends_on or [])  # type: ignore
        handler_func._scorer_config = config  # type: ignore
        handler_func._is_scorer = True  # type: ignore

        return handler_func

    # Handle both @scorer and @scorer() usage
    if _func is not None:
        return decorator(_func)
    return decorator


async def run_scorer(
    scorer_name: str,
    request: ScorerRequest,
    ctx: Optional[ScorerContext] = None,
) -> ScorerResult:
    """Run a registered scorer by name.

    Args:
        scorer_name: Name of the registered scorer
        request: ScorerRequest with output, expected, trace, etc.
        ctx: Optional ScorerContext (created if not provided)

    Returns:
        ScorerResult from the scorer

    Raises:
        ValueError: If scorer is not found
    """
    scorer_config = get_builtin_judge_scorer_config(scorer_name) or ScorerRegistry.get(
        scorer_name
    )
    if scorer_config is None:
        raise ValueError(f"Scorer not found: {scorer_name}")
    try:
        request, binding_metadata = _apply_scorer_field_bindings(request)
    except (KeyError, TypeError) as e:
        return _config_error(f"{scorer_name} field binding error: {e}")

    # Create default context if not provided
    if ctx is None:
        import uuid

        ctx = ScorerContext(
            run_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            parent_correlation_id="",
            peer_scores=request.peer_scores,
        )

    # Call the scorer
    sig = inspect.signature(scorer_config.handler)
    params = list(sig.parameters.values())
    needs_context = bool(params) and params[0].name == "ctx"

    if needs_context:
        result = await scorer_config.handler(ctx, request)
    else:
        result = await scorer_config.handler(request)

    if binding_metadata:
        result.metadata = dict(result.metadata or {})
        result.metadata.update(binding_metadata)

    return result


def is_scorer(func: Callable) -> bool:
    """Check if a function is a registered scorer.

    Args:
        func: Function to check

    Returns:
        True if the function was decorated with @scorer
    """
    return getattr(func, "_is_scorer", False)


def get_scorer_config(func: Callable) -> Optional[ScorerConfig]:
    """Get scorer config from a decorated function.

    Args:
        func: Scorer function

    Returns:
        ScorerConfig or None if not a scorer
    """
    return getattr(func, "_scorer_config", None)
