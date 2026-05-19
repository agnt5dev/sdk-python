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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .context import Context
from .eval.types import ScorerRequest, ScorerResult

T = TypeVar("T")

# Type alias for scorer handler functions
ScorerHandler = Callable[..., ScorerResult]

# Global scorer registry
_SCORER_REGISTRY: Dict[str, "ScorerConfig"] = {}


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
    """Registry for scorer handlers."""

    @staticmethod
    def register(config: ScorerConfig) -> None:
        """Register a scorer handler.

        Args:
            config: Scorer configuration to register

        Raises:
            ValueError: If a scorer with the same name is already registered
        """
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
    expected_type = config.get(type_key)
    if isinstance(expected_type, str) and expected_type.strip():
        if not _value_type_matches(selected, expected_type):
            actual = _value_type_name(selected)
            raise TypeError(f"{field_key} selected {actual}; expected {expected_type}")
        metadata[type_key] = expected_type.strip()
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
    """Register Python handlers for built-in scorers that need Python execution.

    Built-in scorers like llm_judge fall through the Rust fast path and need
    a Python handler in the ScorerRegistry. This is called once during worker startup.
    """
    global _builtin_handlers_registered
    if _builtin_handlers_registered:
        return
    _builtin_handlers_registered = True

    # Register llm_judge handler
    if "llm_judge" not in _SCORER_REGISTRY:

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

        _SCORER_REGISTRY["llm_judge"] = ScorerConfig(
            name="llm_judge",
            handler=_llm_judge_handler,
            description="LLM-as-judge scorer for semantic evaluation",
            scope="item",
            is_async=True,
        )

    if "correctness" not in _SCORER_REGISTRY:

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
            if expected is None:
                return _config_error(
                    "correctness requires expected output or config.reference_field"
                )

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

        _SCORER_REGISTRY["correctness"] = ScorerConfig(
            name="correctness",
            handler=_correctness_handler,
            description="Managed LLM judge preset for answer correctness",
            scope="item",
            is_async=True,
        )

    if "faithfulness" not in _SCORER_REGISTRY:

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

        _SCORER_REGISTRY["faithfulness"] = ScorerConfig(
            name="faithfulness",
            handler=_faithfulness_handler,
            description="Managed LLM judge preset for faithfulness to configured context",
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
    scorer_config = ScorerRegistry.get(scorer_name)
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
