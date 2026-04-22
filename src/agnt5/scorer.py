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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from .context import Context
from .eval.types import ScorerRequest, ScorerResult, TraceEvent

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
        is_async: Whether the handler is async
        input_schema: JSON schema for config (optional)
    """

    name: str
    handler: ScorerHandler
    description: str = ""
    is_async: bool = False
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

    def log(self, message: str, **extra: Any) -> None:
        """Log with structured data: ctx.log("msg", key=value)"""
        self._logger.info(message, extra=extra)


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
                temperature=config.get("temperature", 0.0),
                include_input=config.get("include_input", False),
            )

            result = await llm_judge(
                output=request.output,
                config=llm_config,
                expected=request.expected,
                input_data=request.input,
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
            is_async=True,
        )


def scorer(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., Any]:
    """Decorator to register a function as an AGNT5 scorer.

    Scorers evaluate component outputs and return a ScorerResult.
    They receive a ScorerContext and ScorerRequest as arguments.

    Args:
        name: Custom scorer name (default: function's __name__)
        description: Human-readable description (default: function's docstring)

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

        # Check if function signature includes context
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        needs_context = bool(params) and params[0].name == "ctx"

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
            is_async=True,  # Always async after wrapping
        )

        ScorerRegistry.register(config)

        # Add metadata to the function
        handler_func._scorer_name = scorer_name  # type: ignore
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
    config = ScorerRegistry.get(scorer_name)
    if config is None:
        raise ValueError(f"Scorer not found: {scorer_name}")

    # Create default context if not provided
    if ctx is None:
        import uuid

        ctx = ScorerContext(
            run_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4()),
            parent_correlation_id="",
        )

    # Call the scorer
    sig = inspect.signature(config.handler)
    params = list(sig.parameters.values())
    needs_context = bool(params) and params[0].name == "ctx"

    if needs_context:
        result = await config.handler(ctx, request)
    else:
        result = await config.handler(request)

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
