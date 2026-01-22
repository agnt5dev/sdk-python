"""Custom scorer decorator for Python-defined scorers."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Dict, Optional, TypeVar

from .types import EvalContext, ScorerResultPy

# Type for scorer functions
ScorerFunc = Callable[[EvalContext], ScorerResultPy]
F = TypeVar("F", bound=ScorerFunc)

# Global registry for custom scorers
_custom_scorer_registry: Dict[str, ScorerFunc] = {}


def scorer(
    name: Optional[str] = None,
    description: str = "",
) -> Callable[[F], F]:
    """Decorator to create a custom scorer.

    Usage:
        @scorer(name="my_scorer")
        def check_format(ctx: EvalContext) -> ScorerResultPy:
            valid = is_valid_format(ctx.output)
            return ScorerResultPy(score=1.0 if valid else 0.0, passed=valid)

        # Or without explicit name (uses function name)
        @scorer()
        def check_length(ctx: EvalContext) -> ScorerResultPy:
            length = len(str(ctx.output))
            return ScorerResultPy(
                score=min(1.0, length / 100),
                passed=length >= 10,
                explanation=f"Output length: {length}"
            )

    Args:
        name: Scorer name (defaults to function name)
        description: Human-readable description

    Returns:
        Decorated function registered as a scorer
    """

    def decorator(fn: F) -> F:
        scorer_name = name or fn.__name__
        scorer_description = description or fn.__doc__ or ""

        @wraps(fn)
        def wrapper(ctx: EvalContext) -> ScorerResultPy:
            return fn(ctx)

        # Add metadata attributes
        wrapper._scorer_name = scorer_name  # type: ignore
        wrapper._scorer_description = scorer_description  # type: ignore
        wrapper._is_scorer = True  # type: ignore

        # Register in global registry
        _custom_scorer_registry[scorer_name] = wrapper

        return wrapper  # type: ignore

    return decorator


def get_custom_scorer(name: str) -> Optional[ScorerFunc]:
    """Get a registered custom scorer by name.

    Args:
        name: Scorer name

    Returns:
        Scorer function or None if not found
    """
    return _custom_scorer_registry.get(name)


def list_custom_scorers() -> list[str]:
    """List all registered custom scorer names.

    Returns:
        List of scorer names
    """
    return list(_custom_scorer_registry.keys())


def clear_custom_scorers() -> None:
    """Clear all registered custom scorers.

    Useful for testing.
    """
    _custom_scorer_registry.clear()


def is_scorer(fn: Callable) -> bool:
    """Check if a function is a registered scorer.

    Args:
        fn: Function to check

    Returns:
        True if the function was decorated with @scorer
    """
    return getattr(fn, "_is_scorer", False)


def get_scorer_info(fn: Callable) -> Optional[Dict[str, str]]:
    """Get scorer metadata if available.

    Args:
        fn: Scorer function

    Returns:
        Dict with name and description, or None if not a scorer
    """
    if not is_scorer(fn):
        return None
    return {
        "name": getattr(fn, "_scorer_name", fn.__name__),
        "description": getattr(fn, "_scorer_description", ""),
    }
