"""AGNT5 Evaluation Framework.

This module provides scorers for evaluating AI component outputs.

Built-in scorers (from Rust core):
    - exact_match: Exact string equality check
    - contains: Substring check
    - json_valid: JSON validity check
    - json_schema: Validate output against a JSON Schema
    - numeric_range: Check output is numeric and in [min, max]
    - regex_match: Regex pattern matching
    - levenshtein: Edit distance similarity

Trace assertions (glassbox):
    - TraceAssertion.max_tokens(n): Total tokens <= n
    - TraceAssertion.max_lm_calls(n): LLM calls <= n
    - TraceAssertion.event_sequence([...]): Events in order
    - TraceAssertion.step_memoized(name): Step was cached
    - TraceAssertion.no_errors(): No error events
    - TraceAssertion.duration_under(ms): Duration < ms
    - TraceAssertion.event_count(type, min): Event occurred at least min times

LLM-as-judge:
    For client.eval() - use LLMJudge class:

    >>> from agnt5 import Client
    >>> from agnt5.eval import LLMJudge
    >>> client = Client()
    >>> result = await client.eval(
    ...     component="my_agent",
    ...     input_data={"question": "What is 2+2?"},
    ...     scorers=["exact_match", LLMJudge(criteria="Is the answer correct?")]
    ... )

    For standalone local evaluation - use llm_judge function:

    >>> from agnt5.eval import llm_judge, LLMJudgeConfig
    >>> config = LLMJudgeConfig(criteria="Is the response helpful?")
    >>> result = await llm_judge("The answer is 42.", config)
    >>> print(f"Score: {result.score}, Passed: {result.passed}")

Custom scorers:
    Use the @scorer decorator to create custom Python scorers:

    @scorer(name="my_scorer")
    def check_format(ctx: EvalContext) -> ScorerResultPy:
        valid = is_valid_format(ctx.output)
        return ScorerResultPy(score=1.0 if valid else 0.0, passed=valid)

Example usage:
    >>> from agnt5.eval import exact_match, ScorerInput
    >>> input = ScorerInput(output="hello", expected="hello")
    >>> result = exact_match(input)
    >>> print(f"Score: {result.score}, Passed: {result.passed}")
    Score: 1.0, Passed: True

    >>> from agnt5.eval import TraceAssertion, trace_scorer, ScorerInput
    >>> trace = [
    ...     {"event_type": "run.started", "event_id": "1", "correlation_id": "a", "timestamp_ns": 1000, "data": {}},
    ...     {"event_type": "lm.call.completed", "event_id": "2", "correlation_id": "a", "timestamp_ns": 2000, "data": {"total_tokens": 500}},
    ... ]
    >>> input = ScorerInput(output="result", trace=trace)
    >>> assertions = [TraceAssertion.max_tokens(1000), TraceAssertion.no_errors()]
    >>> result = trace_scorer(input, assertions)
    >>> print(f"Passed: {result.passed}")
    Passed: True
"""

from __future__ import annotations

# Import from Rust core bindings
from .._core import eval as _eval

ScorerInput = _eval.ScorerInput
ScorerResult = _eval.ScorerResult
TraceAssertion = _eval.TraceAssertion
exact_match = _eval.exact_match
contains = _eval.contains
json_valid = _eval.json_valid
json_schema = _eval.json_schema
numeric_range = _eval.numeric_range
regex_match = _eval.regex_match
levenshtein = _eval.levenshtein
trace_scorer = _eval.trace_scorer

# Import Python types and utilities
from .scorer import (
    clear_custom_scorers,
    get_custom_scorer,
    get_scorer_info,
    is_scorer,
    list_custom_scorers,
    scorer,
)
from .types import EvalContext, ScorerRequest, ScorerResult as ScorerResultPy, TraceEvent

# LLM-as-judge
from .llm_judge import (
    Correctness,
    Faithfulness,
    LLMJudge,
    LLMJudgeConfig,
    LLMJudgeResult,
    evaluate_with_criteria,
    llm_judge,
)

__all__ = [
    # Rust core scorers (for ScorerInput-based API)
    "ScorerInput",
    "TraceAssertion",
    "exact_match",
    "contains",
    "json_valid",
    "json_schema",
    "numeric_range",
    "regex_match",
    "levenshtein",
    "trace_scorer",
    # LLM-as-judge
    "LLMJudge",
    "Correctness",
    "Faithfulness",
    "llm_judge",
    "evaluate_with_criteria",
    "LLMJudgeConfig",
    "LLMJudgeResult",
    # Python types for scorer component
    "ScorerRequest",
    "ScorerResultPy",
    "EvalContext",
    "TraceEvent",
    # Custom scorer utilities (legacy API)
    "scorer",
    "get_custom_scorer",
    "list_custom_scorers",
    "clear_custom_scorers",
    "is_scorer",
    "get_scorer_info",
]
