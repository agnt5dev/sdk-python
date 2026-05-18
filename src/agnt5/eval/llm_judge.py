"""LLM-as-judge scorer for semantic evaluation.

Uses a language model to evaluate outputs based on custom criteria.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMJudge:
    """LLM-as-judge scorer for use with client.eval().

    This class is used to configure an LLM judge scorer when running evaluations
    through the platform. For standalone local evaluation, use llm_judge() function
    or LLMJudgeConfig instead.

    Args:
        criteria: Evaluation criteria/rubric for the judge
        model: Model to use in "provider/model" format (default: "openai/gpt-4o-mini")
        include_input: Include original input in judge prompt (default: False)
        temperature: LLM temperature, 0.0 for deterministic (default: 0.0)

    Example:
        >>> from agnt5 import Client
        >>> from agnt5.eval import LLMJudge
        >>>
        >>> client = Client()
        >>> result = await client.eval(
        ...     component="my_agent",
        ...     input_data={"question": "What is 2+2?"},
        ...     expected="4",
        ...     scorers=[
        ...         "exact_match",
        ...         LLMJudge(
        ...             criteria="Is the answer mathematically correct?",
        ...             model="openai/gpt-4o-mini",
        ...         )
        ...     ]
        ... )
        >>> print(f"Passed: {result.passed}")
    """

    criteria: str
    model: str = "openai/gpt-4o-mini"
    include_input: bool = False
    temperature: float = 0.0

    def to_scorer_spec(self) -> Dict[str, Any]:
        """Serialize to platform scorer spec format.

        Returns:
            Dict with "name" and "config" keys for the platform API.
        """
        # Parse model format "provider/model" -> provider, model
        if "/" in self.model:
            provider, model_name = self.model.split("/", 1)
        else:
            provider = "openai"
            model_name = self.model

        return {
            "name": "llm_judge",
            "config": {
                "provider": provider,
                "model": model_name,
                "criteria": self.criteria,
                "include_input": self.include_input,
                "temperature": self.temperature,
            },
        }


@dataclass
class Correctness:
    """Managed correctness judge preset for client.eval() scorers."""

    model: str = "openai/gpt-4o-mini"
    include_input: bool = True
    temperature: float = 0.0
    answer_field: Optional[str] = None
    reference_field: Optional[str] = None

    def to_scorer_spec(self) -> Dict[str, Any]:
        provider, model_name = _split_provider_model(self.model)
        config: Dict[str, Any] = {
            "provider": provider,
            "model": model_name,
            "include_input": self.include_input,
            "temperature": self.temperature,
        }
        if self.answer_field:
            config["answer_field"] = self.answer_field
        if self.reference_field:
            config["reference_field"] = self.reference_field
        return {"name": "correctness", "config": config}


@dataclass
class Faithfulness:
    """Managed faithfulness judge preset with configured context fields."""

    context_fields: List[str] = field(default_factory=list)
    model: str = "openai/gpt-4o-mini"
    include_input: bool = False
    temperature: float = 0.0
    answer_field: Optional[str] = None

    def to_scorer_spec(self) -> Dict[str, Any]:
        provider, model_name = _split_provider_model(self.model)
        config: Dict[str, Any] = {
            "provider": provider,
            "model": model_name,
            "context_fields": self.context_fields,
            "include_input": self.include_input,
            "temperature": self.temperature,
        }
        if self.answer_field:
            config["answer_field"] = self.answer_field
        return {"name": "faithfulness", "config": config}


def _split_provider_model(value: str) -> tuple[str, str]:
    if "/" in value:
        provider, model_name = value.split("/", 1)
        return provider, model_name
    return "openai", value


# Import here to avoid circular import at module level
def _get_generate():
    from ..lm import generate

    return generate


@dataclass
class LLMJudgeConfig:
    """Configuration for LLM-as-judge scorer.

    Attributes:
        criteria: Evaluation criteria/rubric
        model: Model to use (e.g., "openai/gpt-4o-mini")
        system_prompt: Custom system prompt (optional)
        temperature: Temperature for LLM (default: 0.0 for deterministic)
        include_input: Whether to include original input in prompt
    """

    criteria: str
    model: str = "openai/gpt-4o-mini"
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    include_input: bool = False


@dataclass
class LLMJudgeResult:
    """Result from LLM judge evaluation.

    Attributes:
        score: Score between 0.0 and 1.0
        passed: Whether the score passes (score >= 0.7)
        explanation: LLM's explanation of the evaluation
        label: Optional categorical label
        metadata: Raw LLM response and other metadata
    """

    score: float
    passed: bool
    explanation: Optional[str] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


DEFAULT_SYSTEM_PROMPT = """You are an expert evaluator. Your task is to evaluate the given output based on the provided criteria.

Respond with a JSON object containing:
- "score": a number between 0.0 and 1.0
- "passed": boolean (true if score >= 0.7)
- "explanation": brief explanation of your evaluation

Respond ONLY with the JSON object, no other text."""


async def llm_judge(
    output: Any,
    config: LLMJudgeConfig,
    *,
    expected: Optional[Any] = None,
    input_data: Optional[Any] = None,
    context_data: Optional[Any] = None,
) -> LLMJudgeResult:
    """Evaluate output using an LLM as judge.

    Args:
        output: The output to evaluate
        config: LLMJudgeConfig with criteria and settings
        expected: Expected output for comparison (optional)
        input_data: Original input for context (optional)
        context_data: Retrieved or otherwise supplied context for faithfulness-style judging

    Returns:
        LLMJudgeResult with score, passed status, and explanation

    Example:
        >>> config = LLMJudgeConfig(
        ...     criteria="Is the response helpful and accurate?",
        ...     model="openai/gpt-4o-mini"
        ... )
        >>> result = await llm_judge("The capital of France is Paris.", config)
        >>> print(f"Score: {result.score}, Passed: {result.passed}")
    """
    system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Build user prompt
    user_content = f"## Evaluation Criteria\n{config.criteria}\n\n"

    if config.include_input and input_data is not None:
        input_str = _format_value(input_data)
        user_content += f"## Input\n{input_str}\n\n"

    if context_data is not None:
        context_str = _format_value(context_data)
        user_content += f"## Context\n{context_str}\n\n"

    output_str = _format_value(output)
    user_content += f"## Output to Evaluate\n{output_str}\n\n"

    if expected is not None:
        expected_str = _format_value(expected)
        user_content += f"## Expected Output (Reference)\n{expected_str}\n\n"

    user_content += "Please evaluate the output and respond with a JSON object."

    try:
        generate = _get_generate()
        response = await generate(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=config.temperature,
        )

        return _parse_llm_response(response.text)

    except Exception as e:
        return LLMJudgeResult(
            score=0.0,
            passed=False,
            explanation=f"LLM call failed: {str(e)}",
            metadata={"error": str(e)},
        )


def _format_value(value: Any) -> str:
    """Format a value for inclusion in the prompt."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _parse_llm_response(content: str) -> LLMJudgeResult:
    """Parse the LLM's JSON response into an LLMJudgeResult."""
    json_str = _extract_json(content)

    try:
        data = json.loads(json_str)

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        passed = data.get("passed")
        if passed is None:
            passed = score >= 0.7

        explanation = data.get("explanation")
        label = data.get("label")

        # Strip fields already promoted to top-level attributes so
        # metadata only carries extra/custom keys from the LLM response.
        extra = {
            k: v for k, v in data.items() if k not in ("score", "passed", "explanation", "label")
        }

        return LLMJudgeResult(
            score=score,
            passed=passed,
            explanation=explanation,
            label=label,
            metadata=extra,
        )

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return LLMJudgeResult(
            score=0.0,
            passed=False,
            explanation=f"Could not parse LLM response: {content}",
            label="parse_error",
            metadata={"raw_response": content, "error": str(e)},
        )


def _extract_json(s: str) -> str:
    """Extract JSON object from response text (handles markdown code blocks)."""
    s = s.strip()

    # Handle markdown code blocks
    if s.startswith("```json"):
        match = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif s.startswith("```"):
        match = re.search(r"```\s*(.*?)\s*```", s, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to find JSON object in response
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return s[start : end + 1]

    return s


# Convenience function for simple criteria evaluation
async def evaluate_with_criteria(
    output: Any,
    criteria: str,
    *,
    model: str = "openai/gpt-4o-mini",
    expected: Optional[Any] = None,
) -> LLMJudgeResult:
    """Simple interface to evaluate output against criteria.

    Args:
        output: The output to evaluate
        criteria: Evaluation criteria (e.g., "Is the response helpful?")
        model: Model to use for evaluation
        expected: Expected output for comparison (optional)

    Returns:
        LLMJudgeResult with score and explanation

    Example:
        >>> result = await evaluate_with_criteria(
        ...     output="Paris is the capital of France.",
        ...     criteria="Is the answer factually correct?",
        ... )
        >>> print(f"Passed: {result.passed}")
    """
    config = LLMJudgeConfig(criteria=criteria, model=model)
    return await llm_judge(output, config, expected=expected)
