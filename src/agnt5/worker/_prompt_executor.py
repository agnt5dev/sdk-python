"""Built-in prompt executor for eval prompt experiments.

Prompt experiments must execute inside the user's worker so provider credentials
stay in the data plane. The control plane freezes prompt/dataset versions and
the runtime dispatches this built-in component with the frozen prompt envelope.
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from ..types import FunctionConfig

PROMPT_EXECUTOR_COMPONENT_NAME = "agnt5_prompt_executor"
PROMPT_EXECUTOR_ALIASES = frozenset({PROMPT_EXECUTOR_COMPONENT_NAME, "run_prompt"})
PROMPT_WORKER_INPUT_SCHEMA_VERSION = "agnt5.eval.prompt_worker_input.v1"

PROMPT_EXECUTOR_METADATA = {
    "source": "agnt5_builtin",
    "agnt5_builtin": "prompt_executor",
    "schema_version": PROMPT_WORKER_INPUT_SCHEMA_VERSION,
}

PROMPT_EXECUTOR_INPUT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "prompt"],
    "properties": {
        "schema_version": {"const": PROMPT_WORKER_INPUT_SCHEMA_VERSION},
        "input": {},
        "variables": {"type": "object"},
        "prompt": {
            "type": "object",
            "required": ["model", "messages"],
            "properties": {
                "model": {"type": "string"},
                "messages": {"type": "array"},
                "parameters": {"type": "object"},
                "response_format": {"type": "string"},
                "response_schema": {"type": "object"},
            },
        },
    },
}

PROMPT_EXECUTOR_OUTPUT_SCHEMA = {
    "description": "The generated model output, parsed from JSON when possible.",
}

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}")

GenerateFn = Callable[..., Awaitable[Any]]


def is_prompt_executor_component(name: str) -> bool:
    """Return True when a component name targets the built-in prompt executor."""
    return name in PROMPT_EXECUTOR_ALIASES


def prompt_executor_config(name: str = PROMPT_EXECUTOR_COMPONENT_NAME) -> FunctionConfig:
    """Build a synthetic function config for the built-in prompt executor."""
    return FunctionConfig(
        name=name,
        handler=_prompt_executor_handler,
        input_schema=PROMPT_EXECUTOR_INPUT_SCHEMA,
        output_schema=PROMPT_EXECUTOR_OUTPUT_SCHEMA,
        metadata=PROMPT_EXECUTOR_METADATA,
    )


async def _prompt_executor_handler(ctx: Any, **payload: Any) -> Any:
    """Function-compatible handler used by Worker._execute_function."""
    return await execute_prompt_worker_input(payload)


async def execute_prompt_worker_input(
    payload: dict[str, Any],
    *,
    generate_fn: GenerateFn | None = None,
) -> Any:
    """Execute a frozen prompt experiment payload in the worker.

    The return value is intentionally just the generated output. Batch eval
    scoring compares the component output directly to the dataset expected
    output, so wrapping this in metadata would break deterministic scorers.
    """
    _validate_payload(payload)

    prompt = payload["prompt"]
    variables = _resolve_variables(payload)
    messages = _render_messages(prompt.get("messages"), variables)
    model = _normalize_model(prompt.get("model"))
    parameters = prompt.get("parameters") or {}
    temperature = _optional_float(parameters.get("temperature"))
    top_p = _optional_float(parameters.get("top_p"))
    if _is_direct_anthropic_model(model) and temperature is not None and top_p is not None:
        top_p = None
    response_format = _response_format(prompt)

    if generate_fn is None:
        from .. import lm

        generate_fn = lm.generate

    response = await generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=_optional_int(parameters.get("max_tokens")),
        top_p=top_p,
        response_format=response_format,
    )

    structured_output = getattr(response, "structured_output", None)
    if structured_output is not None:
        return structured_output

    text = getattr(response, "text", response)
    if not isinstance(text, str):
        return text

    return _parse_json_output(text)


def _validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Prompt executor input must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != PROMPT_WORKER_INPUT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported prompt executor input schema "
            f"{schema_version!r}; expected {PROMPT_WORKER_INPUT_SCHEMA_VERSION!r}"
        )
    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Prompt executor input must include a prompt object")
    if not prompt.get("model"):
        raise ValueError("Prompt executor prompt.model is required")
    if not prompt.get("messages"):
        raise ValueError("Prompt executor prompt.messages is required")


def _resolve_variables(payload: dict[str, Any]) -> dict[str, Any]:
    variables = payload.get("variables")
    if isinstance(variables, dict):
        return variables

    input_value = payload.get("input")
    if isinstance(input_value, dict):
        return input_value

    return {"input": input_value}


def _is_direct_anthropic_model(model: str) -> bool:
    normalized = model.strip().lower()
    provider, separator, _ = normalized.partition("/")
    if separator:
        return provider == "anthropic"
    return normalized.startswith("claude-")


def _render_messages(messages: Any, variables: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Prompt executor prompt.messages must be a non-empty array")

    rendered: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Prompt executor prompt.messages entries must be objects")

        role = str(message.get("role", "")).strip()
        content = _render_value(message.get("content", ""), variables)
        rendered.append({"role": role, "content": content})

    return rendered


def _render_value(value: Any, variables: dict[str, Any]) -> str:
    if isinstance(value, str):
        return _PLACEHOLDER_RE.sub(
            lambda match: _stringify(_lookup_variable(variables, match)), value
        )
    if isinstance(value, (dict, list)):
        rendered = _render_nested(value, variables)
        return json.dumps(rendered, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def _render_nested(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER_RE.sub(
            lambda match: _stringify(_lookup_variable(variables, match)), value
        )
    if isinstance(value, list):
        return [_render_nested(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _render_nested(item, variables) for key, item in value.items()}
    return value


def _lookup_variable(variables: dict[str, Any], match: re.Match[str]) -> Any:
    expression = match.group(1)
    current: Any = variables
    for part in expression.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return match.group(0)
    return current


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _normalize_model(model: Any) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Prompt executor prompt.model must be a non-empty string")

    normalized = model.strip()
    if "/" in normalized:
        return normalized

    lowered = normalized.lower()
    if lowered.startswith(("gpt-", "chatgpt-")) or re.match(r"^o[1-9](?:-|$)", lowered):
        return f"openai/{normalized}"
    if lowered.startswith("claude-"):
        return f"anthropic/{normalized}"
    if lowered.startswith("gemini-"):
        return f"google/{normalized}"
    if lowered.startswith("mistral-"):
        return f"mistral/{normalized}"

    return normalized


def _response_format(prompt: dict[str, Any]) -> Any | None:
    response_format = str(prompt.get("response_format") or "").strip().lower()
    response_schema = prompt.get("response_schema")

    if response_format == "json_schema" and isinstance(response_schema, dict):
        return response_schema

    return None


def _parse_json_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return text

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
