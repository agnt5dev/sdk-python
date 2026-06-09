"""Tests for the built-in worker prompt executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agnt5.worker._core import _is_system_component
from agnt5.worker._prompt_executor import (
    PROMPT_EXECUTOR_COMPONENT_NAME,
    PROMPT_WORKER_INPUT_SCHEMA_VERSION,
    execute_prompt_worker_input,
    is_prompt_executor_component,
)


@dataclass
class FakeLMResponse:
    text: str


def prompt_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": PROMPT_WORKER_INPUT_SCHEMA_VERSION,
        "input": {"limit": 2, "user": {"name": "Arun"}},
        "variables": {"limit": 2, "user": {"name": "Arun"}},
        "prompt": {
            "model": "gpt-5-mini",
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 50,
                "top_p": 0.9,
            },
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Fetch {{limit}} ids for {{user.name}}."},
            ],
            "response_format": "text",
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_prompt_executor_renders_variables_and_parses_json_output() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate(**kwargs: Any) -> FakeLMResponse:
        calls.append(kwargs)
        return FakeLMResponse("[48138136,48140730]")

    result = await execute_prompt_worker_input(prompt_payload(), generate_fn=fake_generate)

    assert result == [48138136, 48140730]
    assert calls == [
        {
            "model": "openai/gpt-5-mini",
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Fetch 2 ids for Arun."},
            ],
            "temperature": 0.2,
            "max_tokens": 50,
            "top_p": 0.9,
            "response_format": None,
        }
    ]


@pytest.mark.asyncio
async def test_prompt_executor_passes_json_schema_response_format() -> None:
    response_schema = {
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
        "required": ["ids"],
    }
    calls: list[dict[str, Any]] = []

    async def fake_generate(**kwargs: Any) -> FakeLMResponse:
        calls.append(kwargs)
        return FakeLMResponse('{"ids":[48138136]}')

    result = await execute_prompt_worker_input(
        prompt_payload(
            prompt={
                "model": "openai/gpt-5-mini",
                "messages": [{"role": "user", "content": "Return ids"}],
                "response_format": "json_schema",
                "response_schema": response_schema,
            }
        ),
        generate_fn=fake_generate,
    )

    assert result == {"ids": [48138136]}
    assert calls[0]["response_format"] == response_schema


@pytest.mark.asyncio
async def test_prompt_executor_omits_top_p_for_anthropic_when_temperature_is_set() -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate(**kwargs: Any) -> FakeLMResponse:
        calls.append(kwargs)
        return FakeLMResponse("ok")

    payload = prompt_payload(
        prompt={
            "model": "claude-sonnet-4-6",
            "parameters": {"temperature": 0.7, "top_p": 1},
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    assert await execute_prompt_worker_input(payload, generate_fn=fake_generate) == "ok"
    assert calls[0]["model"] == "anthropic/claude-sonnet-4-6"
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["top_p"] is None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5-mini", "openai/gpt-5-mini"),
        ("o3-mini", "openai/o3-mini"),
        ("claude-3-5-sonnet-latest", "anthropic/claude-3-5-sonnet-latest"),
        ("gemini-1.5-pro", "google/gemini-1.5-pro"),
        (
            "openrouter/meta-llama/llama-3.1-70b-instruct",
            "openrouter/meta-llama/llama-3.1-70b-instruct",
        ),
    ],
)
@pytest.mark.asyncio
async def test_prompt_executor_normalizes_common_providerless_models(
    model: str,
    expected: str,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_generate(**kwargs: Any) -> FakeLMResponse:
        calls.append(kwargs)
        return FakeLMResponse("ok")

    payload = prompt_payload(
        prompt={
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )

    assert await execute_prompt_worker_input(payload, generate_fn=fake_generate) == "ok"
    assert calls[0]["model"] == expected


@pytest.mark.asyncio
async def test_prompt_executor_rejects_unknown_schema_version() -> None:
    async def fake_generate(**kwargs: Any) -> FakeLMResponse:
        return FakeLMResponse("never called")

    with pytest.raises(ValueError, match="Unsupported prompt executor input schema"):
        await execute_prompt_worker_input(
            prompt_payload(schema_version="agnt5.eval.prompt_worker_input.v0"),
            generate_fn=fake_generate,
        )


def test_prompt_executor_component_aliases() -> None:
    assert is_prompt_executor_component(PROMPT_EXECUTOR_COMPONENT_NAME)
    assert is_prompt_executor_component("run_prompt")
    assert not is_prompt_executor_component("ordinary_user_function")


def test_prompt_executor_component_is_hidden_from_user_summaries() -> None:
    class Component:
        metadata = {"source": "agnt5_builtin", "agnt5_builtin": "prompt_executor"}
        config = {"builtin": "true"}

    assert _is_system_component(Component())


def test_user_component_is_visible_in_user_summaries() -> None:
    class Component:
        metadata = {"description": "ordinary function"}
        config = {}

    assert not _is_system_component(Component())
