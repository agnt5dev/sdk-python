"""Git-first prompt manifest resolution.

Production prompt refs resolve from prompt manifests bundled with the deployed
code artifact. The control plane remains a scratchpad for draft/test workflows,
but production must not depend on a mutable environment release pointer.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .lm.types import GenerateRequest, GenerationConfig, Message, MessageRole, PromptRef
from .worker._prompt_executor import _normalize_model, _render_messages

PROMPT_MANIFEST_SCHEMA_VERSION = "agnt5.prompts.v1"
DEFAULT_MANIFEST_FILE = "prompts.lock"
DEFAULT_PROMPT_DIR = "prompts"

_PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production"})


class PromptManifestError(ValueError):
    """Raised when a git-first prompt manifest is required but invalid/missing."""


def resolve_prompt_ref_from_manifest(request: GenerateRequest) -> GenerateRequest | None:
    """Return a local GenerateRequest for a prompt ref, or None for CP fallback."""
    prompt_ref = request.prompt_ref
    if prompt_ref is None:
        return None

    prompt = _resolve_prompt(prompt_ref)
    if prompt is None:
        return None

    messages = [
        _message_from_dict(message)
        for message in _render_messages(prompt.get("messages"), prompt_ref.variables)
    ]
    parameters = _prompt_parameters(prompt)
    config = _merge_config(request.config, parameters)
    response_schema = request.response_schema
    if response_schema is None and prompt.get("response_format") == "json_schema":
        schema = prompt.get("response_schema")
        if isinstance(schema, dict):
            response_schema = json.dumps(schema)

    return replace(
        request,
        model=_normalize_model(prompt.get("model")),
        prompt_ref=None,
        messages=messages,
        system_prompt=None,
        config=config,
        response_schema=response_schema,
    )


def _resolve_prompt(prompt_ref: PromptRef) -> dict[str, Any] | None:
    explicit = _has_explicit_manifest_source()
    manifest_required = explicit or _is_production_prompt_ref(prompt_ref)

    for path in _candidate_paths(prompt_ref.id):
        if not path.exists():
            continue

        data = _load_json(path)
        prompt = _find_prompt(data, prompt_ref.id, prompt_ref.version)
        if prompt is not None:
            return prompt

    if manifest_required:
        version_suffix = f" version {prompt_ref.version!r}" if prompt_ref.version else ""
        raise PromptManifestError(
            f"Prompt {prompt_ref.id!r}{version_suffix} was not found in the bundled "
            "prompt manifest. Production prompt refs must be committed to Git and "
            "included in the deploy artifact."
        )

    return None


def _has_explicit_manifest_source() -> bool:
    return bool(os.environ.get("AGNT5_PROMPT_OVERRIDE") or os.environ.get("AGNT5_PROMPTS_MANIFEST"))


def _is_production_prompt_ref(prompt_ref: PromptRef) -> bool:
    candidates = [
        prompt_ref.environment_ref,
        os.environ.get("AGNT5_ENVIRONMENT"),
        os.environ.get("AGNT5_ENVIRONMENT_REF"),
        os.environ.get("AGNT5_ENV"),
    ]
    return any(
        str(value).strip().lower() in _PRODUCTION_ENVIRONMENTS for value in candidates if value
    )


def _candidate_paths(prompt_id: str) -> Iterable[Path]:
    seen: set[Path] = set()

    def add(path: str | Path | None) -> Iterable[Path]:
        if not path:
            return []
        candidate = Path(path).expanduser()
        paths = _paths_for_source(candidate, prompt_id)
        result = []
        for item in paths:
            resolved = item.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                result.append(item)
        return result

    yield from add(os.environ.get("AGNT5_PROMPT_OVERRIDE"))
    yield from add(os.environ.get("AGNT5_PROMPTS_MANIFEST"))
    yield from add(Path.cwd())


def _paths_for_source(source: Path, prompt_id: str) -> list[Path]:
    if source.suffix or source.name == DEFAULT_MANIFEST_FILE:
        return [source]
    return [
        source / DEFAULT_MANIFEST_FILE,
        source / DEFAULT_PROMPT_DIR / f"{prompt_id}.json",
        source / f"{prompt_id}.json",
    ]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptManifestError(f"Invalid prompt manifest JSON at {path}: {exc}") from exc
    except OSError as exc:
        raise PromptManifestError(f"Failed to read prompt manifest at {path}: {exc}") from exc


def _find_prompt(data: Any, prompt_id: str, version: str | None) -> dict[str, Any] | None:
    for candidate in _iter_prompts(data):
        candidate_id = str(candidate.get("id") or candidate.get("public_id") or "")
        if candidate_id != prompt_id:
            continue
        if version is not None and str(candidate.get("version") or "") != str(version):
            continue
        _validate_prompt(candidate, prompt_id)
        return candidate
    return None


def _iter_prompts(data: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(data, dict):
        return

    prompts = data.get("prompts")
    if isinstance(prompts, list):
        for prompt in prompts:
            if isinstance(prompt, dict):
                yield prompt
    elif isinstance(prompts, dict):
        for prompt_id, prompt in prompts.items():
            if isinstance(prompt, dict):
                yield {"id": prompt_id, **prompt}

    if "id" in data and "messages" in data:
        yield data


def _validate_prompt(prompt: dict[str, Any], prompt_id: str) -> None:
    if not prompt.get("model"):
        raise PromptManifestError(f"Prompt {prompt_id!r} is missing model")
    if not prompt.get("messages"):
        raise PromptManifestError(f"Prompt {prompt_id!r} is missing messages")


def _prompt_parameters(prompt: dict[str, Any]) -> dict[str, Any]:
    parameters = prompt.get("parameters")
    if isinstance(parameters, dict):
        return parameters
    model_config = prompt.get("model_config")
    if isinstance(model_config, dict):
        return model_config
    return {}


def _merge_config(current: GenerationConfig, parameters: dict[str, Any]) -> GenerationConfig:
    return replace(
        current,
        temperature=current.temperature
        if current.temperature is not None
        else _optional_float(parameters.get("temperature")),
        max_tokens=current.max_tokens
        if current.max_tokens is not None
        else _optional_int(parameters.get("max_tokens") or parameters.get("max_output_tokens")),
        top_p=current.top_p
        if current.top_p is not None
        else _optional_float(parameters.get("top_p")),
    )


def _message_from_dict(message: dict[str, str]) -> Message:
    role = MessageRole(message["role"])
    if role == MessageRole.SYSTEM:
        return Message.system(message["content"])
    if role == MessageRole.ASSISTANT:
        return Message.assistant(message["content"])
    return Message.user(message["content"])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
