"""Git-first prompt manifest resolution.

Production prompts resolve from prompt manifests bundled with the deployed
code artifact. The control plane remains a scratchpad for draft/test workflows,
but production must not depend on a mutable environment release pointer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .lm.types import GenerateRequest, GenerationConfig, Message, MessageRole, Prompt
from .worker._prompt_executor import _normalize_model, _render_messages

PROMPT_MANIFEST_SCHEMA_VERSION = "agnt5.prompts.v1"
DEFAULT_PROMPT_DIR = "prompts"

_PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production"})
_MDX_EXTENSIONS = frozenset({".md", ".mdx"})
_MDX_ROLE_BLOCK_RE = re.compile(
    r"<(System|User|Assistant)>\s*(.*?)\s*</\1>", re.IGNORECASE | re.DOTALL
)
_FRONTMATTER_PARAMETER_KEYS = frozenset(
    {
        "temperature",
        "max_tokens",
        "max_output_tokens",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop_sequences",
    }
)


class PromptManifestError(ValueError):
    """Raised when a git-first prompt manifest is required but invalid/missing."""


def resolve_prompt_from_manifest(request: GenerateRequest) -> GenerateRequest | None:
    """Return a local GenerateRequest for a Prompt, or None for CP fallback."""
    prompt = request.prompt_ref
    if prompt is None:
        return None

    manifest_prompt = _resolve_prompt(prompt)
    if manifest_prompt is None:
        return None

    messages = [
        _message_from_dict(message)
        for message in _render_messages(manifest_prompt.get("messages"), prompt.variables)
    ]
    parameters = _prompt_parameters(manifest_prompt)
    config = _merge_config(request.config, parameters, prompt)
    response_schema = request.response_schema
    if response_schema is None and manifest_prompt.get("response_format") == "json_schema":
        schema = manifest_prompt.get("response_schema")
        if isinstance(schema, dict):
            response_schema = json.dumps(schema)

    return replace(
        request,
        model=_normalize_model(prompt.model or manifest_prompt.get("model")),
        prompt_ref=None,
        messages=messages,
        system_prompt=None,
        config=config,
        response_schema=response_schema,
    )


resolve_prompt_ref_from_manifest = resolve_prompt_from_manifest


def _resolve_prompt(prompt: Prompt) -> dict[str, Any] | None:
    explicit = _has_explicit_manifest_source()
    manifest_required = explicit or _is_production_prompt(prompt)

    for path in _candidate_paths(prompt.id):
        if not path.exists():
            continue

        data = _load_prompt_source(path)
        manifest_prompt = _find_prompt(data, prompt.id, prompt.version)
        if manifest_prompt is not None:
            return manifest_prompt

    if manifest_required:
        version_suffix = f" version {prompt.version!r}" if prompt.version else ""
        raise PromptManifestError(
            f"Prompt {prompt.id!r}{version_suffix} was not found in the bundled "
            "prompt files. Production prompts must be committed to Git and "
            "included in the deploy artifact."
        )

    return None


def _has_explicit_manifest_source() -> bool:
    return bool(os.environ.get("AGNT5_PROMPT_OVERRIDE") or os.environ.get("AGNT5_PROMPTS_MANIFEST"))


def _is_production_prompt(prompt: Prompt) -> bool:
    candidates = [
        prompt.environment_ref,
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
    if source.suffix:
        if source.suffix.lower() not in _MDX_EXTENSIONS:
            raise PromptManifestError(
                f"Prompt source {source} must be a .md or .mdx file"
            )
        return [source]
    return [
        source / DEFAULT_PROMPT_DIR / f"{prompt_id}.mdx",
        source / DEFAULT_PROMPT_DIR / f"{prompt_id}.md",
        source / f"{prompt_id}.mdx",
        source / f"{prompt_id}.md",
    ]


def _load_prompt_source(path: Path) -> Any:
    if path.suffix.lower() not in _MDX_EXTENSIONS:
        raise PromptManifestError(f"Prompt source {path} must be a .md or .mdx file")
    return _load_mdx_prompt(path)


def _load_mdx_prompt(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptManifestError(f"Failed to read prompt manifest at {path}: {exc}") from exc

    frontmatter, body = _split_mdx_frontmatter(source, path)
    prompt = _normalize_mdx_prompt(_parse_frontmatter(frontmatter, path))
    prompt.setdefault("id", path.stem)
    prompt["messages"] = _parse_mdx_messages(body)
    return prompt


def _split_mdx_frontmatter(source: str, path: Path) -> tuple[str, str]:
    lines = source.splitlines()
    if not lines or lines[0].strip() != "---":
        raise PromptManifestError(f"Prompt MDX at {path} is missing front matter")

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])

    raise PromptManifestError(f"Prompt MDX at {path} has unterminated front matter")


def _parse_frontmatter(source: str, path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise PromptManifestError(f"Invalid front matter line in {path}: {raw_line}")

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value:
            data[key] = _parse_frontmatter_value(value)
            continue

        items: list[Any] = []
        while index < len(lines):
            item_line = lines[index]
            item = item_line.strip()
            if not item:
                index += 1
                continue
            if not item_line.startswith((" ", "\t")):
                break
            if not item.startswith("- "):
                raise PromptManifestError(
                    f"Unsupported nested front matter for {key!r} in {path}; use an "
                    "inline JSON value instead"
                )
            items.append(_parse_frontmatter_value(item[2:].strip()))
            index += 1

        data[key] = items

    return data


def _parse_frontmatter_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] in {'"', "{", "["}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip('"')
    if value[0] == "'" and value.endswith("'"):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", value) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+", value
    ):
        return float(value)
    return value


def _normalize_mdx_prompt(frontmatter: dict[str, Any]) -> dict[str, Any]:
    prompt = dict(frontmatter)

    parameters = prompt.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    else:
        parameters = dict(parameters)

    for key in _FRONTMATTER_PARAMETER_KEYS:
        if key in prompt:
            parameter_key = "max_tokens" if key == "max_output_tokens" else key
            parameters[parameter_key] = prompt.pop(key)

    if parameters:
        prompt["parameters"] = parameters

    if "response_schema_json" in prompt and "response_schema" not in prompt:
        prompt["response_schema"] = _parse_json_frontmatter_field(
            prompt.pop("response_schema_json"), "response_schema_json"
        )
    if "tools_json" in prompt and "tools" not in prompt:
        prompt["tools"] = _parse_json_frontmatter_field(prompt.pop("tools_json"), "tools_json")

    return prompt


def _parse_json_frontmatter_field(value: Any, field: str) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        raise PromptManifestError(f"Prompt MDX front matter field {field!r} must be JSON")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise PromptManifestError(f"Invalid JSON in prompt MDX front matter field {field!r}") from exc


def _parse_mdx_messages(body: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for match in _MDX_ROLE_BLOCK_RE.finditer(body):
        content = match.group(2).strip()
        if content:
            messages.append({"role": match.group(1).lower(), "content": content})

    if messages:
        return messages

    content = body.strip()
    if content:
        return [{"role": "user", "content": content}]
    return []


def _find_prompt(data: Any, prompt_id: str, version: str | None) -> dict[str, Any] | None:
    for candidate in _iter_prompts(data):
        candidate_id = str(candidate.get("id") or candidate.get("public_id") or "")
        if candidate_id != prompt_id:
            continue
        if version is not None:
            candidate_version = str(candidate.get("version") or "")
            candidate_version_id = str(candidate.get("version_id") or "")
            if candidate_version != str(version) and candidate_version_id != str(version):
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


def _merge_config(
    current: GenerationConfig,
    parameters: dict[str, Any],
    prompt: Prompt,
) -> GenerationConfig:
    return replace(
        current,
        temperature=current.temperature
        if current.temperature is not None
        else prompt.temperature
        if prompt.temperature is not None
        else _optional_float(parameters.get("temperature")),
        max_tokens=current.max_tokens
        if current.max_tokens is not None
        else prompt.max_tokens
        if prompt.max_tokens is not None
        else _optional_int(parameters.get("max_tokens") or parameters.get("max_output_tokens")),
        top_p=current.top_p
        if current.top_p is not None
        else prompt.top_p
        if prompt.top_p is not None
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
