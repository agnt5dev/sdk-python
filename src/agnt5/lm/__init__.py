"""Language Model interface for AGNT5 SDK.

Usage:
    >>> from agnt5 import lm
    >>> response = await lm.generate(
    ...     model="openai/gpt-4o-mini",
    ...     prompt="What is love?",
    ... )
    >>> print(response.text)
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from .._schema_utils import detect_format_type
from ..context import get_current_context
from ..events import Event
from .base import LanguageModel
from .client import LMClient
from .events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
    LMFailed,
    LMStarted,
)
from .types import (
    BUILT_IN_TOOL_PROVIDER_NAMES,
    BuiltInTool,
    GenerateRequest,
    GenerateResponse,
    GenerationConfig,
    Message,
    MessageRole,
    Modality,
    ModelConfig,
    Prompt,
    PromptRef,
    ReasoningEffort,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
    built_in_tool_names,
)

SUPPORTED_MODEL_PROVIDERS = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "deepseek",
        "gemini",
        "google",
        "groq",
        "hf",
        "huggingface",
        "mistral",
        "ollama",
        "openai",
        "openrouter",
        "xai",
    }
)

# Re-export types
__all__ = [
    # Public API
    "generate",
    "stream",
    # Client
    "LanguageModel",
    "LMClient",
    # Events
    "LMCompleted",
    "LMContentBlockCompleted",
    "LMContentBlockDelta",
    "LMContentBlockStarted",
    "LMFailed",
    "LMStarted",
    # Types
    "BUILT_IN_TOOL_PROVIDER_NAMES",
    "BuiltInTool",
    "built_in_tool_names",
    "GenerateRequest",
    "GenerateResponse",
    "GenerationConfig",
    "Message",
    "MessageRole",
    "Modality",
    "ModelConfig",
    "Prompt",
    "ReasoningEffort",
    "PromptRef",
    "TokenUsage",
    "ToolChoice",
    "ToolDefinition",
]


def _parse_model(model: str) -> tuple[str, str]:
    """Parse provider and model name from model string."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model must be a non-empty string in 'provider/model' format")

    if "/" not in model:
        raise ValueError(
            f"Model must include provider prefix (e.g., 'openai/{model}'). "
            f"Supported providers: {', '.join(sorted(SUPPORTED_MODEL_PROVIDERS))}"
        )

    provider, model_name = model.split("/", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()

    if not provider or not model_name:
        raise ValueError(
            "Model must be in 'provider/model' format with both provider and model name"
        )

    if provider not in SUPPORTED_MODEL_PROVIDERS:
        raise ValueError(
            f"Unsupported model provider '{provider}' in model '{model}'. "
            f"Did you mean 'openai/{model_name}'? "
            f"Supported providers: {', '.join(sorted(SUPPORTED_MODEL_PROVIDERS))}"
        )

    return (provider, model_name)


def _build_messages(
    prompt: Optional[str],
    messages: Optional[List[Dict[str, str]]],
) -> List[Message]:
    """Convert prompt or messages dict to Message objects."""
    if not prompt and not messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")
    if prompt and messages:
        raise ValueError("Provide either 'prompt' or 'messages', not both")

    if prompt:
        return [Message.user(prompt)]

    result = []
    for msg in messages or []:
        role = MessageRole(msg["role"])
        if role == MessageRole.USER:
            result.append(Message.user(msg["content"]))
        elif role == MessageRole.ASSISTANT:
            result.append(Message.assistant(msg["content"]))
        elif role == MessageRole.SYSTEM:
            result.append(Message.system(msg["content"]))
    return result


def _coerce_prompt_ref(
    value: Optional[Union[Prompt, Dict[str, Any], str]],
    *,
    variables: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    environment: Optional[str] = None,
    environment_id: Optional[str] = None,
    version: Optional[str] = None,
    platform_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[Prompt]:
    if value is None:
        return None
    if isinstance(value, Prompt):
        ref = value
    elif isinstance(value, str):
        ref = Prompt(id=value)
    else:
        ref = Prompt(
            id=value["id"],
            project_id=value.get("project_id"),
            version=value.get("version"),
            environment_id=value.get("environment_id"),
            environment_ref=value.get("environment_ref") or value.get("environment"),
            platform_url=value.get("platform_url"),
            api_key=value.get("api_key"),
            variables=value.get("variables") or {},
            model=value.get("model"),
            temperature=value.get("temperature"),
            max_tokens=value.get("max_tokens") or value.get("max_output_tokens"),
            top_p=value.get("top_p"),
        )
    return replace(
        ref,
        project_id=project_id or ref.project_id,
        version=version or ref.version,
        environment_id=environment_id or ref.environment_id,
        environment_ref=environment or ref.environment_ref,
        platform_url=platform_url or ref.platform_url,
        api_key=api_key or ref.api_key,
        variables=variables if variables is not None else ref.variables,
    )


def _runtime_llm_options(prompt_id: Optional[str] = None) -> Any:
    current_ctx = get_current_context()
    runtime = getattr(current_ctx, "runtime", None)
    if prompt_id:
        prompt_options = getattr(runtime, "prompts", {}).get(prompt_id) if runtime else None
        if prompt_options is not None and getattr(prompt_options, "has_values", lambda: False)():
            return prompt_options
    llm_options = getattr(runtime, "llm", None)
    if llm_options is not None and getattr(llm_options, "has_values", lambda: False)():
        return llm_options
    return None


async def generate(
    model: str,
    prompt: Optional[Union[str, Prompt, Dict[str, Any]]] = None,
    prompt_ref: Optional[Union[Prompt, Dict[str, Any], str]] = None,
    variables: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
    environment: Optional[str] = None,
    environment_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    platform_url: Optional[str] = None,
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    response_format: Optional[Any] = None,
    # Responses API specific
    built_in_tools: Optional[List[BuiltInTool]] = None,
    reasoning_effort: Optional[ReasoningEffort] = None,
    modalities: Optional[List[Modality]] = None,
    store: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
) -> GenerateResponse:
    """Generate text using any LLM provider.

    Args:
        model: Model with provider prefix (e.g., 'openai/gpt-4o-mini')
        prompt: Simple text prompt (single-turn)
        messages: List of message dicts with 'role' and 'content' (multi-turn)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        response_format: Pydantic model, dataclass, or JSON schema for structured output
        built_in_tools: Built-in tools (OpenAI Responses API)
        reasoning_effort: Reasoning effort for o-series models
        modalities: Output modalities (text, audio, image)
        store: Enable server-side conversation state
        previous_response_id: Continue from previous response

    Returns:
        GenerateResponse with text, usage, and optional structured output
    """
    prompt_input = prompt
    if isinstance(prompt_input, (Prompt, dict)):
        if prompt_ref is not None:
            raise ValueError("Provide either 'prompt' or 'prompt_ref', not both")
        prompt_ref = prompt_input
        prompt_input = None
    if prompt_ref is not None and (prompt_input is not None or messages is not None):
        raise ValueError("Provide either 'prompt_ref' or prompt/messages, not both")
    message_objects = [] if prompt_ref is not None else _build_messages(prompt_input, messages)
    prompt_obj = _coerce_prompt_ref(
        prompt_ref,
        variables=variables,
        project_id=project_id,
        environment=environment,
        environment_id=environment_id,
        version=prompt_version,
        platform_url=platform_url,
        api_key=api_key,
    )
    runtime_llm = _runtime_llm_options(prompt_obj.id if prompt_obj is not None else None)
    if runtime_llm is not None:
        if prompt_obj is not None:
            prompt_obj = replace(
                prompt_obj,
                model=runtime_llm.model or prompt_obj.model,
                temperature=(
                    runtime_llm.temperature
                    if runtime_llm.temperature is not None
                    else prompt_obj.temperature
                ),
                max_tokens=(
                    runtime_llm.max_tokens
                    if runtime_llm.max_tokens is not None
                    else prompt_obj.max_tokens
                ),
                top_p=runtime_llm.top_p if runtime_llm.top_p is not None else prompt_obj.top_p,
            )
        if runtime_llm.temperature is not None:
            temperature = runtime_llm.temperature
        if runtime_llm.max_tokens is not None:
            max_tokens = runtime_llm.max_tokens
        if runtime_llm.top_p is not None:
            top_p = runtime_llm.top_p

    effective_model = (
        runtime_llm.model
        if runtime_llm is not None and runtime_llm.model
        else prompt_obj.model
        if prompt_obj and prompt_obj.model
        else model
    )
    provider, model_name = _parse_model(effective_model)
    model = f"{provider}/{model_name}"

    response_schema_json = None
    if response_format is not None:
        _, json_schema = detect_format_type(response_format)
        response_schema_json = json.dumps(json_schema)

    client = LMClient(provider=provider.lower())

    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        built_in_tools=built_in_tools or [],
        reasoning_effort=reasoning_effort,
        modalities=modalities,
        store=store,
        previous_response_id=previous_response_id,
    )

    request = GenerateRequest(
        model=model,
        prompt_ref=prompt_obj,
        messages=message_objects,
        system_prompt=system_prompt,
        config=config,
        response_schema=response_schema_json,
    )

    return await client.generate(request)


async def stream(
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    # Responses API specific
    built_in_tools: Optional[List[BuiltInTool]] = None,
    reasoning_effort: Optional[ReasoningEffort] = None,
    modalities: Optional[List[Modality]] = None,
    store: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
) -> AsyncGenerator[Event, None]:
    """Stream LLM completion as Event objects.

    Args:
        model: Model with provider prefix (e.g., 'openai/gpt-4o-mini')
        prompt: Simple text prompt (single-turn)
        messages: List of message dicts with 'role' and 'content' (multi-turn)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        built_in_tools: Built-in tools (OpenAI Responses API)
        reasoning_effort: Reasoning effort for o-series models
        modalities: Output modalities (text, audio, image)
        store: Enable server-side conversation state
        previous_response_id: Continue from previous response

    Yields:
        Event objects (lm.message.start, lm.message.delta, lm.message.stop)
    """
    provider, model_name = _parse_model(model)
    model = f"{provider}/{model_name}"
    message_objects = _build_messages(prompt, messages)

    client = LMClient(provider=provider.lower())

    config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        built_in_tools=built_in_tools or [],
        reasoning_effort=reasoning_effort,
        modalities=modalities,
        store=store,
        previous_response_id=previous_response_id,
    )

    request = GenerateRequest(
        model=model,
        messages=message_objects,
        system_prompt=system_prompt,
        config=config,
    )

    async for event in client.stream(request):
        yield event
