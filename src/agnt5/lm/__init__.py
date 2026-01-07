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
from typing import Any, AsyncGenerator, Dict, List, Optional

from .._schema_utils import detect_format_type
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
    BuiltInTool,
    GenerateRequest,
    GenerateResponse,
    GenerationConfig,
    Message,
    MessageRole,
    Modality,
    ModelConfig,
    ReasoningEffort,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
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
    "BuiltInTool",
    "GenerateRequest",
    "GenerateResponse",
    "GenerationConfig",
    "Message",
    "MessageRole",
    "Modality",
    "ModelConfig",
    "ReasoningEffort",
    "TokenUsage",
    "ToolChoice",
    "ToolDefinition",
]


def _parse_model(model: str) -> tuple[str, str]:
    """Parse provider and model name from model string."""
    if '/' not in model:
        raise ValueError(
            f"Model must include provider prefix (e.g., 'openai/{model}'). "
            "Supported: openai, anthropic, groq, openrouter, azure, bedrock"
        )
    parts = model.split('/', 1)
    return (parts[0], parts[1])


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


async def generate(
    model: str,
    prompt: Optional[str] = None,
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
    provider, _ = _parse_model(model)
    message_objects = _build_messages(prompt, messages)

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
    provider, _ = _parse_model(model)
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
