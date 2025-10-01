"""Language Model interface for AGNT5 SDK.

Phase 1: Simple Python interface with OpenAI provider.
Phase 2: Python bindings to Rust SDK core for full provider support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

# Phase 1: We define the interface matching Rust SDK core
# Phase 2: This will be replaced with PyO3 bindings to sdk-core


class MessageRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Conversation message."""

    role: MessageRole
    content: str

    @staticmethod
    def system(content: str) -> Message:
        """Create system message."""
        return Message(role=MessageRole.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        """Create user message."""
        return Message(role=MessageRole.USER, content=content)

    @staticmethod
    def assistant(content: str) -> Message:
        """Create assistant message."""
        return Message(role=MessageRole.ASSISTANT, content=content)


@dataclass
class ToolDefinition:
    """Tool definition for LLM."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolChoice(str, Enum):
    """Tool choice mode."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass
class GenerationConfig:
    """LLM generation configuration."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class GenerateResponse:
    """Response from LLM generation."""

    text: str
    usage: Optional[TokenUsage] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class GenerateRequest:
    """Request for LLM generation."""

    model: str
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    tools: List[ToolDefinition] = field(default_factory=list)
    tool_choice: Optional[ToolChoice] = None
    config: GenerationConfig = field(default_factory=GenerationConfig)


class LanguageModel(ABC):
    """Abstract base class for language models.

    Phase 1: Simple Python interface.
    Phase 2: Will use Rust SDK core via PyO3.
    """

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from LLM."""
        pass

    @abstractmethod
    async def stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream completion from LLM."""
        pass


class OpenAILanguageModel(LanguageModel):
    """OpenAI language model implementation.

    Phase 1: Uses OpenAI Python SDK directly.
    Phase 2: Will use Rust SDK core OpenAI provider.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize OpenAI model.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Optional base URL for API
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for OpenAILanguageModel. "
                "Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from OpenAI."""
        # Build messages
        messages = []

        # Add system prompt if provided
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # Add conversation messages
        for msg in request.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        # Add config
        if request.config.temperature is not None:
            kwargs["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            kwargs["max_tokens"] = request.config.max_tokens
        if request.config.top_p is not None:
            kwargs["top_p"] = request.config.top_p

        # Add tools if provided
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters or {},
                    },
                }
                for tool in request.tools
            ]

            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice.value

        # Call OpenAI
        response = await self.client.chat.completions.create(**kwargs)

        # Extract response
        choice = response.choices[0]
        message = choice.message

        # Build response
        text = message.content or ""
        tool_calls = None

        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return GenerateResponse(
            text=text,
            usage=usage,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream completion from OpenAI."""
        # Build messages (same as generate)
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
        }

        if request.config.temperature is not None:
            kwargs["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            kwargs["max_tokens"] = request.config.max_tokens

        # Stream from OpenAI
        stream = await self.client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
