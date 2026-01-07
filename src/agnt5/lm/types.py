"""LM types: enums, messages, and configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @staticmethod
    def system(content: str) -> Message:
        return Message(role=MessageRole.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message(role=MessageRole.USER, content=content)

    @staticmethod
    def assistant(
        content: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        return Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> Message:
        return Message(role=MessageRole.USER, content=content, tool_call_id=tool_call_id)


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


class BuiltInTool(str, Enum):
    """Built-in tools for OpenAI Responses API."""

    WEB_SEARCH = "web_search_preview"
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"


class ReasoningEffort(str, Enum):
    """Reasoning effort level for o-series models."""

    MINIMAL = "minimal"
    MEDIUM = "medium"
    HIGH = "high"


class Modality(str, Enum):
    """Output modalities for multimodal models."""

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


@dataclass
class ModelConfig:
    """Advanced model configuration for custom endpoints."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class GenerationConfig:
    """LLM generation configuration."""

    # Standard parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Responses API specific
    built_in_tools: List[BuiltInTool] = field(default_factory=list)
    reasoning_effort: Optional[ReasoningEffort] = None
    modalities: Optional[List[Modality]] = None
    store: Optional[bool] = None
    previous_response_id: Optional[str] = None


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
    response_id: Optional[str] = None
    _rust_response: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateResponse":
        """Create from dict (for memoization cache)."""
        usage = None
        if data.get("usage"):
            usage = TokenUsage(**data["usage"])
        return cls(
            text=data.get("text", ""),
            usage=usage,
            finish_reason=data.get("finish_reason"),
            tool_calls=data.get("tool_calls"),
            response_id=data.get("response_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for memoization cache)."""
        result: Dict[str, Any] = {"text": self.text}
        if self.usage:
            result["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.response_id:
            result["response_id"] = self.response_id
        return result

    @property
    def structured_output(self) -> Optional[Any]:
        """Parsed structured output (Pydantic model, dataclass, or dict)."""
        if self._rust_response and hasattr(self._rust_response, 'object'):
            return self._rust_response.object
        return None

    @property
    def parsed(self) -> Optional[Any]:
        """Alias for structured_output (OpenAI SDK compatibility)."""
        return self.structured_output

    @property
    def object(self) -> Optional[Any]:
        """Alias for structured_output."""
        return self.structured_output


@dataclass
class GenerateRequest:
    """Request for LLM generation."""

    model: str
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    tools: List[ToolDefinition] = field(default_factory=list)
    tool_choice: Optional[ToolChoice] = None
    config: GenerationConfig = field(default_factory=GenerationConfig)
    response_schema: Optional[str] = None
