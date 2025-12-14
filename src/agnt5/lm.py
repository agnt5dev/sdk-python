"""Language Model interface for AGNT5 SDK.

Simplified API inspired by Vercel AI SDK for seamless multi-provider LLM access.
Uses Rust-backed implementation via PyO3 for performance and reliability.

Basic Usage:
    >>> from agnt5 import lm
    >>>
    >>> # Simple generation
    >>> response = await lm.generate(
    ...     model="openai/gpt-4o-mini",
    ...     prompt="What is love?",
    ...     temperature=0.7
    ... )
    >>> print(response.text)
    >>>
    >>> # Streaming
    >>> async for chunk in lm.stream(
    ...     model="anthropic/claude-3-5-haiku",
    ...     prompt="Write a story"
    ... ):
    ...     print(chunk, end="", flush=True)

Supported Providers (via model prefix):
    - openai/model-name
    - anthropic/model-name
    - groq/model-name
    - openrouter/provider/model-name
    - azure/model-name
    - bedrock/model-name
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from ._schema_utils import detect_format_type
from .context import get_current_context

try:
    from ._core import LanguageModel as RustLanguageModel
    from ._core import LanguageModelConfig as RustLanguageModelConfig
    from ._core import Response as RustResponse
    from ._core import StreamChunk as RustStreamChunk
    from ._core import Usage as RustUsage
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    RustLanguageModel = None
    RustLanguageModelConfig = None
    RustResponse = None
    RustStreamChunk = None
    RustUsage = None


# Keep Python classes for backward compatibility and convenience
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


class BuiltInTool(str, Enum):
    """Built-in tools for OpenAI Responses API.

    These are platform-provided tools that don't require implementation:
    - WEB_SEARCH: Real-time web search capability
    - CODE_INTERPRETER: Execute Python code in a sandboxed environment
    - FILE_SEARCH: Search through uploaded files
    """

    WEB_SEARCH = "web_search_preview"
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"


class ReasoningEffort(str, Enum):
    """Reasoning effort level for o-series models (o1, o3, etc.).

    Controls the amount of reasoning/thinking the model performs:
    - MINIMAL: Fast responses with basic reasoning
    - MEDIUM: Balanced reasoning and speed (default)
    - HIGH: Deep reasoning, slower but more thorough
    """

    MINIMAL = "minimal"
    MEDIUM = "medium"
    HIGH = "high"


class Modality(str, Enum):
    """Output modalities for multimodal models.

    Specifies the types of content the model can generate:
    - TEXT: Standard text output
    - AUDIO: Audio output (e.g., for text-to-speech models)
    - IMAGE: Image generation (future capability)
    """

    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


@dataclass
class ModelConfig:
    """Advanced model configuration for custom endpoints and settings.

    Use this for advanced scenarios like custom API endpoints, special headers,
    or overriding default timeouts. Most users won't need this - the basic
    model string with temperature/max_tokens is sufficient for common cases.

    Example:
        >>> from agnt5.lm import ModelConfig
        >>> from agnt5 import Agent
        >>>
        >>> # Custom API endpoint
        >>> config = ModelConfig(
        ...     base_url="https://custom-api.example.com",
        ...     api_key="custom-key",
        ...     timeout=60,
        ...     headers={"X-Custom-Header": "value"}
        ... )
        >>>
        >>> agent = Agent(
        ...     name="custom_agent",
        ...     model="openai/gpt-4o-mini",
        ...     instructions="...",
        ...     model_config=config
        ... )
    """
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


@dataclass
class GenerationConfig:
    """LLM generation configuration.

    Supports both Chat Completions and Responses API parameters.
    """

    # Standard parameters (both APIs)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Responses API specific parameters
    built_in_tools: List[BuiltInTool] = field(default_factory=list)
    reasoning_effort: Optional[ReasoningEffort] = None
    modalities: Optional[List[Modality]] = None
    store: Optional[bool] = None  # Enable server-side conversation state
    previous_response_id: Optional[str] = None  # Continue previous conversation


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
    response_id: Optional[str] = None  # Response ID for conversation continuation (Responses API)
    _rust_response: Optional[Any] = field(default=None, repr=False)

    @property
    def structured_output(self) -> Optional[Any]:
        """Parsed structured output (Pydantic model, dataclass, or dict).

        Returns the parsed object when response_format is specified.
        This is the recommended property name for accessing structured output.

        Returns:
            Parsed object according to the specified response_format, or None if not available
        """
        if self._rust_response and hasattr(self._rust_response, 'object'):
            return self._rust_response.object
        return None

    @property
    def parsed(self) -> Optional[Any]:
        """Alias for structured_output (OpenAI SDK compatibility).

        Returns:
            Same as structured_output
        """
        return self.structured_output

    @property
    def object(self) -> Optional[Any]:
        """Alias for structured_output.

        Returns:
            Same as structured_output
        """
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
    response_schema: Optional[str] = None  # JSON-encoded schema for structured output


# Abstract base class for language models
# This exists primarily for testing/mocking purposes
class LanguageModel(ABC):
    """Abstract base class for language model implementations.

    This class defines the interface that all language models must implement.
    It's primarily used for testing and mocking, as production code should use
    the module-level generate() and stream() functions instead.
    """

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from LLM.

        Args:
            request: Generation request with model, messages, and configuration

        Returns:
            GenerateResponse with text, usage, and optional tool calls
        """
        pass

    @abstractmethod
    async def stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream completion from LLM.

        Args:
            request: Generation request with model, messages, and configuration

        Yields:
            Text chunks as they are generated
        """
        pass


# Internal wrapper for the Rust-backed implementation
# Users should use the module-level generate() and stream() functions instead
class _LanguageModel(LanguageModel):
    """Internal Language Model wrapper using Rust SDK core.

    This class is for internal use only. Users should use the module-level
    lm.generate() and lm.stream() functions for a simpler interface.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """Initialize language model.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'azure', 'bedrock', 'groq', 'openrouter')
                     If None, provider will be auto-detected from model prefix (e.g., 'openai/gpt-4o')
            default_model: Default model to use if not specified in requests
        """
        if not _RUST_AVAILABLE:
            raise ImportError(
                "Rust extension not available. Please rebuild the SDK with: "
                "cd sdk/sdk-python && maturin develop"
            )

        self._provider = provider
        self._default_model = default_model

        # Create config object for Rust
        config = RustLanguageModelConfig(
            default_model=default_model,
            default_provider=provider,
        )

        self._rust_lm = RustLanguageModel(config=config)

    def _prepare_model_name(self, model: str) -> str:
        """Prepare model name with provider prefix if needed.

        Args:
            model: Model name (e.g., 'gpt-4o-mini' or 'openai/gpt-4o-mini')

        Returns:
            Model name with provider prefix (e.g., 'openai/gpt-4o-mini')
        """
        # If model already has a prefix, return as is
        # This handles cases like OpenRouter where models already have their provider prefix
        # (e.g., 'anthropic/claude-3.5-haiku' for OpenRouter)
        if '/' in model:
            return model

        # If we have a default provider, prefix the model
        if self._provider:
            return f"{self._provider}/{model}"

        # Otherwise return as is and let Rust handle the error
        return model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from LLM.

        Args:
            request: Generation request with model, messages, and configuration

        Returns:
            GenerateResponse with text, usage, and optional tool calls
        """
        # Convert Python request to structured format for Rust
        prompt = self._build_prompt_messages(request)

        # Prepare model name with provider prefix
        model = self._prepare_model_name(request.model)

        # Build kwargs for Rust
        kwargs: dict[str, Any] = {
            "model": model,
        }

        # Always pass provider explicitly if set
        # For gateway providers like OpenRouter, this allows them to handle
        # models with provider prefixes (e.g., openrouter can handle anthropic/claude-3.5-haiku)
        if self._provider:
            kwargs["provider"] = self._provider

        # Pass system prompt separately if provided
        if request.system_prompt:
            kwargs["system_prompt"] = request.system_prompt

        if request.config.temperature is not None:
            kwargs["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            kwargs["max_tokens"] = request.config.max_tokens
        if request.config.top_p is not None:
            kwargs["top_p"] = request.config.top_p

        # Pass response schema for structured output if provided
        if request.response_schema is not None:
            kwargs["response_schema_kw"] = request.response_schema

        # Pass Responses API specific parameters
        if request.config.built_in_tools:
            # Serialize built-in tools to JSON for Rust
            built_in_tools_list = [tool.value for tool in request.config.built_in_tools]
            kwargs["built_in_tools"] = json.dumps(built_in_tools_list)

        if request.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.config.reasoning_effort.value

        if request.config.modalities is not None:
            modalities_list = [modality.value for modality in request.config.modalities]
            kwargs["modalities"] = json.dumps(modalities_list)

        if request.config.store is not None:
            kwargs["store"] = request.config.store

        if request.config.previous_response_id is not None:
            kwargs["previous_response_id"] = request.config.previous_response_id

        # Pass tools and tool_choice to Rust
        if request.tools:
            # Serialize tools to JSON for Rust
            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]
            tools_json = json.dumps(tools_list)
            kwargs["tools"] = tools_json

        if request.tool_choice:
            # Serialize tool_choice to JSON for Rust
            kwargs["tool_choice"] = json.dumps(request.tool_choice.value)

        # Pass runtime_context for proper trace linking
        # Try to get from current context if available
        current_ctx = get_current_context()
        if current_ctx and hasattr(current_ctx, '_runtime_context') and current_ctx._runtime_context:
            kwargs["runtime_context"] = current_ctx._runtime_context

        # Emit checkpoint if called within a workflow context
        from .context import get_workflow_context
        import time
        workflow_ctx = get_workflow_context()

        # Get trace context for event linkage
        trace_id = None
        span_id = None
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                trace_id = format(span_context.trace_id, '032x')
                span_id = format(span_context.span_id, '016x')
        except Exception:
            pass  # Tracing not available, continue without

        # Emit started event (trace_id is optional - emit even without tracing)
        if workflow_ctx:
            event_data = {
                "model": model,
                "provider": self._provider,
                "timestamp": time.time_ns() // 1_000_000,
            }
            if trace_id:
                event_data["trace_id"] = trace_id
                event_data["span_id"] = span_id
            workflow_ctx._send_checkpoint("lm.call.started", event_data)

        try:
            # Call Rust implementation - it returns a proper Python coroutine now
            # Using pyo3-async-runtimes for truly async HTTP calls without blocking
            rust_response = await self._rust_lm.generate(prompt=prompt, **kwargs)

            # Convert Rust response to Python
            response = self._convert_response(rust_response)

            # Emit completion event with token usage and cost
            if workflow_ctx:
                event_data = {
                    "model": model,
                    "provider": self._provider,
                    "timestamp": time.time_ns() // 1_000_000,
                }
                if trace_id:
                    event_data["trace_id"] = trace_id
                    event_data["span_id"] = span_id

                # Add token usage if available
                if response.usage:
                    event_data["input_tokens"] = response.usage.prompt_tokens
                    event_data["output_tokens"] = response.usage.completion_tokens
                    event_data["total_tokens"] = response.usage.total_tokens

                workflow_ctx._send_checkpoint("lm.call.completed", event_data)

            return response
        except Exception as e:
            # Emit failed event
            if workflow_ctx:
                event_data = {
                    "model": model,
                    "provider": self._provider,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time_ns() // 1_000_000,
                }
                if trace_id:
                    event_data["trace_id"] = trace_id
                    event_data["span_id"] = span_id
                workflow_ctx._send_checkpoint("lm.call.failed", event_data)
            raise

    async def stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream completion from LLM.

        Args:
            request: Generation request with model, messages, and configuration

        Yields:
            Text chunks as they are generated
        """
        # Convert Python request to structured format for Rust
        prompt = self._build_prompt_messages(request)

        # Prepare model name with provider prefix
        model = self._prepare_model_name(request.model)

        # Build kwargs for Rust
        kwargs: dict[str, Any] = {
            "model": model,
        }

        # Always pass provider explicitly if set
        # For gateway providers like OpenRouter, this allows them to handle
        # models with provider prefixes (e.g., openrouter can handle anthropic/claude-3.5-haiku)
        if self._provider:
            kwargs["provider"] = self._provider

        # Pass system prompt separately if provided
        if request.system_prompt:
            kwargs["system_prompt"] = request.system_prompt

        if request.config.temperature is not None:
            kwargs["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            kwargs["max_tokens"] = request.config.max_tokens
        if request.config.top_p is not None:
            kwargs["top_p"] = request.config.top_p

        # Pass Responses API specific parameters
        if request.config.built_in_tools:
            # Serialize built-in tools to JSON for Rust
            built_in_tools_list = [tool.value for tool in request.config.built_in_tools]
            kwargs["built_in_tools"] = json.dumps(built_in_tools_list)

        if request.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.config.reasoning_effort.value

        if request.config.modalities is not None:
            modalities_list = [modality.value for modality in request.config.modalities]
            kwargs["modalities"] = json.dumps(modalities_list)

        if request.config.store is not None:
            kwargs["store"] = request.config.store

        if request.config.previous_response_id is not None:
            kwargs["previous_response_id"] = request.config.previous_response_id

        # Pass tools and tool_choice to Rust
        if request.tools:
            # Serialize tools to JSON for Rust
            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]
            kwargs["tools"] = json.dumps(tools_list)

        if request.tool_choice:
            # Serialize tool_choice to JSON for Rust
            kwargs["tool_choice"] = json.dumps(request.tool_choice.value)

        # Emit checkpoint if called within a workflow context
        from .context import get_workflow_context
        import time
        workflow_ctx = get_workflow_context()

        # Get trace context for event linkage
        trace_id = None
        span_id = None
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                trace_id = format(span_context.trace_id, '032x')
                span_id = format(span_context.span_id, '016x')
        except Exception:
            pass  # Tracing not available, continue without

        # Emit started event (trace_id is optional - emit even without tracing)
        if workflow_ctx:
            event_data = {
                "model": model,
                "provider": self._provider,
                "timestamp": time.time_ns() // 1_000_000,
            }
            if trace_id:
                event_data["trace_id"] = trace_id
                event_data["span_id"] = span_id
            workflow_ctx._send_checkpoint("lm.stream.started", event_data)

        try:
            # Call Rust implementation - it returns a proper Python coroutine now
            # Using pyo3-async-runtimes for truly async streaming without blocking
            rust_chunks = await self._rust_lm.stream(prompt=prompt, **kwargs)

            # Yield each delta chunk (skip the final completed chunk which contains
            # the full accumulated text - we only want individual deltas)
            for chunk in rust_chunks:
                if chunk.text and not chunk.finished:
                    yield chunk.text

            # Emit completion event (note: streaming doesn't provide token counts)
            if workflow_ctx:
                event_data = {
                    "model": model,
                    "provider": self._provider,
                    "timestamp": time.time_ns() // 1_000_000,
                }
                if trace_id:
                    event_data["trace_id"] = trace_id
                    event_data["span_id"] = span_id
                workflow_ctx._send_checkpoint("lm.stream.completed", event_data)
        except Exception as e:
            # Emit failed event
            if workflow_ctx:
                event_data = {
                    "model": model,
                    "provider": self._provider,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time_ns() // 1_000_000,
                }
                if trace_id:
                    event_data["trace_id"] = trace_id
                    event_data["span_id"] = span_id
                workflow_ctx._send_checkpoint("lm.stream.failed", event_data)
            raise

    def _build_prompt_messages(self, request: GenerateRequest) -> List[Dict[str, str]]:
        """Build structured message list for Rust.

        Rust expects a list of dicts with 'role' and 'content' keys.
        System prompt is passed separately via kwargs.

        Args:
            request: Generation request with messages

        Returns:
            List of message dicts with role and content
        """
        # Convert messages to Rust format (list of dicts with role and content)
        messages = []
        for msg in request.messages:
            messages.append({
                "role": msg.role.value,  # "system", "user", or "assistant"
                "content": msg.content
            })

        # If no messages and no system prompt, return a default user message
        if not messages and not request.system_prompt:
            messages.append({
                "role": "user",
                "content": ""
            })

        return messages

    def _convert_response(self, rust_response: RustResponse) -> GenerateResponse:
        """Convert Rust response to Python response."""
        usage = None
        if rust_response.usage:
            usage = TokenUsage(
                prompt_tokens=rust_response.usage.prompt_tokens,
                completion_tokens=rust_response.usage.completion_tokens,
                total_tokens=rust_response.usage.total_tokens,
            )

        # Extract tool_calls from Rust response
        tool_calls = None
        if hasattr(rust_response, 'tool_calls') and rust_response.tool_calls:
            tool_calls = rust_response.tool_calls

        # Extract response_id from Rust response (for Responses API)
        response_id = None
        if hasattr(rust_response, 'response_id') and rust_response.response_id:
            response_id = rust_response.response_id

        return GenerateResponse(
            text=rust_response.content,
            usage=usage,
            finish_reason=None,  # TODO: Add finish_reason to Rust response
            tool_calls=tool_calls,
            response_id=response_id,
            _rust_response=rust_response,  # Store for .structured_output access
        )


# ============================================================================
# Simplified API (Recommended)
# ============================================================================
# This is the recommended simple interface for most use cases

async def generate(
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    response_format: Optional[Any] = None,
    # Responses API specific parameters
    built_in_tools: Optional[List[BuiltInTool]] = None,
    reasoning_effort: Optional[ReasoningEffort] = None,
    modalities: Optional[List[Modality]] = None,
    store: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
) -> GenerateResponse:
    """Generate text using any LLM provider (simplified API).

    This is the recommended way to use the LLM API. Provider is auto-detected
    from the model prefix (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-5-haiku').

    Args:
        model: Model identifier with provider prefix (e.g., 'openai/gpt-4o-mini')
        prompt: Simple text prompt (for single-turn requests)
        messages: List of message dicts with 'role' and 'content' (for multi-turn)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        response_format: Pydantic model, dataclass, or JSON schema dict for structured output
        built_in_tools: List of built-in tools (OpenAI Responses API only)
        reasoning_effort: Reasoning effort level for o-series models (OpenAI Responses API only)
        modalities: Output modalities (text, audio, image) (OpenAI Responses API only)
        store: Enable server-side conversation state (OpenAI Responses API only)
        previous_response_id: Continue from previous response (OpenAI Responses API only)

    Returns:
        GenerateResponse with text, usage, and optional structured output

    Examples:
        Simple prompt:
        >>> response = await generate(
        ...     model="openai/gpt-4o-mini",
        ...     prompt="What is love?",
        ...     temperature=0.7
        ... )
        >>> print(response.text)

        Structured output with dataclass:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class CodeReview:
        ...     issues: list[str]
        ...     suggestions: list[str]
        ...     overall_quality: int
        >>>
        >>> response = await generate(
        ...     model="openai/gpt-4o",
        ...     prompt="Analyze this code...",
        ...     response_format=CodeReview
        ... )
        >>> review = response.structured_output  # Returns dict
    """
    # Validate input
    if not prompt and not messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")
    if prompt and messages:
        raise ValueError("Provide either 'prompt' or 'messages', not both")

    # Auto-detect provider from model prefix
    if '/' not in model:
        raise ValueError(
            f"Model must include provider prefix (e.g., 'openai/{model}'). "
            f"Supported providers: openai, anthropic, groq, openrouter, azure, bedrock"
        )

    provider, model_name = model.split('/', 1)

    # Convert response_format to JSON schema if provided
    response_schema_json = None
    if response_format is not None:
        format_type, json_schema = detect_format_type(response_format)
        response_schema_json = json.dumps(json_schema)

    # Create language model client
    lm = _LanguageModel(provider=provider.lower(), default_model=None)

    # Build messages list
    if prompt:
        msg_list = [{"role": "user", "content": prompt}]
    else:
        msg_list = messages or []

    # Convert to Message objects for internal API
    message_objects = []
    for msg in msg_list:
        role = MessageRole(msg["role"])
        if role == MessageRole.USER:
            message_objects.append(Message.user(msg["content"]))
        elif role == MessageRole.ASSISTANT:
            message_objects.append(Message.assistant(msg["content"]))
        elif role == MessageRole.SYSTEM:
            message_objects.append(Message.system(msg["content"]))

    # Build request with Responses API parameters
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

    # Checkpoints are emitted by _LanguageModel.generate() internally
    # to avoid duplication. No need to emit them here.

    # Generate and return
    result = await lm.generate(request)
    return result


async def stream(
    model: str,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    # Responses API specific parameters
    built_in_tools: Optional[List[BuiltInTool]] = None,
    reasoning_effort: Optional[ReasoningEffort] = None,
    modalities: Optional[List[Modality]] = None,
    store: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream text using any LLM provider (simplified API).

    This is the recommended way to use streaming. Provider is auto-detected
    from the model prefix (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-5-haiku').

    Args:
        model: Model identifier with provider prefix (e.g., 'openai/gpt-4o-mini')
        prompt: Simple text prompt (for single-turn requests)
        messages: List of message dicts with 'role' and 'content' (for multi-turn)
        system_prompt: Optional system prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        built_in_tools: List of built-in tools (OpenAI Responses API only)
        reasoning_effort: Reasoning effort level for o-series models (OpenAI Responses API only)
        modalities: Output modalities (text, audio, image) (OpenAI Responses API only)
        store: Enable server-side conversation state (OpenAI Responses API only)
        previous_response_id: Continue from previous response (OpenAI Responses API only)

    Yields:
        Text chunks as they are generated

    Examples:
        Simple streaming:
        >>> async for chunk in stream(
        ...     model="openai/gpt-4o-mini",
        ...     prompt="Write a story"
        ... ):
        ...     print(chunk, end="", flush=True)

        Streaming conversation:
        >>> async for chunk in stream(
        ...     model="groq/llama-3.3-70b-versatile",
        ...     messages=[
        ...         {"role": "user", "content": "Tell me a joke"}
        ...     ],
        ...     temperature=0.9
        ... ):
        ...     print(chunk, end="")
    """
    # Validate input
    if not prompt and not messages:
        raise ValueError("Either 'prompt' or 'messages' must be provided")
    if prompt and messages:
        raise ValueError("Provide either 'prompt' or 'messages', not both")

    # Auto-detect provider from model prefix
    if '/' not in model:
        raise ValueError(
            f"Model must include provider prefix (e.g., 'openai/{model}'). "
            f"Supported providers: openai, anthropic, groq, openrouter, azure, bedrock"
        )

    provider, model_name = model.split('/', 1)

    # Create language model client
    lm = _LanguageModel(provider=provider.lower(), default_model=None)

    # Build messages list
    if prompt:
        msg_list = [{"role": "user", "content": prompt}]
    else:
        msg_list = messages or []

    # Convert to Message objects for internal API
    message_objects = []
    for msg in msg_list:
        role = MessageRole(msg["role"])
        if role == MessageRole.USER:
            message_objects.append(Message.user(msg["content"]))
        elif role == MessageRole.ASSISTANT:
            message_objects.append(Message.assistant(msg["content"]))
        elif role == MessageRole.SYSTEM:
            message_objects.append(Message.system(msg["content"]))

    # Build request with Responses API parameters
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

    # Events are emitted by _LanguageModel.stream() internally
    # (lm.stream.started/completed/failed with trace linkage)

    # Stream and yield chunks
    async for chunk in lm.stream(request):
        yield chunk
