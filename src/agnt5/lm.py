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
from .journal import (
    LMCallStartedEvent,
    LMCallCompletedEvent,
    LMCallFailedEvent,
    write_lm_call_started,
    write_lm_call_completed,
    write_lm_call_failed,
)

try:
    from ._core import LanguageModel as RustLanguageModel
    from ._core import LanguageModelConfig as RustLanguageModelConfig
    from ._core import Response as RustResponse
    from ._core import StreamChunk as RustStreamChunk
    from ._core import AsyncStreamHandle as RustAsyncStreamHandle
    from ._core import Usage as RustUsage
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    RustLanguageModel = None
    RustLanguageModelConfig = None
    RustResponse = None
    RustStreamChunk = None
    RustAsyncStreamHandle = None
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
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @staticmethod
    def system(content: str) -> Message:
        """Create system message."""
        return Message(role=MessageRole.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> Message:
        """Create user message."""
        return Message(role=MessageRole.USER, content=content)

    @staticmethod
    def assistant(
        content: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Create assistant message, optionally with tool calls."""
        return Message(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> Message:
        """Create tool result message.

        Args:
            tool_call_id: The ID of the tool call this is a response to
            content: The result of the tool execution
        """
        return Message(
            role=MessageRole.USER,  # Tool results are sent as user messages in most APIs
            content=content,
            tool_call_id=tool_call_id,
        )


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
    async def stream(self, request: GenerateRequest) -> AsyncIterator["Event"]:
        """Stream completion from LLM as Event objects.

        Yields typed Event objects for real-time SSE streaming:
        - lm.message.start: Beginning of message content
        - lm.message.delta: Token chunk with incremental text
        - lm.message.stop: End of message content

        Args:
            request: Generation request with model, messages, and configuration

        Yields:
            Event objects for streaming
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

        Note:
            If memoization is enabled on the current context, this method will
            check the journal for cached results before executing and cache
            results after successful execution.
        """
        # Check for memoization before expensive LLM call
        current_ctx = get_current_context()
        step_key = None
        content_hash = None

        if current_ctx and hasattr(current_ctx, '_memo') and current_ctx._memo:
            # Generate step_key and content_hash for memoization
            memo = current_ctx._memo
            step_key, content_hash = memo.lm_call_key(
                model=request.model,
                messages=request.messages,
                config={
                    "temperature": request.config.temperature,
                    "max_tokens": request.config.max_tokens,
                }
            )

            # Check cache first - skip expensive LLM call if cached
            cached = await memo.get_cached_lm_result(step_key, content_hash)
            if cached:
                logger.debug(f"LLM call {step_key} served from memoization cache")
                return cached

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

        # Get trace context for event linkage using AGNT5's tracing system
        trace_id = None
        span_id = None
        try:
            from .tracing import get_current_span_info
            span_info = get_current_span_info()
            if span_info:
                trace_id = span_info.trace_id
                span_id = span_info.span_id
        except Exception as e:
            import logging
            logging.getLogger("agnt5.lm").warning(f"🔍 LM-DEBUG: Failed to get span info: {e}")

        # Get run_id for journal events - use runtime_context.run_id (base invocation_id)
        # NOT current_ctx.run_id which may have :agent:name suffix
        run_id = None
        if current_ctx and hasattr(current_ctx, '_runtime_context') and current_ctx._runtime_context:
            run_id = current_ctx._runtime_context.run_id
        tenant_id = None  # TODO: Get from context when available

        # Track start time for latency calculation (nanoseconds for precision)
        start_time_ns = time.time_ns()

        # Write journal event for LLM observability (in addition to checkpoint for streaming)
        if run_id and trace_id and span_id:
            started_event = LMCallStartedEvent(
                model=model,
                provider=self._provider or "unknown",
                temperature=request.config.temperature,
                max_tokens=request.config.max_tokens,
                tools_count=len(request.config.tools) if request.config.tools else 0,
                timestamp_ns=start_time_ns,
            )
            await write_lm_call_started(
                run_id=run_id,
                trace_id=trace_id,
                span_id=span_id,
                event=started_event,
                tenant_id=tenant_id,
            )
        # Emit checkpoint for real-time streaming (separate from journal)
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

            # Calculate latency (in ms for human readability)
            end_time_ns = time.time_ns()
            latency_ms = (end_time_ns - start_time_ns) // 1_000_000

            # Write journal event for LLM observability
            if run_id and trace_id and span_id:
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                completed_event = LMCallCompletedEvent(
                    model=model,
                    provider=self._provider or "unknown",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    finish_reason=response.finish_reason,
                    tool_calls_count=len(response.tool_calls) if response.tool_calls else 0,
                    timestamp_ns=end_time_ns,
                )
                await write_lm_call_completed(
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    event=completed_event,
                    tenant_id=tenant_id,
                )

            # Emit checkpoint for real-time streaming (separate from journal)
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

            # Cache result for replay if memoization is enabled
            if current_ctx and current_ctx._memo and step_key:
                await current_ctx._memo.cache_lm_result(step_key, content_hash, response)

            return response
        except Exception as e:
            # Calculate latency for failed call (in ms for human readability)
            end_time_ns = time.time_ns()
            latency_ms = (end_time_ns - start_time_ns) // 1_000_000

            # Write journal event for LLM failure
            if run_id and trace_id and span_id:
                failed_event = LMCallFailedEvent(
                    model=model,
                    provider=self._provider or "unknown",
                    error_code=type(e).__name__,
                    error_message=str(e),
                    latency_ms=latency_ms,
                    timestamp_ns=end_time_ns,
                )
                await write_lm_call_failed(
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    event=failed_event,
                    tenant_id=tenant_id,
                )

            # Emit checkpoint for real-time streaming (separate from journal)
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

    async def stream(self, request: GenerateRequest) -> AsyncIterator["Event"]:
        """Stream completion from LLM as Event objects for SSE delivery.

        This method yields typed Event objects suitable for real-time streaming
        via SSE. It emits content block events following the pattern:
        - lm.message.start / lm.thinking.start: Beginning of content block
        - lm.message.delta / lm.thinking.delta: Token chunk with incremental text
        - lm.message.stop / lm.thinking.stop: End of content block

        Extended thinking models (Claude with extended thinking) emit thinking blocks
        before text blocks, allowing you to see the model's reasoning process.

        Args:
            request: Generation request with model, messages, and configuration

        Yields:
            Event objects for streaming

        Example:
            ```python
            async for event in lm_instance.stream(request):
                if event.event_type == EventType.LM_MESSAGE_DELTA:
                    print(event.data.get("content", ""), end="", flush=True)
                elif event.event_type == EventType.LM_THINKING_DELTA:
                    # Handle thinking content (optional)
                    pass
            ```
        """
        from .events import Event, EventType
        from .context import get_current_context

        current_ctx = get_current_context()

        # Convert Python request to structured format for Rust
        prompt = self._build_prompt_messages(request)

        # Prepare model name with provider prefix
        model = self._prepare_model_name(request.model)

        # Build kwargs for Rust
        kwargs: dict[str, Any] = {
            "model": model,
        }

        # Always pass provider explicitly if set
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
            kwargs["tool_choice"] = json.dumps(request.tool_choice.value)

        import time
        sequence = 0
        # Track block types by index since content_block_stop doesn't include block_type
        block_types: Dict[int, str] = {}

        # Get trace context for journal events
        trace_id = None
        span_id = None
        try:
            from .tracing import get_current_span_info
            span_info = get_current_span_info()
            if span_info:
                trace_id = span_info.trace_id
                span_id = span_info.span_id
        except Exception:
            pass

        # Get run_id for journal events - use runtime_context.run_id (base invocation_id)
        # NOT current_ctx.run_id which may have :agent:name suffix
        run_id = None
        if current_ctx and hasattr(current_ctx, '_runtime_context') and current_ctx._runtime_context:
            run_id = current_ctx._runtime_context.run_id
        tenant_id = None

        # Track timing (nanoseconds for precision)
        start_time_ns = time.time_ns()

        # Write lm.call.started journal event
        if run_id and trace_id and span_id:
            started_event = LMCallStartedEvent(
                model=model,
                provider=self._provider or "unknown",
                temperature=request.config.temperature,
                max_tokens=request.config.max_tokens,
                tools_count=len(request.tools) if request.tools else 0,
                timestamp_ns=start_time_ns,
            )
            await write_lm_call_started(
                run_id=run_id,
                trace_id=trace_id,
                span_id=span_id,
                event=started_event,
                tenant_id=tenant_id,
            )

        try:
            # Use stream_iter for true async streaming - yields chunks as they arrive
            # instead of collecting all chunks first
            async for chunk in self._rust_lm.stream_iter(prompt=prompt, **kwargs):
                chunk_type = chunk.chunk_type
                block_type = chunk.block_type  # "text" or "thinking" (None for stop/completed)
                index = chunk.index if chunk.index is not None else 0

                if chunk_type == "content_block_start":
                    # Track block type for this index
                    block_types[index] = block_type or "text"
                    # Emit start event based on block type
                    if block_type == "thinking":
                        yield Event.thinking_start(
                            index=index,
                            sequence=sequence,
                        )
                    else:
                        yield Event.message_start(
                            index=index,
                            sequence=sequence,
                        )
                    sequence += 1

                elif chunk_type == "delta":
                    # Emit delta event based on block type
                    if block_type == "thinking":
                        yield Event.thinking_delta(
                            content=chunk.text,
                            index=index,
                            sequence=sequence,
                        )
                    else:
                        yield Event.message_delta(
                            content=chunk.text,
                            index=index,
                            sequence=sequence,
                        )
                    sequence += 1

                elif chunk_type == "content_block_stop":
                    # Look up block type from when we saw content_block_start
                    tracked_block_type = block_types.get(index, "text")
                    # Emit stop event based on tracked block type
                    if tracked_block_type == "thinking":
                        yield Event.thinking_stop(
                            index=index,
                            sequence=sequence,
                        )
                    else:
                        yield Event.message_stop(
                            index=index,
                            sequence=sequence,
                        )
                    sequence += 1

                elif chunk_type == "completed":
                    # Final response - emit completion event
                    completion_data = {
                        "text": chunk.text,
                        "model": chunk.model,
                        "timestamp": time.time_ns() // 1_000_000,
                    }
                    if chunk.finish_reason:
                        completion_data["finish_reason"] = chunk.finish_reason
                    if chunk.usage:
                        completion_data["usage"] = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    yield Event(
                        event_type=EventType.LM_STREAM_COMPLETED,
                        data=completion_data,
                        sequence=sequence,
                    )
                    sequence += 1

                    # Write lm.call.completed journal event
                    if run_id and trace_id and span_id:
                        end_time_ns = time.time_ns()
                        latency_ms = (end_time_ns - start_time_ns) // 1_000_000
                        completed_event = LMCallCompletedEvent(
                            model=model,
                            provider=self._provider or "unknown",
                            input_tokens=chunk.usage.prompt_tokens if chunk.usage else 0,
                            output_tokens=chunk.usage.completion_tokens if chunk.usage else 0,
                            total_tokens=chunk.usage.total_tokens if chunk.usage else 0,
                            latency_ms=latency_ms,
                            finish_reason=chunk.finish_reason,
                            tool_calls_count=0,  # TODO: track tool calls in streaming
                            timestamp_ns=end_time_ns,
                        )
                        await write_lm_call_completed(
                            run_id=run_id,
                            trace_id=trace_id,
                            span_id=span_id,
                            event=completed_event,
                            tenant_id=tenant_id,
                        )

        except Exception as e:
            # Write lm.call.failed journal event
            if run_id and trace_id and span_id:
                end_time_ns = time.time_ns()
                latency_ms = (end_time_ns - start_time_ns) // 1_000_000
                failed_event = LMCallFailedEvent(
                    model=model,
                    provider=self._provider or "unknown",
                    error_code=type(e).__name__,
                    error_message=str(e),
                    latency_ms=latency_ms,
                    timestamp_ns=end_time_ns,
                )
                await write_lm_call_failed(
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    event=failed_event,
                    tenant_id=tenant_id,
                )

            # Emit error as a failed event (caller can handle)
            yield Event(
                event_type=EventType.LM_STREAM_FAILED,
                data={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time_ns() // 1_000_000,
                },
                sequence=sequence,
            )
            raise

    def _build_prompt_messages(self, request: GenerateRequest) -> List[Dict[str, Any]]:
        """Build structured message list for Rust.

        Rust expects a list of dicts with 'role', 'content', and optional fields:
        - tool_calls: List of tool calls for assistant messages
        - tool_call_id: ID of the tool call this message responds to
        System prompt is passed separately via kwargs.

        Args:
            request: Generation request with messages

        Returns:
            List of message dicts with role, content, and optional tool fields
        """
        # Convert messages to Rust format (list of dicts with role, content, and optional fields)
        messages = []
        for msg in request.messages:
            msg_dict: Dict[str, Any] = {
                "role": msg.role.value,  # "system", "user", or "assistant"
                "content": msg.content
            }
            # Include tool_calls for assistant messages that have them
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            # Include tool_call_id for tool result messages
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages.append(msg_dict)

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
        # PyResponse exposes .id which is the response ID for conversation continuation
        response_id = None
        if hasattr(rust_response, 'id') and rust_response.id:
            response_id = rust_response.id

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
) -> AsyncIterator["Event"]:
    """Stream LLM completion as Event objects (simplified API).

    This is the recommended way to use streaming. Provider is auto-detected
    from the model prefix (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-5-haiku').

    Yields Event objects for real-time SSE streaming:
    - lm.message.start: Beginning of message content
    - lm.message.delta: Token chunk with incremental text
    - lm.message.stop: End of message content

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
        Event objects for streaming

    Examples:
        Simple streaming:
        >>> from agnt5.events import EventType
        >>> async for event in stream(
        ...     model="openai/gpt-4o-mini",
        ...     prompt="Write a story"
        ... ):
        ...     if event.event_type == EventType.LM_MESSAGE_DELTA:
        ...         print(event.data.get("content", ""), end="", flush=True)

        Streaming conversation:
        >>> async for event in stream(
        ...     model="groq/llama-3.3-70b-versatile",
        ...     messages=[{"role": "user", "content": "Tell me a joke"}],
        ...     temperature=0.9
        ... ):
        ...     if event.event_type == EventType.LM_MESSAGE_DELTA:
        ...         print(event.data.get("content", ""), end="")
    """
    from .events import Event
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
