"""
Language Model API for AGNT5 SDK

This module provides simplified access to language models with two main classes:
- LanguageModel: Async-first interface (default)
- SyncLanguageModel: Synchronous wrapper for REPL/Jupyter convenience

The API follows the principle that async is the default, with sync as a convenience wrapper.
It supports plain-text completions, JSON outputs, JSON Schema-constrained responses, and
OpenAI-style tool invocation when the provider implements it.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
import json

try:
    from agnt5._core import (
        LanguageModel as _CoreLanguageModel,
        LanguageModelConfig as _CoreLanguageModelConfig,
        Response as _CoreResponse,
        StreamChunk as _CoreStreamChunk,
        Usage as _CoreUsage,
    )
except ImportError:
    # Fallback for when the Rust extension is not available
    _CoreLanguageModel = None
    _CoreLanguageModelConfig = None
    _CoreResponse = None
    _CoreStreamChunk = None
    _CoreUsage = None


class Response:
    """Response from language model generation."""

    def __init__(self, core_response: _CoreResponse):
        self._core = core_response

    @property
    def text(self) -> str:
        """The generated text content."""
        return self._core.text

    @property
    def content(self) -> str:
        """Alias for text (Anthropic-style)."""
        return self._core.content

    @property
    def model(self) -> str:
        """The model that generated this response."""
        return self._core.model

    @property
    def object(self) -> Optional[Any]:
        """Parsed structured object (if response_format was specified)."""
        return self._core.object

    @property
    def usage(self) -> Optional[Usage]:
        """Token usage information."""
        core_usage = self._core.usage
        return Usage(core_usage) if core_usage else None

    @property
    def finish_reason(self) -> Optional[str]:
        """Finish reason from the model."""
        return self._core.finish_reason

    @property
    def raw(self) -> Optional[Dict[str, Any]]:
        """Raw response data from the provider."""
        return self._core.raw


class StreamChunk:
    """A chunk from streaming generation."""

    def __init__(self, core_chunk: _CoreStreamChunk):
        self._core = core_chunk

    @property
    def text(self) -> str:
        """The text delta for this chunk."""
        return self._core.text

    @property
    def model(self) -> str:
        """The model generating this chunk."""
        return self._core.model

    @property
    def object(self) -> Optional[Any]:
        """Partial structured object (if parseable)."""
        return self._core.object

    @property
    def finished(self) -> bool:
        """Whether this is the final chunk."""
        return self._core.finished

    @property
    def finish_reason(self) -> Optional[str]:
        """Finish reason (if this is the final chunk)."""
        return self._core.finish_reason

    @property
    def raw(self) -> Optional[Dict[str, Any]]:
        """Raw chunk data from the provider."""
        return self._core.raw


class Usage:
    """Token usage information."""

    def __init__(self, core_usage: _CoreUsage):
        self._core = core_usage

    @property
    def input_tokens(self) -> Optional[int]:
        """Number of tokens in the input (alias for prompt_tokens)."""
        return self._core.input_tokens

    @property
    def prompt_tokens(self) -> Optional[int]:
        """Number of tokens in the prompt."""
        return self._core.prompt_tokens

    @property
    def output_tokens(self) -> Optional[int]:
        """Number of tokens in the output (alias for completion_tokens)."""
        return self._core.output_tokens

    @property
    def completion_tokens(self) -> Optional[int]:
        """Number of tokens in the completion."""
        return self._core.completion_tokens

    @property
    def total_tokens(self) -> Optional[int]:
        """Total number of tokens (input + output)."""
        return self._core.total_tokens


class LanguageModelConfig:
    """Configuration for LanguageModel."""

    def __init__(
        self,
        default_model: Optional[str] = None,
        default_provider: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize configuration.

        Args:
            default_model: Default model to use
            default_provider: Default provider to use
            **kwargs: Reserved for future use
        """
        if _CoreLanguageModelConfig is None:
            raise ImportError("Language model functionality requires the Rust extension")

        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(f"Unsupported configuration options: {unsupported}")

        self._core = _CoreLanguageModelConfig(
            default_model=default_model,
            default_provider=default_provider,
        )

    @classmethod
    def from_environment(cls) -> LanguageModelConfig:
        """Create configuration from environment variables.

        Environment variables:
        - LM_MODEL / LM_DEFAULT_MODEL: Default model
        - LM_PROVIDER: Default provider
        - Provider-specific keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
        """
        if _CoreLanguageModelConfig is None:
            raise ImportError("Language model functionality requires the Rust extension")

        config = cls.__new__(cls)
        config._core = _CoreLanguageModelConfig.from_environment()
        return config


class LanguageModel:
    """
    Async-first language model client.

    This is the main interface for language model operations. It provides
    two core methods:
    - generate(): For non-streaming generation (text and structured)
    - stream(): For streaming generation (text and structured)

    Examples:
        ```python
        from agnt5 import LanguageModel

        lm = LanguageModel()

        # Simple text generation
        response = await lm.generate("What is love?")
        print(response.text)

        # Chat-style generation
        response = await lm.generate([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ])
        print(response.text)

        # Structured JSON output
        response = await lm.generate(
            "Give me a name and age as JSON",
            response_format="json",
        )
        print(response.object)

        response = await lm.generate(
            "Return a task with status",
            response_format="json_schema",
            schema={
                "name": "task",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "status": {"type": "string", "enum": ["todo", "doing", "done"]}
                    },
                    "required": ["title", "status"]
                },
            },
        )
        print(response.object)

        # Streaming
        async for chunk in lm.stream("Tell me a story"):
            print(chunk.text, end="")
        ```
    """

    def __init__(self, config: Optional[LanguageModelConfig] = None):
        """Initialize LanguageModel.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        if _CoreLanguageModel is None:
            raise ImportError("Language model functionality requires the Rust extension")

        core_config = config._core if config else None
        self._core = _CoreLanguageModel(config=core_config)

    async def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Response:
        """Generate text or structured data.

        Args:
            prompt: Text prompt or list of chat messages
            model: Model to use (overrides default)
            provider: Provider to use (overrides default)
            system_prompt: Optional system prompt to apply
            response_format: Response format (``"text"``, ``"json"``, ``"json_schema"`` or schema dict)
            schema: Optional JSON schema dict. Supply either a raw JSON schema or a dict with
                ``{"name": ..., "schema": {...}, "strict": bool}`` to mirror OpenAI's schema API.
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0.0 to 2.0)
            top_p: Top-p nucleus sampling (0.0 to 1.0)
            stop: Stop sequences (not yet supported)
            seed: Seed for deterministic generation (not yet supported)
            user: User identifier
            tools: Optional OpenAI-compatible tool definitions (list of dicts containing
                ``name`` plus optional ``description``, ``parameters`` JSON schema, and ``strict``)
            tool_choice: Tool selection strategy ("auto", "none", or a tool descriptor such as
                ``{"type": "function", "function": {"name": "..."}}``)
            **kwargs: Additional provider-specific parameters

        Returns:
            Response object with generated content
        """
        response_format_arg: Optional[str] = None
        response_schema_arg: Optional[str] = None

        if response_format is not None:
            if isinstance(response_format, dict):
                response_format_arg = "json_schema"
                response_schema_arg = json.dumps(response_format)
            elif isinstance(response_format, str):
                fmt = response_format.lower()
                if fmt in {"json", "json_object"}:
                    response_format_arg = "json"
                elif fmt in {"json_schema", "schema"}:
                    response_format_arg = "json_schema"
                elif fmt in {"text", "plain", "default"}:
                    response_format_arg = "text"
                else:
                    raise NotImplementedError(
                        "response_format must be 'text', 'json', or 'json_schema'"
                    )
            else:
                raise NotImplementedError("response_format must be a string or dict")

        if schema is not None:
            response_schema_arg = json.dumps(schema)
            if response_format_arg is None:
                response_format_arg = "json_schema"
            elif response_format_arg not in {"json", "json_schema"}:
                raise ValueError(
                    "schema can only be provided with response_format='json' or 'json_schema'"
                )
            else:
                response_format_arg = "json_schema"

        tools_arg: Optional[str] = None
        if tools is not None:
            tools_arg = json.dumps(tools)

        tool_choice_arg: Optional[str] = None
        if tool_choice is not None:
            tool_choice_arg = json.dumps(tool_choice)

        if stop is not None:
            raise NotImplementedError("stop sequences are not supported yet")

        if seed is not None:
            raise NotImplementedError("seed is not supported yet")

        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(f"Unsupported options: {unsupported}")

        core_response = self._core.generate(
            prompt,
            model=model,
            provider=provider,
            system_prompt=system_prompt,
            response_format=response_format_arg,
            response_schema=response_schema_arg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            user=user,
            tools=tools_arg,
            tool_choice=tool_choice_arg,
        )
        return Response(core_response)

    async def stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream text or structured data generation.

        Args:
            prompt: Text prompt or list of chat messages
            model: Model to use (overrides default)
            provider: Provider to use (overrides default)
            system_prompt: Optional system prompt to apply
            response_format: Response format (``"text"``, ``"json"``, ``"json_schema"`` or schema dict)
            schema: Optional JSON schema dict used for structured streaming responses
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling (0.0 to 2.0)
            top_p: Top-p nucleus sampling (0.0 to 1.0)
            stop: Stop sequences (not yet supported)
            seed: Seed for deterministic generation (not yet supported)
            user: User identifier
            tools: Optional OpenAI-compatible tool definitions
            tool_choice: Tool selection strategy ("auto", "none", or tool descriptor)
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with generated content
        """
        response_format_arg: Optional[str] = None
        response_schema_arg: Optional[str] = None

        if response_format is not None:
            if isinstance(response_format, dict):
                response_format_arg = "json_schema"
                response_schema_arg = json.dumps(response_format)
            elif isinstance(response_format, str):
                fmt = response_format.lower()
                if fmt in {"json", "json_object"}:
                    response_format_arg = "json"
                elif fmt in {"json_schema", "schema"}:
                    response_format_arg = "json_schema"
                elif fmt in {"text", "plain", "default"}:
                    response_format_arg = "text"
                else:
                    raise NotImplementedError(
                        "response_format must be 'text', 'json', or 'json_schema'"
                    )
            else:
                raise NotImplementedError("response_format must be a string or dict")

        if schema is not None:
            response_schema_arg = json.dumps(schema)
            if response_format_arg is None:
                response_format_arg = "json_schema"
            elif response_format_arg not in {"json", "json_schema"}:
                raise ValueError(
                    "schema can only be provided with response_format='json' or 'json_schema'"
                )
            else:
                response_format_arg = "json_schema"

        tools_arg: Optional[str] = None
        if tools is not None:
            tools_arg = json.dumps(tools)

        tool_choice_arg: Optional[str] = None
        if tool_choice is not None:
            tool_choice_arg = json.dumps(tool_choice)

        if stop is not None:
            raise NotImplementedError("stop sequences are not supported yet")

        if seed is not None:
            raise NotImplementedError("seed is not supported yet")

        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(f"Unsupported options: {unsupported}")

        core_chunks = self._core.stream(
            prompt,
            model=model,
            provider=provider,
            system_prompt=system_prompt,
            response_format=response_format_arg,
            response_schema=response_schema_arg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            user=user,
            tools=tools_arg,
            tool_choice=tool_choice_arg,
        )
        for core_chunk in core_chunks:
            yield StreamChunk(core_chunk)

    def list_providers(self) -> List[str]:
        """List available providers."""
        return self._core.list_providers()


class SyncLanguageModel:
    """
    Synchronous wrapper for LanguageModel.

    This class provides a synchronous interface to the async LanguageModel,
    primarily for convenience in REPL and Jupyter notebook environments.

    For production code, prefer the async LanguageModel.

    Examples:
        ```python
        from agnt5 import SyncLanguageModel

        lm = SyncLanguageModel()

        # Simple text generation (sync)
        response = lm.generate("What is love?")
        print(response.text)

        # Streaming (sync)
        for chunk in lm.stream("Tell me a story"):
            print(chunk.text, end="")
        ```
    """

    def __init__(self, config: Optional[LanguageModelConfig] = None):
        """Initialize SyncLanguageModel.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self._lm = LanguageModel(config)
        self._loop = None

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any,
    ) -> Response:
        """Generate text or structured data (synchronous).

        Args:
            prompt: Text prompt or list of chat messages
            **kwargs: Same arguments as LanguageModel.generate()

        Returns:
            Response object with generated content
        """
        return self._run_async(self._lm.generate(prompt, **kwargs))

    def stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Stream text or structured data generation (synchronous).

        Args:
            prompt: Text prompt or list of chat messages
            **kwargs: Same arguments as LanguageModel.stream()

        Yields:
            StreamChunk objects with generated content
        """

        async def collect_chunks():
            chunks = []
            async for chunk in self._lm.stream(prompt, **kwargs):
                chunks.append(chunk)
            return chunks

        chunks = self._run_async(collect_chunks())
        for chunk in chunks:
            yield chunk

    def list_providers(self) -> List[str]:
        """List available providers."""
        return self._lm.list_providers()

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            # Try to handle nested event loops (Jupyter)
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                pass

            # Try to run in new event loop
            try:
                return asyncio.run(coro)
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    # We're in an existing event loop (Jupyter), need to use different approach
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a task and wait for it
                        import concurrent.futures
                        import threading

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(coro)
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            return future.result()
                    else:
                        return loop.run_until_complete(coro)
                else:
                    raise
        except Exception as e:
            raise RuntimeError(f"Failed to run async operation: {e}") from e


__all__ = [
    "LanguageModel",
    "SyncLanguageModel",
    "LanguageModelConfig",
    "Response",
    "StreamChunk",
    "Usage",
]
