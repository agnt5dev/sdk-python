"""Rust-backed language model client implementation."""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..context import get_current_context
from ..events import Event
from .base import LanguageModel
from .events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
    LMFailed,
    LMStarted,
)
from .types import (
    GenerateRequest,
    GenerateResponse,
    TokenUsage,
)

logger = logging.getLogger(__name__)

try:
    from .._core import LanguageModel as RustLanguageModel
    from .._core import LanguageModelConfig as RustLanguageModelConfig
    from .._core import Response as RustResponse

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    RustLanguageModel = None  # type: ignore
    RustLanguageModelConfig = None  # type: ignore
    RustResponse = None  # type: ignore


class LMClient(LanguageModel):
    """Language Model client using Rust SDK core."""

    def __init__(
        self,
        provider: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        if not _RUST_AVAILABLE:
            raise ImportError(
                "Rust extension not available. Rebuild with: cd sdk/sdk-python && maturin develop"
            )

        self._provider = provider
        self._default_model = default_model

        assert RustLanguageModelConfig is not None
        assert RustLanguageModel is not None
        config = RustLanguageModelConfig(
            default_model=default_model,
            default_provider=provider,
        )
        self._rust_lm = RustLanguageModel(config=config)

    def _prepare_model_name(self, model: str) -> str:
        """Add provider prefix if needed."""
        if '/' in model:
            return model
        if self._provider:
            return f"{self._provider}/{model}"
        return model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from LLM."""
        current_ctx = get_current_context()
        step_key = None
        content_hash = None

        # Check memoization cache
        if current_ctx and hasattr(current_ctx, '_memo') and current_ctx._memo:
            memo = current_ctx._memo
            step_key, content_hash = memo.lm_call_key(
                model=request.model,
                messages=request.messages,
                config={
                    "temperature": request.config.temperature,
                    "max_tokens": request.config.max_tokens,
                }
            )
            cached = await memo.get_cached_lm_result(step_key, content_hash)
            if cached:
                logger.debug(f"LLM call {step_key} served from cache")
                return cached

        prompt = self._build_prompt_messages(request)
        model = self._prepare_model_name(request.model)
        kwargs = self._build_kwargs(request, model)

        # Pass runtime_context for trace linking
        if current_ctx and hasattr(current_ctx, '_runtime_context') and current_ctx._runtime_context:
            kwargs["runtime_context"] = current_ctx._runtime_context

        start_time_ns = time.time_ns()
        correlation_id = f"lm-{secrets.token_hex(5)}"

        if current_ctx:
            self._emit_started(current_ctx, model, request, start_time_ns, correlation_id)

        try:
            rust_response = await self._rust_lm.generate(prompt=prompt, **kwargs)
            response = self._convert_response(rust_response)

            end_time_ns = time.time_ns()
            latency_ms = (end_time_ns - start_time_ns) // 1_000_000

            if current_ctx:
                self._emit_completed(current_ctx, model, response, latency_ms, end_time_ns, correlation_id)

            # Cache result
            if current_ctx and current_ctx._memo and step_key and content_hash:
                await current_ctx._memo.cache_lm_result(step_key, content_hash, response)

            return response

        except Exception as e:
            end_time_ns = time.time_ns()
            latency_ms = (end_time_ns - start_time_ns) // 1_000_000
            if current_ctx:
                self._emit_failed(current_ctx, model, e, latency_ms, end_time_ns, correlation_id)
            raise

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Event, None]:
        """Stream completion from LLM as Event objects."""
        current_ctx = get_current_context()

        prompt = self._build_prompt_messages(request)
        model = self._prepare_model_name(request.model)
        kwargs = self._build_kwargs(request, model)

        block_types: Dict[int, str] = {}
        start_time_ns = time.time_ns()
        correlation_id = f"lm-{secrets.token_hex(5)}"
        parent_correlation_id = current_ctx.correlation_id if current_ctx else ""

        if current_ctx:
            self._emit_started(current_ctx, model, request, start_time_ns, correlation_id)

        try:
            async for chunk in self._rust_lm.stream_iter(prompt=prompt, **kwargs):
                chunk_type = chunk.chunk_type
                block_type = chunk.block_type
                index = chunk.index if chunk.index is not None else 0

                if chunk_type == "content_block_start":
                    block_types[index] = block_type or "text"
                    yield LMContentBlockStarted(
                        name=model,
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                        block_type=block_type or "text",
                        index=index,
                    )

                elif chunk_type == "delta":
                    yield LMContentBlockDelta(
                        name=model,
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                        content=chunk.text,
                        block_type=block_type or "text",
                        index=index,
                    )

                elif chunk_type == "content_block_stop":
                    tracked_type = block_types.get(index, "text")
                    yield LMContentBlockCompleted(
                        name=model,
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                        block_type=tracked_type,
                        index=index,
                    )

                elif chunk_type == "completed":
                    end_time_ns = time.time_ns()
                    latency_ms = (end_time_ns - start_time_ns) // 1_000_000
                    usage = chunk.usage
                    yield LMCompleted(
                        name=model,
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                        model=chunk.model or model,
                        provider=self._provider or "unknown",
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                        total_tokens=usage.total_tokens if usage else 0,
                        finish_reason=chunk.finish_reason,
                        output_data={
                            "text": chunk.text,
                            "model": chunk.model,
                            "timestamp": time.time_ns() // 1_000_000,
                        },
                        duration_ms=latency_ms,
                    )

                    if current_ctx:
                        self._emit_completed(current_ctx, model, chunk, latency_ms, end_time_ns, correlation_id)

        except Exception as e:
            if current_ctx:
                end_time_ns = time.time_ns()
                latency_ms = (end_time_ns - start_time_ns) // 1_000_000
                self._emit_failed(current_ctx, model, e, latency_ms, end_time_ns, correlation_id)

            yield LMFailed(
                name=model,
                correlation_id=correlation_id,
                parent_correlation_id=parent_correlation_id,
                model=model,
                provider=self._provider or "unknown",
                error_code=type(e).__name__,
                error_message=str(e),
            )
            raise

    def _build_kwargs(self, request: GenerateRequest, model: str) -> Dict[str, Any]:
        """Build kwargs dict for Rust call."""
        kwargs: Dict[str, Any] = {"model": model}

        if self._provider:
            kwargs["provider"] = self._provider
        if request.system_prompt:
            kwargs["system_prompt"] = request.system_prompt
        if request.config.temperature is not None:
            kwargs["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            kwargs["max_tokens"] = request.config.max_tokens
        if request.config.top_p is not None:
            kwargs["top_p"] = request.config.top_p
        if request.response_schema is not None:
            kwargs["response_schema_kw"] = request.response_schema

        # Responses API parameters
        if request.config.built_in_tools:
            kwargs["built_in_tools"] = json.dumps([t.value for t in request.config.built_in_tools])
        if request.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.config.reasoning_effort.value
        if request.config.modalities is not None:
            kwargs["modalities"] = json.dumps([m.value for m in request.config.modalities])
        if request.config.store is not None:
            kwargs["store"] = request.config.store
        if request.config.previous_response_id is not None:
            kwargs["previous_response_id"] = request.config.previous_response_id

        # Tools
        if request.tools:
            tools_list = [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in request.tools
            ]
            kwargs["tools"] = json.dumps(tools_list)
        if request.tool_choice:
            kwargs["tool_choice"] = json.dumps(request.tool_choice.value)

        return kwargs

    def _build_prompt_messages(self, request: GenerateRequest) -> List[Dict[str, Any]]:
        """Build message list for Rust."""
        messages = []
        for msg in request.messages:
            msg_dict: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages.append(msg_dict)

        if not messages and not request.system_prompt:
            messages.append({"role": "user", "content": ""})

        return messages

    def _convert_response(self, rust_response: Any) -> GenerateResponse:
        """Convert Rust response to Python response."""
        usage = None
        if rust_response.usage:
            usage = TokenUsage(
                prompt_tokens=rust_response.usage.prompt_tokens,
                completion_tokens=rust_response.usage.completion_tokens,
                total_tokens=rust_response.usage.total_tokens,
            )

        tool_calls = None
        if hasattr(rust_response, 'tool_calls') and rust_response.tool_calls:
            tool_calls = rust_response.tool_calls

        response_id = None
        if hasattr(rust_response, 'id') and rust_response.id:
            response_id = rust_response.id

        return GenerateResponse(
            text=rust_response.content,
            usage=usage,
            finish_reason=None,
            tool_calls=tool_calls,
            response_id=response_id,
            _rust_response=rust_response,
        )

    def _emit_started(
        self,
        ctx: Any,
        model: str,
        request: GenerateRequest,
        timestamp_ns: int,
        correlation_id: str,
    ) -> None:
        # Serialize messages for event
        messages = [
            {"role": m.role.value, "content": m.content}
            for m in request.messages
        ]
        started_event = LMStarted(
            name=model,
            correlation_id=correlation_id,
            parent_correlation_id=ctx.correlation_id,
            model=model,
            provider=self._provider or "unknown",
            input_data={
                "system_prompt": request.system_prompt,
                "messages": messages,
                "temperature": request.config.temperature,
                "max_tokens": request.config.max_tokens,
                "tools_count": len(request.tools) if request.tools else 0,
            },
            metadata={"name": model},
        )
        ctx.emit(started_event)

    def _emit_completed(
        self,
        ctx: Any,
        model: str,
        response: Any,  # Can be GenerateResponse or chunk
        latency_ms: int,
        timestamp_ns: int,
        correlation_id: str,
    ) -> None:
        # Handle both GenerateResponse and streaming chunk
        if hasattr(response, 'text'):
            output_text = response.text
        else:
            output_text = getattr(response, 'content', str(response))

        usage = getattr(response, 'usage', None)
        completed_event = LMCompleted(
            name=model,
            correlation_id=correlation_id,
            parent_correlation_id=ctx.correlation_id,
            model=model,
            provider=self._provider or "unknown",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=getattr(response, 'finish_reason', None),
            output_data={
                "output": output_text,
                "tool_calls": getattr(response, 'tool_calls', None),
            },
            duration_ms=latency_ms,
            metadata={"name": model},
        )
        ctx.emit(completed_event)

    def _emit_failed(
        self,
        ctx: Any,
        model: str,
        error: Exception,
        latency_ms: int,
        timestamp_ns: int,
        correlation_id: str,
    ) -> None:
        failed_event = LMFailed(
            name=model,
            correlation_id=correlation_id,
            parent_correlation_id=ctx.correlation_id,
            model=model,
            provider=self._provider or "unknown",
            error_code=type(error).__name__,
            error_message=str(error),
            duration_ms=latency_ms,
            metadata={"name": model},
        )
        ctx.emit(failed_event)
