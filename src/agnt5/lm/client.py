"""Rust-backed language model client implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .._ids import generate_cid
from .._serialization import deserialize, serialize
from ..activation import (
    ActivationDecision,
    ActivationDecisionKind,
    ActivationEvidence,
    ActivationKind,
    ActivationRecoveryPolicy,
    ActivationUsage,
    _reset_current_activation,
    _set_current_activation,
    activation_request_from_context,
)
from ..context import get_current_context
from ..events import Event
from ..prompt_manifest import resolve_prompt_from_manifest
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
_model_activation_execute: ContextVar[bool] = ContextVar(
    "agnt5_model_activation_execute",
    default=False,
)
_model_stream_activation_execute: ContextVar[bool] = ContextVar(
    "agnt5_model_stream_activation_execute",
    default=False,
)

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


def _model_recovery_policy(value: str | None) -> ActivationRecoveryPolicy:
    normalized = (value or "unknown_outcome").strip().lower()
    mapping = {
        "idempotent_retry": ActivationRecoveryPolicy.IDEMPOTENT_RETRY,
        "durable_steps": ActivationRecoveryPolicy.DURABLE_STEPS,
        "unknown_outcome": ActivationRecoveryPolicy.UNKNOWN_OUTCOME,
        "compensate": ActivationRecoveryPolicy.COMPENSATE,
        "fail": ActivationRecoveryPolicy.FAIL,
    }
    try:
        return mapping[normalized]
    except KeyError as error:
        from ..exceptions import ActivationError, ActivationErrorCode

        raise ActivationError(
            ActivationErrorCode.INVALID_ARGUMENT,
            f"Unsupported model recovery policy: {value!r}",
        ) from error


def _model_activation_input(request: GenerateRequest) -> Dict[str, Any]:
    cache = request.config.cache
    return {
        "model": request.model,
        "system_prompt": request.system_prompt,
        "messages": [
            {
                "role": message.role.value,
                "content": message.content,
                "tool_calls": message.tool_calls,
                "tool_call_id": message.tool_call_id,
            }
            for message in request.messages
        ],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in request.tools
        ],
        "tool_choice": request.tool_choice.value if request.tool_choice else None,
        "config": {
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_tokens,
            "top_p": request.config.top_p,
            "cache": (
                {
                    "enabled": cache.enabled,
                    "ttl": cache.ttl,
                    "key": cache.key,
                    "retention": cache.retention,
                    "resource": cache.resource,
                }
                if cache
                else None
            ),
            "built_in_tools": [tool.value for tool in request.config.built_in_tools],
            "reasoning_effort": (
                request.config.reasoning_effort.value
                if request.config.reasoning_effort
                else None
            ),
            "modalities": [value.value for value in request.config.modalities or []],
            "store": request.config.store,
            "previous_response_id": request.config.previous_response_id,
        },
        "response_schema": request.response_schema,
    }


def _model_terminal_evidence(response: GenerateResponse) -> tuple[ActivationEvidence, ...]:
    payload = json.dumps(
        {
            "schema": "agnt5.model_provider_terminal.v1",
            "classification": "accepted_final",
            "finish_reason": response.finish_reason,
            "response_id": response.response_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (ActivationEvidence.inline("model_provider_terminal_v1", payload),)


class LMClient(LanguageModel):
    """Language Model client using Rust SDK core."""

    _STREAMING_TOOL_PROVIDERS = frozenset({"openai"})

    def __init__(
        self,
        provider: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        if not _RUST_AVAILABLE:
            raise ImportError(
                "Rust extension not available. Rebuild with: maturin develop"
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

    @classmethod
    def supports_streaming_tools_for_provider(cls, provider: Optional[str]) -> bool:
        """Return whether the Rust provider preserves tool calls while streaming."""
        return bool(provider and provider.lower() in cls._STREAMING_TOOL_PROVIDERS)

    @property
    def supports_streaming_tools(self) -> bool:
        return self.supports_streaming_tools_for_provider(self._provider)

    def _prepare_model_name(self, model: str) -> str:
        """Add provider prefix if needed."""
        if "/" in model:
            return model
        if self._provider:
            return f"{self._provider}/{model}"
        return model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from LLM."""
        if request.prompt_ref is not None:
            return await self._run_managed_prompt(request)

        current_ctx = get_current_context()
        durable_execute = _model_activation_execute.get()
        if (
            current_ctx
            and not durable_execute
            and (getattr(current_ctx, "_trace_metadata", None) or {}).get(
                "durable_activation_v1"
            )
            == "true"
        ):
            return await self._generate_durable(request, current_ctx)
        step_key = None
        content_hash = None

        # Check memoization cache
        if (
            not durable_execute
            and current_ctx
            and hasattr(current_ctx, "_memo")
            and current_ctx._memo
        ):
            memo = current_ctx._memo
            step_key, content_hash = memo.lm_call_key(
                model=request.model,
                messages=request.messages,
                config={
                    "temperature": request.config.temperature,
                    "max_tokens": request.config.max_tokens,
                },
            )
            cached = await memo.get_cached_lm_result(step_key, content_hash)
            if cached:
                logger.debug(f"LLM call {step_key} served from cache")
                return cached

        prompt = self._build_prompt_messages(request)
        model = self._prepare_model_name(request.model)
        kwargs = self._build_kwargs(request, model)

        # Pass runtime_context for trace linking
        if (
            current_ctx
            and hasattr(current_ctx, "_runtime_context")
            and current_ctx._runtime_context
        ):
            kwargs["runtime_context"] = current_ctx._runtime_context

        start_time_ns = time.time_ns()
        correlation_id = generate_cid()

        if current_ctx:
            self._emit_started(current_ctx, model, request, start_time_ns, correlation_id)

        try:
            rust_response = await self._rust_lm.generate(prompt=prompt, **kwargs)
            response = self._convert_response(rust_response)

            end_time_ns = time.time_ns()
            latency_ms = (end_time_ns - start_time_ns) // 1_000_000

            if current_ctx:
                self._emit_completed(
                    current_ctx, model, response, latency_ms, end_time_ns, correlation_id
                )

            # Cache result
            if (
                not durable_execute
                and current_ctx
                and current_ctx._memo
                and step_key
                and content_hash
            ):
                await current_ctx._memo.cache_lm_result(step_key, content_hash, response)

            return response

        except Exception as e:
            end_time_ns = time.time_ns()
            latency_ms = (end_time_ns - start_time_ns) // 1_000_000
            if current_ctx:
                self._emit_failed(current_ctx, model, e, latency_ms, end_time_ns, correlation_id)
            raise

    async def _generate_durable(self, request: GenerateRequest, context: Any) -> GenerateResponse:
        activation_client = getattr(context, "_activation_client", None)
        if activation_client is None:
            from ..exceptions import ActivationError, ActivationErrorCode

            raise ActivationError(
                ActivationErrorCode.DURABILITY_UNAVAILABLE,
                "runtime negotiated durable_activation_v1 but no activation client is available",
            )
        policy = _model_recovery_policy(request.config.recovery_policy)
        stable_key = context.allocate_activation_key("model", request.model)
        activation_request = activation_request_from_context(
            context,
            kind=ActivationKind.MODEL,
            stable_key=stable_key,
            input_value=_model_activation_input(request),
            recovery_policy=policy,
        )
        admitted: ActivationDecision | None = None
        started = time.monotonic()

        def on_admitted(decision: ActivationDecision) -> None:
            nonlocal admitted
            admitted = decision

        async def execute() -> GenerateResponse:
            if admitted is None:
                from ..exceptions import ActivationError, ActivationErrorCode

                raise ActivationError(
                    ActivationErrorCode.UNKNOWN_OUTCOME,
                    "model activation executed without admitted authority",
                )
            activation_token = _set_current_activation(admitted)
            execute_token = _model_activation_execute.set(True)
            try:
                return await self.generate(request)
            finally:
                _model_activation_execute.reset(execute_token)
                _reset_current_activation(activation_token)

        result, _receipt = await activation_client.run(
            activation_request,
            execute,
            encode_output=lambda response: serialize(response.to_dict()),
            decode_output=lambda value: GenerateResponse.from_dict(deserialize(value)),
            latency_ms=lambda: int((time.monotonic() - started) * 1000),
            on_admitted=on_admitted,
            failure_error_code="MODEL_FAILED",
            failure_retryable=policy
            in {
                ActivationRecoveryPolicy.IDEMPOTENT_RETRY,
                ActivationRecoveryPolicy.DURABLE_STEPS,
            },
            failure_external_outcome_certainty="UNKNOWN",
            completion_usage=lambda response: ActivationUsage(
                tokens_in=response.usage.prompt_tokens if response.usage else 0,
                tokens_out=response.usage.completion_tokens if response.usage else 0,
                provider=(
                    request.model.split("/", 1)[0]
                    if "/" in request.model
                    else (self._provider or "")
                ),
                model=request.model,
            ),
            completion_evidence=_model_terminal_evidence,
        )
        return result

    async def _run_managed_prompt(self, request: GenerateRequest) -> GenerateResponse:
        manifest_request = resolve_prompt_from_manifest(request)
        if manifest_request is not None:
            return await self.generate(manifest_request)

        prompt_ref = request.prompt_ref
        assert prompt_ref is not None
        project_id = (
            prompt_ref.project_id
            or os.environ.get("AGNT5_PROJECT_ID")
            or os.environ.get("AGNT5_PROJECT_REF")
        )
        if not project_id:
            raise ValueError(
                "Managed prompts require a project context. Set project_id on the call "
                "or AGNT5_PROJECT_ID in the environment."
            )

        platform_url = (
            prompt_ref.platform_url
            or os.environ.get("AGNT5_PLATFORM_URL")
            or os.environ.get("AGNT5_CONTROL_PLANE_URL")
            or "https://api.agnt5.com"
        ).rstrip("/")
        body: Dict[str, Any] = {"variables": prompt_ref.variables}
        if prompt_ref.version:
            body["version_id"] = prompt_ref.version
        if prompt_ref.model:
            body["model"] = prompt_ref.model
        if request.config.temperature is not None:
            body["temperature"] = request.config.temperature
        if request.config.max_tokens is not None:
            body["max_tokens"] = request.config.max_tokens

        headers = {"Content-Type": "application/json"}
        api_key = prompt_ref.api_key or os.environ.get("AGNT5_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-KEY"] = api_key

        url = f"{platform_url}/api/v1/projects/{project_id}/prompts/{prompt_ref.id}/run"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        data = payload.get("data", payload)
        usage = None
        if data.get("total_tokens") is not None:
            usage = TokenUsage(
                prompt_tokens=int(data.get("prompt_tokens", 0)),
                completion_tokens=int(data.get("completion_tokens", 0)),
                total_tokens=int(data.get("total_tokens", 0)),
            )
        return GenerateResponse(
            text=data.get("content", ""),
            usage=usage,
            finish_reason=data.get("finish_reason"),
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Event, None]:
        """Stream completion from LLM as Event objects."""
        current_ctx = get_current_context()
        if (
            current_ctx
            and not _model_stream_activation_execute.get()
            and (getattr(current_ctx, "_trace_metadata", None) or {}).get(
                "durable_activation_v1"
            )
            == "true"
        ):
            async for event in self._stream_durable(request, current_ctx):
                yield event
            return

        prompt = self._build_prompt_messages(request)
        model = self._prepare_model_name(request.model)
        kwargs = self._build_kwargs(request, model)

        block_types: Dict[int, str] = {}
        start_time_ns = time.time_ns()
        correlation_id = generate_cid()
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
                    tool_calls = getattr(chunk, "tool_calls", None)
                    yield LMCompleted(
                        name=model,
                        correlation_id=correlation_id,
                        parent_correlation_id=parent_correlation_id,
                        model=chunk.model or model,
                        provider=self._provider or "unknown",
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                        total_tokens=usage.total_tokens if usage else 0,
                        cached_tokens=getattr(usage, "cached_tokens", 0) if usage else 0,
                        finish_reason=chunk.finish_reason,
                        output_data={
                            "text": chunk.text,
                            "tool_calls": tool_calls,
                            "model": chunk.model,
                            "timestamp": time.time_ns() // 1_000_000,
                        },
                        duration_ms=latency_ms,
                    )

                    if current_ctx and not _model_stream_activation_execute.get():
                        self._emit_completed(
                            current_ctx, model, chunk, latency_ms, end_time_ns, correlation_id
                        )

        except Exception as e:
            if current_ctx and not _model_stream_activation_execute.get():
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

    async def _stream_durable(
        self,
        request: GenerateRequest,
        context: Any,
    ) -> AsyncGenerator[Event, None]:
        activation_client = getattr(context, "_activation_client", None)
        if activation_client is None:
            from ..exceptions import ActivationError, ActivationErrorCode

            raise ActivationError(
                ActivationErrorCode.DURABILITY_UNAVAILABLE,
                "runtime negotiated durable_activation_v1 but no activation client is available",
            )
        policy = _model_recovery_policy(request.config.recovery_policy)
        model = self._prepare_model_name(request.model)
        activation_request = activation_request_from_context(
            context,
            kind=ActivationKind.MODEL,
            stable_key=context.allocate_activation_key("model", model),
            input_value=_model_activation_input(request),
            recovery_policy=policy,
        )
        decision = await activation_client.begin(activation_request)
        if decision.kind is ActivationDecisionKind.REPLAY:
            if decision.replay_output is None:
                from ..exceptions import ActivationError, ActivationErrorCode

                raise ActivationError(
                    ActivationErrorCode.UNKNOWN_OUTCOME,
                    "REPLAY receipt is missing its canonical output",
                    activation_id=decision.activation_id,
                    attempt=decision.attempt,
                )
            replayed = GenerateResponse.from_dict(deserialize(decision.replay_output))
            correlation_id = generate_cid()
            parent_correlation_id = context.correlation_id
            started_event = LMContentBlockStarted(
                name=model,
                correlation_id=correlation_id,
                parent_correlation_id=parent_correlation_id,
                block_type="text",
                index=0,
            )
            self._emit_started(context, model, request, time.time_ns(), correlation_id)
            yield started_event
            if replayed.text:
                yield LMContentBlockDelta(
                    name=model,
                    correlation_id=correlation_id,
                    parent_correlation_id=parent_correlation_id,
                    content=replayed.text,
                    block_type="text",
                    index=0,
                )
            yield LMContentBlockCompleted(
                name=model,
                correlation_id=correlation_id,
                parent_correlation_id=parent_correlation_id,
                block_type="text",
                index=0,
            )
            usage = replayed.usage
            completed_event = LMCompleted(
                name=model,
                correlation_id=correlation_id,
                parent_correlation_id=parent_correlation_id,
                model=model,
                provider=self._provider or "unknown",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                cached_tokens=usage.cached_tokens if usage else 0,
                finish_reason=replayed.finish_reason,
                output_data={
                    "text": replayed.text,
                    "tool_calls": replayed.tool_calls,
                    "model": model,
                    "replayed": True,
                },
                duration_ms=0,
            )
            self._emit_completed(
                context,
                model,
                replayed,
                0,
                time.time_ns(),
                correlation_id,
            )
            yield completed_event
            return
        if decision.kind is not ActivationDecisionKind.EXECUTE:
            from ..activation import _decision_error

            raise _decision_error(decision)

        activation_token = _set_current_activation(decision)
        execute_token = _model_stream_activation_execute.set(True)
        started = time.monotonic()
        partial_hash = hashlib.sha256()
        partial_chunks = 0
        partial_bytes = 0
        terminal: LMCompleted | None = None
        terminal_response: GenerateResponse | None = None
        failure_event: LMFailed | None = None
        provider_error: Exception | None = None
        try:
            try:
                async for event in self.stream(request):
                    if isinstance(event, LMContentBlockDelta) and event.content:
                        encoded = str(event.content).encode("utf-8")
                        partial_hash.update(encoded)
                        partial_chunks += 1
                        partial_bytes += len(encoded)
                    if isinstance(event, LMCompleted):
                        terminal = event
                        output = event.output_data if isinstance(event.output_data, dict) else {}
                        terminal_response = GenerateResponse(
                            text=str(output.get("text") or ""),
                            usage=TokenUsage(
                                prompt_tokens=event.input_tokens,
                                completion_tokens=event.output_tokens,
                                total_tokens=event.total_tokens,
                                cached_tokens=getattr(event, "cached_tokens", 0),
                            ),
                            finish_reason=event.finish_reason,
                            tool_calls=output.get("tool_calls") or None,
                        )
                        continue
                    if isinstance(event, LMFailed):
                        failure_event = event
                        continue
                    yield event
            except Exception as error:
                provider_error = error

            if provider_error is not None or terminal_response is None or terminal is None:
                from ..exceptions import ActivationError, ActivationErrorCode

                error = provider_error or ActivationError(
                    ActivationErrorCode.UNKNOWN_OUTCOME,
                    "model stream ended without a terminal provider response",
                    activation_id=decision.activation_id,
                    attempt=decision.attempt,
                )
                evidence_payload = json.dumps(
                    {
                        "schema": "agnt5.model_stream_interruption.v1",
                        "attempt": decision.attempt,
                        "partial_chunks": partial_chunks,
                        "partial_utf8_bytes": partial_bytes,
                        "partial_sha256": partial_hash.hexdigest(),
                        "classification": "provider_interrupted",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                error_data = json.dumps(
                    {"message": str(error), "type": type(error).__name__},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                await activation_client.fail(
                    activation_request,
                    decision,
                    error_code="MODEL_STREAM_INTERRUPTED",
                    error_data=error_data,
                    retryable=policy
                    in {
                        ActivationRecoveryPolicy.IDEMPOTENT_RETRY,
                        ActivationRecoveryPolicy.DURABLE_STEPS,
                    },
                    external_outcome_certainty="UNKNOWN",
                    evidence=(
                        ActivationEvidence.inline(
                            "model_stream_interruption_v1",
                            evidence_payload,
                        ),
                    ),
                )
                self._emit_failed(
                    context,
                    model,
                    error,
                    int((time.monotonic() - started) * 1000),
                    time.time_ns(),
                    failure_event.correlation_id if failure_event else generate_cid(),
                )
                if failure_event is not None:
                    yield failure_event
                raise error

            usage = terminal_response.usage
            await activation_client.complete(
                activation_request,
                decision,
                output=serialize(terminal_response.to_dict()),
                usage=ActivationUsage(
                    tokens_in=usage.prompt_tokens if usage else 0,
                    tokens_out=usage.completion_tokens if usage else 0,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    provider=self._provider or "",
                    model=model,
                ),
                evidence=_model_terminal_evidence(terminal_response),
            )
            self._emit_completed(
                context,
                model,
                terminal_response,
                int((time.monotonic() - started) * 1000),
                time.time_ns(),
                terminal.correlation_id,
            )
            yield terminal
        finally:
            _model_stream_activation_execute.reset(execute_token)
            _reset_current_activation(activation_token)

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
        if request.config.cache is not None:
            kwargs["prompt_cache"] = json.dumps(
                {
                    "enabled": request.config.cache.enabled,
                    "ttl": request.config.cache.ttl,
                    "key": request.config.cache.key,
                    "retention": request.config.cache.retention,
                    "resource": request.config.cache.resource,
                }
            )
        if request.config.cache_control:
            kwargs["cache_control"] = request.config.cache_control
        if request.config.cache_ttl is not None:
            kwargs["cache_ttl"] = request.config.cache_ttl
        if request.config.google_cached_content is not None:
            kwargs["google_cached_content"] = request.config.google_cached_content
        if request.response_schema is not None:
            kwargs["response_schema_kw"] = request.response_schema
        if request.prompt_ref is not None:
            kwargs["prompt_ref"] = json.dumps(
                {
                    "id": request.prompt_ref.id,
                    "project_id": request.prompt_ref.project_id,
                    "version": request.prompt_ref.version,
                    "environment_id": request.prompt_ref.environment_id,
                    "environment_ref": request.prompt_ref.environment_ref,
                    "platform_url": request.prompt_ref.platform_url,
                    "variables": request.prompt_ref.variables,
                }
            )

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

    async def create_cache(
        self,
        *,
        model: str,
        contents: List[str],
        system_prompt: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Create a Gemini explicit context cache and return its resource name."""
        contents_json = json.dumps(contents)
        return await self._rust_lm.create_cache(
            model=model,
            contents=contents_json,
            system_prompt=system_prompt,
            ttl_seconds=ttl_seconds,
            provider=self._provider,
        )

    async def delete_cache(self, name: str) -> None:
        """Delete a Gemini explicit context cache by resource name."""
        await self._rust_lm.delete_cache(name=name, provider=self._provider)

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
            ru = rust_response.usage
            usage = TokenUsage(
                prompt_tokens=ru.prompt_tokens,
                completion_tokens=ru.completion_tokens,
                total_tokens=ru.total_tokens,
                cached_tokens=getattr(ru, "cached_tokens", None) or 0,
                cache_creation_tokens=getattr(ru, "cache_creation_tokens", None) or 0,
            )

        tool_calls = None
        if hasattr(rust_response, "tool_calls") and rust_response.tool_calls:
            tool_calls = rust_response.tool_calls

        response_id = None
        if hasattr(rust_response, "id") and rust_response.id:
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
        # Serialize messages for event. Include tool_calls and tool_call_id so
        # the trace shows the OpenAI/Anthropic protocol shape the Rust LM
        # client actually sends — tool result messages render as role:"tool"
        # with tool_call_id, and assistant turns that requested tools include
        # the tool_calls block. The Rust ApiMessage builder in
        # sdk-core/src/lm/openai_common.rs:179 routes by presence of
        # tool_call_id; mirror that here for display fidelity.
        def _serialize_msg(m):
            role = "tool" if m.tool_call_id else m.role.value
            out: Dict[str, Any] = {"role": role, "content": m.content}
            if m.tool_call_id:
                out["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                out["tool_calls"] = m.tool_calls
            return out

        messages = [_serialize_msg(m) for m in request.messages]
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
        if hasattr(response, "text"):
            output_text = response.text
        else:
            output_text = getattr(response, "content", str(response))

        usage = getattr(response, "usage", None)
        completed_event = LMCompleted(
            name=model,
            correlation_id=correlation_id,
            parent_correlation_id=ctx.correlation_id,
            model=model,
            provider=self._provider or "unknown",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cached_tokens=getattr(usage, "cached_tokens", 0) if usage else 0,
            finish_reason=getattr(response, "finish_reason", None),
            output_data={
                "output": output_text,
                "tool_calls": getattr(response, "tool_calls", None),
            },
            duration_ms=latency_ms,
            metadata={
                "name": model,
                "model": model,
                "provider": self._provider or "unknown",
                "duration_ms": str(latency_ms),
                "input_tokens": str(usage.prompt_tokens if usage else 0),
                "output_tokens": str(usage.completion_tokens if usage else 0),
                "total_tokens": str(usage.total_tokens if usage else 0),
                "cached_tokens": str(getattr(usage, "cached_tokens", 0) if usage else 0),
            },
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
