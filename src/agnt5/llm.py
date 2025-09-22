"""Lightweight LLM client facade used by the Python SDK context."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Sequence, Union
import urllib.error


class Transport(Protocol):
    async def __call__(
        self, method: str, url: str, *, headers: Dict[str, str], data: bytes
    ) -> "TransportResponse":
        ...


@dataclass
class TransportResponse:
    status: int
    headers: Dict[str, str]
    body: bytes


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


@dataclass
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LlmResponse:
    model: str
    output: str
    usage: LlmUsage = field(default_factory=LlmUsage)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmStreamChunk:
    index: int
    content: Any
    model: str
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class LlmError(RuntimeError):
    """Raised when the LLM proxy returns an error response."""

    def __init__(self, status: int, message: str, *, code: Optional[str] = None, details: Any = None, raw: Any = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.details = details
        self.raw = raw

    @classmethod
    def from_payload(cls, status: int, payload: Dict[str, Any]) -> "LlmError":
        error = payload.get("error") or {}
        message = error.get("message") or payload.get("message") or "LLM proxy error"
        code = error.get("code") or payload.get("code")
        details = error.get("details") or payload.get("details")
        return cls(status, message, code=code, details=details, raw=payload)


async def _default_transport(  # pragma: no cover - exercised implicitly via client calls
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    data: bytes,
) -> TransportResponse:
    def request() -> TransportResponse:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as response:
            return TransportResponse(
                status=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )

    return await asyncio.to_thread(request)


class LlmClient:
    """HTTP-based LLM client that talks to the platform's LLM proxy."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport: Transport = transport or _default_transport

    @classmethod
    def from_environment(cls) -> Optional["LlmClient"]:
        base_url = os.getenv("AGNT5_LLM_PROXY_URL")
        if not base_url:
            return None
        api_key = os.getenv("AGNT5_LLM_PROXY_KEY")
        return cls(base_url=base_url, api_key=api_key)

    async def complete(
        self,
        *,
        prompt: str,
        model: str,
        schema: Optional[Dict[str, Any]],
        context: Any,
    ) -> LlmResponse:
        payload = {
            "mode": "completion",
            "model": model,
            "prompt": prompt,
        }
        if schema is not None:
            payload["schema"] = schema
        return await self._send("/v1/llm/complete", payload)

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
        model: str,
        tools: Optional[Sequence[Tool]],
        context: Any,
        stream: bool = False,
        on_chunk: Optional[Callable[[LlmStreamChunk], Union[Awaitable[None], None]]] = None,
    ) -> LlmResponse:
        payload: Dict[str, Any] = {
            "mode": "chat",
            "model": model,
            "messages": [message.__dict__ for message in messages],
        }
        if tools:
            payload["tools"] = [self._tool_to_payload(tool) for tool in tools]
        if stream:
            payload["stream"] = True
            return await self._stream_chat("/v1/llm/chat", payload, model=model, on_chunk=on_chunk)
        return await self._send("/v1/llm/chat", payload)

    async def _send(self, path: str, payload: Dict[str, Any]) -> LlmResponse:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await self._transport(
            "POST",
            url,
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )

        text = response.body.decode("utf-8")
        data = self._loads(text)

        if response.status >= 400:
            raise LlmError.from_payload(response.status, data)
        output = data.get("output") or data.get("content") or ""
        usage_payload = data.get("usage") or {}
        usage = LlmUsage(
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
            completion_tokens=int(usage_payload.get("completion_tokens", 0)),
            total_tokens=int(usage_payload.get("total_tokens", 0)),
        )

        return LlmResponse(model=data.get("model", "unknown"), output=output, usage=usage, raw=data)

    async def _stream_chat(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        model: str,
        on_chunk: Optional[Callable[[LlmStreamChunk], Union[Awaitable[None], None]]],
    ) -> LlmResponse:
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        def open_stream() -> urllib.request.addinfourl:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                return urllib.request.urlopen(req, timeout=30)
            except urllib.error.HTTPError as exc:  # pragma: no cover - defensive
                body = exc.read().decode("utf-8") if exc.fp else ""
                data = self._safe_load_event(body) or {}
                raise LlmError.from_payload(exc.code, data) from exc

        response = await asyncio.to_thread(open_stream)

        text_parts: list[str] = []
        raw_events: list[Dict[str, Any]] = []
        usage = LlmUsage()
        current_model = model

        try:
            while True:
                line = await asyncio.to_thread(response.readline)
                if not line:
                    break
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text in ("", "[DONE]"):
                    break
                event = self._safe_load_event(text)
                if event is None:
                    continue
                event_type = event.get("type", "chunk")
                if event_type == "error":
                    status = int(event.get("status", 500))
                    raise LlmError.from_payload(status, event)
                if event_type == "metadata":
                    usage_payload = event.get("usage") or {}
                    usage = LlmUsage(
                        prompt_tokens=int(usage_payload.get("prompt_tokens", 0)),
                        completion_tokens=int(usage_payload.get("completion_tokens", 0)),
                        total_tokens=int(usage_payload.get("total_tokens", 0)),
                    )
                    current_model = event.get("model", current_model)
                    raw_events.append(event)
                    continue

                chunk_payload = event.get("chunk") or event
                delta = chunk_payload.get("delta")
                if delta is None:
                    delta = chunk_payload.get("content")
                chunk = LlmStreamChunk(
                    index=int(chunk_payload.get("index", 0)),
                    content=delta,
                    model=chunk_payload.get("model", current_model),
                    finish_reason=chunk_payload.get("finish_reason"),
                    raw=event,
                )
                if isinstance(delta, str):
                    text_parts.append(delta)
                raw_events.append(event)
                if on_chunk is not None:
                    maybe_awaitable = on_chunk(chunk)
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
        finally:
            await asyncio.to_thread(response.close)

        raw_payload: Dict[str, Any] = {
            "events": raw_events,
            "usage": asdict(usage),
        }

        return LlmResponse(
            model=current_model,
            output="".join(text_parts),
            usage=usage,
            raw=raw_payload,
        )

    @staticmethod
    def _tool_to_payload(tool: Tool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": tool.name}
        if tool.description:
            payload["description"] = tool.description
        if tool.input_schema is not None:
            payload["input_schema"] = tool.input_schema
        return payload

    @staticmethod
    def _loads(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LlmError(500, "Invalid JSON returned from LLM proxy", raw=text) from exc

    @staticmethod
    def _safe_load_event(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


__all__ = [
    "ChatMessage",
    "LlmError",
    "LlmClient",
    "LlmResponse",
    "LlmUsage",
    "LlmStreamChunk",
    "Tool",
]
