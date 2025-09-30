"""LLM client facade for Context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union

if TYPE_CHECKING:
    from ..context import Context
    from ..llm import LlmClient


class LlmFacade:
    """Expose ``ctx.llm`` with ``complete`` and ``chat`` helpers."""

    def __init__(self, context: Context) -> None:
        self._context = context

    async def complete(
        self,
        *,
        prompt: str,
        model: str = "gpt-4o-mini",
        schema: Optional[Dict[str, Any]] = None,
    ) -> LlmResponse:
        return await self._context.llm_complete(prompt=prompt, model=model, schema=schema)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str = "gpt-4o-mini",
        tools: Optional[Sequence[Tool]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[LlmStreamChunk], Union[Awaitable[None], None]]] = None,
    ) -> LlmResponse:
        return await self._context.llm_chat(
            messages,
            model=model,
            tools=tools,
            stream=stream,
            on_chunk=on_chunk,
        )


ToolHandler = Callable[..., Union[Awaitable[Any], Any]]


@dataclass
