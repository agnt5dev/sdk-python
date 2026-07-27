"""Abstract base class for language model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncGenerator

from .types import GenerateRequest, GenerateResponse

if TYPE_CHECKING:
    from ..events import Event


class LanguageModel(ABC):
    """Abstract base class for language models (for testing/mocking)."""

    # Custom model implementations can opt in when ``stream()`` returns complete
    # tool-call information in the terminal ``LMCompleted`` event. Agents keep
    # using ``generate()`` for tools unless this is explicitly enabled.
    supports_streaming_tools: bool = False

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate completion from LLM."""
        pass

    @abstractmethod
    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Event, None]:
        """Stream completion from LLM as Event objects."""
        yield  # type: ignore
