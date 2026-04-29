"""Shared types for the AGNT5 memory system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MemoryScope(str, Enum):
    """Scope for memory isolation.

    Determines how memory is partitioned:
    - SESSION: Per conversation session (default)
    - USER: Per user, across sessions
    - RUN: Per workflow run (ephemeral)
    - GLOBAL: Tenant-wide shared state
    """

    SESSION = "session"
    USER = "user"
    RUN = "run"
    GLOBAL = "global"


@dataclass
class ConversationMessage:
    """Message in a conversation.

    Attributes:
        role: Message role (user, assistant, system, tool)
        content: Message content text
        timestamp: Unix timestamp when message was added
        metadata: Optional additional metadata
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationMessage:
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )

    def to_lm_message(self):
        """Convert to LM Message for agent prompts."""
        from ..lm import Message, MessageRole

        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        return Message(
            role=role_map.get(self.role, MessageRole.USER),
            content=self.content,
        )


@dataclass
class SemanticSearchResult:
    """Result from a semantic memory search.

    Attributes:
        id: Unique identifier for this memory
        content: The original text content that was stored
        score: Similarity score (0.0 to 1.0, higher is more similar)
        metadata: Optional metadata associated with the memory
    """

    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRecord:
    """AGNT5-native semantic memory record.

    This is a scoped, tenant-isolated memory item stored by the configured
    MemoryService backend. It is not raw conversation history.
    """

    id: str
    tenant_id: str
    deployment_id: str
    scope: str
    scope_id: str
    kind: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_dim: int = 0
    source_session_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_event_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class MemorySearchResult:
    """Result from AGNT5-native semantic memory search."""

    record: MemoryRecord
    score: float
    distance: float
