"""Memory classes for AGNT5 SDK.

Provides memory abstractions for workflows and agents:
- ConversationMemory: KV-backed message history for sessions
- SemanticMemory: Vector-backed semantic search for user/tenant memory (Phase 3)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._telemetry import setup_module_logger
from .lm import Message, MessageRole

logger = setup_module_logger(__name__)


@dataclass
class MemoryMessage:
    """Message stored in conversation memory.

    Attributes:
        role: Message role (user, assistant, system)
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
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMessage":
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )

    def to_lm_message(self) -> Message:
        """Convert to LM Message for agent prompts."""
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        return Message(
            role=role_map.get(self.role, MessageRole.USER),
            content=self.content,
        )


class ConversationMemory:
    """KV-backed conversation memory for session history.

    Stores sequential message history for a session, enabling multi-turn
    conversations. Messages are persisted to the platform and loaded on demand.

    Example:
        ```python
        # In a workflow
        @workflow
        async def chat_workflow(ctx: WorkflowContext, message: str) -> str:
            # Load conversation history
            conversation = ConversationMemory(ctx.session_id)
            history = await conversation.get_messages()

            # Process with agent
            result = await agent.run(message, history=history)

            # Save new messages
            await conversation.add("user", message)
            await conversation.add("assistant", result.output)

            return result.output
        ```
    """

    def __init__(self, session_id: str) -> None:
        """Initialize conversation memory for a session.

        Args:
            session_id: Unique identifier for the conversation session
        """
        self.session_id = session_id
        self._entity_key = f"conversation:{session_id}"
        self._entity_type = "ConversationMemory"
        self._state_adapter = None
        self._cache: Optional[List[MemoryMessage]] = None

    def _get_adapter(self):
        """Get or create state adapter for persistence."""
        if self._state_adapter is None:
            from .entity import _get_state_adapter, EntityStateAdapter
            try:
                self._state_adapter = _get_state_adapter()
            except RuntimeError:
                # Not in worker context - create standalone adapter
                self._state_adapter = EntityStateAdapter()
        return self._state_adapter

    async def get_messages(self, limit: int = 50) -> List[MemoryMessage]:
        """Get recent messages from conversation history.

        Args:
            limit: Maximum number of messages to return (most recent)

        Returns:
            List of MemoryMessage objects, ordered chronologically
        """
        adapter = self._get_adapter()

        # Load session data from storage
        session_data = await adapter.load_state(self._entity_type, self._entity_key)

        if not session_data:
            return []

        messages_data = session_data.get("messages", [])

        # Convert to MemoryMessage objects
        messages = [MemoryMessage.from_dict(m) for m in messages_data]

        # Apply limit (return most recent)
        if limit and len(messages) > limit:
            messages = messages[-limit:]

        # Cache for potential add() calls
        self._cache = messages

        return messages

    async def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content text
            metadata: Optional additional metadata to store
        """
        adapter = self._get_adapter()

        # Load current state with version for optimistic locking
        current_state, current_version = await adapter.load_with_version(
            self._entity_type, self._entity_key
        )

        # Get existing messages or start fresh
        messages_data = current_state.get("messages", []) if current_state else []

        # Create new message
        new_message = MemoryMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Append message
        messages_data.append(new_message.to_dict())

        # Build session data
        now = time.time()
        session_data = {
            "session_id": self.session_id,
            "created_at": current_state.get("created_at", now) if current_state else now,
            "last_message_at": now,
            "message_count": len(messages_data),
            "messages": messages_data,
        }

        # Save to storage
        try:
            await adapter.save_state(
                self._entity_type,
                self._entity_key,
                session_data,
                current_version,
            )
            logger.debug(f"Saved message to conversation {self.session_id}: {role}")
        except Exception as e:
            logger.error(f"Failed to save message to conversation {self.session_id}: {e}")
            raise

    async def clear(self) -> None:
        """Clear all messages in this conversation."""
        adapter = self._get_adapter()

        # Load current version for optimistic locking
        _, current_version = await adapter.load_with_version(
            self._entity_type, self._entity_key
        )

        # Save empty session
        now = time.time()
        session_data = {
            "session_id": self.session_id,
            "created_at": now,
            "last_message_at": now,
            "message_count": 0,
            "messages": [],
        }

        try:
            await adapter.save_state(
                self._entity_type,
                self._entity_key,
                session_data,
                current_version,
            )
            self._cache = []
            logger.info(f"Cleared conversation {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to clear conversation {self.session_id}: {e}")
            raise

    async def get_as_lm_messages(self, limit: int = 50) -> List[Message]:
        """Get messages formatted for LLM consumption.

        Convenience method that returns messages as LM Message objects,
        ready to pass to agent.run() or lm.generate().

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of Message objects for LM API
        """
        messages = await self.get_messages(limit=limit)
        return [m.to_lm_message() for m in messages]


# Placeholder for Phase 3: SemanticMemory
class SemanticMemory:
    """Vector-backed semantic memory for user/tenant knowledge.

    NOT YET IMPLEMENTED - Placeholder for Phase 3.

    Will provide:
    - Vector similarity search for relevant memories
    - Scoped to user or tenant
    - Integration with embeddings (OpenAI, local models)
    """

    def __init__(self, scope: str, scope_id: str) -> None:
        """Initialize semantic memory for a scope.

        Args:
            scope: Memory scope ("user" or "tenant")
            scope_id: The user_id or tenant_id
        """
        self.scope = scope
        self.scope_id = scope_id
        raise NotImplementedError(
            "SemanticMemory is not yet implemented. "
            "See Phase 3 of entity-memory-dx-improvements for details."
        )

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant memories using vector similarity."""
        raise NotImplementedError("SemanticMemory.search() not yet implemented")

    async def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add content to memory, returns memory_id."""
        raise NotImplementedError("SemanticMemory.add() not yet implemented")

    async def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        raise NotImplementedError("SemanticMemory.delete() not yet implemented")
