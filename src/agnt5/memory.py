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
            result = await agent.run_sync(message, history=history)

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


class MemoryScope:
    """Memory scope for semantic memory.

    Scopes determine the isolation level of memories:
    - USER: Isolated per user (most common)
    - TENANT: Shared across users in a tenant
    - AGENT: Isolated per agent instance
    - SESSION: Isolated per session (ephemeral)
    - GLOBAL: Shared across all users/tenants
    """
    USER = "user"
    TENANT = "tenant"
    AGENT = "agent"
    SESSION = "session"
    GLOBAL = "global"

    @classmethod
    def valid_scopes(cls) -> List[str]:
        """Return list of valid scope strings."""
        return [cls.USER, cls.TENANT, cls.AGENT, cls.SESSION, cls.GLOBAL]


@dataclass
class MemoryResult:
    """Result from semantic memory search.

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


@dataclass
class MemoryMetadata:
    """Metadata for storing with a memory.

    Attributes:
        source: Optional source identifier (e.g., "chat", "document", "api")
        created_at: Optional timestamp string
        extra: Additional key-value metadata
    """
    source: Optional[str] = None
    created_at: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


class SemanticMemory:
    """Vector-backed semantic memory for user/tenant knowledge.

    Provides semantic search capabilities over stored memories using
    vector embeddings. Memories are automatically embedded and indexed
    for fast similarity search.

    Requires:
    - OPENAI_API_KEY for embeddings
    - One of: QDRANT_URL, PINECONE_API_KEY+PINECONE_HOST, or POSTGRES_URL for vector storage

    Example:
        ```python
        from agnt5 import SemanticMemory, MemoryScope

        # Create memory scoped to a user
        memory = SemanticMemory(MemoryScope.USER, "user-123")

        # Store some memories
        await memory.store("User prefers dark mode")
        await memory.store("User's favorite color is blue")

        # Search for relevant memories
        results = await memory.search("color preferences")
        for result in results:
            print(f"{result.content} (score: {result.score:.2f})")

        # Delete a memory
        await memory.forget(results[0].id)
        ```
    """

    def __init__(self, scope: str, scope_id: str) -> None:
        """Initialize semantic memory for a scope.

        Args:
            scope: Memory scope (use MemoryScope constants: USER, TENANT, AGENT, SESSION, GLOBAL)
            scope_id: The unique identifier for the scope (e.g., user_id, tenant_id)

        Raises:
            ValueError: If scope is not a valid scope string
        """
        if scope not in MemoryScope.valid_scopes():
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of: {MemoryScope.valid_scopes()}"
            )
        self.scope = scope
        self.scope_id = scope_id
        self._inner = None  # Lazy initialization

    async def _get_inner(self):
        """Get or create the underlying Rust SemanticMemory instance."""
        if self._inner is None:
            try:
                from ._core import PySemanticMemory, PyMemoryScope
            except ImportError as e:
                raise ImportError(
                    "SemanticMemory requires the agnt5 Rust extension. "
                    f"Import error: {e}"
                ) from e

            # Map scope string to PyMemoryScope
            scope_map = {
                MemoryScope.USER: PyMemoryScope.user(),
                MemoryScope.TENANT: PyMemoryScope.tenant(),
                MemoryScope.AGENT: PyMemoryScope.agent(),
                MemoryScope.SESSION: PyMemoryScope.session(),
                MemoryScope.GLOBAL: PyMemoryScope.global_(),
            }
            py_scope = scope_map.get(self.scope)
            if py_scope is None:
                py_scope = PyMemoryScope.from_str(self.scope)

            self._inner = await PySemanticMemory.from_env(py_scope, self.scope_id)
        return self._inner

    async def store(self, content: str, metadata: Optional[MemoryMetadata] = None) -> str:
        """Store content in semantic memory.

        The content is automatically embedded and indexed for semantic search.

        Args:
            content: Text content to store
            metadata: Optional metadata to associate with the memory

        Returns:
            The unique ID of the stored memory

        Raises:
            RuntimeError: If embedder or vector database is not configured
        """
        inner = await self._get_inner()
        if metadata is not None:
            from ._core import PyMemoryMetadata
            py_metadata = PyMemoryMetadata(
                source=metadata.source,
                created_at=metadata.created_at,
                extra=metadata.extra,
            )
            return await inner.store_with_metadata(content, py_metadata)
        return await inner.store(content)

    async def store_batch(
        self,
        contents: List[str],
        metadata: Optional[List[MemoryMetadata]] = None,
    ) -> List[str]:
        """Store multiple contents in batch (more efficient for RAG indexing).

        Uses batch embedding and batch upsert for better performance when
        indexing many documents.

        Args:
            contents: List of text contents to store
            metadata: Optional list of metadata (must match contents length)

        Returns:
            List of unique IDs for all stored memories

        Raises:
            RuntimeError: If embedder or vector database is not configured
            ValueError: If metadata length doesn't match contents length

        Example:
            ```python
            # Index documents in batch
            docs = ["Doc 1 content...", "Doc 2 content...", "Doc 3 content..."]
            ids = await memory.store_batch(docs)

            # With metadata for source tracking
            metadata = [
                MemoryMetadata(source="file1.pdf"),
                MemoryMetadata(source="file2.pdf"),
                MemoryMetadata(source="file3.pdf"),
            ]
            ids = await memory.store_batch(docs, metadata=metadata)
            ```
        """
        inner = await self._get_inner()
        if metadata is not None:
            if len(metadata) != len(contents):
                raise ValueError(
                    f"Metadata length ({len(metadata)}) must match contents length ({len(contents)})"
                )
            from ._core import PyMemoryMetadata
            py_metadata = [
                PyMemoryMetadata(
                    source=m.source,
                    created_at=m.created_at,
                    extra=m.extra,
                )
                for m in metadata
            ]
            return await inner.store_batch_with_metadata(contents, py_metadata)
        return await inner.store_batch(contents)

    async def search(self, query: str, limit: int = 10, min_score: Optional[float] = None) -> List[MemoryResult]:
        """Search for relevant memories using vector similarity.

        Args:
            query: Search query text (will be embedded)
            limit: Maximum number of results to return (default: 10)
            min_score: Optional minimum similarity score filter (0.0 to 1.0)

        Returns:
            List of MemoryResult objects, ranked by similarity score (highest first)

        Raises:
            RuntimeError: If embedder or vector database is not configured
        """
        inner = await self._get_inner()
        if min_score is not None:
            results = await inner.search_with_options(query, limit, min_score)
        else:
            results = await inner.search(query, limit)

        return [
            MemoryResult(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=dict(r.metadata.extra) if r.metadata else {},
            )
            for r in results
        ]

    async def forget(self, memory_id: str) -> bool:
        """Delete a memory by its ID.

        Args:
            memory_id: The unique ID of the memory to delete

        Returns:
            True if the memory was deleted, False if it wasn't found

        Raises:
            RuntimeError: If vector database is not configured
        """
        inner = await self._get_inner()
        return await inner.forget(memory_id)

    async def get(self, memory_id: str) -> Optional[MemoryResult]:
        """Get a specific memory by ID.

        Args:
            memory_id: The unique ID of the memory to retrieve

        Returns:
            MemoryResult if found, None otherwise

        Raises:
            RuntimeError: If vector database is not configured
        """
        inner = await self._get_inner()
        result = await inner.get(memory_id)
        if result is None:
            return None
        return MemoryResult(
            id=result.id,
            content=result.content,
            score=result.score,
            metadata=dict(result.metadata.extra) if result.metadata else {},
        )
