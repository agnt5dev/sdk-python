"""Semantic Memory Provider — pluggable interface for vector-backed memory.

AGNT5 does NOT ship a built-in vector database. Users bring their own
semantic memory provider by implementing this interface and registering
it at worker level.

Example:
    ```python
    from agnt5 import Worker
    from agnt5_qdrant import QdrantSemanticProvider  # separate package

    worker = Worker(
        service_name="my-service",
        semantic_provider=QdrantSemanticProvider(url="http://localhost:6333"),
    )
    ```

Then in workflow/agent code:
    ```python
    if ctx.memory.semantic:
        results = await ctx.memory.semantic.search("user preferences", limit=5)
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import SemanticSearchResult


class SemanticMemoryProvider(ABC):
    """Abstract base class for semantic memory providers.

    Implement this to integrate any vector database (Qdrant, Pinecone,
    pgvector, Chroma, Weaviate, etc.) with AGNT5's memory system.

    Implementations must be async and thread-safe.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's identifier (e.g., "qdrant", "pinecone")."""
        ...

    @abstractmethod
    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store content in semantic memory.

        The implementation should embed the content and store it in the
        vector database.

        Args:
            content: Text content to store
            metadata: Optional metadata to associate with the memory

        Returns:
            Unique ID of the stored memory
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[SemanticSearchResult]:
        """Search for similar content.

        Args:
            query: Search query text (will be embedded by the provider)
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of results ranked by similarity (highest first)
        """
        ...

    @abstractmethod
    async def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: ID returned by store()

        Returns:
            True if deleted, False if not found
        """
        ...

    async def store_batch(
        self,
        contents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Store multiple items in batch (override for efficiency).

        Default implementation calls store() in sequence.
        Providers should override this with batch embedding/upsert.

        Args:
            contents: List of text contents to store
            metadata: Optional list of metadata dicts (must match contents length)

        Returns:
            List of unique IDs for stored memories
        """
        ids = []
        for i, content in enumerate(contents):
            meta = metadata[i] if metadata else None
            memory_id = await self.store(content, meta)
            ids.append(memory_id)
        return ids

    async def get(self, memory_id: str) -> Optional[SemanticSearchResult]:
        """Get a specific memory by ID.

        Default returns None. Override if your provider supports point lookups.

        Args:
            memory_id: ID returned by store()

        Returns:
            SemanticSearchResult if found, None otherwise
        """
        return None

    async def health_check(self) -> None:
        """Check provider health. Override to add connectivity checks.

        Raises:
            Exception: If the provider is unhealthy
        """
        pass
