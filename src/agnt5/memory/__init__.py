"""AGNT5 Memory System.

Built-in memory (zero-config, backed by runtime RocksDB):
- KVMemory: Scoped key-value storage
- WorkingMemory: Structured JSON scratchpad
- MemoryAccessor: Unified facade (attached to ctx.memory)

AGNT5-native semantic memory:
- ctx.memory.session.save/search: Session-scoped semantic memory
- ctx.memory.user.save/search: User-scoped semantic memory
- NativeMemoryService: Lazy bridge to the configured Rust MemoryRuntime

Pluggable memory (user brings their own):
- SemanticMemoryProvider: ABC for vector-backed semantic search

Types:
- MemoryScope: Scope enum (SESSION, USER, RUN, GLOBAL)
- ConversationMessage: Message in a conversation
- MemoryRecord: Saved semantic memory record
- MemorySearchResult: Result from AGNT5-native semantic search
- SemanticSearchResult: Result from semantic search
"""

# --------------------------------------------------------------------------- #
# Backward-compatible re-exports from the legacy memory module.
# These emit DeprecationWarnings on first use via __getattr__.
# --------------------------------------------------------------------------- #
# Legacy types that are re-exported directly (no deprecation needed — they are
# simple data classes that still work fine).
from .._memory_legacy import MemoryMessage, MemoryMetadata, MemoryResult
from .accessor import MemoryAccessor
from .conversation import ConversationAccessor
from .kv import KVMemory
from .native import DisabledMemoryService, MemoryService, NativeMemoryService
from .semantic import SemanticMemoryProvider
from .types import (
    ConversationMessage,
    MemoryRecord,
    MemoryScope,
    MemorySearchResult,
    SemanticSearchResult,
)
from .working import WorkingMemory

# Legacy classes that get deprecation warnings are exposed via __getattr__
# below so we don't import them eagerly.

_DEPRECATED_NAMES = {
    "ConversationMemory": "Use ctx.conversation instead.",
    "SemanticMemory": "Use ctx.memory.semantic with a SemanticMemoryProvider instead.",
    "GraphMemory": "GraphMemory has been removed (it was in-memory only with no persistence).",
    "GraphNode": "GraphMemory has been removed.",
    "GraphRelationship": "GraphMemory has been removed.",
    "GraphTraversalResult": "GraphMemory has been removed.",
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        import warnings

        msg = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"{name} is deprecated. {msg}",
            DeprecationWarning,
            stacklevel=2,
        )
        # Graph types were removed — raise ImportError after warning
        if "removed" in msg.lower():
            raise ImportError(f"{name} has been removed from agnt5.memory.")
        # For ConversationMemory / SemanticMemory, return the legacy class
        from .. import _memory_legacy

        return getattr(_memory_legacy, name)
    raise AttributeError(f"module 'agnt5.memory' has no attribute {name!r}")


__all__ = [
    # Facades
    "MemoryAccessor",
    "ConversationAccessor",
    # Built-in memory
    "KVMemory",
    "WorkingMemory",
    "DisabledMemoryService",
    "MemoryService",
    "NativeMemoryService",
    # Pluggable memory
    "SemanticMemoryProvider",
    # Types
    "ConversationMessage",
    "MemoryRecord",
    "MemoryScope",
    "MemorySearchResult",
    "SemanticSearchResult",
    # Legacy re-exports (no deprecation)
    "MemoryMessage",
    "MemoryMetadata",
    "MemoryResult",
]
