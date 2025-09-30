"""Session management for multi-turn conversations with scoped state."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Session:
    """Conversation container with scoped state and message management.

    Sessions provide structured conversation history, state management with
    multiple scopes (session, user, app, temp), and automatic retention policies.

    Example:
        ```python
        session = Session(
            id="conv-123",
            user_id="user-456",
            metadata={"project": "research"}
        )

        await session.send_message({"role": "user", "content": "Hello"})
        messages = await session.history(limit=10)
        session.set_state("current_topic", "AI", scope="session")
        ```
    """

    def __init__(
        self,
        id: str,
        app_name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        retention: Optional[Dict[str, Any]] = None,
    ):
        """Create a Session instance.

        Args:
            id: Unique session identifier
            app_name: Application name for scoping (optional)
            user_id: User identifier for cross-session state (optional)
            metadata: Arbitrary session metadata (optional)
            retention: Retention policy configuration (optional)
                - ttl_days: Days to retain data
                - auto_prune: Enable automatic pruning (bool)
                - immutable: Prevent deletion (bool)
                - prune_strategy: Strategy for pruning
        """
        self.id = id
        self.app_name = app_name
        self.user_id = user_id
        self.metadata = metadata or {}
        self.retention = retention or {}

        # In-memory storage for messages and state (will be backed by entity/backend)
        self._messages: List[Dict[str, Any]] = []
        self._state: Dict[str, Dict[str, Any]] = {"session": {}, "user": {}, "app": {}, "temp": {}}

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history.

        Args:
            message: Message dict with 'role' and 'content' fields
                Example: {"role": "user", "content": "Hello"}

        Raises:
            ValueError: If message format is invalid
        """
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        if "role" not in message or "content" not in message:
            raise ValueError("Message must have 'role' and 'content' fields")

        # Add timestamp
        enriched_message = {
            **message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.id,
        }

        self._messages.append(enriched_message)
        logger.debug(f"Added message to session '{self.id}': {message.get('role')}")

    async def history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            limit: Maximum number of messages to return (most recent first)

        Returns:
            List of message dictionaries
        """
        if limit is None:
            return list(self._messages)

        # Return most recent messages
        return self._messages[-limit:] if limit > 0 else []

    def set_state(self, key: str, value: Any, scope: str = "session") -> None:
        """Set state value in specified scope.

        Args:
            key: State key
            value: State value (must be JSON-serializable)
            scope: One of "session", "user", "app", "temp" (default: "session")

        Raises:
            ValueError: If scope is invalid
        """
        if scope not in self._state:
            raise ValueError(f"Invalid scope '{scope}'. Must be one of: session, user, app, temp")

        # Validate JSON-serializable
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"State value must be JSON-serializable: {e}")

        self._state[scope][key] = value
        logger.debug(f"Set state '{key}' in scope '{scope}' for session '{self.id}'")

    def get_state(self, key: str, scope: str = "session", default: Any = None) -> Any:
        """Get state value from specified scope.

        Args:
            key: State key
            scope: One of "session", "user", "app", "temp" (default: "session")
            default: Default value if key not found

        Returns:
            State value or default

        Raises:
            ValueError: If scope is invalid
        """
        if scope not in self._state:
            raise ValueError(f"Invalid scope '{scope}'. Must be one of: session, user, app, temp")

        return self._state[scope].get(key, default)

    async def export(self, format: str = "jsonl") -> str:
        """Export session data.

        Args:
            format: Export format, currently only "jsonl" supported

        Returns:
            Formatted session data as string

        Raises:
            ValueError: If format is unsupported
        """
        if format != "jsonl":
            raise ValueError(f"Unsupported export format: {format}")

        # Export messages as JSON lines
        lines = []
        for message in self._messages:
            lines.append(json.dumps(message))

        return "\n".join(lines)

    async def events(
        self, since: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get session events for audit trail.

        Args:
            since: ISO timestamp to filter events after
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        # TODO: Implement event tracking
        # For now, return messages as events
        events = [
            {"type": "message", "data": msg, "timestamp": msg.get("timestamp")}
            for msg in self._messages
        ]

        # Filter by timestamp if since is provided
        if since:
            events = [e for e in events if e.get("timestamp", "") >= since]

        # Apply limit
        if limit is not None and limit > 0:
            events = events[-limit:]

        return events

    async def prune(self, strategy: str, **kwargs) -> None:
        """Prune conversation history using specified strategy.

        Args:
            strategy: Pruning strategy
                - "keep_last_n": Keep most recent N messages
                - "keep_important": LLM-based importance filtering
                - "sliding_window": Keep last N with window_size parameter
                - "summarize_old": Summarize messages older than threshold
            **kwargs: Strategy-specific parameters

        Raises:
            ValueError: If strategy is unsupported
        """
        if strategy == "keep_last_n":
            n = kwargs.get("n", 50)
            if len(self._messages) > n:
                self._messages = self._messages[-n:]
                logger.info(f"Pruned session '{self.id}' to last {n} messages")

        elif strategy == "sliding_window":
            window_size = kwargs.get("window_size", 50)
            if len(self._messages) > window_size:
                self._messages = self._messages[-window_size:]
                logger.info(f"Pruned session '{self.id}' with sliding window of {window_size}")

        elif strategy == "keep_important":
            # TODO: Implement LLM-based importance filtering
            logger.warning("keep_important strategy not yet implemented")

        elif strategy == "summarize_old":
            # TODO: Implement summarization
            logger.warning("summarize_old strategy not yet implemented")

        else:
            raise ValueError(f"Unsupported prune strategy: {strategy}")

    def clear(self) -> None:
        """Clear all messages and temporary state."""
        self._messages.clear()
        self._state["temp"].clear()
        logger.info(f"Cleared session '{self.id}'")

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, user_id={self.user_id!r}, messages={len(self._messages)})"


__all__ = ["Session"]
