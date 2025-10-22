"""Agent session management for multi-turn conversations.

Provides AgentSession entity to track and persist conversation metadata
across multiple agent invocations. Session state is backed by the entity
infrastructure for durability.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

from .entity import Entity


class AgentSession(Entity):
    """
    Entity representing an agent conversation session.

    Tracks metadata about multi-turn conversations including:
    - Session creation and last activity timestamps
    - Message count
    - Agent identification
    - Custom metadata

    The actual conversation messages are stored separately in AgentContext
    using the session_id as the key. This entity only tracks session metadata.

    Example:
        ```python
        # Create new session
        session = AgentSession("session-123")
        session.create(agent_name="tutor", metadata={"user_id": "user-456"})

        # Update on each message
        session.add_message()

        # Get session summary
        summary = session.get_summary()
        ```
    """

    def __init__(self, session_id: str):
        """
        Initialize agent session entity.

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.agent_name: str = ""
        self.created_at: float = 0.0
        self.last_message_time: float = 0.0
        self.message_count: int = 0
        self.metadata: Dict[str, Any] = {}

    def create(self, agent_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new session.

        Args:
            agent_name: Name of the agent for this session
            metadata: Optional additional session metadata
        """
        current_time = time.time()
        self.agent_name = agent_name
        self.created_at = current_time
        self.last_message_time = current_time
        self.message_count = 0
        self.metadata = metadata or {}

    def add_message(self) -> None:
        """
        Record a new message in the session.

        Updates message count and last activity timestamp.
        """
        self.message_count += 1
        self.last_message_time = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get session metadata summary.

        Returns:
            Dictionary with session metadata including:
            - session_id
            - agent_name
            - created_at
            - last_message_time
            - message_count
            - metadata
        """
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "last_message_time": self.last_message_time,
            "message_count": self.message_count,
            "metadata": self.metadata,
        }

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Update session metadata.

        Args:
            metadata: Metadata to merge into existing metadata
        """
        self.metadata.update(metadata)
