"""Conversation Memory — message history backed by runtime storage.

Uses native CF_SESSIONS + CF_MESSAGES when the Rust core is available
(production with runtime). Falls back to CF_STATE via StateAdapter in
standalone/testing mode.

Usage:
    conversation = ConversationAccessor(state_adapter, session_id="sess-1")
    await conversation.add("user", "Hello!")
    await conversation.add("assistant", "Hi there!")
    messages = await conversation.get_messages(limit=20)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .types import ConversationMessage

logger = logging.getLogger(__name__)


class ConversationAccessor:
    """Conversation memory accessor attached to ``ctx.conversation``.

    Manages per-session message history with add/get/clear operations.
    Messages are persisted to the runtime and survive process restarts.

    When connected to the runtime (production), uses native session/message
    operations backed by CF_SESSIONS + CF_MESSAGES in RocksDB. In standalone
    or testing mode, falls back to CF_STATE.

    Example:
        ```python
        @workflow
        async def chat(ctx: WorkflowContext, user_input: str) -> str:
            history = await ctx.conversation.get_as_lm_messages(limit=20)
            await ctx.conversation.add("user", user_input)
            result = await agent.run(user_input, history=history)
            await ctx.conversation.add("assistant", result.output)
            return result.output
        ```
    """

    # Entity type/key for CF_STATE fallback (standalone mode)
    _ENTITY_TYPE = "conversation"

    def __init__(
        self,
        state_adapter,
        session_id: str,
        component_name: Optional[str] = None,
    ) -> None:
        """Initialize conversation accessor.

        Args:
            state_adapter: StateAdapter for runtime state operations
            session_id: Session identifier (used as entity key and scope_id)
            component_name: Optional component name (agent/workflow name)
        """
        self._state_adapter = state_adapter
        self._session_id = session_id
        self._component_name = component_name or ""
        self._entity_key = f"history:{session_id}"
        self._session_created = False

    @property
    def _use_native(self) -> bool:
        """Whether to use native session/message operations."""
        return getattr(self._state_adapter, 'has_native_messages', False)

    async def _ensure_session(self) -> None:
        """Ensure native session exists (called once per accessor)."""
        if self._session_created or not self._use_native:
            return
        try:
            await self._state_adapter.create_session(
                self._session_id,
                self._component_name,
                "conversation",
            )
            self._session_created = True
        except Exception:
            # Session may already exist — that's fine
            self._session_created = True

    async def add(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a message to the conversation.

        Args:
            role: Message role ("user", "assistant", "system", "tool")
            content: Message content text
            metadata: Optional metadata to attach to the message

        Returns:
            Message ID

        Example:
            await ctx.conversation.add("user", "What's the weather?")
            await ctx.conversation.add("assistant", "It's sunny!")
        """
        if self._use_native:
            return await self._add_native(role, content, metadata)
        return await self._add_fallback(role, content, metadata)

    async def _add_native(self, role: str, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Add via native message.enqueued journal event."""
        await self._ensure_session()

        payload = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }).encode("utf-8")

        try:
            message_id = await self._state_adapter.send_message(
                correlation_id=self._session_id,
                message_type=role,
                payload=payload,
            )
            return message_id
        except Exception as e:
            logger.error(f"Native message send failed for {self._session_id}: {e}")
            raise

    async def _add_fallback(self, role: str, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Add via CF_STATE (standalone/testing mode)."""
        current_state, current_version = await self._state_adapter.load_with_version(
            entity_type=self._ENTITY_TYPE,
            entity_key=self._entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        messages_data = current_state.get("messages", []) if current_state else []

        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        messages_data.append(msg.to_dict())

        now = time.time()
        session_data = {
            "session_id": self._session_id,
            "component_name": self._component_name,
            "created_at": current_state.get("created_at", now) if current_state else now,
            "last_message_at": now,
            "message_count": len(messages_data),
            "messages": messages_data,
        }

        try:
            await self._state_adapter.save_state(
                entity_type=self._ENTITY_TYPE,
                entity_key=self._entity_key,
                state=session_data,
                expected_version=current_version,
                scope="session",
                scope_id=self._session_id,
            )
        except Exception as e:
            logger.error(f"Failed to add message to conversation {self._session_id}: {e}")
            raise

        return str(len(messages_data) - 1)

    async def get_messages(self, limit: int = 50) -> List[ConversationMessage]:
        """Get recent messages from conversation history.

        Args:
            limit: Maximum number of messages to return (most recent)

        Returns:
            List of ConversationMessage objects, ordered chronologically

        Example:
            messages = await ctx.conversation.get_messages(limit=20)
            for msg in messages:
                print(f"{msg.role}: {msg.content}")
        """
        if self._use_native:
            return await self._get_messages_native(limit)
        return await self._get_messages_fallback(limit)

    async def _get_messages_native(self, limit: int) -> List[ConversationMessage]:
        """Get via native CF_MESSAGES list_messages."""
        try:
            raw_messages = await self._state_adapter.list_messages(
                correlation_id=self._session_id,
                limit=limit,
            )

            messages = []
            for raw in raw_messages:
                # raw is proto-encoded ChatMessage bytes
                # Parse the payload field which contains our JSON
                try:
                    # Try to decode as proto ChatMessage and extract payload
                    # Decode the payload field as JSON when possible.
                    # The proto ChatMessage has payload as bytes field
                    # Simple approach: try JSON decode directly
                    msg_data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    if isinstance(msg_data, dict) and "payload" in msg_data:
                        payload = msg_data["payload"]
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        messages.append(ConversationMessage.from_dict(payload))
                    else:
                        messages.append(ConversationMessage.from_dict(msg_data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Proto bytes — fall back to extracting what we can
                    logger.debug("Could not parse native message, skipping")
                    continue

            return messages
        except Exception as e:
            logger.warning(f"Native list_messages failed, falling back to CF_STATE: {e}")
            return await self._get_messages_fallback(limit)

    async def _get_messages_fallback(self, limit: int) -> List[ConversationMessage]:
        """Get via CF_STATE (standalone/testing mode)."""
        session_data = await self._state_adapter.load_state(
            entity_type=self._ENTITY_TYPE,
            entity_key=self._entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        if not session_data:
            return []

        messages_data = session_data.get("messages", [])
        messages = [ConversationMessage.from_dict(m) for m in messages_data]

        if limit and len(messages) > limit:
            messages = messages[-limit:]

        return messages

    async def get_as_lm_messages(self, limit: int = 50):
        """Get messages formatted for LLM consumption.

        Convenience method that returns messages as LM Message objects,
        ready to pass to agent.run() or lm.generate().

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of Message objects for LM API

        Example:
            history = await ctx.conversation.get_as_lm_messages(limit=20)
            result = await agent.run("Continue", history=history)
        """
        messages = await self.get_messages(limit=limit)
        return [m.to_lm_message() for m in messages]

    async def get_state(self) -> Dict[str, Any]:
        """Get session-level metadata (not individual messages).

        Returns:
            Dict with session_id, created_at, last_message_at, message_count
        """
        session_data = await self._state_adapter.load_state(
            entity_type=self._ENTITY_TYPE,
            entity_key=self._entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        if not session_data:
            return {
                "session_id": self._session_id,
                "created_at": None,
                "last_message_at": None,
                "message_count": 0,
            }

        return {
            "session_id": session_data.get("session_id", self._session_id),
            "created_at": session_data.get("created_at"),
            "last_message_at": session_data.get("last_message_at"),
            "message_count": session_data.get("message_count", 0),
        }

    async def clear(self) -> None:
        """Clear all messages in this conversation.

        Example:
            await ctx.conversation.clear()
        """
        _, current_version = await self._state_adapter.load_with_version(
            entity_type=self._ENTITY_TYPE,
            entity_key=self._entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        now = time.time()
        session_data = {
            "session_id": self._session_id,
            "component_name": self._component_name,
            "created_at": now,
            "last_message_at": now,
            "message_count": 0,
            "messages": [],
        }

        try:
            await self._state_adapter.save_state(
                entity_type=self._ENTITY_TYPE,
                entity_key=self._entity_key,
                state=session_data,
                expected_version=current_version,
                scope="session",
                scope_id=self._session_id,
            )
            logger.info(f"Cleared conversation {self._session_id}")
        except Exception as e:
            logger.error(f"Failed to clear conversation {self._session_id}: {e}")
            raise

    @property
    def session_id(self) -> str:
        """Get the session ID for this conversation."""
        return self._session_id
