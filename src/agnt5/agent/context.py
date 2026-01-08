"""Agent execution context with conversation state management."""

import logging
import os
import secrets
import time
import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..context import Context
from ..lm import Message

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Gateway URL for session history API (defaults to localhost dev server)
DEFAULT_GATEWAY_URL = "http://localhost:34181"


class AgentContext(Context):
    """
    Context for agent execution with conversation state management.

    Extends base Context with:
    - State management via EntityStateManager
    - Conversation history persistence
    - Context inheritance (child agents share parent's state)

    Three initialization modes:
    1. Standalone: Creates own state manager (playground testing)
    2. Inherit WorkflowContext: Shares parent's state manager
    3. Inherit parent AgentContext: Shares parent's state manager

    Example:
        ```python
        # Standalone agent with conversation history
        ctx = AgentContext(run_id="session-1", agent_name="tutor")
        result = await agent.run_sync("Hello", context=ctx)
        result = await agent.run_sync("Continue", context=ctx)  # Remembers previous message

        # Agent in workflow - shares workflow state
        @workflow
        async def research_workflow(ctx: WorkflowContext):
            agent_result = await research_agent.run_sync("Find AI trends", context=ctx)
            # Agent has access to workflow state via inherited context
        ```
    """

    def __init__(
        self,
        run_id: str,
        agent_name: str,
        session_id: Optional[str] = None,
        state_manager: Optional[Any] = None,
        parent_context: Optional[Context] = None,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
        is_streaming: bool = False,
        worker: Optional[Any] = None,
        correlation_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
    ):
        """
        Initialize agent context.

        Args:
            run_id: Unique execution identifier
            agent_name: Name of the agent
            session_id: Session identifier for conversation history (default: run_id)
            state_manager: Optional state manager (for context inheritance)
            parent_context: Parent context to inherit state from
            attempt: Retry attempt number
            runtime_context: RuntimeContext for trace correlation
            is_streaming: Whether this is a streaming request (for real-time SSE log delivery)
            worker: PyWorker instance for event queueing
            correlation_id: Unique identifier for this agent execution (auto-generated if not provided)
            parent_correlation_id: Parent's correlation ID for event hierarchy
        """
        # Inherit is_streaming and worker from parent context if not explicitly provided
        if parent_context and not is_streaming:
            is_streaming = getattr(parent_context, '_is_streaming', False)
        if parent_context and not worker:
            worker = getattr(parent_context, '_worker', None)

        # Generate correlation IDs for agent execution if not provided
        if not correlation_id:
            correlation_id = f"agent-{secrets.token_hex(5)}"
        if not parent_correlation_id:
            parent_correlation_id = getattr(parent_context, '_correlation_id', '') if parent_context else ''

        # Initialize parent Context with memoization enabled by default for agents
        # This ensures LLM and tool calls are automatically journaled for replay
        super().__init__(
            run_id=run_id,
            correlation_id=correlation_id,
            parent_correlation_id=parent_correlation_id,
            attempt=attempt,
            runtime_context=runtime_context,
            is_streaming=is_streaming,
            session_id=session_id,
            enable_memoization=True,  # Agents get memoization by default
            worker=worker,
        )

        self._agent_name = agent_name
        self._session_id = session_id or run_id
        self.parent_context = parent_context  # Store for context chain traversal

        # Determine state adapter based on parent context
        from ..entity import EntityStateAdapter, _get_state_adapter

        if state_manager:
            # Explicit state adapter provided (parameter name kept for backward compat)
            self._state_adapter = state_manager
        elif parent_context:
            # Try to inherit state adapter from parent
            try:
                # Check if parent is WorkflowContext or AgentContext
                if hasattr(parent_context, '_workflow_entity'):
                    # WorkflowContext - get state adapter from worker context
                    self._state_adapter = _get_state_adapter()
                elif hasattr(parent_context, '_state_adapter'):
                    # Parent AgentContext - share state adapter
                    self._state_adapter = parent_context._state_adapter
                elif hasattr(parent_context, '_state_manager'):
                    # Backward compatibility: parent has old _state_manager
                    self._state_adapter = parent_context._state_manager
                else:
                    # FunctionContext or base Context - create new state adapter
                    self._state_adapter = EntityStateAdapter()
            except RuntimeError:
                # _get_state_adapter() failed (not in worker context) - create standalone
                self._state_adapter = EntityStateAdapter()
        else:
            # Try to get from worker context first
            try:
                self._state_adapter = _get_state_adapter()
            except RuntimeError:
                # Standalone - create new state adapter
                self._state_adapter = EntityStateAdapter()

        # Conversation key for state storage (used for in-memory state)
        self._conversation_key = f"agent:{agent_name}:{self._session_id}:messages"
        # Entity key for database persistence (without :messages suffix to match API expectations)
        self._entity_key = f"agent:{agent_name}:{self._session_id}"

        # Determine storage mode: "workflow" if parent is WorkflowContext, else "standalone"
        self._storage_mode = "standalone"  # Default mode
        self._workflow_entity = None

        if parent_context and hasattr(parent_context, '_workflow_entity'):
            # Agent is running within a workflow - store conversation in workflow state
            self._storage_mode = "workflow"
            self._workflow_entity = parent_context._workflow_entity
            logger.debug(
                f"Agent '{agent_name}' using workflow storage mode "
                f"(workflow entity: {self._workflow_entity.key})"
            )

    @property
    def state(self):
        """
        Get state interface for agent state management.

        Note: This is a simplified in-memory state interface for agent-specific data.
        Conversation history is managed separately via get_conversation_history() and
        save_conversation_history() which use the Rust-backed persistence layer.

        Returns:
            Dict-like object for state operations

        Example:
            # Store agent-specific data (in-memory only)
            ctx.state["research_results"] = data
            ctx.state["iteration_count"] = 5
        """
        # Simple dict-based state for agent-specific data
        # This is in-memory only and not persisted to platform
        if not hasattr(self, '_agent_state'):
            self._agent_state = {}
        return self._agent_state

    @property
    def session_id(self) -> str:
        """Get session identifier for this agent context."""
        return self._session_id

    async def get_conversation_history(self) -> List[Message]:
        """
        Retrieve conversation history, preferring runs-based history from the platform.

        Load order (as of Phase 5.2 - runs-first architecture):
        1. For workflow mode: Load from workflow entity state (shared state)
        2. For standalone mode:
           a. Try loading from runs via gateway API (/v1/sessions/{id}/history)
           b. Fall back to entity storage for legacy sessions (with deprecation warning)

        Returns:
            List of Message objects from conversation history
        """
        if self._storage_mode == "workflow":
            return await self._load_from_workflow_state()
        else:
            # Try runs-based API first (Phase 5.2 architecture)
            messages = await self._load_from_runs_api()
            if messages:
                return messages

            # Fall back to entity storage for legacy sessions
            legacy_messages = await self._load_from_entity_storage()
            if legacy_messages:
                warnings.warn(
                    "Loading conversation history from entity storage is deprecated. "
                    "New sessions use runs-based history. Consider migrating this session.",
                    DeprecationWarning,
                    stacklevel=2
                )
            return legacy_messages

    async def _load_from_workflow_state(self) -> List[Message]:
        """Load conversation history from workflow entity state."""
        key = f"agent.{self._agent_name}"
        agent_data = self._workflow_entity.state.get(key, {})
        messages_data = agent_data.get("messages", [])

        # Convert dict representations back to Message objects
        return self._convert_dicts_to_messages(messages_data)

    async def _load_from_runs_api(self) -> List[Message]:
        """
        Load conversation history from runs via gateway API.

        This is the new Phase 5.2 architecture where conversation history
        is derived from runs (each run = one conversation turn) rather than
        stored in entity state.

        Returns:
            List of Message objects, or empty list if no runs found or API fails
        """
        import httpx

        gateway_url = os.environ.get("AGNT5_GATEWAY_URL", DEFAULT_GATEWAY_URL)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{gateway_url}/v1/sessions/{self._session_id}/history"

                response = await client.get(url)

                if response.status_code == 404:
                    # Session not found - this might be a new session or legacy session
                    logger.debug(f"Session {self._session_id} not found in runs API")
                    return []

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to load session history from runs API: "
                        f"status={response.status_code}"
                    )
                    return []

                data = response.json()
                messages_data = data.get("messages", [])

                if not messages_data:
                    return []

                # Convert API response to Message objects
                messages = []
                for msg in messages_data:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    # Content might be JSON-encoded if it was stored as structured data
                    if isinstance(content, dict):
                        # Extract text content if it's a structured message
                        content = content.get("text", content.get("message", str(content)))

                    if role == "user":
                        messages.append(Message.user(content))
                    elif role == "assistant":
                        messages.append(Message.assistant(content))
                    else:
                        # Handle other roles (system, etc.)
                        from ..lm import MessageRole
                        msg_role = MessageRole(role) if role in ("user", "assistant", "system") else MessageRole.USER
                        messages.append(Message(role=msg_role, content=content))

                logger.debug(
                    f"Loaded {len(messages)} messages from runs API for session {self._session_id}"
                )
                return messages

        except httpx.HTTPError as e:
            logger.debug(f"HTTP error loading from runs API: {e}")
            return []
        except Exception as e:
            logger.debug(f"Error loading from runs API: {e}")
            return []

    async def _load_from_entity_storage(self) -> List[Message]:
        """Load conversation history from AgentSession entity (standalone mode)."""
        entity_type = "AgentSession"
        entity_key = self._entity_key

        # Load session data via adapter (Rust handles cache + platform load)
        # Use session scope with session_id for proper entity isolation
        session_data = await self._state_adapter.load_state(
            entity_type,
            entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        # Extract messages from session object
        if isinstance(session_data, dict) and "messages" in session_data:
            # New format with session metadata
            messages_data = session_data["messages"]
        elif isinstance(session_data, list):
            # Old format - just messages array
            messages_data = session_data
        else:
            # No messages found
            messages_data = []

        # Convert dict representations back to Message objects
        return self._convert_dicts_to_messages(messages_data)

    def _convert_dicts_to_messages(self, messages_data: list) -> List[Message]:
        """Convert list of message dicts to Message objects."""
        from ..lm import MessageRole

        messages = []
        for msg_dict in messages_data:
            if isinstance(msg_dict, dict):
                role = msg_dict.get("role", "user")
                content = msg_dict.get("content", "")
                if role == "user":
                    messages.append(Message.user(content))
                elif role == "assistant":
                    messages.append(Message.assistant(content))
                else:
                    # Generic message - create with MessageRole enum
                    msg_role = (
                        MessageRole(role)
                        if role in ("user", "assistant", "system")
                        else MessageRole.USER
                    )
                    msg = Message(role=msg_role, content=content)
                    messages.append(msg)
            else:
                # Already a Message object
                messages.append(msg_dict)

        return messages

    async def save_conversation_history(self, messages: List[Message]) -> None:
        """
        Save conversation history to state and persist to database.

        Uses the EntityStateAdapter which delegates to Rust core for version-checked saves.
        If running within a workflow, saves to workflow entity state instead.

        Args:
            messages: List of Message objects to persist
        """
        if self._storage_mode == "workflow":
            await self._save_to_workflow_state(messages)
        else:
            await self._save_to_entity_storage(messages)

    async def _save_to_workflow_state(self, messages: List[Message]) -> None:
        """Save conversation history to workflow entity state."""
        # Convert Message objects to dict for JSON serialization
        messages_data = []
        for msg in messages:
            messages_data.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content,
                "timestamp": time.time()
            })

        # Build agent data structure
        key = f"agent.{self._agent_name}"
        current_data = self._workflow_entity.state.get(key, {})
        now = time.time()

        agent_data = {
            "session_id": self._session_id,
            "agent_name": self._agent_name,
            "created_at": current_data.get("created_at", now),
            "last_message_time": now,
            "message_count": len(messages_data),
            "messages": messages_data,
            "metadata": getattr(self, '_custom_metadata', {})
        }

        # Store in workflow state (WorkflowEntity handles persistence)
        self._workflow_entity.state.set(key, agent_data)
        logger.info(f"Saved conversation to workflow state: {key} ({len(messages_data)} messages)")

    async def _save_to_entity_storage(self, messages: List[Message]) -> None:
        """
        Save conversation history to AgentSession entity (standalone mode).

        DEPRECATED: This method saves to entity storage which is the legacy approach.
        In the Phase 5.2 architecture, conversation history is derived from runs
        (each run = one turn). New conversations should not need to call this
        as the platform automatically records run inputs/outputs.
        """
        warnings.warn(
            "Saving conversation history to entity storage is deprecated. "
            "In the new architecture, conversation history is derived from runs. "
            "This method will be removed in a future version.",
            DeprecationWarning,
            stacklevel=3
        )

        # Convert Message objects to dict for JSON serialization
        messages_data = []
        for msg in messages:
            messages_data.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content,
                "timestamp": time.time()  # Add timestamp for each message
            })

        entity_type = "AgentSession"
        entity_key = self._entity_key

        # Load current state with version for optimistic locking
        # Use session scope with session_id for proper entity isolation
        current_state, current_version = await self._state_adapter.load_with_version(
            entity_type,
            entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        # Build session object with metadata
        now = time.time()

        # Get custom metadata from instance variable or preserve from loaded state
        custom_metadata = getattr(self, '_custom_metadata', current_state.get("metadata", {}))

        session_data = {
            "session_id": self._session_id,
            "agent_name": self._agent_name,
            "created_at": current_state.get("created_at", now),  # Preserve existing or set new
            "last_message_time": now,
            "message_count": len(messages_data),
            "messages": messages_data,
            "metadata": custom_metadata  # Save custom metadata
        }

        # Save to platform via adapter (Rust handles optimistic locking)
        # Use session scope with session_id for proper entity isolation
        try:
            new_version = await self._state_adapter.save_state(
                entity_type,
                entity_key,
                session_data,
                current_version,
                scope="session",
                scope_id=self._session_id,
            )
            logger.info(
                f"Persisted conversation history: {entity_key} "
                f"(version {current_version} -> {new_version})"
            )
        except Exception as e:
            logger.error(f"Failed to persist conversation history to database: {e}")
            # Don't fail - conversation is still in memory for this execution

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get conversation session metadata.

        Returns session metadata including:
        - created_at: Timestamp of first message (float, Unix timestamp)
        - last_activity: Timestamp of last message (float, Unix timestamp)
        - message_count: Number of messages in conversation (int)
        - custom: Dict of user-provided custom metadata

        Returns:
            Dictionary with metadata. If no conversation exists yet, returns defaults.

        Example:
            ```python
            metadata = await context.get_metadata()
            print(f"Session created: {metadata['created_at']}")
            print(f"User ID: {metadata['custom'].get('user_id')}")
            ```
        """
        if self._storage_mode == "workflow":
            return await self._get_metadata_from_workflow()
        else:
            return await self._get_metadata_from_entity()

    async def _get_metadata_from_workflow(self) -> Dict[str, Any]:
        """Get metadata from workflow entity state."""
        key = f"agent.{self._agent_name}"
        agent_data = self._workflow_entity.state.get(key, {})

        if not agent_data:
            # No conversation exists yet - return defaults
            return {
                "created_at": None,
                "last_activity": None,
                "message_count": 0,
                "custom": getattr(self, '_custom_metadata', {})
            }

        messages = agent_data.get("messages", [])
        return {
            "created_at": agent_data.get("created_at"),
            "last_activity": agent_data.get("last_message_time"),
            "message_count": len(messages),
            "custom": agent_data.get("metadata", {})
        }

    async def _get_metadata_from_entity(self) -> Dict[str, Any]:
        """Get metadata from AgentSession entity (standalone mode)."""
        entity_type = "AgentSession"
        entity_key = self._entity_key

        # Load session data with session scope
        session_data = await self._state_adapter.load_state(
            entity_type,
            entity_key,
            scope="session",
            scope_id=self._session_id,
        )

        if not session_data:
            # No conversation exists yet - return defaults
            return {
                "created_at": None,
                "last_activity": None,
                "message_count": 0,
                "custom": getattr(self, '_custom_metadata', {})
            }

        messages = session_data.get("messages", [])

        # Derive timestamps from messages if available
        created_at = session_data.get("created_at")
        last_activity = session_data.get("last_message_time")

        return {
            "created_at": created_at,
            "last_activity": last_activity,
            "message_count": len(messages),
            "custom": session_data.get("metadata", {})
        }

    def update_metadata(self, **kwargs) -> None:
        """
        Update custom session metadata.

        Metadata will be persisted alongside conversation history on next save.
        Use this to store application-specific data like user_id, preferences, etc.

        Args:
            **kwargs: Key-value pairs to store as metadata

        Example:
            ```python
            # Store user identification and preferences
            context.update_metadata(
                user_id="user-123",
                subscription_tier="premium",
                preferences={"theme": "dark", "language": "en"}
            )

            # Later retrieve it
            metadata = await context.get_metadata()
            user_id = metadata["custom"]["user_id"]
            ```

        Note:
            - Metadata is merged with existing metadata (doesn't replace)
            - Changes persist on next save_conversation_history() call
            - Use simple JSON-serializable types (str, int, float, dict, list)
        """
        if not hasattr(self, '_custom_metadata'):
            self._custom_metadata = {}
        self._custom_metadata.update(kwargs)
