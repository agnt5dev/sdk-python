"""ChatBot — connects an AGNT5 Agent to chat platforms (Slack, Discord, etc.)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Awaitable

import httpx

from .config import PlatformConfig, SlackConfig

if TYPE_CHECKING:
    from agnt5.agent import Agent

logger = logging.getLogger("agnt5.chat")

# Rust core bindings
from agnt5._core import chat as _chat  # type: ignore[attr-defined]


class ChatBot:
    """Wraps an :class:`Agent` and bridges it to one or more chat platforms.

    The bot registers as a regular Agent component — no new proto types needed.
    When a webhook arrives it is parsed/verified by the Rust core, routed to
    a handler, and the agent's response is sent back via the platform API.

    Example::

        from agnt5 import Agent, Worker
        from agnt5.chat import ChatBot, SlackConfig

        agent = Agent(name="support-bot", model="anthropic/claude-sonnet-4-20250514",
                      instructions="You are a helpful support agent.")

        bot = ChatBot(agent=agent, adapters=[
            SlackConfig(bot_token=os.environ["SLACK_BOT_TOKEN"],
                        signing_secret=os.environ["SLACK_SIGNING_SECRET"]),
        ])

        worker = Worker()
        worker.register(bot)
        worker.start()
    """

    def __init__(
        self,
        agent: Agent,
        adapters: list[PlatformConfig],
    ) -> None:
        self._agent = agent
        self._adapters: dict[str, PlatformConfig] = {a.platform: a for a in adapters}
        self._handlers: dict[str, Callable[..., Awaitable[str | None]]] = {}

    # -- Public properties -----------------------------------------------------

    @property
    def name(self) -> str:
        """Component name (delegates to the wrapped agent)."""
        return self._agent.name

    @property
    def agent(self) -> Agent:
        return self._agent

    # -- Handler registration --------------------------------------------------

    def on_mention(self, fn: Callable[..., Awaitable[str | None]]) -> Callable[..., Awaitable[str | None]]:
        """Register a handler for bot mention events. The handler receives a
        ``ChatEvent`` and should return a response string (or None to skip)."""
        self._handlers["mention"] = fn
        return fn

    def on_message(self, fn: Callable[..., Awaitable[str | None]]) -> Callable[..., Awaitable[str | None]]:
        """Register a handler for all message events."""
        self._handlers["message"] = fn
        return fn

    def on_reaction(self, fn: Callable[..., Awaitable[str | None]]) -> Callable[..., Awaitable[str | None]]:
        """Register a handler for emoji reaction events."""
        self._handlers["reaction"] = fn
        return fn

    def on_action(self, fn: Callable[..., Awaitable[str | None]]) -> Callable[..., Awaitable[str | None]]:
        """Register a handler for interactive component actions."""
        self._handlers["action"] = fn
        return fn

    def on_slash_command(self, fn: Callable[..., Awaitable[str | None]]) -> Callable[..., Awaitable[str | None]]:
        """Register a handler for slash commands."""
        self._handlers["slash_command"] = fn
        return fn

    # -- Webhook handling (called by worker) -----------------------------------

    async def handle_webhook(
        self,
        platform: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any] | None:
        """Process an incoming webhook from a chat platform.

        This method is called by the worker's message handler when a webhook
        dispatch arrives. It:

        1. Verifies the webhook signature (Rust core)
        2. Parses the payload into a normalized ChatEvent (Rust core)
        3. Routes to the appropriate handler (or default agent handler)
        4. Sends the response back via platform API

        Returns a dict with the url_verification challenge if applicable,
        or None for normal event processing.
        """
        config = self._adapters.get(platform)
        if config is None:
            raise ValueError(f"No adapter configured for platform: {platform}")

        platform_obj = _chat.Platform.from_str(platform)

        # Verify webhook signature
        signing_secret = self._get_signing_secret(config)
        if signing_secret:
            valid = _chat.verify_webhook(platform_obj, signing_secret, headers, body)
            if not valid:
                raise ValueError("Invalid webhook signature")

        # Parse event
        event = _chat.parse_event(platform_obj, body)

        # Handle URL verification challenge (Slack sends this during setup)
        if event.event_type == "url_verification":
            return {"challenge": event.challenge}

        # Route to handler
        handler = self._handlers.get(event.event_type, self._default_handler)
        response_text = await handler(event)

        if response_text is None:
            return None

        # Send response back to the platform
        await self._send_response(config, event, response_text)
        return None

    async def handle_webhook_streaming(
        self,
        platform: str,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any] | None:
        """Process a webhook with streaming agent response (post-then-edit).

        Same as handle_webhook but uses the StreamingMessageBuffer for
        progressive message updates during agent execution.
        """
        config = self._adapters.get(platform)
        if config is None:
            raise ValueError(f"No adapter configured for platform: {platform}")

        platform_obj = _chat.Platform.from_str(platform)

        # Verify + parse
        signing_secret = self._get_signing_secret(config)
        if signing_secret:
            valid = _chat.verify_webhook(platform_obj, signing_secret, headers, body)
            if not valid:
                raise ValueError("Invalid webhook signature")

        event = _chat.parse_event(platform_obj, body)

        if event.event_type == "url_verification":
            return {"challenge": event.challenge}

        msg = event.message
        if msg is None:
            return None

        # Determine thread context
        # Use existing thread_id, or start a new thread from the message ts
        thread_ts = msg.thread_id or msg.id

        # Post initial "thinking" message
        assert isinstance(config, SlackConfig)
        initial_req = _chat.slack_post_message(
            config.bot_token, msg.channel_id, "Thinking...", thread_ts,
        )
        initial_resp = await self._execute_request(initial_req)
        resp_data = initial_resp.json()
        message_ts = resp_data.get("ts", "")

        # Stream agent response with buffer
        buffer = _chat.StreamingMessageBuffer(flush_interval_ms=500)

        async for event_obj in self._agent.run_stream(msg.content):
            # Check for output delta events
            token = getattr(event_obj, "delta", None)
            if token is None:
                continue

            buffer.push(token)

            if buffer.should_flush():
                content = buffer.flush()
                if content is not None:
                    update_req = _chat.slack_update_message(
                        config.bot_token, msg.channel_id, message_ts, content,
                    )
                    await self._execute_request(update_req)

        # Final update with complete content
        final_content = buffer.finalize()
        if final_content:
            final_req = _chat.slack_update_message(
                config.bot_token, msg.channel_id, message_ts, final_content,
            )
            await self._execute_request(final_req)

        return None

    # -- Internal helpers ------------------------------------------------------

    async def _default_handler(self, event: Any) -> str | None:
        """Default handler: run the agent with the message content.

        Maps the platform thread ID to an AGNT5 session_id so multi-turn
        conversations within the same thread share conversation context.
        """
        from agnt5.agent import AgentContext

        msg = event.message
        if msg is None:
            return None

        # Derive session_id from thread context:
        # - If in a thread, use thread_ts (all replies share the same session)
        # - If a top-level message, use the message ts (starts a new thread/session)
        thread_key = msg.thread_id or msg.id
        session_id = self._thread_to_session(str(msg.platform), msg.channel_id, thread_key)

        ctx = AgentContext(
            run_id=session_id,
            agent_name=self._agent.name,
            session_id=session_id,
        )

        result = await self._agent.run_sync(msg.content, context=ctx)
        return str(result.output) if result.output is not None else None

    async def _send_response(
        self,
        config: PlatformConfig,
        event: Any,
        text: str,
    ) -> None:
        """Send a response message back to the platform."""
        if isinstance(config, SlackConfig):
            msg = event.message
            thread_ts = None
            if msg is not None:
                thread_ts = msg.thread_id or msg.id

            req = _chat.slack_post_message(
                config.bot_token,
                event.channel_id,
                text,
                thread_ts,
            )
            await self._execute_request(req)
        else:
            raise NotImplementedError(f"Platform {config.platform} not yet supported")

    async def _execute_request(self, req: Any) -> httpx.Response:
        """Execute a PlatformRequest using httpx."""
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=req.method,
                url=req.url,
                headers=req.headers,
                content=req.body,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Platform API error: %s %s -> %s %s",
                    req.method, req.url, response.status_code, response.text[:200],
                )
            return response

    def _thread_to_session(self, platform: str, channel_id: str, thread_id: str) -> str:
        """Derive a deterministic AGNT5 session_id from a platform thread.

        Uses a UUID5 namespace so the same thread always maps to the same
        session, enabling multi-turn conversation continuity.
        """
        import uuid
        # Namespace UUID for AGNT5 chat sessions (deterministic, never changes)
        CHAT_SESSION_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        key = f"{platform}:{channel_id}:{thread_id}"
        return str(uuid.uuid5(CHAT_SESSION_NS, key))

    def _get_signing_secret(self, config: PlatformConfig) -> str | None:
        """Extract the signing secret from a platform config."""
        if isinstance(config, SlackConfig):
            return config.signing_secret
        # TODO: Discord public_key, Teams app_password, Telegram webhook_secret
        return None
