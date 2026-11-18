"""The built-in tools of agents declared ``per="user"``.

- ``schedule_followup`` wakes the agent again for the same user later. The
  platform schedules a timer on the user's instance and, when it is due,
  delivers the follow-up's note to the instance's inbox, which starts the
  agent's next turn.
- ``send`` messages the user. The platform delivers it to the app's webhook,
  in the order the agent sent its messages, with an id a retry keeps.

Both ask the platform on the run's own journal; the platform does the rest
for the run's own instance only.
"""

from __future__ import annotations

import contextvars
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..events import Event
from ..tool import Tool

#: The longest follow-up, in seconds from now (30 days).
MAX_FOLLOWUP_DELAY_SECONDS = 30 * 24 * 3600

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.:#-]{1,128}$")


@dataclass
class _InstanceTurn:
    run_id: str
    conversation_id: str


#: The per-user turn being run, set by the worker for its duration.
current_instance_turn: contextvars.ContextVar[Optional[_InstanceTurn]] = contextvars.ContextVar(
    "agnt5_instance_turn", default=None
)


def enter_instance_turn(run_id: str, conversation_id: str) -> contextvars.Token:
    """Mark the code that follows as running a per-user agent's turn."""
    return current_instance_turn.set(_InstanceTurn(run_id, conversation_id))


@dataclass
class FollowupRequested(Event):
    """Asks the platform to schedule a follow-up for this run's instance."""

    followup_key: str = ""
    fire_at_ns: int = 0
    conversation_id: str = ""
    content: Any = None

    event_type: str = field(default="instance.followup_requested", init=False)


def followup_timer_id(run_id: str, followup_key: str) -> str:
    """The follow-up's timer id, as the platform names it."""
    return f"followup-{run_id}-{followup_key}"


def _default_key(delay_seconds: int, message: str) -> str:
    # The same call in a replay asks again with the same key, and so gets the
    # same timer.
    digest = hashlib.sha256(f"{delay_seconds}\0{message}".encode()).hexdigest()
    return f"auto-{digest[:24]}"


async def _schedule_followup(
    ctx: Any, delay_seconds: int, message: str, key: Optional[str] = None
) -> dict:
    """Wake yourself again for this user later, with a note to yourself.

    Use it to check back on something: a PR that needs a review, a reply
    you're waiting for. The note arrives as a new message in this
    conversation when it is due.

    Args:
        delay_seconds: How long from now, in seconds (at most 30 days).
        message: The note you'll receive when it is due.
        key: Optional name for this follow-up; calling again with the same key
            returns the same follow-up.
    """
    turn = current_instance_turn.get()
    if turn is None:
        return {"error": "follow-ups are only available to agents declared per='user'"}
    if not 0 <= int(delay_seconds) <= MAX_FOLLOWUP_DELAY_SECONDS:
        return {"error": f"delay_seconds must be between 0 and {MAX_FOLLOWUP_DELAY_SECONDS}"}
    followup_key = key or _default_key(int(delay_seconds), message)
    if not _KEY_PATTERN.match(followup_key):
        return {"error": "key must be 1 to 128 letters, digits or -_.:#"}
    emitter = getattr(ctx, "_get_emitter", None)
    if emitter is None:
        return {"error": "follow-ups need the run's context"}
    await emitter().emit_now_async(
        FollowupRequested(
            name="schedule_followup",
            correlation_id=followup_key,
            parent_correlation_id="",
            followup_key=followup_key,
            fire_at_ns=time.time_ns() + int(delay_seconds) * 1_000_000_000,
            conversation_id=turn.conversation_id,
            content={"message": message},
        )
    )
    return {
        "timer_id": followup_timer_id(turn.run_id, followup_key),
        "due_in_seconds": int(delay_seconds),
    }


def schedule_followup_tool() -> Tool:
    """The built-in ``schedule_followup`` tool every per-user agent gets."""
    return Tool(
        name="schedule_followup",
        description=(
            "Wake yourself again for this user later, with a note to yourself. "
            "The note arrives as a new message in this conversation when it is due."
        ),
        handler=_schedule_followup,
        auto_schema=True,
    )


@dataclass
class MessageRequested(Event):
    """Asks the platform to send this run's user a message."""

    message_key: str = ""
    conversation_id: str = ""
    content: Any = None

    event_type: str = field(default="instance.message_requested", init=False)


def message_id(run_id: str, message_key: str) -> str:
    """The message's id, as the platform names it and the app receives it."""
    return f"msg-{run_id}-{message_key}"


async def _send(ctx: Any, text: str, key: Optional[str] = None) -> dict:
    """Send the user a message now, in this conversation.

    Use it when something is worth telling the user without waiting for them
    to ask: a review landed, a check failed. Your final answer to a message
    from the user is sent anyway; you don't need this for it.

    Args:
        text: What to tell the user.
        key: Optional name for this message; sending the same text with the
            same key again sends nothing new.
    """
    turn = current_instance_turn.get()
    if turn is None:
        return {"error": "send is only available to agents declared per='user'"}
    if not text:
        return {"error": "text must not be empty"}
    # The call's key plus a hash of what it sends: a replay sends nothing
    # new, and a different message under the same key is a different one.
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    message_key = f"{key or 'send'}.{digest}"
    if not _KEY_PATTERN.match(message_key):
        return {"error": "key must be letters, digits or -_.:#, and short"}
    emitter = getattr(ctx, "_get_emitter", None)
    if emitter is None:
        return {"error": "send needs the run's context"}
    await emitter().emit_now_async(
        MessageRequested(
            name="send",
            correlation_id=message_key,
            parent_correlation_id="",
            message_key=message_key,
            conversation_id=turn.conversation_id,
            content={"text": text},
        )
    )
    return {"message_id": message_id(turn.run_id, message_key)}


def send_tool() -> Tool:
    """The built-in ``send`` tool every per-user agent gets."""
    return Tool(
        name="send",
        description="Send the user a message now, in this conversation.",
        handler=_send,
        auto_schema=True,
    )
