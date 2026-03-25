"""AGNT5 Chat SDK — connect agents to Slack, Discord, Teams, and Telegram."""

from .bot import ChatBot
from .config import (
    DiscordConfig,
    PlatformConfig,
    SlackConfig,
    TeamsConfig,
    TelegramConfig,
)

__all__ = [
    "ChatBot",
    "DiscordConfig",
    "PlatformConfig",
    "SlackConfig",
    "TeamsConfig",
    "TelegramConfig",
]
