"""Platform configuration dataclasses for the AGNT5 Chat SDK."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlackConfig:
    """Configuration for connecting to Slack.

    Args:
        bot_token: Slack Bot User OAuth Token (xoxb-...).
        signing_secret: Slack app signing secret for webhook verification.
        app_token: Optional Slack App-Level Token (xapp-...) for Socket Mode.
    """

    bot_token: str
    signing_secret: str
    app_token: str | None = None
    platform: str = "slack"


@dataclass(frozen=True)
class DiscordConfig:
    """Configuration for connecting to Discord (Phase 3).

    Args:
        bot_token: Discord bot token.
        public_key: Discord app public key for webhook verification (Ed25519).
        application_id: Discord application ID.
    """

    bot_token: str
    public_key: str
    application_id: str
    platform: str = "discord"


@dataclass(frozen=True)
class TeamsConfig:
    """Configuration for connecting to Microsoft Teams (Phase 4).

    Args:
        app_id: Microsoft App ID.
        app_password: Microsoft App Password.
        tenant_id: Azure AD tenant ID (optional, for single-tenant apps).
    """

    app_id: str
    app_password: str
    tenant_id: str | None = None
    platform: str = "teams"


@dataclass(frozen=True)
class TelegramConfig:
    """Configuration for connecting to Telegram (Phase 3).

    Args:
        bot_token: Telegram Bot API token from @BotFather.
        webhook_secret: Optional secret token for webhook verification.
    """

    bot_token: str
    webhook_secret: str | None = None
    platform: str = "telegram"


# Union type for any platform config
PlatformConfig = SlackConfig | DiscordConfig | TeamsConfig | TelegramConfig
