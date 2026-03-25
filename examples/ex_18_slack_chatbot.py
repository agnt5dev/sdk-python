"""
Example: Slack ChatBot

This example demonstrates connecting an AGNT5 Agent to Slack:
- Creating a ChatBot with a Slack adapter
- Default behavior: agent processes every mention automatically
- Custom handlers: override behavior for mentions, messages, reactions
- Streaming: progressive message updates while the agent thinks
- Thread continuity: same Slack thread = same AGNT5 session

The bot is registered as a regular agent component — no new infrastructure
needed. The Gateway webhook endpoint dispatches Slack events to your worker.

Setup:
1. Create a Slack app at https://api.slack.com/apps
2. Enable Events API, subscribe to: app_mention, message.im
3. Add bot scopes: chat:write, app_mentions:read, im:history
4. Set the webhook URL to: https://<gateway>/v1/chat/webhooks/slack/<bot-name>
5. Set environment variables:
   - SLACK_BOT_TOKEN: xoxb-... token from OAuth & Permissions
   - SLACK_SIGNING_SECRET: from Basic Information > App Credentials
   - OPENAI_API_KEY or ANTHROPIC_API_KEY: for the agent's LLM

Run:
    python ex_18_slack_chatbot.py
"""

import os

from agnt5 import Agent, Worker
from agnt5.chat import ChatBot, SlackConfig

# =============================================================================
# BASIC CHATBOT — Agent handles all mentions automatically
# =============================================================================

support_agent = Agent(
    name="support-bot",
    model="anthropic/claude-sonnet-4-20250514",
    instructions=(
        "You are a helpful support agent in Slack. "
        "Keep responses concise and use Slack-compatible markdown. "
        "If you don't know the answer, say so honestly."
    ),
    temperature=0.3,
    max_iterations=5,
)

# Wrap the agent in a ChatBot with Slack credentials
bot = ChatBot(
    agent=support_agent,
    adapters=[
        SlackConfig(
            bot_token=os.environ.get("SLACK_BOT_TOKEN", "xoxb-placeholder"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET", "placeholder"),
        ),
    ],
)


# =============================================================================
# CUSTOM HANDLERS (optional) — override default agent behavior
# =============================================================================

# Uncomment any of these to customize how the bot responds to specific events.

# @bot.on_mention
# async def handle_mention(event):
#     """Custom mention handler — runs instead of the default agent.run()."""
#     user = event.user.name if event.user else "someone"
#     msg = event.message.content if event.message else ""
#     return f"Hey {user}! You said: {msg}"


# @bot.on_reaction
# async def handle_reaction(event):
#     """React to emoji reactions (e.g., auto-translate on :flag-es:)."""
#     if event.emoji == "flag-es":
#         return "¡Hola! Detected a Spanish flag reaction."
#     return None  # Return None to skip sending a response


# @bot.on_slash_command
# async def handle_slash(event):
#     """Handle slash commands like /support-status."""
#     if event.command == "/support-status":
#         return "All systems operational."
#     return None


# =============================================================================
# REGISTER AND RUN
# =============================================================================

if __name__ == "__main__":
    worker = Worker(service_name="support-bot")
    worker.register(bot)
    worker.start()
