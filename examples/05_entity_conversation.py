"""
Example: Conversation/Session Pattern with Entity

This example demonstrates:
- Using Entity to implement conversation sessions
- Managing conversation history and context
- Multi-turn dialogues with state persistence
- Session pattern (replacing the need for a separate Session component)
"""

import asyncio
from datetime import datetime
from typing import Dict, List

from agnt5 import Context, entity

# Create Conversation entity type
Conversation = entity("Conversation")


@Conversation.method
async def add_message(ctx: Context, role: str, content: str) -> Dict:
    """Add a message to the conversation history."""
    messages = ctx.get("messages", [])
    timestamp = datetime.now().isoformat()

    message = {
        "role": role,
        "content": content,
        "timestamp": timestamp
    }

    messages.append(message)
    ctx.set("messages", messages)

    # Update conversation metadata
    ctx.set("last_updated", timestamp)
    turn_count = ctx.get("turn_count", 0)
    ctx.set("turn_count", turn_count + 1)

    ctx.logger.info(f"Message added ({role}): {content[:50]}...")

    return {
        "message_id": len(messages) - 1,
        "timestamp": timestamp
    }


@Conversation.method
async def get_history(ctx: Context, limit: int = None) -> List[Dict]:
    """Get conversation history, optionally limited to last N messages."""
    messages = ctx.get("messages", [])

    if limit:
        return messages[-limit:]
    return messages


@Conversation.method
async def get_context(ctx: Context) -> Dict:
    """Get conversation context and metadata."""
    return {
        "conversation_id": ctx.object_id,
        "turn_count": ctx.get("turn_count", 0),
        "message_count": len(ctx.get("messages", [])),
        "last_updated": ctx.get("last_updated"),
        "metadata": ctx.get("metadata", {})
    }


@Conversation.method
async def set_metadata(ctx: Context, key: str, value: any) -> None:
    """Set conversation metadata (user preferences, settings, etc.)."""
    metadata = ctx.get("metadata", {})
    metadata[key] = value
    ctx.set("metadata", metadata)
    ctx.logger.info(f"Metadata updated: {key} = {value}")


@Conversation.method
async def clear_history(ctx: Context) -> None:
    """Clear conversation history."""
    ctx.set("messages", [])
    ctx.set("turn_count", 0)
    ctx.set("last_updated", datetime.now().isoformat())
    ctx.logger.info("Conversation history cleared")


@Conversation.method
async def get_summary(ctx: Context) -> Dict:
    """Get a summary of the conversation."""
    messages = ctx.get("messages", [])

    user_messages = [m for m in messages if m["role"] == "user"]
    assistant_messages = [m for m in messages if m["role"] == "assistant"]

    return {
        "total_messages": len(messages),
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "turn_count": ctx.get("turn_count", 0),
        "metadata": ctx.get("metadata", {})
    }


async def simulate_conversation():
    """Simulate a multi-turn conversation."""
    print("=== Conversation/Session Pattern Example ===\n")

    # Create a conversation session
    session_id = "chat-12345"
    conv = Conversation(key=session_id)

    # Set conversation metadata
    await conv.set_metadata("user_id", "user-789")
    await conv.set_metadata("user_name", "Alice")
    await conv.set_metadata("language", "en")

    print("1. Starting conversation...\n")

    # Turn 1
    print("User: Hello! I need help with Python.")
    await conv.add_message(
        role="user",
        content="Hello! I need help with Python."
    )

    await conv.add_message(
        role="assistant",
        content="Hi Alice! I'd be happy to help you with Python. What specific topic are you interested in?"
    )
    print("Assistant: Hi Alice! I'd be happy to help you with Python. What specific topic are you interested in?\n")

    # Turn 2
    print("User: How do I work with async/await?")
    await conv.add_message(
        role="user",
        content="How do I work with async/await?"
    )

    await conv.add_message(
        role="assistant",
        content="Great question! async/await in Python allows you to write asynchronous code. Here's a basic example..."
    )
    print("Assistant: Great question! async/await in Python allows you to write asynchronous code. Here's a basic example...\n")

    # Turn 3
    print("User: Can you show me a practical example?")
    await conv.add_message(
        role="user",
        content="Can you show me a practical example?"
    )

    await conv.add_message(
        role="assistant",
        content="Absolutely! Here's a practical example of using async/await for making concurrent API calls..."
    )
    print("Assistant: Absolutely! Here's a practical example of using async/await for making concurrent API calls...\n")

    # Get conversation context
    print("2. Conversation Context:")
    context = await conv.get_context()
    print(f"   Session ID: {context['conversation_id']}")
    print(f"   Turn Count: {context['turn_count']}")
    print(f"   Message Count: {context['message_count']}")
    print(f"   User: {context['metadata'].get('user_name')}\n")

    # Get recent history
    print("3. Recent History (last 4 messages):")
    recent = await conv.get_history(limit=4)
    for i, msg in enumerate(recent):
        print(f"   [{msg['role']}]: {msg['content'][:60]}...")

    print()

    # Get summary
    print("4. Conversation Summary:")
    summary = await conv.get_summary()
    print(f"   Total messages: {summary['total_messages']}")
    print(f"   User messages: {summary['user_messages']}")
    print(f"   Assistant messages: {summary['assistant_messages']}")
    print(f"   Total turns: {summary['turn_count']}\n")


async def multi_session_example():
    """Demonstrate multiple independent conversation sessions."""
    print("=== Multiple Independent Sessions ===\n")

    # Create multiple conversation sessions
    conv1 = Conversation(key="chat-alice")
    conv2 = Conversation(key="chat-bob")

    # Set up first conversation
    await conv1.set_metadata("user_name", "Alice")
    await conv1.add_message("user", "I want to learn about decorators")
    await conv1.add_message("assistant", "Decorators are a powerful feature...")

    # Set up second conversation
    await conv2.set_metadata("user_name", "Bob")
    await conv2.add_message("user", "How do I use generators?")
    await conv2.add_message("assistant", "Generators are functions that yield values...")

    # Get summaries
    summary1 = await conv1.get_summary()
    summary2 = await conv2.get_summary()

    print("Alice's conversation:")
    print(f"   User: {summary1['metadata']['user_name']}")
    print(f"   Messages: {summary1['total_messages']}\n")

    print("Bob's conversation:")
    print(f"   User: {summary2['metadata']['user_name']}")
    print(f"   Messages: {summary2['total_messages']}\n")

    print("State is completely isolated between sessions!")


async def main():
    await simulate_conversation()
    print("\n" + "=" * 50 + "\n")
    await multi_session_example()


if __name__ == "__main__":
    asyncio.run(main())
