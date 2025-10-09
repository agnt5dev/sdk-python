"""
Conversation entity using Entity with session management pattern.

This demonstrates:
- Implementing session/conversation pattern with Entity
- Managing conversation history manually
- Auto-trimming when max messages exceeded
- Multi-turn conversations with memory

Note: SessionEntity was removed from the SDK core. This example shows
how to implement the session pattern using the base Entity class.

Run as worker:
    uv run 19_session_entity_conversation.py

Test with client:
    uv run 19_session_entity_conversation.py --client
"""

import asyncio
from typing import Optional

from agnt5 import Entity


class Conversation(Entity):
    """
    Conversation entity with automatic history management.

    Implements the session pattern using Entity base class.
    History is trimmed to last max_messages (default: 20 messages = 10 turns).
    """

    # Configuration (can be overridden in subclasses)
    max_messages: int = 20  # Keep last 20 messages (10 turns)

    async def chat(self, message: str) -> dict:
        """
        Chat method - automatically manages history.

        Args:
            message: User's message

        Returns:
            dict with response and history info
        """
        # Add user message to history
        await self.add_message("user", message)

        # Get conversation history
        history = self.state.get("messages", [])

        # Simple echo bot for now (in real app, call AI model with history)
        response = f"Echo: {message} (Turn {len(history) // 2})"

        # Add assistant response to history
        await self.add_message("assistant", response)

        # Get updated history
        history = self.state.get("messages", [])

        return {
            "response": response,
            "turn": len(history) // 2,
            "history_length": len(history)
        }

    async def add_message(self, role: str, content: str) -> dict:
        """Add a message to conversation history with auto-trimming."""
        messages = self.state.get("messages", [])

        # Add new message
        messages.append({"role": role, "content": content})

        # Auto-trim if exceeds max_messages
        if len(messages) > self.max_messages:
            # Keep only the last max_messages
            messages = messages[-self.max_messages:]

        self.state.set("messages", messages)

        return {
            "message_index": len(messages) - 1,
            "total_messages": len(messages)
        }

    async def get_history(self, limit: Optional[int] = None) -> list:
        """Get conversation history with optional limit."""
        messages = self.state.get("messages", [])

        if limit:
            return messages[-limit:]
        return messages

    async def get_conversation_history(self, limit: int = 10) -> dict:
        """Get conversation history with formatting."""
        history = await self.get_history(limit=limit)

        return {
            "messages": history,
            "total_messages": len(history),
            "turns": len(history) // 2
        }

    async def clear_history(self) -> dict:
        """Clear conversation history."""
        messages = self.state.get("messages", [])
        message_count = len(messages)

        self.state.set("messages", [])

        return {
            "message_count": message_count,
            "cleared": True
        }

    async def reset_conversation(self) -> dict:
        """Reset the conversation history."""
        result = await self.clear_history()
        return {
            **result,
            "message": "Conversation reset successfully"
        }


class SmartConversation(Entity):
    """
    Advanced conversation with summarization support.

    This example shows how you might implement summarization
    when history gets too long.
    """

    max_messages: int = 10  # Keep last 5 turns only

    async def chat(self, message: str) -> dict:
        """
        Chat with automatic summarization.

        Args:
            message: User's message

        Returns:
            dict with response and summary info
        """
        # Add user message
        await self.add_message("user", message)

        # Get history and summary
        history = self.state.get("messages", [])
        summary = self.state.get("summary")

        # Simple response
        response = f"Message received: '{message}'. I remember {len(history)} messages."

        if summary:
            response += f"\nSummary of earlier conversation: {summary}"

        # Add response
        await self.add_message("assistant", response)

        return {
            "response": response,
            "history_length": len(history),
            "has_summary": summary is not None,
            "summary": summary
        }

    async def add_message(self, role: str, content: str) -> dict:
        """Add message with summarization when history exceeds limit."""
        messages = self.state.get("messages", [])

        # Add new message
        messages.append({"role": role, "content": content})

        # If exceeds limit, create summary and trim
        if len(messages) > self.max_messages:
            # In a real implementation, you'd use an LLM to summarize
            old_messages = messages[:-self.max_messages]
            summary = f"Discussed {len(old_messages)} earlier messages"

            # Store summary and keep only recent messages
            self.state.set("summary", summary)
            messages = messages[-self.max_messages:]

        self.state.set("messages", messages)

        return {
            "message_index": len(messages) - 1,
            "total_messages": len(messages)
        }

    async def get_history(self, limit: Optional[int] = None) -> list:
        """Get conversation history."""
        messages = self.state.get("messages", [])

        if limit:
            return messages[-limit:]
        return messages

    async def get_summary(self) -> Optional[str]:
        """Get conversation summary."""
        return self.state.get("summary")


async def test_conversation():
    """Test basic conversation entity."""
    print("\n=== Testing Conversation Entity ===\n")

    conv = Conversation(key="user-alice")

    # Multi-turn conversation
    print("Turn 1:")
    result = await conv.chat("Hello!")
    print(f"  User: Hello!")
    print(f"  Bot: {result['response']}")
    print(f"  History: {result['history_length']} messages\n")

    print("Turn 2:")
    result = await conv.chat("How are you?")
    print(f"  User: How are you?")
    print(f"  Bot: {result['response']}")
    print(f"  History: {result['history_length']} messages\n")

    print("Turn 3:")
    result = await conv.chat("Tell me a joke")
    print(f"  User: Tell me a joke")
    print(f"  Bot: {result['response']}")
    print(f"  History: {result['history_length']} messages\n")

    # Get full history
    print("Full conversation history:")
    history_result = await conv.get_conversation_history()
    for msg in history_result["messages"]:
        print(f"  [{msg['role']}]: {msg['content']}")
    print()

    # Reset
    print("Resetting conversation...")
    reset_result = await conv.reset_conversation()
    print(f"  {reset_result['message']}")
    print(f"  Cleared {reset_result['message_count']} messages\n")


async def test_smart_conversation():
    """Test conversation with auto-summarization."""
    print("\n=== Testing SmartConversation (with auto-summarization) ===\n")

    conv = SmartConversation(key="user-bob")

    # Have many turns to trigger summarization
    messages = [
        "Hi there!",
        "What's the weather?",
        "Tell me about Python",
        "How do I use async?",
        "What is AGNT5?",
        "Explain workflows",
        "Tell me about entities",
        "How does state work?",
        "What about durability?",
        "Can you help me?",
        "This should trigger summarization",
    ]

    for i, message in enumerate(messages, 1):
        result = await conv.chat(message)
        print(f"Turn {i}:")
        print(f"  User: {message}")
        print(f"  Bot: {result['response'][:100]}...")
        print(f"  History: {result['history_length']} messages")
        if result['has_summary']:
            print(f"  📝 Summary exists!")
        print()


async def test_with_client():
    """Test using HTTP client with entity() API."""
    print("\n=== Testing Conversation via Client (entity API) ===\n")

    from agnt5 import Client

    client = Client("http://localhost:34181")

    # Create conversation proxy using entity() API
    conv = client.entity("Conversation", "user-charlie")

    print("Turn 1:")
    result = conv.chat(message="Hello from client!")
    print(f"  Response: {result}")
    print()

    print("Turn 2:")
    result = conv.chat(message="How are you doing?")
    print(f"  Response: {result}")
    print()

    print("Get history:")
    history = conv.get_conversation_history(limit=10)
    print(f"  Total messages: {history['total_messages']}")
    print(f"  Turns: {history['turns']}")
    for msg in history['messages']:
        print(f"    [{msg['role']}]: {msg['content']}")


async def main():
    """Main entry point."""
    import sys

    print("=" * 60)
    print("Session Entity Conversation Example")
    print("=" * 60)

    if "--client" in sys.argv:
        # Test with client
        await test_with_client()
    elif "--test" in sys.argv:
        # Run local tests
        print("\nRunning local tests...")
        await test_conversation()
        await test_smart_conversation()
        print("\n✅ Local tests complete!")
    else:
        # Start as worker
        from agnt5 import Worker
        from agnt5.entity import EntityRegistry

        print("\n🚀 Starting Worker...")
        print("Entities registered:")
        for entity_name in EntityRegistry.all().keys():
            print(f"  - {entity_name}")
        print("\nWaiting for client requests...")
        print()

        worker = Worker(service_name="default-service")
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
