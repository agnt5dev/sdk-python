"""
Conversation entity using SessionEntity with auto-history management.

This demonstrates:
- SessionEntity base class with built-in history
- Automatic message tracking
- Auto-trimming when max_turns exceeded
- Multi-turn conversations with memory

Run as worker:
    uv run 19_session_entity_conversation.py

Test with client:
    uv run 19_session_entity_conversation.py --client
"""

import asyncio

from agnt5 import SessionEntity


class Conversation(SessionEntity):
    """
    Simple conversation entity with automatic history.
    History is trimmed to last 20 turns (40 messages).
    """

    max_turns: int = 10  # Keep last 10 turns (20 messages)
    auto_summarize: bool = False  # For now, just trim

    async def chat(self, message: str) -> dict:
        """
        Chat method - automatically manages history.

        Args:
            message: User's message

        Returns:
            dict with response and history info
        """
        # Add user message to history (automatic)
        await self.add_message("user", message)

        # Get conversation history
        history = await self.get_history()

        # Simple echo bot for now (in real app, call AI model with history)
        response = f"Echo: {message} (Turn {len(history) // 2})"

        # Add assistant response to history (automatic)
        await self.add_message("assistant", response)

        # Get updated history
        history = await self.get_history()

        return {
            "response": response,
            "turn": len(history) // 2,
            "history_length": len(history)
        }

    async def get_conversation_history(self, limit: int = 10) -> dict:
        """Get conversation history with formatting."""
        history = await self.get_history(limit=limit)

        return {
            "messages": history,
            "total_messages": len(history),
            "turns": len(history) // 2
        }

    async def reset_conversation(self) -> dict:
        """Reset the conversation history."""
        result = await self.clear_history()
        return {
            **result,
            "message": "Conversation reset successfully"
        }


class SmartConversation(SessionEntity):
    """
    Advanced conversation with summarization.
    Auto-summarizes when history gets too long.
    """

    max_turns: int = 5  # Keep last 5 turns only
    auto_summarize: bool = True  # Enable summarization

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
        history = await self.get_history()
        summary = await self.get_summary()

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
    """Test using HTTP client with entity() API (Restate-style)."""
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


async def test_with_session_api():
    """Test using HTTP client with session() API (OpenAI/ADK-style)."""
    print("\n=== Testing Conversation via Client (session API) ===\n")
    print("This demonstrates the OpenAI/ADK-style session API\n")

    from agnt5 import Client

    client = Client("http://localhost:34181")

    # Create session using session() API - cleaner for conversation use cases
    session = client.session("Conversation", "user-diana")

    print("Turn 1:")
    response = session.chat("Hello! Tell me about yourself.")
    print(f"  User: Hello! Tell me about yourself.")
    print(f"  Bot: {response}\n")

    print("Turn 2:")
    response = session.chat("What can you help me with?")
    print(f"  User: What can you help me with?")
    print(f"  Bot: {response}\n")

    print("Turn 3:")
    response = session.chat("Great, thanks!")
    print(f"  User: Great, thanks!")
    print(f"  Bot: {response}\n")

    # Get conversation history using session helper
    print("Conversation history:")
    history = session.get_history()
    for msg in history:
        print(f"  [{msg['role']}]: {msg['content']}")
    print()

    # Can also add messages directly
    print("Adding system message...")
    session.add_message("system", "You are a helpful AI assistant")
    print()

    # Clear history
    print("Clearing history...")
    session.clear_history()
    print("  ✅ History cleared\n")


async def main():
    """Main entry point."""
    import sys

    print("=" * 60)
    print("SessionEntity Conversation Example")
    print("=" * 60)

    if "--client" in sys.argv:
        # Test with both client APIs
        await test_with_client()
        await test_with_session_api()
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
