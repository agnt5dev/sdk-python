"""
Example: Chatflow - Multi-Turn Conversations

This example demonstrates:
- Using @chatflow decorator for conversation workflows
- Managing conversation state across multiple turns
- Tracking message history
- Turn counting and conversation metrics
- Integration with AI agents for chat responses

The @chatflow decorator is identical to @workflow but adds metadata {"chat": "true"}
to indicate this workflow is designed for multi-turn conversation scenarios.
"""

import asyncio
from typing import Dict, List

from agnt5 import WorkflowContext, chatflow, function, with_entity_context


# Define a function to generate AI responses (simulated)
@function
async def generate_ai_response(ctx, messages: List[Dict], user_message: str) -> str:
    """Generate an AI response based on conversation history."""
    ctx.logger.info(f"Generating response for: {user_message[:50]}...")

    # Simulate AI response generation
    # In a real scenario, this would call an LLM API
    await asyncio.sleep(0.1)

    # Simple rule-based responses for demo
    if "hello" in user_message.lower():
        return "Hello! How can I assist you today?"
    elif "weather" in user_message.lower():
        return "I don't have access to real-time weather data, but I can help you with other questions!"
    elif "help" in user_message.lower():
        return "I'm here to help! You can ask me questions and I'll do my best to assist you."
    elif len(messages) > 4:
        return f"We've had {len(messages) // 2} exchanges so far. Is there anything else I can help you with?"
    else:
        return f"Thanks for your message: '{user_message}'. How else can I help you?"


# Basic chatflow example
@chatflow
async def customer_support_chat(ctx: WorkflowContext, message: str) -> Dict:
    """
    Simple customer support chatflow.

    Manages conversation state and generates responses for each user message.
    """
    ctx.logger.info(f"Processing chat message: {message[:50]}...")

    # Initialize conversation state if needed
    if not ctx.state.get("messages"):
        ctx.state.set("messages", [])
        ctx.state.set("session_start", "2024-01-01T00:00:00Z")  # In real app, use actual timestamp

    # Get current messages
    messages = ctx.state.get("messages")

    # Add user message
    messages.append({"role": "user", "content": message})
    ctx.state.set("messages", messages)

    # Generate AI response
    response = await ctx.task(generate_ai_response, messages=messages, user_message=message)

    # Add assistant response
    messages.append({"role": "assistant", "content": response})
    ctx.state.set("messages", messages)

    # Calculate conversation metrics
    turn_count = len(messages) // 2

    return {
        "response": response,
        "turn_count": turn_count,
        "total_messages": len(messages)
    }


# Advanced chatflow with context management
@chatflow
async def contextual_chat(ctx: WorkflowContext, message: str, user_id: str) -> Dict:
    """
    Advanced chatflow with user context and conversation management.

    Demonstrates:
    - User-specific conversation state
    - Conversation context tracking
    - Metadata management
    """
    ctx.logger.info(f"Processing message from user {user_id}: {message[:30]}...")

    # Initialize state
    if not ctx.state.get("initialized"):
        ctx.state.set("initialized", True)
        ctx.state.set("user_id", user_id)
        ctx.state.set("messages", [])
        ctx.state.set("context", {
            "topics": [],
            "sentiment": "neutral",
            "language": "en"
        })

    # Get current state
    messages = ctx.state.get("messages")
    context = ctx.state.get("context")

    # Add user message
    messages.append({"role": "user", "content": message})

    # Extract topics (simple keyword extraction for demo)
    if "weather" in message.lower():
        if "weather" not in context["topics"]:
            context["topics"].append("weather")
    elif "help" in message.lower():
        if "support" not in context["topics"]:
            context["topics"].append("support")

    # Update context
    ctx.state.set("context", context)
    ctx.state.set("messages", messages)

    # Generate response
    response = await ctx.task(generate_ai_response, messages=messages, user_message=message)

    # Add assistant response
    messages.append({"role": "assistant", "content": response})
    ctx.state.set("messages", messages)

    return {
        "response": response,
        "turn_count": len(messages) // 2,
        "topics_discussed": context["topics"],
        "user_id": user_id
    }


# Multi-session chatflow
@chatflow(name="support_session")
async def support_session_chat(ctx: WorkflowContext, message: str, session_id: str) -> Dict:
    """
    Chatflow with session management.

    Demonstrates how to maintain conversation state across multiple sessions
    using session IDs for proper state isolation.
    """
    ctx.logger.info(f"Session {session_id}: {message[:30]}...")

    # Initialize session state
    if not ctx.state.get(f"session_{session_id}_initialized"):
        ctx.state.set(f"session_{session_id}_initialized", True)
        ctx.state.set(f"session_{session_id}_messages", [])
        ctx.state.set(f"session_{session_id}_metadata", {
            "created_at": "2024-01-01T00:00:00Z",
            "last_active": "2024-01-01T00:00:00Z",
            "message_count": 0
        })
        ctx.logger.info(f"Initialized new session: {session_id}")

    # Get session-specific state
    messages = ctx.state.get(f"session_{session_id}_messages")
    metadata = ctx.state.get(f"session_{session_id}_metadata")

    # Add user message
    messages.append({"role": "user", "content": message})
    metadata["message_count"] += 1
    metadata["last_active"] = "2024-01-01T00:00:01Z"  # In real app, use actual timestamp

    ctx.state.set(f"session_{session_id}_messages", messages)
    ctx.state.set(f"session_{session_id}_metadata", metadata)

    # Generate response
    response = await ctx.task(generate_ai_response, messages=messages, user_message=message)

    # Add assistant response
    messages.append({"role": "assistant", "content": response})
    metadata["message_count"] += 1

    ctx.state.set(f"session_{session_id}_messages", messages)
    ctx.state.set(f"session_{session_id}_metadata", metadata)

    return {
        "response": response,
        "session_id": session_id,
        "turn_count": len(messages) // 2,
        "total_messages": metadata["message_count"],
        "last_active": metadata["last_active"]
    }


@with_entity_context
async def main():
    print("=== Chatflow Examples ===\n")

    # Example 1: Basic customer support chat
    print("1. Customer Support Chat:\n")

    result1 = await customer_support_chat(message="Hello, I need help")
    print(f"   User: Hello, I need help")
    print(f"   Assistant: {result1['response']}")
    print(f"   Turn: {result1['turn_count']}\n")

    result2 = await customer_support_chat(message="What's the weather like?")
    print(f"   User: What's the weather like?")
    print(f"   Assistant: {result2['response']}")
    print(f"   Turn: {result2['turn_count']}\n")

    result3 = await customer_support_chat(message="Thanks for your help!")
    print(f"   User: Thanks for your help!")
    print(f"   Assistant: {result3['response']}")
    print(f"   Turn: {result3['turn_count']}")
    print(f"   Total messages: {result3['total_messages']}\n")

    # Example 2: Contextual chat
    print("2. Contextual Chat:\n")

    result = await contextual_chat(
        message="I need help with the weather forecast",
        user_id="user-123"
    )
    print(f"   User: I need help with the weather forecast")
    print(f"   Assistant: {result['response']}")
    print(f"   Topics discussed: {result['topics_discussed']}")
    print(f"   Turn: {result['turn_count']}\n")

    # Example 3: Multi-session chat
    print("3. Multi-Session Support Chat:\n")

    # Session 1
    result_s1 = await support_session_chat(
        message="Hello!",
        session_id="session-001"
    )
    print(f"   [Session 001] User: Hello!")
    print(f"   [Session 001] Assistant: {result_s1['response']}")
    print(f"   [Session 001] Messages: {result_s1['total_messages']}\n")

    # Session 2 (different session, separate state)
    result_s2 = await support_session_chat(
        message="Hi there!",
        session_id="session-002"
    )
    print(f"   [Session 002] User: Hi there!")
    print(f"   [Session 002] Assistant: {result_s2['response']}")
    print(f"   [Session 002] Messages: {result_s2['total_messages']}\n")

    # Continue session 1
    result_s1_2 = await support_session_chat(
        message="Can you help me?",
        session_id="session-001"
    )
    print(f"   [Session 001] User: Can you help me?")
    print(f"   [Session 001] Assistant: {result_s1_2['response']}")
    print(f"   [Session 001] Messages: {result_s1_2['total_messages']}\n")

    print("=== Chatflow Metadata ===\n")

    # Verify that chatflow has the correct metadata
    import agnt5
    config = agnt5.WorkflowRegistry.get("customer_support_chat")
    if config:
        print(f"   Workflow: {config.name}")
        print(f"   Chat metadata: {config.metadata.get('chat', 'not set')}")
        print(f"   Description: {config.metadata.get('description', 'N/A')}")

    print("\n✅ All chatflow examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
