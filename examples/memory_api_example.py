"""
Example: Memory API usage - State vs Memory distinction

Demonstrates the difference between State API and Memory API:
- State API (ctx.state): For coordination, control flow, and temporary data
- Memory API (ctx.memory): For domain knowledge, facts, and semantic search

This example shows:
1. ConversationMemory: Session-based message history for multi-turn conversations
2. SemanticMemory: Vector-based semantic search for facts and knowledge
3. State API: Simple key-value coordination data

Requirements:
- OPENAI_API_KEY for embeddings
- One of: QDRANT_URL, PINECONE_API_KEY+PINECONE_HOST, or POSTGRES_URL for vectors
"""

import asyncio
import os
from agnt5 import (
    Agent,
    AgentContext,
    ConversationMemory,
    SemanticMemory,
    MemoryScope,
)


async def demo_conversation_memory():
    """Demonstrate ConversationMemory for multi-turn chat history."""
    print("=" * 60)
    print("ConversationMemory Demo - Session Message History")
    print("=" * 60)

    session_id = "demo-session-123"
    conversation = ConversationMemory(session_id)

    # Simulate a multi-turn conversation
    await conversation.add("user", "Hello! I'm planning a trip to Japan.")
    await conversation.add(
        "assistant",
        "That's wonderful! Japan is a beautiful country. When are you planning to go?"
    )
    await conversation.add("user", "Probably in April for the cherry blossoms.")
    await conversation.add(
        "assistant",
        "Perfect timing! April is cherry blossom season. Would you like recommendations for cities to visit?"
    )
    await conversation.add("user", "Yes, especially Kyoto and Tokyo.")

    # Retrieve conversation history
    messages = await conversation.get_messages(limit=10)

    print(f"\nConversation has {len(messages)} messages:")
    for msg in messages:
        print(f"  [{msg.role}]: {msg.content[:60]}...")

    # Convert to LM messages for agent input
    lm_messages = await conversation.get_as_lm_messages()
    print(f"\nConverted to {len(lm_messages)} LM messages for agent context")

    print()


async def demo_semantic_memory():
    """Demonstrate SemanticMemory for vector-based semantic search."""
    print("=" * 60)
    print("SemanticMemory Demo - Semantic Search Over Facts")
    print("=" * 60)

    # Check if vector database is configured
    if not (os.getenv("QDRANT_URL") or os.getenv("PINECONE_API_KEY") or os.getenv("POSTGRES_URL")):
        print("\n⚠️  Skipping SemanticMemory demo - no vector database configured")
        print("   Set QDRANT_URL, PINECONE_API_KEY, or POSTGRES_URL to enable")
        print()
        return

    # Create user-scoped memory
    user_id = "user-456"
    memory = SemanticMemory(MemoryScope.USER, user_id)

    print("\n1. Storing facts about user preferences...")

    # Store various facts
    facts = [
        "User prefers dark mode in applications",
        "User's favorite programming language is Python",
        "User is interested in AI and machine learning",
        "User prefers concise documentation over verbose explanations",
        "User lives in San Francisco",
        "User enjoys hiking and outdoor activities",
    ]

    fact_ids = []
    for fact in facts:
        fact_id = await memory.store(fact)
        fact_ids.append(fact_id)
        print(f"   ✓ Stored: {fact[:50]}... (id: {fact_id[:8]}...)")

    print("\n2. Semantic search for related facts...")

    # Search for UI preferences
    query = "What are the user's UI and design preferences?"
    results = await memory.search(query, limit=3)

    print(f"\n   Query: '{query}'")
    print("   Results:")
    for i, result in enumerate(results, 1):
        print(f"     {i}. [score: {result.score:.3f}] {result.content}")

    # Search for technical interests
    query = "What technologies does the user like?"
    results = await memory.search(query, limit=3)

    print(f"\n   Query: '{query}'")
    print("   Results:")
    for i, result in enumerate(results, 1):
        print(f"     {i}. [score: {result.score:.3f}] {result.content}")

    # Search for personal info
    query = "Where does the user live and what hobbies do they have?"
    results = await memory.search(query, limit=3)

    print(f"\n   Query: '{query}'")
    print("   Results:")
    for i, result in enumerate(results, 1):
        print(f"     {i}. [score: {result.score:.3f}] {result.content}")

    print("\n3. Forgetting specific facts...")
    # Delete the first fact
    deleted = await memory.forget(fact_ids[0])
    print(f"   ✓ Deleted fact: {facts[0][:50]}... (success: {deleted})")

    print()


async def demo_agent_context_memory():
    """Demonstrate memory access through AgentContext."""
    print("=" * 60)
    print("AgentContext Memory Integration")
    print("=" * 60)

    # Create agent context
    ctx = AgentContext(
        run_id="run-789",
        agent_name="demo-agent",
        session_id="session-abc",
        user_id="user-456",
    )

    print("\n1. State API - For coordination and control flow")
    print("   Use ctx.state for temporary coordination data:")

    # Set some coordination state
    await ctx.state.set("phase", "research")
    await ctx.state.set("iteration_count", 3)
    await ctx.state.set("agents_completed", ["researcher", "analyst"])

    phase = await ctx.state.get("phase")
    count = await ctx.state.get("iteration_count")
    agents = await ctx.state.get("agents_completed")

    print(f"   - Current phase: {phase}")
    print(f"   - Iteration count: {count}")
    print(f"   - Agents completed: {agents}")

    print("\n2. Memory API - For domain knowledge and facts")
    print("   Use ctx.memory for semantic search over facts:")

    # Check if vector database is configured
    if not (os.getenv("QDRANT_URL") or os.getenv("PINECONE_API_KEY") or os.getenv("POSTGRES_URL")):
        print("\n   ⚠️  Skipping memory demo - no vector database configured")
    else:
        # ctx.memory is automatically scoped to user if user_id provided
        await ctx.memory.store("User asked about memory architecture")
        await ctx.memory.store("User wants to understand State vs Memory distinction")

        # Search for what user asked about
        results = await ctx.memory.search("What did the user ask?", limit=2)
        print("\n   Recent queries:")
        for result in results:
            print(f"   - {result.content}")

    print("\n3. Key Difference:")
    print("   - State API: Simple KV store for coordination (phase, counters, flags)")
    print("   - Memory API: Vector search for domain knowledge (facts, preferences)")

    print()


async def demo_state_vs_memory_comparison():
    """Clear comparison of when to use State vs Memory."""
    print("=" * 60)
    print("When to Use State API vs Memory API")
    print("=" * 60)

    print("\n✓ Use STATE API (ctx.state) for:")
    print("  - Workflow phase tracking (phase: 'research', 'analysis', 'synthesis')")
    print("  - Iteration counters (attempt_count, retry_count)")
    print("  - Agent coordination flags (needs_approval, is_paused)")
    print("  - Simple configuration (max_retries, timeout_seconds)")
    print("  - Temporary execution state (cleared when run completes)")

    print("\n✓ Use MEMORY API (ctx.memory) for:")
    print("  - User preferences and settings (prefers_dark_mode, language)")
    print("  - Domain facts and knowledge (user_interests, past_interactions)")
    print("  - Semantic search queries (find similar user questions)")
    print("  - Long-term user profile data (skills, background, context)")
    print("  - RAG knowledge bases (documents, FAQs, knowledge articles)")

    print("\n✗ DON'T use State API for:")
    print("  - Long-term user data (use Memory API)")
    print("  - Semantic search (use Memory API)")
    print("  - Complex domain entities (use Entity API)")

    print("\n✗ DON'T use Memory API for:")
    print("  - Workflow coordination (use State API)")
    print("  - Simple counters and flags (use State API)")
    print("  - Temporary execution state (use State API)")

    print()


async def main():
    """Run all memory API demos."""
    print("\n")
    print("=" * 60)
    print("AGNT5 Memory API Examples")
    print("=" * 60)
    print()

    # Run demos
    await demo_conversation_memory()
    await demo_semantic_memory()
    await demo_agent_context_memory()
    await demo_state_vs_memory_comparison()

    print("=" * 60)
    print("Memory API Demo Complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
