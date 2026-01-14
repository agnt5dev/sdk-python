"""
Example: Simple State API usage across run/session/user scopes

Demonstrates the new State API for coordination and control flow:
- Run-scoped state (ephemeral, cleared when run completes)
- Session-scoped state (multi-turn conversation state)
- User-scoped state (long-term user preferences)
"""

import asyncio
from agnt5 import Agent, AgentContext


async def main():
    """Demonstrate state API across different scopes."""

    # Create agent context with all scope identifiers
    ctx = AgentContext(
        run_id="run-123",
        agent_name="demo-agent",
        session_id="session-abc",
        user_id="user-456",
    )

    print("=" * 60)
    print("State API Demo - Run/Session/User Scopes")
    print("=" * 60)

    # =================================================================
    # Run-scoped state (ephemeral, cleared when run completes)
    # =================================================================
    print("\n1. Run-scoped State (Coordination)")
    print("-" * 60)

    # Set run state for coordination
    await ctx.state.set("phase", "research")
    await ctx.state.set("agents_completed", ["researcher"])
    await ctx.state.set("iteration_count", 1)

    # Retrieve run state
    phase = await ctx.state.get("phase")
    agents = await ctx.state.get("agents_completed")
    count = await ctx.state.get("iteration_count", 0)

    print(f"Current phase: {phase}")
    print(f"Agents completed: {agents}")
    print(f"Iteration count: {count}")

    # List all run state keys
    keys = await ctx.state.keys()
    print(f"All run state keys: {keys}")

    # =================================================================
    # Session-scoped state (multi-turn conversation)
    # =================================================================
    print("\n2. Session-scoped State (Conversation)")
    print("-" * 60)

    # Set session state for conversation context
    await ctx.session.state.set("context", {
        "topic": "travel",
        "destination": "Japan",
        "budget": 5000
    })
    await ctx.session.state.set("turn_count", 3)

    # Retrieve session state
    context = await ctx.session.state.get("context")
    turns = await ctx.session.state.get("turn_count")

    print(f"Conversation context: {context}")
    print(f"Turn count: {turns}")

    # List all session state keys
    session_keys = await ctx.session.state.keys()
    print(f"All session state keys: {session_keys}")

    # =================================================================
    # User-scoped state (long-term preferences)
    # =================================================================
    print("\n3. User-scoped State (Preferences)")
    print("-" * 60)

    # Set user state for preferences
    if ctx.user:
        await ctx.user.state.set("theme", "dark")
        await ctx.user.state.set("language", "en")
        await ctx.user.state.set("notifications", True)

        # Retrieve user state
        theme = await ctx.user.state.get("theme")
        language = await ctx.user.state.get("language")
        notifications = await ctx.user.state.get("notifications")

        print(f"User theme: {theme}")
        print(f"User language: {language}")
        print(f"Notifications enabled: {notifications}")

        # List all user state keys
        user_keys = await ctx.user.state.keys()
        print(f"All user state keys: {user_keys}")
    else:
        print("No user_id provided - user state not available")

    # =================================================================
    # State Operations
    # =================================================================
    print("\n4. State Operations")
    print("-" * 60)

    # Delete a key
    await ctx.state.delete("iteration_count")
    count_after = await ctx.state.get("iteration_count")
    print(f"After delete: iteration_count = {count_after}")

    # Clear all state (run scope only for demo)
    print("\nClearing run state...")
    await ctx.state.clear()
    keys_after_clear = await ctx.state.keys()
    print(f"Keys after clear: {keys_after_clear}")

    print("\n" + "=" * 60)
    print("State API Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
