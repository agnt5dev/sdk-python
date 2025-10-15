"""
Test Agent Tools Example

A simple example to test agent tools functionality with OpenAI.
"""

import asyncio
import os
from datetime import datetime

from agnt5 import Agent, Context, tool, workflow


@tool(auto_schema=True)
async def get_time(ctx: Context) -> str:
    """Get the current time."""
    ctx.logger.info("Getting current time")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool(auto_schema=True)
async def add_numbers(ctx: Context, a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    ctx.logger.info(f"Adding {a} + {b}")
    return a + b


@tool(auto_schema=True)
async def greet_user(ctx: Context, name: str) -> str:
    """Greet a user by name.
    
    Args:
        name: Name of the user to greet
    """
    ctx.logger.info(f"Greeting user: {name}")
    return f"Hello, {name}! Nice to meet you."


@workflow
async def test_agent_tools(ctx: Context):
    """Test agent with simple tools."""
    print("=== Testing Agent Tools ===\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    print("✅ OpenAI API key found")
    
    try:
        # Create agent with tools
        agent = Agent(
            name="test_agent",
            model="openai/gpt-4o-mini",
            instructions="You are a helpful assistant. Use the available tools to help users. Strictly use one of the provided tools",
            tools=[get_time, add_numbers, greet_user],
            temperature=0.7,
            max_iterations=3,
        )
        
        print("✅ Agent created successfully")
        
        # Test 1: Simple greeting
        print("\n--- Test 1: Greeting ---")
        result = await agent.run("Please greet me. My name is Alice.", context=ctx)
        print(f"Response: {result.output}")
        print(f"Tools used: {len(result.tool_calls)}")
        
        # Test 2: Math calculation
        print("\n--- Test 2: Math ---")
        result = await agent.run("What is 15 + 27?", context=ctx)
        print(f"Response: {result.output}")
        print(f"Tools used: {len(result.tool_calls)}")
        
        # Test 3: Time query
        print("\n--- Test 3: Time ---")
        result = await agent.run("What time is it?", context=ctx)
        print(f"Response: {result.output}")
        print(f"Tools used: {len(result.tool_calls)}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_agent_tools())
