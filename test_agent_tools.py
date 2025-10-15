"""
Test Agent-as-Tool Example

Demonstrates using one agent as a tool for another agent (agents-as-tools pattern).
This example shows:
1. Specialist agents with specific skills
2. A coordinator agent that uses specialist agents as tools
3. How agents can delegate tasks to other agents
"""

import asyncio
import os

from agnt5 import Agent, Context, tool


# Define some basic tools for the specialist agents
@tool(auto_schema=True)
async def search_python_docs(ctx: Context, query: str) -> str:
    """Search Python documentation.

    Args:
        query: What to search for in Python docs
    """
    ctx.logger.info(f"Searching Python docs for: {query}")
    # Simulated search result
    return f"Python documentation for '{query}': Use the built-in function with proper syntax."


@tool(auto_schema=True)
async def analyze_code(ctx: Context, code: str) -> str:
    """Analyze code for issues.

    Args:
        code: The code to analyze
    """
    ctx.logger.info(f"Analyzing code: {code[:50]}...")
    # Simulated analysis
    return "Code analysis: No major issues found. Consider adding error handling."


@tool(auto_schema=True)
async def write_tests(ctx: Context, function_name: str) -> str:
    """Generate unit tests for a function.

    Args:
        function_name: Name of the function to test
    """
    ctx.logger.info(f"Writing tests for: {function_name}")
    # Simulated test generation
    return f"""
def test_{function_name}():
    result = {function_name}()
    assert result is not None
"""


async def test_agents_as_tools():
    """Test using agents as tools for other agents."""
    print("=== Testing Agents as Tools ===\n")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return

    print("✅ OpenAI API key found")

    try:
        # Create specialist agents with specific skills
        print("\n📝 Creating specialist agents...")

        # 1. Python expert agent
        python_expert = Agent(
            name="python_expert",
            model="openai/gpt-4o-mini",
            instructions="You are a Python programming expert. Help users with Python-related questions and provide accurate technical guidance.",
            tools=[search_python_docs],
            temperature=0.3,
            max_iterations=5,
        )
        print("  ✅ Python expert created")

        # 2. Code reviewer agent
        code_reviewer = Agent(
            name="code_reviewer",
            model="openai/gpt-4o-mini",
            instructions="You are a code review specialist. Analyze code for bugs, performance issues, and best practices.",
            tools=[analyze_code],
            temperature=0.3,
            max_iterations=5,
        )
        print("  ✅ Code reviewer created")

        # 3. Test writer agent
        test_writer = Agent(
            name="test_writer",
            model="openai/gpt-4o-mini",
            instructions="You are a testing specialist. Generate comprehensive unit tests for functions and classes.",
            tools=[write_tests],
            temperature=0.3,
            max_iterations=5,
        )
        print("  ✅ Test writer created")

        # 4. Coordinator agent that uses specialists as tools
        print("\n🎯 Creating coordinator agent with specialist agents as tools...")
        coordinator = Agent(
            name="coordinator",
            model="openai/gpt-4o-mini",
            instructions="""You are a software development coordinator. You have access to specialist agents:
            - python_expert: For Python programming questions and guidance
            - code_reviewer: For code analysis and review
            - test_writer: For generating unit tests

            Delegate tasks to the appropriate specialist agent based on the user's needs.""",
            tools=[python_expert, code_reviewer, test_writer],  # Agents as tools!
            temperature=0.5,
            max_iterations=10,
        )
        print("  ✅ Coordinator created with 3 specialist agents as tools")

        # Test 1: Ask a Python question (should use python_expert)
        print("\n" + "="*60)
        print("Test 1: Python Programming Question")
        print("="*60)
        print("User: How do I read a file in Python?")

        result = await coordinator.run("How do I read a file in Python?")
        print(f"\n📤 Coordinator Response:\n{result.output}")
        print(f"\n📊 Tools/Agents used: {len(result.tool_calls)}")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"  {i}. {tc['name']}")

        # Test 2: Ask for code review (should use code_reviewer)
        print("\n" + "="*60)
        print("Test 2: Code Review Request")
        print("="*60)
        print("User: Please review this code: def add(a, b): return a + b")

        result = await coordinator.run("Please review this code: def add(a, b): return a + b")
        print(f"\n📤 Coordinator Response:\n{result.output}")
        print(f"\n📊 Tools/Agents used: {len(result.tool_calls)}")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"  {i}. {tc['name']}")

        # Test 3: Ask for tests (should use test_writer)
        print("\n" + "="*60)
        print("Test 3: Test Generation Request")
        print("="*60)
        print("User: Write unit tests for a function called calculate_total")

        result = await coordinator.run("Write unit tests for a function called calculate_total")
        print(f"\n📤 Coordinator Response:\n{result.output}")
        print(f"\n📊 Tools/Agents used: {len(result.tool_calls)}")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"  {i}. {tc['name']}")

        # Test 4: Complex multi-step task (should use multiple agents)
        print("\n" + "="*60)
        print("Test 4: Multi-Step Task")
        print("="*60)
        print("User: I need help with Python file handling, then review my code, and write tests")

        result = await coordinator.run(
            "I need help with Python file handling. My code is: def read_file(path): f = open(path); return f.read(). "
            "Also write tests for the read_file function."
        )
        print(f"\n📤 Coordinator Response:\n{result.output}")
        print(f"\n📊 Tools/Agents used: {len(result.tool_calls)}")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"  {i}. {tc['name']}")

        print("\n" + "="*60)
        print("✅ All agent-as-tool tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_agents_as_tools())
