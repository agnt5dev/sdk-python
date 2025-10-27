"""
Example: Multi-Agent Systems with Handoffs and Agents-as-Tools

This example demonstrates two patterns for multi-agent collaboration:

1. **Agents-as-Tools**: One agent uses another agent like a regular tool
   - The coordinator agent decides when to invoke specialist agents
   - Results are returned back to the coordinator for further processing

2. **Handoffs**: Explicit transfer of control from one agent to another
   - Uses `transfer_to_{agent_name}` tools for delegation
   - Target agent takes over completely and returns the final result
   - Conversation history and context are passed to the target agent

Prerequisites:
    Set environment variables:
    - OPENAI_API_KEY=your-key
    - ANTHROPIC_API_KEY=your-key (optional)
"""

import asyncio
import os
from typing import Dict, List

from agnt5 import Agent, Context, handoff, tool


# Define some regular tools for demonstration
@tool(auto_schema=True)
async def search_database(ctx: Context, query: str) -> List[Dict]:
    """Search internal database for information.

    Args:
        query: Search query
    """
    ctx.logger.info(f"Searching database for: {query}")

    # Simulate database search
    results = [
        {"id": 1, "title": f"Document about {query}", "relevance": 0.95},
        {"id": 2, "title": f"Related info on {query}", "relevance": 0.82},
    ]

    return results


@tool(auto_schema=True)
async def calculate_metrics(ctx: Context, data: List[float]) -> Dict:
    """Calculate statistical metrics on data.

    Args:
        data: List of numeric values
    """
    if not data:
        return {"error": "No data provided"}

    return {
        "count": len(data),
        "sum": sum(data),
        "average": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
    }


async def example_agents_as_tools():
    """Example 1: Using agents as tools (implicit invocation)."""
    print("=== Example 1: Agents as Tools ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Create specialist agents
    research_agent = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="""You are a research specialist.
        When given a topic, search for information and provide a comprehensive summary.""",
        tools=[search_database],
        temperature=0.7,
    )

    analyst_agent = Agent(
        name="analyst",
        model="openai/gpt-4o-mini",
        instructions="""You are a data analyst.
        Analyze data and provide insights with statistical metrics.""",
        tools=[calculate_metrics],
        temperature=0.7,
    )

    # Create coordinator that uses specialist agents as tools
    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="""You are a coordinator agent. You can delegate tasks to:
        - researcher: For research and information gathering
        - analyst: For data analysis and statistics

        Use these specialists when appropriate, then synthesize their results.""",
        tools=[research_agent, analyst_agent],  # Pass agents directly as tools
        temperature=0.7,
    )

    # Run a task that requires multiple specialists
    result = await coordinator.run(
        """I need you to:
        1. Research information about 'machine learning trends'
        2. Analyze this dataset: [85, 92, 78, 90, 88, 95]
        3. Provide a summary combining both insights"""
    )

    print(f"Coordinator output:\n{result.output}\n")
    print(f"Tool calls made: {len(result.tool_calls)}")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"  {i}. {tc['name']}")
    print()


async def example_explicit_handoffs():
    """Example 2: Explicit handoffs (transfer of control)."""
    print("=== Example 2: Explicit Handoffs ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Create specialist agents for different domains
    technical_specialist = Agent(
        name="technical_specialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a technical specialist for software engineering questions.
        Provide detailed technical answers with code examples when appropriate.""",
        temperature=0.7,
    )

    business_specialist = Agent(
        name="business_specialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a business specialist for business and strategy questions.
        Provide business-focused advice with practical recommendations.""",
        temperature=0.7,
    )

    writing_specialist = Agent(
        name="writing_specialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a writing specialist for content creation.
        Help craft well-written documents, emails, and content.""",
        temperature=0.7,
    )

    # Create triage agent that routes to specialists via handoffs
    # Two ways to configure handoffs:

    # 1. Simple: Pass agents directly (auto-wrapped with defaults)
    triage_agent = Agent(
        name="triage",
        model="openai/gpt-4o-mini",
        instructions="""You are a triage agent that routes requests to specialists.

        Available specialists:
        - technical_specialist: For technical/engineering questions
        - business_specialist: For business/strategy questions
        - writing_specialist: For writing/content creation

        Analyze the user's request and transfer to the appropriate specialist using
        the transfer_to_{specialist_name} tools. Only handle simple general questions yourself.""",
        handoffs=[
            technical_specialist,  # Simple: Just pass the agent directly!
            business_specialist,   # AGNT5 auto-wraps with sensible defaults
            writing_specialist,
        ],
        temperature=0.7,
    )

    # 2. Advanced: Use handoff() for custom descriptions (optional)
    # Uncomment to try this alternative approach:
    # triage_agent = Agent(
    #     name="triage",
    #     model="openai/gpt-4o-mini",
    #     instructions="...",
    #     handoffs=[
    #         handoff(technical_specialist, "Transfer to technical specialist"),
    #         handoff(business_specialist, "Transfer to business specialist"),
    #         handoff(writing_specialist, "Transfer to writing specialist"),
    #     ],
    #     temperature=0.7,
    # )

    # Test different types of requests
    test_cases = [
        "How do I implement a binary search tree in Python?",
        "What's a good go-to-market strategy for a B2B SaaS product?",
        "Help me write a professional email declining a job offer",
    ]

    for i, user_request in enumerate(test_cases, 1):
        print(f"Test {i}: {user_request}")
        result = await triage_agent.run(user_request)

        if result.handoff_to:
            print(f"✓ Handed off to: {result.handoff_to}")
            print(f"Specialist response:\n{result.output[:200]}...\n")
        else:
            print(f"✓ Handled by triage agent")
            print(f"Response:\n{result.output}\n")

        print()


async def example_handoff_with_context():
    """Example 3: Handoffs with conversation history and shared state."""
    print("=== Example 3: Handoffs with Context Sharing ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Tool to save user preferences
    @tool(auto_schema=True)
    async def save_preference(ctx: Context, key: str, value: str) -> str:
        """Save a user preference.

        Args:
            key: Preference key
            value: Preference value
        """
        prefs = ctx.get("user_preferences", {})
        prefs[key] = value
        ctx.set("user_preferences", prefs)
        return f"Saved preference: {key}={value}"

    # Front-desk agent that collects info
    front_desk = Agent(
        name="front_desk",
        model="openai/gpt-4o-mini",
        instructions="""You are a front desk agent that greets users and collects their preferences.
        Ask for their name and preferred communication style, then transfer to the specialist.""",
        tools=[save_preference],
        temperature=0.7,
    )

    # Specialist that uses the collected context
    specialist = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="""You are a specialist agent. You receive requests with context.
        Check for user preferences in the context and personalize your response accordingly.""",
        temperature=0.7,
    )

    # Add handoff from front desk to specialist
    front_desk_with_handoff = Agent(
        name="front_desk",
        model="openai/gpt-4o-mini",
        instructions="""You are a front desk agent. Collect the user's name and save it,
        then transfer them to the specialist for help.""",
        tools=[save_preference],
        handoffs=[
            handoff(
                specialist,
                "Transfer to specialist after collecting info",
                pass_full_history=True,
            )
        ],
        temperature=0.7,
    )

    # Create shared context
    ctx = Context(run_id="handoff-demo")

    # Simulate multi-turn interaction
    print("User: Hi, my name is Alice and I prefer concise answers")
    result = await front_desk_with_handoff.run(
        "Hi, my name is Alice and I prefer concise answers. "
        "Please help me understand what cloud computing is.",
        context=ctx,
    )

    print(f"\nAgent: {result.output}\n")

    # Check if handoff occurred
    if result.handoff_to:
        print(f"✓ Request was handed off to: {result.handoff_to}")
        print(f"✓ User preferences saved: {ctx.get('user_preferences', {})}")

    print()


async def example_complex_workflow():
    """Example 4: Complex multi-agent workflow combining both patterns."""
    print("=== Example 4: Complex Workflow (Both Patterns) ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipped: OPENAI_API_KEY not set\n")
        return

    # Research agent
    researcher = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="Research topics and gather information.",
        tools=[search_database],
        temperature=0.7,
    )

    # Analysis agent
    analyst = Agent(
        name="analyst",
        model="openai/gpt-4o-mini",
        instructions="Analyze data and provide insights.",
        tools=[calculate_metrics],
        temperature=0.7,
    )

    # Report writer (uses other agents as tools)
    report_writer = Agent(
        name="report_writer",
        model="openai/gpt-4o-mini",
        instructions="""You are a report writer. You can:
        - Use the researcher agent to gather information
        - Use the analyst agent to analyze data
        - Synthesize findings into a well-structured report""",
        tools=[researcher, analyst],  # Use agents as tools
        temperature=0.7,
    )

    # Executive agent (can handoff to report writer)
    executive = Agent(
        name="executive",
        model="openai/gpt-4o-mini",
        instructions="""You are an executive assistant. For simple questions, answer directly.
        For complex requests requiring reports, transfer to the report writer.""",
        handoffs=[handoff(report_writer, "Transfer to report writer for detailed reports")],
        temperature=0.7,
    )

    # Complex request
    result = await executive.run(
        """I need a comprehensive report on AI adoption trends including:
        1. Research findings on AI adoption
        2. Analysis of adoption metrics: [25, 45, 62, 78, 85]
        3. Executive summary"""
    )

    print(f"Result:\n{result.output}\n")

    if result.handoff_to:
        print(f"✓ Handoff occurred to: {result.handoff_to}")
        print(f"✓ Total tool calls in workflow: {len(result.tool_calls)}")

    print()


async def main():
    print("=== AGNT5 Multi-Agent Systems Example ===\n")
    print("Demonstrating agents-as-tools and explicit handoffs\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("Set OPENAI_API_KEY to run examples\n")
        return

    try:
        await example_agents_as_tools()
        await example_explicit_handoffs()
        await example_handoff_with_context()
        await example_complex_workflow()

        print("=== All Examples Complete ===")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
