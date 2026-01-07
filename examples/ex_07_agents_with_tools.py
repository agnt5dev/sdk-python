"""
Example: AGNT5 Agents with Tools

This example demonstrates agents with tool capabilities:
- Defining tools with @tool decorator
- Agents that can use tools
- Multi-agent patterns (agents-as-tools, handoffs)

Tools allow agents to:
- Perform calculations
- Access external data
- Execute specific operations
- Delegate to other agents

Each component can be:
- Run standalone: python ex_07_agents_with_tools.py
- Registered with platform: imported by app.py
- Invoked via client: client.run("calculator_agent", {"question": "What is 15 * 23?"})

Prerequisites:
- Set OPENAI_API_KEY for OpenAI models
"""

import asyncio
import os
from typing import List, Dict

from agnt5 import function, FunctionContext, Agent, Context, tool, handoff


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


@tool(auto_schema=True)
async def calculate(ctx: Context, expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2", "15 * 23")

    Returns:
        Result of the calculation as a string
    """
    ctx.logger.info(f"[calculate] Evaluating: {expression}")
    try:
        # Simple evaluation (in production, use a safer parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@tool(auto_schema=True)
async def get_weather(ctx: Context, city: str) -> str:
    """
    Get current weather for a city (simulated).

    Args:
        city: Name of the city

    Returns:
        Weather information for the city
    """
    ctx.logger.info(f"[get_weather] Getting weather for: {city}")
    # Simulated weather data
    weather_data = {
        "new york": {"temp": 72, "condition": "Sunny", "humidity": 45},
        "london": {"temp": 58, "condition": "Cloudy", "humidity": 75},
        "tokyo": {"temp": 68, "condition": "Partly cloudy", "humidity": 60},
        "paris": {"temp": 65, "condition": "Rainy", "humidity": 80},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        w = weather_data[city_lower]
        return f"Weather in {city}: {w['temp']}F, {w['condition']}, Humidity: {w['humidity']}%"
    return f"Weather data not available for {city}. Try: New York, London, Tokyo, or Paris."


@tool(auto_schema=True)
async def search_database(ctx: Context, query: str, limit: int = 5) -> List[Dict]:
    """
    Search a database for information (simulated).

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        List of matching records
    """
    ctx.logger.info(f"[search_database] Query: {query}, Limit: {limit}")
    # Simulated database results
    results = [
        {"id": 1, "title": f"Result for '{query}' #1", "relevance": 0.95},
        {"id": 2, "title": f"Result for '{query}' #2", "relevance": 0.87},
        {"id": 3, "title": f"Result for '{query}' #3", "relevance": 0.75},
    ]
    return results[:limit]


@tool(auto_schema=True)
async def send_email(ctx: Context, to: str, subject: str, body: str) -> dict:
    """
    Send an email (simulated).

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content

    Returns:
        Status of email sending operation
    """
    ctx.logger.info(f"[send_email] To: {to}, Subject: {subject}")
    # Simulated email sending
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "message_id": f"MSG-{hash(to + subject) % 10000:04d}",
    }


# =============================================================================
# AGENTS WITH TOOLS
# =============================================================================


@function
async def calculator_agent(ctx: FunctionContext, question: str) -> dict:
    """
    Agent that can perform calculations using tools.

    Args:
        question: Math question to answer

    Returns:
        Dict with agent response

    Example:
        result = client.run("calculator_agent", {
            "question": "What is 15 multiplied by 23, then add 100?"
        })
    """
    ctx.logger.info(f"[calculator_agent] Question: {question[:50]}...")

    agent = Agent(
        name="calculator",
        model="openai/gpt-4o-mini",
        instructions="""You are a helpful math assistant.
Use the calculate tool to perform mathematical operations.
Break down complex calculations into steps.
Always show your work.""",
        tools=[calculate],
        temperature=0.3,
        max_iterations=10,
    )

    result = await agent.run_sync(question)

    return {
        "question": question,
        "response": result.output,
        "tool_calls": len(result.tool_calls),
    }


@function
async def weather_assistant(ctx: FunctionContext, query: str) -> dict:
    """
    Agent that can check weather using tools.

    Args:
        query: Weather-related question

    Returns:
        Dict with weather information

    Example:
        result = client.run("weather_assistant", {
            "query": "Compare the weather in New York and London"
        })
    """
    ctx.logger.info(f"[weather_assistant] Query: {query[:50]}...")

    agent = Agent(
        name="weather_assistant",
        model="openai/gpt-4o-mini",
        instructions="""You are a weather assistant.
Use the get_weather tool to fetch current weather information.
Provide helpful summaries and comparisons when asked.""",
        tools=[get_weather],
        temperature=0.5,
        max_iterations=10,
    )

    result = await agent.run_sync(query)

    return {
        "query": query,
        "response": result.output,
        "tool_calls": len(result.tool_calls),
    }


@function
async def multi_tool_agent(ctx: FunctionContext, task: str) -> dict:
    """
    Agent with multiple tools for complex tasks.

    Args:
        task: Task description

    Returns:
        Dict with task results

    Example:
        result = client.run("multi_tool_agent", {
            "task": "Search for project updates, calculate the budget (1500 * 12), and draft an email to team@example.com summarizing findings"
        })
    """
    ctx.logger.info(f"[multi_tool_agent] Task: {task[:50]}...")

    agent = Agent(
        name="assistant",
        model="openai/gpt-4o-mini",
        instructions="""You are a capable assistant with access to multiple tools.
Use the appropriate tools to complete complex tasks:
- calculate: For math operations
- search_database: For finding information
- send_email: For sending communications

Plan your approach, use tools as needed, and summarize results.""",
        tools=[calculate, search_database, send_email],
        temperature=0.5,
        max_iterations=15,
    )

    result = await agent.run_sync(task)

    return {
        "task": task,
        "response": result.output,
        "tool_calls": len(result.tool_calls),
        "tools_used": [tc.get("name", "unknown") for tc in result.tool_calls],
    }


# =============================================================================
# MULTI-AGENT PATTERNS
# =============================================================================


@function
async def research_coordinator(ctx: FunctionContext, topic: str) -> dict:
    """
    Coordinator agent that delegates to specialist agents.

    Demonstrates agents-as-tools pattern where one agent
    can invoke other agents as tools.

    Args:
        topic: Research topic

    Returns:
        Dict with coordinated research results

    Example:
        result = client.run("research_coordinator", {
            "topic": "AI trends in 2025"
        })
    """
    ctx.logger.info(f"[research_coordinator] Topic: {topic[:50]}...")

    # Create specialist agents
    researcher = Agent(
        name="researcher",
        model="openai/gpt-4o-mini",
        instructions="""You are a research specialist.
When given a topic, provide detailed information and key findings.
Be thorough but concise.""",
        tools=[search_database],
        temperature=0.5,
        max_iterations=5,
    )

    analyst = Agent(
        name="analyst",
        model="openai/gpt-4o-mini",
        instructions="""You are a data analyst.
Analyze information and provide insights, patterns, and recommendations.
Use clear structure in your analysis.""",
        temperature=0.5,
        max_iterations=5,
    )

    # Coordinator uses specialists as tools
    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="""You are a research coordinator.
Delegate tasks to your team:
- Use the researcher for gathering information
- Use the analyst for analyzing findings
Synthesize their work into a cohesive response.""",
        tools=[researcher, analyst],  # Agents as tools!
        temperature=0.5,
        max_iterations=10,
    )

    result = await coordinator.run_sync(f"Research and analyze: {topic}")

    return {
        "topic": topic,
        "response": result.output,
        "tool_calls": len(result.tool_calls),
    }


@function
async def support_router(ctx: FunctionContext, query: str) -> dict:
    """
    Support agent that routes to specialists via handoffs.

    Demonstrates explicit handoff pattern where control
    is transferred to a specialist agent.

    Args:
        query: Customer support query

    Returns:
        Dict with support response

    Example:
        result = client.run("support_router", {
            "query": "How do I reset my password?"
        })
    """
    ctx.logger.info(f"[support_router] Query: {query[:50]}...")

    # Create specialist agents
    technical_support = Agent(
        name="technical_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a technical support specialist.
Help with technical issues like:
- Password resets
- Account access
- Software problems
- Technical troubleshooting
Provide step-by-step solutions.""",
        temperature=0.5,
        max_iterations=5,
    )

    billing_support = Agent(
        name="billing_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a billing support specialist.
Help with billing issues like:
- Invoice questions
- Payment problems
- Subscription changes
- Refund requests
Be clear about policies and procedures.""",
        temperature=0.5,
        max_iterations=5,
    )

    general_support = Agent(
        name="general_support",
        model="openai/gpt-4o-mini",
        instructions="""You are a general support specialist.
Help with general questions about:
- Product features
- How to use the service
- General inquiries
Be friendly and helpful.""",
        temperature=0.5,
        max_iterations=5,
    )

    # Router with handoffs
    router = Agent(
        name="support_router",
        model="openai/gpt-4o-mini",
        instructions="""You are a support router.
Analyze the customer's query and transfer to the appropriate specialist:
- Technical issues (passwords, access, bugs) -> technical_support
- Billing issues (payments, invoices, subscriptions) -> billing_support
- General questions (features, how-to) -> general_support

Transfer immediately without trying to answer yourself.""",
        handoffs=[
            handoff(technical_support, "Transfer to technical support for tech issues"),
            handoff(billing_support, "Transfer to billing support for payment issues"),
            handoff(general_support, "Transfer to general support for other questions"),
        ],
        temperature=0.3,
        max_iterations=5,
    )

    result = await router.run_sync(query)

    return {
        "query": query,
        "response": result.output,
        "handoff_to": result.handoff_to,
        "routed": result.handoff_to is not None,
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================


async def main() -> None:
    """Run examples standalone (without platform)."""
    import logging
    from agnt5 import Context

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Agent calls will fail.")
        print("   Set it with: export OPENAI_API_KEY=sk-...")
        return

    print("=" * 60)
    print("AGNT5 Agents with Tools Example")
    print("=" * 60)

    ctx = Context(
        run_id="tools-example",
        correlation_id="tools-example",
        parent_correlation_id="",
    )

    # Calculator agent
    print("\n--- Calculator Agent ---")
    result = await calculator_agent(ctx, "What is 15 * 23 + 100?")
    print(f"Response: {result['response'][:200]}...")
    print(f"Tool calls: {result['tool_calls']}")

    # Weather assistant
    print("\n--- Weather Assistant ---")
    result = await weather_assistant(ctx, "What's the weather in Tokyo?")
    print(f"Response: {result['response'][:200]}...")

    # Multi-tool agent
    print("\n--- Multi-Tool Agent ---")
    result = await multi_tool_agent(ctx, "Calculate 500 * 3 and search for 'quarterly report'")
    print(f"Response: {result['response'][:300]}...")
    print(f"Tools used: {result['tools_used']}")

    # Research coordinator (agents as tools)
    print("\n--- Research Coordinator ---")
    result = await research_coordinator(ctx, "Machine learning applications")
    print(f"Response: {result['response'][:300]}...")

    # Support router (handoffs)
    print("\n--- Support Router ---")
    result = await support_router(ctx, "How do I reset my password?")
    print(f"Response: {result['response'][:200]}...")
    print(f"Routed to: {result['handoff_to']}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
