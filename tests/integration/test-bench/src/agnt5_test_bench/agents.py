"""
Reusable Agent Components for Integration Testing

This module contains common, reusable agents that can be used across multiple
test workflows. Test-specific agents should be defined inline in their workflow files.

Canonical Agents:
- chat_agent: Chatbot for session testing
- research_specialist: Research with search and analysis tools
- researcher_agent: Researcher with web search tool
- analyst_agent: Analyst with data analysis tool
"""

from agnt5 import Agent, agent
# Import tools from tools module (avoid circular imports)
from agnt5_test_bench.tools import search_web, domain_specific_analysis


@agent
def chat_agent():
    """Chat agent for session testing.

    This agent is used to verify:
    - Agent sessions are created automatically
    - session_id is returned in response metadata
    - Conversation history is maintained across multiple turns
    - Agent remembers context from previous messages
    """
    return Agent(
        name="chat_agent",
        model="openai/gpt-4o-mini",
        instructions="You are a helpful chatbot. Remember what the user tells you and maintain conversation context.",
        temperature=0.7,
    )


# ============================================================================
# CANONICAL AGENTS - Reusable across workflows
# ============================================================================

# Research specialist with search and analysis capabilities
research_specialist = Agent(
    name="ResearchSpecialist",
    model="openai/gpt-4o-mini",
    instructions="""You are a research specialist. You can:
    - Search the web for information
    - Perform data analysis
    Be thorough and provide detailed insights.""",
    tools=[search_web, domain_specific_analysis],
    temperature=0.7,
    max_iterations=5,
)

# Researcher agent for gathering information
researcher_agent = Agent(
    name="Researcher",
    model="openai/gpt-4o-mini",
    instructions="Research topics and gather information using web search.",
    tools=[search_web],
    temperature=0.7,
    max_iterations=3,
)

# Analyst agent for data analysis
analyst_agent = Agent(
    name="Analyst",
    model="openai/gpt-4o-mini",
    instructions="Analyze data and provide insights using analysis tools.",
    tools=[domain_specific_analysis],
    temperature=0.7,
    max_iterations=3,
)


__all__ = [
    "chat_agent",
    "research_specialist",
    "researcher_agent",
    "analyst_agent",
]
