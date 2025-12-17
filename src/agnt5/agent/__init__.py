"""Agent module - AI agents with streaming execution.

This module provides the core agent primitives for building AI-powered
applications with tool orchestration and multi-agent collaboration.

Example:
    ```python
    from agnt5.agent import Agent, AgentResult, handoff

    # Create an agent
    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        instructions="You are a research assistant.",
    )

    # Streaming execution (recommended)
    async for event in agent.run("Find recent AI papers"):
        if event.event_type == EventType.LM_MESSAGE_DELTA:
            print(event.data, end="")  # data is raw content string for deltas

    # Non-streaming execution
    result = await agent.run_sync("Find recent AI papers")
    print(result.output)
    ```
"""

# Import from split modules
from .context import AgentContext
from .result import AgentResult
from .handoff import Handoff, handoff
from .registry import AgentRegistry
from .core import Agent
from .decorator import agent

__all__ = [
    # Core classes
    "Agent",
    "AgentContext",
    "AgentResult",
    # Handoff support
    "Handoff",
    "handoff",
    # Registry
    "AgentRegistry",
    # Decorator
    "agent",
]
