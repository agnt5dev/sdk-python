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
        if event.event_type == "lm.content_block.delta":
            print(event.content, end="")

    # Non-streaming execution
    result = await agent.run_sync("Find recent AI papers")
    print(result.output)
    ```
"""

# Import from split modules
from .context import AgentContext
from .core import Agent
from .decorator import agent
from .events import (
    AgentCompleted,
    AgentFailed,
    AgentIterationCompleted,
    AgentIterationStarted,
    AgentStarted,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from .handoff import Handoff, handoff
from .registry import AgentRegistry
from .result import AgentResult

__all__ = [
    # Core classes
    "Agent",
    "AgentContext",
    "AgentResult",
    # Events
    "AgentCompleted",
    "AgentFailed",
    "AgentIterationCompleted",
    "AgentIterationStarted",
    "AgentStarted",
    "ToolCallCompleted",
    "ToolCallFailed",
    "ToolCallStarted",
    # Handoff support
    "Handoff",
    "handoff",
    # Registry
    "AgentRegistry",
    # Decorator
    "agent",
]
