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

    # Streaming execution
    async for event in agent.stream("Find recent AI papers"):
        if event.event_type == "lm.content_block.delta":
            print(event.content, end="")

    # Non-streaming execution (recommended)
    result = await agent.run("Find recent AI papers")
    print(result.output)
    ```
"""

# Import from split modules
from .agents_md import discover_agents_md, load_agents_md
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
from .skill_events import SkillLoaded
from .skills import Skill

__all__ = [
    # Core classes
    "Agent",
    "AgentContext",
    "AgentResult",
    "Skill",
    "SkillLoaded",
    # AGENTS.md guidance
    "discover_agents_md",
    "load_agents_md",
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
