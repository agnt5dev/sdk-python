"""Agent handoff support for multi-agent systems.

Handoffs enable one agent to delegate control to another specialized agent,
following the pattern popularized by LangGraph and OpenAI Agents SDK.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Agent


class Handoff:
    """Configuration for agent-to-agent handoff.

    The handoff is exposed to the LLM as a tool named 'transfer_to_{agent_name}'
    that allows explicit delegation with conversation history.

    Example:
        ```python
        specialist = Agent(name="specialist", ...)

        # Simple: Pass agent directly (auto-wrapped with defaults)
        coordinator = Agent(
            name="coordinator",
            handoffs=[specialist]  # Agent auto-converted to Handoff
        )

        # Advanced: Use Handoff for custom configuration
        coordinator = Agent(
            name="coordinator",
            handoffs=[
                Handoff(
                    agent=specialist,
                    description="Custom description for LLM",
                    tool_name="custom_transfer_name",
                    pass_full_history=False
                )
            ]
        )
        ```
    """

    def __init__(
        self,
        agent: "Agent",
        description: Optional[str] = None,
        tool_name: Optional[str] = None,
        pass_full_history: bool = True,
    ):
        """Initialize handoff configuration.

        Args:
            agent: Target agent to hand off to
            description: Description shown to LLM (defaults to agent instructions)
            tool_name: Custom tool name (defaults to 'transfer_to_{agent_name}')
            pass_full_history: Whether to pass full conversation history to target agent
        """
        self.agent = agent
        self.description = description or agent.instructions or f"Transfer to {agent.name}"
        self.tool_name = tool_name or f"transfer_to_{agent.name}"
        self.pass_full_history = pass_full_history


def handoff(
    agent: "Agent",
    description: Optional[str] = None,
    tool_name: Optional[str] = None,
    pass_full_history: bool = True,
) -> Handoff:
    """Create a handoff configuration for agent-to-agent delegation.

    This is a convenience function for creating Handoff instances with a clean API.

    Args:
        agent: Target agent to hand off to
        description: Description shown to LLM
        tool_name: Custom tool name
        pass_full_history: Whether to pass full conversation history

    Returns:
        Handoff configuration

    Example:
        ```python
        from agnt5 import Agent, handoff

        research_agent = Agent(name="researcher", ...)
        writer_agent = Agent(name="writer", ...)

        coordinator = Agent(
            name="coordinator",
            handoffs=[
                handoff(research_agent, "Transfer for research tasks"),
                handoff(writer_agent, "Transfer for writing tasks"),
            ]
        )
        ```
    """
    return Handoff(
        agent=agent,
        description=description,
        tool_name=tool_name,
        pass_full_history=pass_full_history,
    )
