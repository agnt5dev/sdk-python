"""Global agent registry for lookups."""

import logging
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Agent

logger = logging.getLogger(__name__)

# Global registry for agents
_AGENT_REGISTRY: Dict[str, "Agent"] = {}


class AgentRegistry:
    """Registry for looking up agents by name.

    This provides a global registry where agents can be registered
    and looked up by name. Useful for multi-agent systems where
    agents need to discover each other.

    Example:
        ```python
        # Register agent
        agent = Agent(name="researcher", ...)
        AgentRegistry.register(agent)

        # Look up agent
        found = AgentRegistry.get("researcher")
        ```
    """

    @staticmethod
    def register(agent: "Agent") -> None:
        """Register an agent.

        Args:
            agent: Agent to register
        """
        if agent.name in _AGENT_REGISTRY:
            logger.warning(f"Overwriting existing agent '{agent.name}'")
        _AGENT_REGISTRY[agent.name] = agent

    @staticmethod
    def get(name: str) -> Optional["Agent"]:
        """Get agent by name.

        Args:
            name: Name of agent to look up

        Returns:
            Agent if found, None otherwise
        """
        return _AGENT_REGISTRY.get(name)

    @staticmethod
    def all() -> Dict[str, "Agent"]:
        """Get all registered agents.

        Returns:
            Copy of the agent registry
        """
        return _AGENT_REGISTRY.copy()

    @staticmethod
    def clear() -> None:
        """Clear all registered agents."""
        _AGENT_REGISTRY.clear()
