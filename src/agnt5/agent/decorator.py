"""Agent decorator for registering agents."""

from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional

from ..lm import LanguageModel

from .core import Agent
from .registry import AgentRegistry


def agent(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    model: Optional[LanguageModel] = None,
    instructions: Optional[str] = None,
    tools: Optional[List[Any]] = None,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_iterations: int = 10,
) -> Callable:
    """
    Decorator to register a function as an agent and automatically register it.

    This decorator allows you to define agents as functions that create and return Agent instances.
    The agent will be automatically registered in the AgentRegistry for discovery by the worker.

    Args:
        name: Agent name (defaults to function name)
        model: Language model instance (required if not provided in function)
        instructions: System instructions (required if not provided in function)
        tools: List of tools available to the agent
        model_name: Model name to use
        temperature: LLM temperature
        max_iterations: Maximum reasoning iterations

    Returns:
        Decorated function that returns an Agent instance

    Example:
        ```python
        from agnt5 import agent, tool
        from agnt5.lm import OpenAILanguageModel

        @agent(
            name="research_agent",
            model=OpenAILanguageModel(),
            instructions="You are a research assistant.",
            tools=[search_web, analyze_data]
        )
        def create_researcher():
            # Agent is created and registered automatically
            pass

        # Or create agent directly
        @agent
        def my_agent():
            from agnt5.lm import OpenAILanguageModel
            return Agent(
                name="my_agent",
                model=OpenAILanguageModel(),
                instructions="You are a helpful assistant."
            )
        ```
    """

    def decorator(func: Callable) -> Callable:
        # Determine agent name
        agent_name = name or func.__name__

        # Create the agent
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Agent:
            # Check if function returns an Agent
            result = func(*args, **kwargs)
            if isinstance(result, Agent):
                # Function creates its own agent
                agent_instance = result
            elif model is not None and instructions is not None:
                # Create agent from decorator parameters
                agent_instance = Agent(
                    name=agent_name,
                    model=model,
                    instructions=instructions,
                    tools=tools,
                    model_name=model_name,
                    temperature=temperature,
                    max_iterations=max_iterations,
                )
            else:
                raise ValueError(
                    f"Agent decorator for '{agent_name}' requires either "
                    "the decorated function to return an Agent instance, "
                    "or 'model' and 'instructions' parameters to be provided"
                )

            # Register agent
            AgentRegistry.register(agent_instance)
            return agent_instance

        # Create agent immediately and store reference
        agent_instance = wrapper()

        # Return the agent instance itself (so it can be used directly)
        return agent_instance

    if _func is None:
        return decorator
    return decorator(_func)
