"""Agent component implementation for AGNT5 SDK.

Phase 1: Simple agent with external LLM integration and tool orchestration.
Phase 2: Platform-backed agents with durable execution and multi-agent coordination.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .context import Context
from . import lm
from .lm import GenerateRequest, GenerateResponse, Message, ModelConfig, ToolDefinition
from .tool import Tool, ToolRegistry
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Global agent registry
_AGENT_REGISTRY: Dict[str, "Agent"] = {}


class AgentRegistry:
    """Registry for agents."""

    @staticmethod
    def register(agent: "Agent") -> None:
        """Register an agent."""
        if agent.name in _AGENT_REGISTRY:
            logger.warning(f"Overwriting existing agent '{agent.name}'")
        _AGENT_REGISTRY[agent.name] = agent
        logger.debug(f"Registered agent '{agent.name}'")

    @staticmethod
    def get(name: str) -> Optional["Agent"]:
        """Get agent by name."""
        return _AGENT_REGISTRY.get(name)

    @staticmethod
    def all() -> Dict[str, "Agent"]:
        """Get all registered agents."""
        return _AGENT_REGISTRY.copy()

    @staticmethod
    def clear() -> None:
        """Clear all registered agents."""
        _AGENT_REGISTRY.clear()
        logger.debug("Cleared agent registry")


class AgentResult:
    """Result from agent execution."""

    def __init__(self, output: str, tool_calls: List[Dict[str, Any]], context: Context):
        self.output = output
        self.tool_calls = tool_calls
        self.context = context


class Agent:
    """Autonomous LLM-driven agent with tool orchestration.

    Phase 1: Simple agent with:
    - LLM integration (OpenAI, Anthropic, etc.)
    - Tool selection and execution
    - Multi-turn reasoning
    - Context and state management

    Phase 2 will add:
    - Durable execution with checkpointing
    - Multi-agent coordination
    - Platform-backed tool execution
    - Streaming responses

    Example:
        ```python
        from agnt5 import Agent, tool, Context

        @tool(auto_schema=True)
        async def search_web(ctx: Context, query: str) -> List[Dict]:
            # Search implementation
            return [{"title": "Result", "url": "..."}]

        # Simple usage with model string
        agent = Agent(
            name="researcher",
            model="openai/gpt-4o-mini",
            instructions="You are a research assistant.",
            tools=[search_web],
            temperature=0.7
        )

        result = await agent.run("What are the latest AI trends?")
        print(result.output)
        ```
    """

    def __init__(
        self,
        name: str,
        model: str,
        instructions: str,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model_config: Optional[ModelConfig] = None,
        max_iterations: int = 10,
    ):
        """Initialize agent.

        Args:
            name: Agent name/identifier
            model: Model string with provider prefix (e.g., "openai/gpt-4o-mini")
            instructions: System instructions for the agent
            tools: List of tools available to the agent (functions with @tool decorator)
            temperature: LLM temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            model_config: Optional advanced configuration (custom endpoints, headers, etc.)
            max_iterations: Maximum reasoning iterations
        """
        self.name = name
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.model_config = model_config
        self.max_iterations = max_iterations

        # Build tool registry
        self.tools: Dict[str, Tool] = {}
        if tools:
            for tool_item in tools:
                # Check if it's a Tool instance
                if isinstance(tool_item, Tool):
                    self.tools[tool_item.name] = tool_item
                # Check if it's a decorated function with config
                elif hasattr(tool_item, "_agnt5_config"):
                    # Try to get from ToolRegistry first
                    tool_config = tool_item._agnt5_config
                    tool_instance = ToolRegistry.get(tool_config.name)
                    if tool_instance:
                        self.tools[tool_instance.name] = tool_instance
                # Otherwise try to look up by function name
                elif callable(tool_item):
                    # Try to find in registry by function name
                    tool_name = tool_item.__name__
                    tool_instance = ToolRegistry.get(tool_name)
                    if tool_instance:
                        self.tools[tool_instance.name] = tool_instance

        self.logger = logging.getLogger(f"agnt5.agent.{name}")

        # Define schemas based on the run method signature
        # Input: user_message (string)
        self.input_schema = {
            "type": "object",
            "properties": {
                "user_message": {"type": "string"}
            },
            "required": ["user_message"]
        }
        # Output: AgentResult with output and tool_calls
        self.output_schema = {
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        }

        # Store metadata
        self.metadata = {
            "description": instructions,
            "model": model_name
        }

    async def run(
        self,
        user_message: str,
        context: Optional[Context] = None,
    ) -> AgentResult:
        """Run agent to completion.

        Args:
            user_message: User's input message
            context: Optional context (auto-created if not provided)

        Returns:
            AgentResult with output and execution details

        Example:
            ```python
            result = await agent.run("Analyze recent tech news")
            print(result.output)
            ```
        """
        # Create context if not provided
        if context is None:
            import uuid

            context = Context(
                run_id=f"agent-{self.name}-{uuid.uuid4().hex[:8]}",
                component_type="agent",
            )

        # Initialize conversation
        messages: List[Message] = [Message.user(user_message)]
        all_tool_calls: List[Dict[str, Any]] = []

        # Reasoning loop
        for iteration in range(self.max_iterations):
            self.logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            # Build tool definitions for LLM
            tool_defs = [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                )
                for tool in self.tools.values()
            ]

            # Convert messages to dict format for lm.generate()
            messages_dict = []
            for msg in messages:
                messages_dict.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

            # Call LLM using simplified API
            # TODO: Support tools in lm.generate() - for now using GenerateRequest internally
            request = GenerateRequest(
                model=self.model,
                system_prompt=self.instructions,
                messages=messages,
                tools=tool_defs if tool_defs else [],
            )
            request.config.temperature = self.temperature
            if self.max_tokens:
                request.config.max_tokens = self.max_tokens
            if self.top_p:
                request.config.top_p = self.top_p

            # Create internal LM instance for generation
            # TODO: Use model_config when provided
            from .lm import _LanguageModel
            provider, model_name = self.model.split('/', 1)
            internal_lm = _LanguageModel(provider=provider.lower(), default_model=None)
            response = await internal_lm.generate(request)

            # Add assistant response to messages
            messages.append(Message.assistant(response.text))

            # Check if LLM wants to use tools
            if response.tool_calls:
                self.logger.info(f"Agent calling {len(response.tool_calls)} tool(s)")

                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args_str = tool_call["arguments"]

                    # Track tool call
                    all_tool_calls.append(
                        {
                            "name": tool_name,
                            "arguments": tool_args_str,
                            "iteration": iteration + 1,
                        }
                    )

                    # Execute tool
                    try:
                        # Parse arguments
                        tool_args = json.loads(tool_args_str)

                        # Get tool
                        tool = self.tools.get(tool_name)
                        if not tool:
                            result_text = f"Error: Tool '{tool_name}' not found"
                        else:
                            # Execute tool
                            result = await tool.invoke(context, **tool_args)
                            result_text = json.dumps(result) if result else "null"

                        tool_results.append(
                            {"tool": tool_name, "result": result_text, "error": None}
                        )

                    except Exception as e:
                        self.logger.error(f"Tool execution error: {e}")
                        tool_results.append(
                            {"tool": tool_name, "result": None, "error": str(e)}
                        )

                # Add tool results to conversation
                results_text = "\n".join(
                    [
                        f"Tool: {tr['tool']}\nResult: {tr['result']}"
                        if tr["error"] is None
                        else f"Tool: {tr['tool']}\nError: {tr['error']}"
                        for tr in tool_results
                    ]
                )
                messages.append(Message.user(f"Tool results:\n{results_text}"))

                # Continue loop for agent to process results

            else:
                # No tool calls - agent is done
                self.logger.info(f"Agent completed after {iteration + 1} iterations")
                return AgentResult(
                    output=response.text,
                    tool_calls=all_tool_calls,
                    context=context,
                )

        # Max iterations reached
        self.logger.warning(f"Agent reached max iterations ({self.max_iterations})")
        final_output = messages[-1].content if messages else "No output generated"
        return AgentResult(
            output=final_output,
            tool_calls=all_tool_calls,
            context=context,
        )

    async def chat(
        self,
        user_message: str,
        messages: List[Message],
        context: Optional[Context] = None,
    ) -> tuple[str, List[Message]]:
        """Continue multi-turn conversation.

        Args:
            user_message: New user message
            messages: Previous conversation messages
            context: Optional context

        Returns:
            Tuple of (assistant_response, updated_messages)

        Example:
            ```python
            messages = []
            response, messages = await agent.chat("Hello", messages)
            response, messages = await agent.chat("Tell me more", messages)
            ```
        """
        if context is None:
            import uuid

            context = Context(
                run_id=f"agent-chat-{self.name}-{uuid.uuid4().hex[:8]}",
                component_type="agent",
            )

        # Add user message
        conversation = messages + [Message.user(user_message)]

        # Build request (no tools for simple chat)
        request = GenerateRequest(
            model=self.model,
            system_prompt=self.instructions,
            messages=conversation,
        )
        request.config.temperature = self.temperature
        if self.max_tokens:
            request.config.max_tokens = self.max_tokens
        if self.top_p:
            request.config.top_p = self.top_p

        # Create internal LM instance for generation
        from .lm import _LanguageModel
        provider, model_name = self.model.split('/', 1)
        internal_lm = _LanguageModel(provider=provider.lower(), default_model=None)
        response = await internal_lm.generate(request)

        # Add assistant response
        conversation.append(Message.assistant(response.text))

        return response.text, conversation


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
