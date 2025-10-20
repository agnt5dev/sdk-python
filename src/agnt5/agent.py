"""Agent component implementation for AGNT5 SDK.

Provides simple agent with external LLM integration and tool orchestration.
Future: Platform-backed agents with durable execution and multi-agent coordination.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .context import Context
from . import lm
from .lm import GenerateRequest, GenerateResponse, LanguageModel, Message, ModelConfig, ToolDefinition
from .tool import Tool, ToolRegistry
from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

# Global agent registry
_AGENT_REGISTRY: Dict[str, "Agent"] = {}


class AgentContext(Context):
    """
    Context for agent execution with conversation state management.

    Extends base Context with:
    - State management via EntityStateManager
    - Conversation history persistence
    - Context inheritance (child agents share parent's state)

    Three initialization modes:
    1. Standalone: Creates own state manager (playground testing)
    2. Inherit WorkflowContext: Shares parent's state manager
    3. Inherit parent AgentContext: Shares parent's state manager

    Example:
        ```python
        # Standalone agent with conversation history
        ctx = AgentContext(run_id="session-1", agent_name="tutor")
        result = await agent.run("Hello", context=ctx)
        result = await agent.run("Continue", context=ctx)  # Remembers previous message

        # Agent in workflow - shares workflow state
        @workflow
        async def research_workflow(ctx: WorkflowContext):
            agent_result = await research_agent.run("Find AI trends", context=ctx)
            # Agent has access to workflow state via inherited context
        ```
    """

    def __init__(
        self,
        run_id: str,
        agent_name: str,
        session_id: Optional[str] = None,
        state_manager: Optional[Any] = None,
        parent_context: Optional[Context] = None,
        attempt: int = 0,
        runtime_context: Optional[Any] = None,
    ):
        """
        Initialize agent context.

        Args:
            run_id: Unique execution identifier
            agent_name: Name of the agent
            session_id: Session identifier for conversation history (default: run_id)
            state_manager: Optional state manager (for context inheritance)
            parent_context: Parent context to inherit state from
            attempt: Retry attempt number
            runtime_context: RuntimeContext for trace correlation
        """
        super().__init__(run_id, attempt, runtime_context)

        self._agent_name = agent_name
        self._session_id = session_id or run_id

        # Determine state manager based on parent context
        from .entity import EntityStateManager, _get_state_manager

        if state_manager:
            # Explicit state manager provided
            self._state_manager = state_manager
            logger.debug(f"AgentContext using provided state manager")
        elif parent_context:
            # Try to inherit state manager from parent
            try:
                # Check if parent is WorkflowContext or AgentContext
                if hasattr(parent_context, '_workflow_entity'):
                    # WorkflowContext - get state manager from worker context
                    self._state_manager = _get_state_manager()
                    logger.debug(f"AgentContext inheriting state from WorkflowContext")
                elif hasattr(parent_context, '_state_manager'):
                    # Parent AgentContext - share state manager
                    self._state_manager = parent_context._state_manager
                    logger.debug(f"AgentContext inheriting state from parent AgentContext")
                else:
                    # FunctionContext or base Context - create new state manager
                    self._state_manager = EntityStateManager()
                    logger.debug(f"AgentContext created new state manager (parent has no state)")
            except RuntimeError:
                # _get_state_manager() failed (not in worker context) - create standalone
                self._state_manager = EntityStateManager()
                logger.debug(f"AgentContext created standalone state manager")
        else:
            # Standalone - create new state manager
            self._state_manager = EntityStateManager()
            logger.debug(f"AgentContext created standalone state manager")

        # Conversation key for state storage
        self._conversation_key = f"agent:{agent_name}:{self._session_id}:messages"

    @property
    def state(self):
        """
        Get state interface for agent state management.

        Returns:
            EntityState instance for state operations

        Example:
            # Store conversation history
            messages = ctx.state.get(f"agent:{agent_name}:{session_id}:messages", [])
            messages.append({"role": "user", "content": "Hello"})
            ctx.state.set(f"agent:{agent_name}:{session_id}:messages", messages)

            # Store agent-specific data
            ctx.state.set("research_results", data)
        """
        from .entity import EntityState

        # Use agent's conversation key as the state key
        state_key = ("agent", self._conversation_key)
        state_dict = self._state_manager.get_or_create_state(state_key)
        return EntityState(state_dict)

    @property
    def session_id(self) -> str:
        """Get session identifier for this agent context."""
        return self._session_id

    def get_conversation_history(self) -> List[Message]:
        """
        Retrieve conversation history from state.

        Returns:
            List of Message objects from conversation history
        """
        messages_data = self.state.get(self._conversation_key, [])

        # Convert dict representations back to Message objects
        messages = []
        for msg_dict in messages_data:
            if isinstance(msg_dict, dict):
                role = msg_dict.get("role", "user")
                content = msg_dict.get("content", "")
                if role == "user":
                    messages.append(Message.user(content))
                elif role == "assistant":
                    messages.append(Message.assistant(content))
                else:
                    # Generic message - create with MessageRole enum
                    from .lm import MessageRole
                    msg_role = MessageRole(role) if role in ("user", "assistant", "system") else MessageRole.USER
                    msg = Message(role=msg_role, content=content)
                    messages.append(msg)
            else:
                # Already a Message object
                messages.append(msg_dict)

        return messages

    def save_conversation_history(self, messages: List[Message]) -> None:
        """
        Save conversation history to state.

        Args:
            messages: List of Message objects to persist
        """
        # Convert Message objects to dict for JSON serialization
        messages_data = []
        for msg in messages:
            messages_data.append({
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "content": msg.content
            })

        self.state.set(self._conversation_key, messages_data)


class Handoff:
    """Configuration for agent-to-agent handoff.

    Handoffs enable one agent to delegate control to another specialized agent,
    following the pattern popularized by LangGraph and OpenAI Agents SDK.

    The handoff is exposed to the LLM as a tool named 'transfer_to_{agent_name}'
    that allows explicit delegation with conversation history.

    Example:
        ```python
        specialist = Agent(name="specialist", ...)

        # Create handoff configuration
        handoff_to_specialist = Handoff(
            agent=specialist,
            description="Transfer to specialist for detailed analysis"
        )

        # Use in coordinator agent
        coordinator = Agent(
            name="coordinator",
            handoffs=[handoff_to_specialist]
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

    def __init__(
        self,
        output: str,
        tool_calls: List[Dict[str, Any]],
        context: Context,
        handoff_to: Optional[str] = None,
        handoff_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output = output
        self.tool_calls = tool_calls
        self.context = context
        self.handoff_to = handoff_to  # Name of agent that was handed off to
        self.handoff_metadata = handoff_metadata or {}  # Additional handoff info


class Agent:
    """Autonomous LLM-driven agent with tool orchestration.

    Current features:
    - LLM integration (OpenAI, Anthropic, etc.)
    - Tool selection and execution
    - Multi-turn reasoning
    - Context and state management

    Future enhancements:
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
        model: Any,  # Can be string like "openai/gpt-4o-mini" OR LanguageModel instance
        instructions: str,
        tools: Optional[List[Any]] = None,
        handoffs: Optional[List[Handoff]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model_config: Optional[ModelConfig] = None,
        max_iterations: int = 10,
        model_name: Optional[str] = None,  # For backwards compatibility with tests
    ):
        """Initialize agent.

        Args:
            name: Agent name/identifier
            model: Model string with provider prefix (e.g., "openai/gpt-4o-mini") OR LanguageModel instance
            instructions: System instructions for the agent
            tools: List of tools available to the agent (functions, Tool instances, or Agent instances)
            handoffs: List of Handoff configurations for agent-to-agent delegation
            temperature: LLM temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            model_config: Optional advanced configuration (custom endpoints, headers, etc.)
            max_iterations: Maximum reasoning iterations
            model_name: Optional model name (for backwards compatibility, used when model is a LanguageModel instance)
        """
        self.name = name
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.model_config = model_config
        self.max_iterations = max_iterations

        # Support both string model names and LanguageModel instances
        if isinstance(model, str):
            # New API: model is a string like "openai/gpt-4o-mini"
            self.model = model
            self.model_name = model_name or model
            self._language_model = None  # Will create on demand
        elif isinstance(model, LanguageModel):
            # Old API (for tests): model is a LanguageModel instance
            self._language_model = model
            self.model = model  # Keep for backwards compatibility
            self.model_name = model_name or "mock-model"
        else:
            raise TypeError(f"model must be a string or LanguageModel instance, got {type(model)}")

        # Store handoffs for building handoff tools
        self.handoffs = handoffs or []

        # Build tool registry (includes regular tools, agent-as-tools, and handoff tools)
        self.tools: Dict[str, Tool] = {}
        if tools:
            for tool_item in tools:
                # Check if it's an Agent instance (agents-as-tools pattern)
                if isinstance(tool_item, Agent):
                    agent_tool = tool_item.to_tool()
                    self.tools[agent_tool.name] = agent_tool
                    logger.info(f"Added agent '{tool_item.name}' as tool to '{self.name}'")
                # Check if it's a Tool instance
                elif isinstance(tool_item, Tool):
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

        # Build handoff tools
        for handoff_config in self.handoffs:
            handoff_tool = self._create_handoff_tool(handoff_config)
            self.tools[handoff_tool.name] = handoff_tool
            logger.info(f"Added handoff tool '{handoff_tool.name}' to '{self.name}'")

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
            "model": model
        }

    def to_tool(self, description: Optional[str] = None) -> Tool:
        """Convert this agent to a Tool that can be used by other agents.

        This enables agents-as-tools pattern where one agent can invoke another
        agent as if it were a regular tool.

        Args:
            description: Optional custom description (defaults to agent instructions)

        Returns:
            Tool instance that wraps this agent

        Example:
            ```python
            research_agent = Agent(
                name="researcher",
                model="openai/gpt-4o-mini",
                instructions="You are a research specialist."
            )

            # Use research agent as a tool for another agent
            coordinator = Agent(
                name="coordinator",
                model="openai/gpt-4o-mini",
                instructions="Coordinate tasks using specialist agents.",
                tools=[research_agent.to_tool()]
            )
            ```
        """
        agent_name = self.name

        # Handler that runs the agent
        async def agent_tool_handler(ctx: Context, user_message: str) -> str:
            """Execute agent and return output."""
            ctx.logger.info(f"Invoking agent '{agent_name}' as tool")

            # Run the agent with the user message
            result = await self.run(user_message, context=ctx)

            return result.output

        # Create tool with agent's schema
        tool_description = description or self.instructions or f"Agent: {self.name}"

        agent_tool = Tool(
            name=self.name,
            description=tool_description,
            handler=agent_tool_handler,
            input_schema=self.input_schema,
            auto_schema=False,
        )

        return agent_tool

    def _create_handoff_tool(self, handoff_config: Handoff, current_messages_callback: Optional[Callable] = None) -> Tool:
        """Create a tool for handoff to another agent.

        Args:
            handoff_config: Handoff configuration
            current_messages_callback: Optional callback to get current conversation messages

        Returns:
            Tool instance that executes the handoff
        """
        target_agent = handoff_config.agent
        tool_name = handoff_config.tool_name

        # Handler that executes the handoff
        async def handoff_handler(ctx: Context, message: str) -> Dict[str, Any]:
            """Transfer control to target agent."""
            ctx.logger.info(
                f"Handoff from '{self.name}' to '{target_agent.name}': {message}"
            )

            # If we should pass conversation history, add it to context
            if handoff_config.pass_full_history:
                # Get current conversation from the agent's run loop
                # (This will be set when we detect the handoff in run())
                conversation_history = getattr(ctx, '_agent_data', {}).get("_current_conversation", [])

                if conversation_history:
                    ctx.logger.info(
                        f"Passing {len(conversation_history)} messages to target agent"
                    )
                    # Store in context for target agent to optionally use
                    if not hasattr(ctx, '_agent_data'):
                        ctx._agent_data = {}
                    ctx._agent_data["_handoff_conversation_history"] = conversation_history

            # Execute target agent with the message and shared context
            result = await target_agent.run(message, context=ctx)

            # Store handoff metadata - this signals that a handoff occurred
            handoff_data = {
                "_handoff": True,
                "from_agent": self.name,
                "to_agent": target_agent.name,
                "message": message,
                "output": result.output,
                "tool_calls": result.tool_calls,
            }

            if not hasattr(ctx, '_agent_data'):
                ctx._agent_data = {}
            ctx._agent_data["_handoff_result"] = handoff_data

            # Return the handoff data (will be detected in run() loop)
            return handoff_data

        # Create tool with handoff schema
        handoff_tool = Tool(
            name=tool_name,
            description=handoff_config.description,
            handler=handoff_handler,
            input_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message or task to pass to the target agent"
                    }
                },
                "required": ["message"]
            },
            auto_schema=False,
        )

        return handoff_tool

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
        # Create or adapt context
        if context is None:
            # Standalone execution - create AgentContext
            import uuid
            run_id = f"agent-{self.name}-{uuid.uuid4().hex[:8]}"
            context = AgentContext(
                run_id=run_id,
                agent_name=self.name,
            )
        elif isinstance(context, AgentContext):
            # Already AgentContext - use as-is
            pass
        elif hasattr(context, '_workflow_entity'):
            # WorkflowContext - create AgentContext that inherits state
            import uuid
            run_id = f"{context.run_id}:agent:{self.name}"
            context = AgentContext(
                run_id=run_id,
                agent_name=self.name,
                session_id=context.run_id,  # Share workflow's session
                parent_context=context,
            )
        else:
            # FunctionContext or other - create new AgentContext
            import uuid
            run_id = f"{context.run_id}:agent:{self.name}"
            context = AgentContext(
                run_id=run_id,
                agent_name=self.name,
            )

        # Load conversation history from state (if AgentContext)
        if isinstance(context, AgentContext):
            messages: List[Message] = context.get_conversation_history()
            # Add new user message
            messages.append(Message.user(user_message))
            # Save updated conversation
            context.save_conversation_history(messages)
        else:
            # Fallback for non-AgentContext (shouldn't happen with code above)
            messages = [Message.user(user_message)]

        # Create span for agent execution with trace linking
        from ._core import create_span

        with create_span(
            self.name,
            "agent",
            context._runtime_context if hasattr(context, "_runtime_context") else None,
            {
                "agent.name": self.name,
                "agent.model": self.model_name,  # Use model_name (always a string)
                "agent.max_iterations": str(self.max_iterations),
            },
        ) as span:
            all_tool_calls: List[Dict[str, Any]] = []

            # Reasoning loop
            for iteration in range(self.max_iterations):
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

                # Call LLM
                # Check if we have a legacy LanguageModel instance or need to create one
                if self._language_model is not None:
                    # Legacy API: use provided LanguageModel instance
                    request = GenerateRequest(
                        model="mock-model",  # Not used by MockLanguageModel
                        system_prompt=self.instructions,
                        messages=messages,
                        tools=tool_defs if tool_defs else [],
                    )
                    request.config.temperature = self.temperature
                    if self.max_tokens:
                        request.config.max_tokens = self.max_tokens
                    if self.top_p:
                        request.config.top_p = self.top_p
                    response = await self._language_model.generate(request)
                else:
                    # New API: model is a string, create internal LM instance
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
                    self.logger.debug(f"Agent calling {len(response.tool_calls)} tool(s)")

                    # Store current conversation in context for potential handoffs
                    # Use a simple dict attribute since we don't need full state persistence for this
                    if not hasattr(context, '_agent_data'):
                        context._agent_data = {}
                    context._agent_data["_current_conversation"] = messages

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

                                # Check if this was a handoff
                                if isinstance(result, dict) and result.get("_handoff"):
                                    self.logger.info(
                                        f"Handoff detected to '{result['to_agent']}', "
                                        f"terminating current agent"
                                    )
                                    # Save conversation before returning
                                    if isinstance(context, AgentContext):
                                        context.save_conversation_history(messages)
                                    # Return immediately with handoff result
                                    return AgentResult(
                                        output=result["output"],
                                        tool_calls=all_tool_calls + result.get("tool_calls", []),
                                        context=context,
                                        handoff_to=result["to_agent"],
                                        handoff_metadata=result,
                                    )

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
                    messages.append(Message.user(f"Tool results:\n{results_text}\n\nPlease provide your final answer based on these results."))

                    # Continue loop for agent to process results

                else:
                    # No tool calls - agent is done
                    self.logger.debug(f"Agent completed after {iteration + 1} iterations")
                    # Save conversation before returning
                    if isinstance(context, AgentContext):
                        context.save_conversation_history(messages)
                    return AgentResult(
                        output=response.text,
                        tool_calls=all_tool_calls,
                        context=context,
                    )

            # Max iterations reached
            self.logger.warning(f"Agent reached max iterations ({self.max_iterations})")
            final_output = messages[-1].content if messages else "No output generated"
            # Save conversation before returning
            if isinstance(context, AgentContext):
                context.save_conversation_history(messages)
            return AgentResult(
                output=final_output,
                tool_calls=all_tool_calls,
                context=context,
            )


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
