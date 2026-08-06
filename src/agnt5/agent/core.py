"""Agent class - core LLM-driven agent with tool orchestration."""

import inspect
import json
import logging
import secrets
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Union

from .. import lm
from .._ids import generate_cid
from .._serialization import deserialize, serialize, serialize_to_str
from .._telemetry import setup_module_logger, truncate_span_attribute_value
from ..activation import (
    ActivationRecoveryPolicy,
    ChildJoinPolicy,
    _reset_current_activation,
    _set_current_activation,
    child_activation_request_from_context,
)
from ..callbacks import (
    AfterAgentCallback,
    AfterModelCallback,
    AfterToolCallback,
    AgentCallbackContext,
    AgentCallbacks,
    BeforeAgentCallback,
    BeforeModelCallback,
    BeforeToolCallback,
    CallbackOverride,
    ModelCallbackContext,
    ToolCallbackContext,
)
from ..context import Context, get_current_context, set_current_context
from ..events import Event
from ..exceptions import WaitingForUserInputException
from ..lm import (
    BuiltInTool,
    GenerateRequest,
    GenerateResponse,
    LanguageModel,
    Message,
    ModelConfig,
    ToolDefinition,
    built_in_tool_names,
)
from ..lm.events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)
from ..tool import Tool, ToolRegistry
from .agents_md import AgentsMdSource, load_agents_md, render_guidance
from .context import AgentContext
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
from .handoff import Handoff
from .registry import AgentRegistry
from .result import AgentResult
from .skills import Skill, make_load_skill_tool, render_catalog, resolve_skills

logger = setup_module_logger(__name__)

class _DefaultTemperature(float):
    pass


_DEFAULT_AGENT_TEMPERATURE = _DefaultTemperature(0.7)


def _is_openai_reasoning_model(model: str) -> bool:
    if not model.startswith("openai/"):
        return False

    model_name = model.split("/", 1)[1]
    return (
        model_name.startswith("gpt-5")
        or model_name == "o1"
        or model_name.startswith("o1-")
        or model_name == "o3"
        or model_name.startswith("o3-")
        or model_name == "o4"
        or model_name.startswith("o4-")
    )


def _serialize_tool_result(result: Any) -> str:
    """Serialize a tool result to JSON string, handling Pydantic models and other complex types.

    Args:
        result: The tool execution result (may be Pydantic model, dataclass, dict, etc.)

    Returns:
        JSON string representation of the result
    """
    if result is None:
        return "null"

    # Use centralized serialization that handles Pydantic models, dataclasses, etc.
    return serialize_to_str(result)


def _serialize_span_data(value: Any) -> str:
    """Serialize and bound data stored as OpenTelemetry span attributes."""
    return truncate_span_attribute_value(_serialize_tool_result(value))


def _resolved_handoff_history(history: Sequence[Any]) -> List[Any]:
    """Exclude the unresolved assistant tool call that triggered a handoff."""
    resolved = list(history)
    if not resolved:
        return resolved
    last = resolved[-1]
    if isinstance(last, Message):
        has_active_tool_call = bool(last.tool_calls)
    elif isinstance(last, dict):
        has_active_tool_call = bool(last.get("tool_calls") or last.get("toolCalls"))
    else:
        has_active_tool_call = False
    if has_active_tool_call:
        resolved.pop()
    return resolved


@dataclass
class _StreamedLMResponse:
    """Result from streaming LLM call - contains collected text and any tool calls."""
    text: str
    tool_calls: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


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

    Example:
        ```python
        from agnt5 import Agent, tool

        @tool
        async def search_web(query: str) -> str:
            '''Search the web for information.'''
            return f"Results for: {query}"

        agent = Agent(
            name="researcher",
            model="openai/gpt-4o-mini",
            instructions="You are a research assistant.",
            tools=[search_web],
        )

        result = await agent.run("Find recent AI developments")
        print(result.output)
        ```
    """

    def __init__(
        self,
        name: str,
        model: Union[str, LanguageModel],
        instructions: str,
        tools: Optional[List[Any]] = None,
        sandbox: Optional[Any] = None,
        built_in_tools: Optional[List[BuiltInTool]] = None,
        model_config: Optional[ModelConfig] = None,
        handoffs: Optional[List[Union["Agent", Handoff]]] = None,
        callbacks: Optional[AgentCallbacks] = None,
        before_agent_callback: Optional[BeforeAgentCallback] = None,
        after_agent_callback: Optional[AfterAgentCallback] = None,
        before_model_callback: Optional[BeforeModelCallback] = None,
        after_model_callback: Optional[AfterModelCallback] = None,
        before_tool_callback: Optional[BeforeToolCallback] = None,
        after_tool_callback: Optional[AfterToolCallback] = None,
        # Legacy parameters (kept for backward compatibility)
        model_name: Optional[str] = None,
        temperature: Optional[float] = _DEFAULT_AGENT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        cache: Optional[Union[bool, lm.PromptCache, lm.ContextCache, str]] = None,
        # Deprecated compatibility aliases. Prefer cache=True or
        # cache=lm.PromptCache(...).
        cache_control: bool = False,
        cache_ttl: Optional[str] = None,
        max_iterations: int = 10,
        skills: Optional[Sequence[Union[str, "Skill"]]] = None,
        skills_dir: Optional[Union[str, Path]] = None,
        agents_md: Optional[AgentsMdSource] = None,
    ):
        """Initialize agent.

        Args:
            name: Agent identifier
            model: Model specification. Either:
                   - String like "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"
                   - LanguageModel instance (legacy, for backward compatibility)
            instructions: System prompt for the agent
            tools: List of tools, Tool instances, or Agents (used as tools)
            sandbox: Optional sandbox workspace. When provided, standard
                sandbox tools are added automatically and custom tools can
                access the same workspace via ``ctx.sandbox``.
            built_in_tools: Provider-hosted tools (currently OpenAI Responses API only:
                BuiltInTool.WEB_SEARCH, CODE_INTERPRETER, FILE_SEARCH). These run
                server-side; results are baked into the assistant message and the
                Agent records the call for tracing without local dispatch.
            model_config: Model configuration (temperature, max_tokens, etc.)
            handoffs: List of agents to hand off to (creates transfer_to_* tools)
            callbacks: Grouped execution callbacks for agent/model/tool stages
            before_agent_callback: Callback before the agent loop starts
            after_agent_callback: Callback after a successful agent result
            before_model_callback: Callback before each model request is sent
            after_model_callback: Callback after each model response is received
            before_tool_callback: Callback before each tool call executes
            after_tool_callback: Callback after each successful tool call
            model_name: Deprecated - use `model` parameter instead
            temperature: LLM temperature (0-1). Legacy parameter - prefer model_config.
            max_tokens: Maximum tokens in response. Legacy parameter - prefer model_config.
            top_p: Top-p sampling. Legacy parameter - prefer model_config.
            cache: Enable provider-native prompt caching with ``True``, pass
                ``lm.PromptCache(...)`` for TTL/key/retention hints, or pass a
                Gemini ``lm.ContextCache`` for reusable explicit caches.
            cache_control: Deprecated compatibility alias for ``cache=True``.
            cache_ttl: Deprecated compatibility alias for ``cache=lm.PromptCache(ttl=...)``.
            max_iterations: Maximum reasoning iterations
            skills: On-demand skills — names (resolved against ``skills_dir``) or
                ``Skill`` objects. Only name+description sit in context until the
                agent calls ``load_skill`` to load a skill's full instructions.
            skills_dir: Directory pool that skill names resolve against. With no
                ``skills`` selection, every skill in the pool is loaded.
            agents_md: Always-on project/area guidance (``AGENTS.md``). A file
                path, a directory (uses its ``AGENTS.md``), or an ordered list
                where later entries are more specific. Injected into every prompt.
        """
        self.name = name
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.sandbox = sandbox
        self.logger = logging.getLogger(f"agnt5.agent.{name}")
        base_callbacks = callbacks or AgentCallbacks()
        self.callbacks = AgentCallbacks(
            before_agent=before_agent_callback or base_callbacks.before_agent,
            after_agent=after_agent_callback or base_callbacks.after_agent,
            before_model=before_model_callback or base_callbacks.before_model,
            after_model=after_model_callback or base_callbacks.after_model,
            before_tool=before_tool_callback or base_callbacks.before_tool,
            after_tool=after_tool_callback or base_callbacks.after_tool,
        )

        # Handle model parameter: string or LanguageModel
        if isinstance(model, str):
            # New API: model is a string like "openai/gpt-4o-mini"
            provider, model_name = lm._parse_model(model)
            self.model = f"{provider}/{model_name}"
            self.model_name = self.model  # For compatibility
            self._language_model = None
        elif isinstance(model, LanguageModel):
            # Legacy API: model is a LanguageModel instance
            self._language_model = model
            self.model = model_name or "mock-model"
            self.model_name = model_name or "mock-model"
        else:
            raise ValueError("model must be a string (e.g., 'openai/gpt-4o-mini') or LanguageModel instance")

        # Model configuration (legacy params take precedence for backward compat)
        self.model_config = model_config
        self._temperature_explicit = temperature is not _DEFAULT_AGENT_TEMPERATURE
        self.temperature = None if temperature is None else float(temperature)
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.cache = lm._coerce_prompt_cache(cache)
        if cache_control or cache_ttl is not None:
            if self.cache is None:
                self.cache = lm.PromptCache(
                    enabled=cache_control or cache_ttl is not None,
                    ttl=cache_ttl,
                )
            elif cache_ttl is not None and self.cache.ttl is None:
                self.cache = replace(self.cache, enabled=True, ttl=cache_ttl)
        if (
            self.cache is not None
            and self.cache.resource is not None
            and "/" in self.model
            and self.model.split("/", 1)[0] not in {"google", "gemini"}
        ):
            raise ValueError("Explicit context caches can only be used with google/gemini models")
        self._built_in_tools: List[BuiltInTool] = list(built_in_tools or [])

        # Cost tracking
        self._cumulative_cost_usd: float = 0.0

        # Initialize tools registry
        self.tools: Dict[str, Tool] = {}

        if sandbox is not None:
            from ..sandbox_tools import sandbox_tools

            for item in sandbox_tools(sandbox=sandbox):
                self.tools[item.name] = item

        # Resolve on-demand skills and register the loader tool. Empty when no
        # skills are configured, so skill-less agents are unchanged.
        self._skills: Dict[str, Skill] = resolve_skills(skills, skills_dir)
        self._skills_catalog: str = render_catalog(self._skills)
        if self._skills:
            load_tool = make_load_skill_tool(self._skills, sandbox=self.sandbox)
            self.tools[load_tool.name] = load_tool

        # Always-on project/area guidance (AGENTS.md). Empty string when unset,
        # so guidance-less agents are unchanged.
        self._agents_md_guidance: str = render_guidance(load_agents_md(agents_md))

        if tools:
            for item in tools:
                if isinstance(item, Tool):
                    # Tool instance (including @tool decorated functions)
                    self.tools[item.name] = item
                elif isinstance(item, Agent):
                    # Agent as tool - wrap it
                    agent_tool = item.to_tool()
                    self.tools[agent_tool.name] = agent_tool
                    self.logger.debug(f"Wrapped agent '{item.name}' as tool")
                else:
                    self.logger.warning(
                        f"Skipping unknown tool type: {type(item)}. "
                        f"Expected Tool (from @tool decorator) or Agent."
                    )

        # Store handoffs for introspection
        self.handoffs: List[Handoff] = []

        # Process handoffs: create transfer_to_* tools for each target agent
        if handoffs:
            for item in handoffs:
                if isinstance(item, Agent):
                    # Auto-wrap Agent in Handoff with defaults
                    handoff_config = Handoff(agent=item)
                elif isinstance(item, Handoff):
                    handoff_config = item
                else:
                    self.logger.warning(f"Skipping unknown handoff type: {type(item)}")
                    continue

                # Store the handoff configuration
                self.handoffs.append(handoff_config)

                # Create handoff tool
                handoff_tool = self._create_handoff_tool(handoff_config)
                self.tools[handoff_tool.name] = handoff_tool
                self.logger.debug(f"Added handoff tool '{handoff_tool.name}'")

        # Auto-register agent in registry (similar to Entity auto-registration)
        AgentRegistry.register(self)
        self.logger.debug(f"Auto-registered agent '{self.name}'")

    @property
    def cumulative_cost_usd(self) -> float:
        """Get cumulative cost of all LLM calls for this agent.

        Returns:
            Total cost in USD
        """
        return self._cumulative_cost_usd

    def _track_llm_cost(self, response: GenerateResponse, context: Optional[Context] = None) -> None:
        """Track LLM call cost.

        Args:
            response: LLM response containing usage/cost info
            context: Optional context for emitting cost events
        """
        cost_usd = getattr(response, 'cost_usd', None)
        if cost_usd:
            self._cumulative_cost_usd += cost_usd
            self.logger.debug(
                f"LLM call cost: ${cost_usd:.6f}, "
                f"cumulative: ${self._cumulative_cost_usd:.6f}"
            )

    def _temperature_for_request(self) -> Optional[float]:
        if self._temperature_explicit:
            return self.temperature
        if _is_openai_reasoning_model(self.model):
            return None
        return self.temperature

    def _apply_generation_config(
        self,
        request: GenerateRequest,
        *,
        include_built_in_tools: bool = False,
    ) -> None:
        temperature = self._temperature_for_request()
        if temperature is not None:
            request.config.temperature = temperature
        if self.max_tokens is not None:
            request.config.max_tokens = self.max_tokens
        if self.top_p is not None:
            request.config.top_p = self.top_p
        if self.cache is not None:
            request.config.cache = self.cache
        if include_built_in_tools and self._built_in_tools:
            request.config.built_in_tools = list(self._built_in_tools)

    def _model_config_snapshot(self) -> Dict[str, Any]:
        cache = None
        if self.cache is not None:
            cache = {
                "enabled": self.cache.enabled,
                "ttl": self.cache.ttl,
                "key": self.cache.key,
                "retention": self.cache.retention,
                "resource": self.cache.resource,
            }
        return {
            "model": self.model,
            "temperature": self._temperature_for_request(),
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "cache": cache,
        }

    def to_tool(self) -> Tool:
        """Convert this agent to a tool that can be used by other agents.

        The tool will run this agent and return its output.

        Returns:
            Tool instance that wraps this agent

        Example:
            ```python
            # Create specialist agents
            researcher = Agent(name="researcher", ...)
            analyst = Agent(name="analyst", ...)

            # Use them as tools
            coordinator = Agent(
                name="coordinator",
                tools=[researcher.to_tool(), analyst.to_tool()]
            )
            ```
        """
        from ..tool import tool as tool_decorator

        # Capture agent reference
        agent = self

        @tool_decorator(
            name=f"ask_{agent.name}",
            description=agent.instructions or f"Ask the {agent.name} agent for help",
            recovery_policy=ActivationRecoveryPolicy.DURABLE_STEPS,
        )
        async def agent_as_tool(ctx: Context, message: str) -> str:
            """Invoke the agent with a message and return its response."""
            result = await agent._run_delegated_child(
                ctx,
                message,
                history=None,
                join_policy=ChildJoinPolicy.REQUIRED,
            )
            return result["output"]

        # Get the tool from registry
        return ToolRegistry.get(f"ask_{agent.name}")

    def _create_handoff_tool(self, handoff: Handoff) -> Tool:
        """Create a handoff tool for transferring control to another agent.

        Args:
            handoff: Handoff configuration

        Returns:
            Tool that performs the handoff
        """
        from ..tool import tool as tool_decorator

        target_agent = handoff.agent
        pass_history = handoff.pass_full_history

        @tool_decorator(
            name=handoff.tool_name,
            description=handoff.description,
            recovery_policy=ActivationRecoveryPolicy.DURABLE_STEPS,
        )
        async def transfer_tool(ctx: Context, message: str) -> Dict[str, Any]:
            """Transfer control to another agent.

            Args:
                ctx: Execution context (auto-injected)
                message: Message to pass to the target agent

            Returns:
                Dict with handoff marker and target agent's result
            """
            # Get conversation history if available and requested
            history = None
            if pass_history and ctx:
                if hasattr(ctx, '_agent_data') and "_current_conversation" in ctx._agent_data:
                    history = _resolved_handoff_history(
                        ctx._agent_data["_current_conversation"]
                    )

            # Run target agent (using run for non-streaming invocation)
            result = await target_agent._run_delegated_child(
                ctx,
                message,
                history=history,
                join_policy=handoff.join_policy,
            )

            # Return with handoff marker
            return {
                "_handoff": True,
                "to_agent": target_agent.name,
                "output": result["output"],
                "tool_calls": result["tool_calls"],
            }

        return ToolRegistry.get(handoff.tool_name)

    async def _run_delegated_child(
        self,
        ctx: Context,
        message: str,
        *,
        history: Optional[List[Message]],
        join_policy: ChildJoinPolicy,
    ) -> Dict[str, Any]:
        """Run or replay this agent as one logical durable child."""

        metadata = getattr(ctx, "_trace_metadata", None) or {}
        activation_client = getattr(ctx, "_activation_client", None)
        if metadata.get("durable_activation_v1") != "true" or activation_client is None:
            result = await self.run(message, context=ctx, history=history)
            return {"output": result.output, "tool_calls": result.tool_calls}

        stable_key = ctx.allocate_activation_key("child", self.name)
        request = child_activation_request_from_context(
            ctx,
            child_name=self.name,
            stable_key=stable_key,
            input_value={"agent": self.name, "message": message, "history": history},
            join_policy=join_policy,
        )
        activation_token = None

        def on_admitted(decision):
            nonlocal activation_token
            activation_token = _set_current_activation(decision)

        async def execute() -> Dict[str, Any]:
            result = await self.run(message, context=ctx, history=history)
            return {"output": result.output, "tool_calls": result.tool_calls}

        try:
            result, _receipt = await activation_client.run(
                request,
                execute,
                encode_output=serialize,
                decode_output=deserialize,
                latency_ms=lambda: 0,
                on_admitted=on_admitted,
                failure_error_code="CHILD_FAILED",
                failure_retryable=True,
                failure_external_outcome_certainty="UNKNOWN",
            )
            return result
        finally:
            if activation_token is not None:
                _reset_current_activation(activation_token)

    def _render_prompt(
        self,
        template: str,
        context_vars: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render system prompt template with context variables.

        Args:
            template: System prompt with {{variable_name}} placeholders
            context_vars: Variables to substitute

        Returns:
            Rendered prompt string
        """
        if not context_vars:
            return template

        rendered = template
        for key, value in context_vars.items():
            placeholder = "{{" + key + "}}"
            if placeholder in rendered:
                rendered = rendered.replace(placeholder, str(value))

        return rendered

    def _compose_system_prompt(self, context_vars: Optional[Dict[str, Any]] = None) -> str:
        """Render instructions and append the on-demand skills catalog.

        Single choke point for system-prompt assembly so every request path
        carries the same context. The catalog is empty for skill-less agents,
        leaving their prompt byte-for-byte unchanged.
        """
        rendered = self._render_prompt(self.instructions, context_vars)
        blocks = [rendered]
        if self._agents_md_guidance:
            blocks.append(self._agents_md_guidance)
        if self._skills_catalog:
            blocks.append(self._skills_catalog)
        return "\n\n".join(blocks)

    def _detect_memory_scope(
        self,
        context: Optional[Context] = None
    ) -> tuple[str, str]:
        """Detect memory scope from context.

        Priority: user_id > session_id > run_id

        Returns:
            Tuple of (entity_key, scope) where:
            - entity_key: e.g., "user:user-456", "session:abc-123", "run:xyz-789"
            - scope: "user", "session", or "run"

        Example:
            entity_key, scope = agent._detect_memory_scope(ctx)
            # If ctx.user_id="user-123": ("user:user-123", "user")
            # If ctx.session_id="sess-456": ("session:sess-456", "session")
            # Otherwise: ("run:run-789", "run")
        """
        # Extract identifiers from context
        user_id = getattr(context, 'user_id', None) if context else None
        session_id = getattr(context, 'session_id', None) if context else None
        run_id = getattr(context, 'run_id', None) if context else None

        # Priority: user_id > session_id > run_id
        if user_id:
            return (f"user:{user_id}", "user")
        elif session_id and session_id != run_id:  # Explicit session (not defaulting to run_id)
            return (f"session:{session_id}", "session")
        elif run_id:
            return (f"run:{run_id}", "run")
        else:
            # Fallback: create ephemeral key
            import uuid
            fallback_run_id = f"agent-{self.name}-{uuid.uuid4().hex[:8]}"
            return (f"run:{fallback_run_id}", "run")

    async def _maybe_await_callback(self, value: Any) -> Any:
        """Await a callback result when it is awaitable."""
        if inspect.isawaitable(value):
            return await value
        return value

    def _callback_value(self, result: Any) -> tuple[bool, Any]:
        """Return whether a callback supplied an override and its value."""
        if isinstance(result, CallbackOverride):
            return True, result.value
        if result is None:
            return False, None
        return True, result

    def _agent_result_from_callback(self, value: Any, context: Context) -> AgentResult:
        """Convert a callback replacement value into an AgentResult."""
        if isinstance(value, AgentResult):
            if value.context is None:
                value.context = context
            return value

        if isinstance(value, dict):
            output_value = value.get("output", value)
            output = output_value if isinstance(output_value, str) else _serialize_tool_result(output_value)
            tool_calls = value.get("tool_calls") or value.get("toolCalls") or []
            handoff_to = value.get("handoff_to") or value.get("handoffTo")
            return AgentResult(
                output=output,
                tool_calls=tool_calls,
                context=context,
                handoff_to=handoff_to,
                handoff_metadata=value if handoff_to else None,
            )

        output = value if isinstance(value, str) else _serialize_tool_result(value)
        return AgentResult(output=output, tool_calls=[], context=context)

    async def _run_before_agent_callback(
        self,
        context: Context,
        user_message: str,
        history: Optional[List[Message]],
        prompt_context: Optional[Dict[str, Any]],
    ) -> Optional[AgentResult]:
        callback = self.callbacks.before_agent
        if callback is None:
            return None

        callback_context = AgentCallbackContext(
            agent=self,
            context=context,
            user_message=user_message,
            history=history,
            prompt_context=prompt_context,
        )
        raw_result = await self._maybe_await_callback(callback(callback_context))
        has_value, value = self._callback_value(raw_result)
        if not has_value:
            return None
        return self._agent_result_from_callback(value, context)

    async def _run_after_agent_callback(
        self,
        context: Context,
        user_message: str,
        history: Optional[List[Message]],
        prompt_context: Optional[Dict[str, Any]],
        result: AgentResult,
    ) -> AgentResult:
        callback = self.callbacks.after_agent
        if callback is None:
            return result

        callback_context = AgentCallbackContext(
            agent=self,
            context=context,
            user_message=user_message,
            history=history,
            prompt_context=prompt_context,
        )
        raw_result = await self._maybe_await_callback(callback(callback_context, result))
        has_value, value = self._callback_value(raw_result)
        if not has_value:
            return result
        return self._agent_result_from_callback(value, context)

    async def _call_model_generate(self, request: GenerateRequest) -> GenerateResponse:
        from ..lm import LMClient as _LanguageModel

        if self._language_model is not None:
            return await self._language_model.generate(request)

        provider, _model_name = self.model.split('/', 1)
        internal_lm = _LanguageModel(provider=provider.lower(), default_model=None)
        return await internal_lm.generate(request)

    def _validate_model_response(self, response: Any) -> GenerateResponse:
        if not isinstance(response, GenerateResponse):
            raise TypeError(
                "model callbacks must return GenerateResponse, CallbackOverride, or None"
            )
        return response

    async def _generate_with_callbacks(
        self,
        context: Context,
        request: GenerateRequest,
        iteration: int,
        messages: List[Message],
        tool_definitions: List[ToolDefinition],
    ) -> GenerateResponse:
        callback_context = ModelCallbackContext(
            agent=self,
            context=context,
            iteration=iteration,
            messages=messages,
            tool_definitions=tool_definitions,
        )

        response: Optional[GenerateResponse] = None
        before = self.callbacks.before_model
        if before is not None:
            raw_result = await self._maybe_await_callback(before(callback_context, request))
            has_value, value = self._callback_value(raw_result)
            if has_value:
                response = self._validate_model_response(value)

        if response is None:
            response = await self._call_model_generate(request)
            self._track_llm_cost(response, context)

        after = self.callbacks.after_model
        if after is not None:
            raw_result = await self._maybe_await_callback(after(callback_context, request, response))
            has_value, value = self._callback_value(raw_result)
            if has_value:
                response = self._validate_model_response(value)

        return response

    async def _invoke_tool_with_callbacks(
        self,
        callback_context: ToolCallbackContext,
    ) -> Any:
        before = self.callbacks.before_tool
        if before is not None:
            raw_result = await self._maybe_await_callback(before(callback_context))
            has_value, value = self._callback_value(raw_result)
            if has_value:
                return value

        if callback_context.tool is None:
            raise ValueError(f"Tool '{callback_context.tool_name}' not found")

        result = await callback_context.tool.invoke_with_stable_key(
            callback_context.context,
            callback_context.arguments,
            stable_key=callback_context.tool_call_id or None,
        )

        after = self.callbacks.after_tool
        if after is not None:
            raw_result = await self._maybe_await_callback(after(callback_context, result))
            has_value, value = self._callback_value(raw_result)
            if has_value:
                return value

        return result

    async def _run_core(
        self,
        user_message: str,
        context: Optional[Context] = None,
        history: Optional[List[Message]] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
        sequence_start: int = 0,
    ) -> AsyncGenerator[Union[Event, AgentResult], None]:
        """Core streaming execution loop.

        This async generator yields events during execution and returns
        the final AgentResult as the last yielded item.

        Yields:
            Event objects (LM events, tool events) during execution
            AgentResult as the final item

        Used by:
            - stream(): Wraps with agent.started/completed events
            - run(): Consumes events and extracts final result
        """
        self.logger.debug(f"_run_core entered for agent '{self.name}', tools={list(self.tools.keys())}")
        if self._built_in_tools:
            self.logger.info(f"Built-in tools enabled: {[t.value for t in self._built_in_tools]}")
        sequence = sequence_start

        # Create or adapt context
        if context is None:
            context = get_current_context()

        # Capture workflow context for checkpoints
        from ..workflow import WorkflowContext
        workflow_ctx = context if isinstance(context, WorkflowContext) else None

        # Generate correlation_id for pairing agent.started ↔ agent.completed/failed
        agent_correlation_id = generate_cid()

        if context is None:
            import uuid
            # Standalone agent - generate a proper UUID for run_id
            run_id = str(uuid.uuid4())
            context = AgentContext(
                run_id=run_id,
                agent_name=self.name,
            )
        elif isinstance(context, AgentContext):
            pass
        elif hasattr(context, '_workflow_entity'):
            entity_key, scope = self._detect_memory_scope(context)
            detected_session_id = entity_key.split(":", 1)[1] if ":" in entity_key else context.run_id
            context = AgentContext(
                run_id=context.run_id,
                agent_name=self.name,
                session_id=detected_session_id,
                parent_context=context,
                runtime_context=getattr(context, '_runtime_context', None),
                trace_metadata=getattr(context, '_trace_metadata', None),
            )
        else:
            context = AgentContext(
                run_id=context.run_id,
                agent_name=self.name,
                parent_context=context,
                runtime_context=getattr(context, '_runtime_context', None),
                trace_metadata=getattr(context, '_trace_metadata', None),
            )

        if self.sandbox is not None:
            setattr(context, "sandbox", self.sandbox)

        # Emit agent.started checkpoint for journal persistence
        # Skip if executor already emitted (to avoid duplicate events)
        # Use _parent_correlation_id to link agent to parent step in hierarchy
        if context and not getattr(context, '_executor_managed_lifecycle', False):
            context.emit(AgentStarted(
                name=self.name,
                correlation_id=agent_correlation_id,
                parent_correlation_id=context._parent_correlation_id,
                agent_model=self.model_name,
                tool_names=list(self.tools.keys()),
                max_iterations=self.max_iterations,
                input_data={"task": user_message},
                metadata={"name": self.name},
            ))

        # Set agent as parent for iteration events (using Context-based tracking)
        original_agent_parent = context.set_as_parent(agent_correlation_id)

        # Check for HITL resume
        if workflow_ctx and hasattr(workflow_ctx, "_agent_resume_info"):
            resume_info = workflow_ctx._agent_resume_info
            if resume_info["agent_name"] == self.name:
                self.logger.info("Detected HITL resume, calling resume_from_hitl()")
                delattr(workflow_ctx, "_agent_resume_info")
                result = await self.resume_from_hitl(
                    context=workflow_ctx,
                    agent_context=resume_info["agent_context"],
                    user_response=resume_info["user_response"],
                )
                result = await self._run_after_agent_callback(
                    result.context or context,
                    user_message,
                    history,
                    prompt_context,
                    result,
                )
                yield result
                return

        # Set context in task-local storage
        token = set_current_context(context)
        try:
            before_agent_result = await self._run_before_agent_callback(
                context,
                user_message,
                history,
                prompt_context,
            )
            if before_agent_result is not None:
                context.restore_parent(original_agent_parent)
                if context and not getattr(context, '_executor_managed_lifecycle', False):
                    context.emit(AgentCompleted(
                        name=self.name,
                        correlation_id=agent_correlation_id,
                        parent_correlation_id=context._parent_correlation_id,
                        iterations=0,
                        tool_calls_count=len(before_agent_result.tool_calls),
                        handoff_to=before_agent_result.handoff_to,
                        output_length=len(before_agent_result.output),
                        metadata={"name": self.name},
                    ))
                yield before_agent_result
                return

            # Build conversation messages
            messages: List[Message] = []

            if history:
                # Convert dicts to Message objects if needed (for JSON history from platform)
                for msg in history:
                    if isinstance(msg, Message):
                        messages.append(msg)
                    elif isinstance(msg, dict):
                        role_str = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role_str == "user":
                            messages.append(Message.user(content))
                        elif role_str == "assistant":
                            messages.append(Message.assistant(content))
                        elif role_str == "system":
                            messages.append(Message.system(content))
                        else:
                            messages.append(Message.user(content))
                    else:
                        # Try to use it as a Message anyway
                        messages.append(msg)
                self.logger.debug(f"Prepended {len(history)} messages from explicit history")

            if isinstance(context, AgentContext):
                stored_messages = await context.get_conversation_history()
                messages.extend(stored_messages)

            messages.append(Message.user(user_message))

            if isinstance(context, AgentContext):
                messages_to_save = stored_messages + [Message.user(user_message)] if history else messages
                await context.save_conversation_history(messages_to_save)

            # Create span for tracing (uses contextvar for async-safe parent-child linking)
            from ..tracing import create_span

            with create_span(
                self.name,
                "agent",
                context._runtime_context if hasattr(context, "_runtime_context") else None,
                {
                    "agent.name": self.name,
                    "agent.model": self.model_name,
                    "agent.max_iterations": str(self.max_iterations),
                    "input.data": _serialize_span_data({"message": user_message}),
                },
            ) as span:
                all_tool_calls: List[Dict[str, Any]] = []
                import time as _time

                # Render system prompt
                rendered_instructions = self._compose_system_prompt(prompt_context)

                # Reasoning loop
                for iteration in range(self.max_iterations):
                    iteration_start_time = _time.time()
                    # Generate correlation_id for pairing agent.iteration.started ↔ agent.iteration.completed
                    iteration_correlation_id = generate_cid()

                    if context:
                        context.emit(AgentIterationStarted(
                            name=self.name,
                            correlation_id=iteration_correlation_id,
                            parent_correlation_id=agent_correlation_id,
                            iteration=iteration + 1,
                            input_data={"iteration": iteration + 1, "max_iterations": self.max_iterations},
                            metadata={"name": self.name},
                        ))

                    # Set iteration as parent for lm.call and tool events
                    original_iteration_parent = context.set_as_parent(iteration_correlation_id)

                    # Build tool definitions
                    tool_defs = [
                        ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.input_schema,
                        )
                        for tool in self.tools.values()
                    ]

                    # Build request
                    request = GenerateRequest(
                        model=self.model if not self._language_model else "mock-model",
                        system_prompt=rendered_instructions,
                        messages=messages,
                        tools=tool_defs if tool_defs else [],
                    )
                    self._apply_generation_config(request, include_built_in_tools=True)

                    # Stream LLM call and yield events
                    response_text = ""
                    response_tool_calls = []

                    async for item, seq in self._stream_lm_call(
                        request,
                        sequence,
                        iteration_correlation_id,
                        context,
                        iteration + 1,
                        messages,
                        tool_defs,
                    ):
                        if isinstance(item, _StreamedLMResponse):
                            response_text = item.text
                            response_tool_calls = item.tool_calls
                            sequence = seq
                        else:
                            # Yield LM event
                            yield item
                            sequence = seq

                    # Server-side built-in tools (Responses API) execute on the
                    # provider; their results are already baked into the assistant
                    # message. Record them in the trace and exclude from local
                    # dispatch so the rest of the loop only handles user tools.
                    built_in_names = set()
                    if self._built_in_tools:
                        built_in_names = built_in_tool_names(self._built_in_tools)
                        if response_tool_calls:
                            used = [
                                tool_call["name"]
                                for tool_call in response_tool_calls
                                if tool_call.get("name") in built_in_names
                            ]
                            if used:
                                self.logger.info(f"Built-in tools used by model: {used}")
                            else:
                                self.logger.info(
                                    f"Built-in tools configured {[t.value for t in self._built_in_tools]} "
                                    "but model answered without triggering them"
                                )
                        else:
                            self.logger.info(
                                f"Built-in tools configured {[t.value for t in self._built_in_tools]} "
                                "but no provider-side tool calls were surfaced this iteration"
                            )

                    if response_tool_calls and self._built_in_tools:
                        partitioned_user_calls: List[Dict[str, Any]] = []
                        for built_in_idx, tool_call in enumerate(response_tool_calls):
                            if tool_call.get("name") not in built_in_names:
                                partitioned_user_calls.append(tool_call)
                                continue

                            tool_name = tool_call["name"]
                            tool_args_str = tool_call.get("arguments", "{}")
                            tool_call_id = tool_call.get("id")
                            tool_correlation_id = f"tool-{secrets.token_hex(5)}"

                            all_tool_calls.append({
                                "name": tool_name,
                                "arguments": tool_args_str,
                                "iteration": iteration + 1,
                                "id": tool_call_id,
                                "built_in": True,
                            })

                            started_event = ToolCallStarted(
                                name=tool_name,
                                correlation_id=tool_correlation_id,
                                parent_correlation_id=iteration_correlation_id,
                                tool_name=tool_name,
                                tool_call_id=tool_call_id or "",
                                input_data={"arguments": tool_args_str, "built_in": True},
                                index=built_in_idx,
                            )
                            if context:
                                context.emit(started_event)
                            yield started_event
                            sequence += 1

                            completed_event = ToolCallCompleted(
                                name=tool_name,
                                correlation_id=tool_correlation_id,
                                parent_correlation_id=iteration_correlation_id,
                                tool_name=tool_name,
                                tool_call_id=tool_call_id or "",
                                output_data={"server_side": True},
                                duration_ms=0,
                                index=built_in_idx,
                            )
                            if context:
                                context.emit(completed_event)
                            yield completed_event
                            sequence += 1

                        response_tool_calls = partitioned_user_calls

                    # Add assistant response to messages. Preserve only user
                    # tool calls so the next iteration's request includes the
                    # tool_use blocks that require local tool_result messages.
                    messages.append(Message.assistant(
                        response_text,
                        tool_calls=response_tool_calls if response_tool_calls else None,
                    ))

                    # Check if LLM wants to use tools
                    self.logger.debug(f"response_tool_calls count={len(response_tool_calls) if response_tool_calls else 0}")
                    if response_tool_calls:
                        self.logger.debug(f"Agent calling {len(response_tool_calls)} tool(s): {[tc.get('name') for tc in response_tool_calls]}")

                        if not hasattr(context, '_agent_data'):
                            context._agent_data = {}
                        context._agent_data["_current_conversation"] = messages

                        # Execute tool calls
                        tool_results = []
                        for tool_idx, tool_call in enumerate(response_tool_calls):
                            tool_name = tool_call["name"]
                            tool_args_str = tool_call["arguments"]
                            tool_call_id = tool_call.get("id")  # From LLM response

                            all_tool_calls.append({
                                "name": tool_name,
                                "arguments": tool_args_str,
                                "iteration": iteration + 1,
                                "id": tool_call_id,
                            })

                            # Yield tool call started event with unique content_index
                            tool_correlation_id = f"tool-{secrets.token_hex(5)}"
                            tool_start_time = _time.time()
                            tool_started_event = ToolCallStarted(
                                name=tool_name,
                                correlation_id=tool_correlation_id,
                                parent_correlation_id=iteration_correlation_id,
                                tool_name=tool_name,
                                tool_call_id=tool_call_id or "",
                                input_data={"arguments": tool_args_str},
                                index=tool_idx,
                            )
                            # Emit to platform for persistence
                            self.logger.debug(f"Emitting ToolCallStarted: tool={tool_name}")
                            if context:
                                self.logger.debug(f"context.emit(ToolCallStarted) for {tool_name}")
                                context.emit(tool_started_event)
                            yield tool_started_event
                            sequence += 1

                            try:
                                tool_args = json.loads(tool_args_str)
                                tool = self.tools.get(tool_name)

                                if not tool:
                                    result_text = f"Error: Tool '{tool_name}' not found"
                                else:
                                    result = await self._invoke_tool_with_callbacks(
                                        ToolCallbackContext(
                                            agent=self,
                                            context=context,
                                            iteration=iteration + 1,
                                            tool_name=tool_name,
                                            tool_call_id=tool_call_id or "",
                                            tool_call=tool_call,
                                            arguments=tool_args,
                                            tool=tool,
                                        )
                                    )

                                    if isinstance(result, dict) and result.get("_handoff"):
                                        self.logger.info(f"Handoff to '{result['to_agent']}'")
                                        if isinstance(context, AgentContext):
                                            await context.save_conversation_history(messages)

                                        # Yield tool completed and final result
                                        tool_duration_ms = int((_time.time() - tool_start_time) * 1000)
                                        tool_completed_event = ToolCallCompleted(
                                            name=tool_name,
                                            correlation_id=tool_correlation_id,
                                            parent_correlation_id=iteration_correlation_id,
                                            tool_name=tool_name,
                                            tool_call_id=tool_call_id or "",
                                            output_data={"result": _serialize_tool_result(result["output"])},
                                            duration_ms=tool_duration_ms,
                                            index=tool_idx,
                                        )
                                        # Emit to platform for persistence
                                        if context:
                                            context.emit(tool_completed_event)
                                        yield tool_completed_event
                                        sequence += 1

                                        # Add output data to span for trace visibility
                                        span.set_attribute("output.data", _serialize_span_data(result["output"]))

                                        agent_result = AgentResult(
                                            output=result["output"],
                                            tool_calls=all_tool_calls + result.get("tool_calls", []),
                                            context=context,
                                            handoff_to=result["to_agent"],
                                            handoff_metadata=result,
                                        )
                                        agent_result = await self._run_after_agent_callback(
                                            context,
                                            user_message,
                                            history,
                                            prompt_context,
                                            agent_result,
                                        )
                                        yield agent_result
                                        return

                                    result_text = _serialize_tool_result(result)

                                tool_results.append({
                                    "tool": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "result": result_text,
                                    "error": None,
                                })

                                # Yield tool completed event
                                tool_duration_ms = int((_time.time() - tool_start_time) * 1000)
                                tool_completed_event = ToolCallCompleted(
                                    name=tool_name,
                                    correlation_id=tool_correlation_id,
                                    parent_correlation_id=iteration_correlation_id,
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id or "",
                                    output_data={"result": result_text},
                                    duration_ms=tool_duration_ms,
                                    index=tool_idx,
                                )
                                # Emit to platform for persistence
                                if context:
                                    context.emit(tool_completed_event)
                                yield tool_completed_event
                                sequence += 1

                            except WaitingForUserInputException as e:
                                self.logger.info(f"Agent pausing for user input at iteration {iteration}")
                                messages_dict = [
                                    {"role": msg.role.value, "content": msg.content}
                                    for msg in messages
                                ]
                                raise WaitingForUserInputException(
                                    question=e.question,
                                    input_type=e.input_type,
                                    options=e.options,
                                    checkpoint_state=e.checkpoint_state,
                                    pause_index=e.pause_index,
                                    step_name=e.step_name,
                                    step_correlation_id=e.step_correlation_id,
                                    allow_custom=e.allow_custom,
                                    skippable=e.skippable,
                                    checkpoint_metadata=e.checkpoint_metadata,
                                    agent_context={
                                        "agent_name": self.name,
                                        "iteration": iteration,
                                        "messages": messages_dict,
                                        "tool_results": tool_results,
                                        "pending_tool_call": {
                                            "name": tool_call["name"],
                                            "arguments": tool_call["arguments"],
                                            "tool_call_index": response_tool_calls.index(tool_call),
                                        },
                                        "all_tool_calls": all_tool_calls,
                                        "model_config": self._model_config_snapshot(),
                                    },
                                ) from e

                            except Exception as e:
                                self.logger.error(f"Tool execution error: {e}")
                                tool_results.append({
                                    "tool": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "result": None,
                                    "error": str(e),
                                })
                                tool_failed_event = ToolCallFailed(
                                    name=tool_name,
                                    correlation_id=tool_correlation_id,
                                    parent_correlation_id=iteration_correlation_id,
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id or "",
                                    error_code=type(e).__name__,
                                    error_message=str(e),
                                )
                                # Emit to platform for persistence
                                if context:
                                    context.emit(tool_failed_event)
                                yield tool_failed_event
                                sequence += 1

                        # Append one tool message per result, keyed by
                        # tool_call_id so OpenAI/Anthropic can map each result
                        # back to the assistant message's tool_use block.
                        # The Rust LM client (openai_common.rs:179) routes
                        # messages with tool_call_id to role="tool".
                        for tr in tool_results:
                            tool_content = (
                                tr["result"] if tr["error"] is None
                                else f"Error: {tr['error']}"
                            )
                            messages.append(Message.tool_result(
                                tool_call_id=tr.get("tool_call_id") or "",
                                content=tool_content or "",
                            ))

                        # Reset parent before emitting iteration.completed
                        context.restore_parent(original_iteration_parent)

                        iteration_duration_ms = int((_time.time() - iteration_start_time) * 1000)
                        if context:
                            context.emit(AgentIterationCompleted(
                                name=self.name,
                                correlation_id=iteration_correlation_id,
                                parent_correlation_id=agent_correlation_id,
                                iteration=iteration + 1,
                                duration_ms=iteration_duration_ms,
                                has_tool_calls=True,
                                tool_calls_count=len(tool_results),
                                metadata={"name": self.name},
                            ))

                    else:
                        # No tool calls - agent is done
                        self.logger.debug(f"Agent completed after {iteration + 1} iterations")

                        # Reset parent before emitting iteration.completed
                        context.restore_parent(original_iteration_parent)

                        iteration_duration_ms = int((_time.time() - iteration_start_time) * 1000)
                        if context:
                            context.emit(AgentIterationCompleted(
                                name=self.name,
                                correlation_id=iteration_correlation_id,
                                parent_correlation_id=agent_correlation_id,
                                iteration=iteration + 1,
                                duration_ms=iteration_duration_ms,
                                has_tool_calls=False,
                                metadata={"name": self.name},
                            ))

                        if isinstance(context, AgentContext):
                            await context.save_conversation_history(messages)

                        # Reset parent to workflow before emitting agent.completed
                        context.restore_parent(original_agent_parent)

                        # Emit agent.completed checkpoint for journal persistence
                        # Skip if executor already manages lifecycle (to avoid duplicate events)
                        if context and not getattr(context, '_executor_managed_lifecycle', False):
                            context.emit(AgentCompleted(
                                name=self.name,
                                correlation_id=agent_correlation_id,
                                parent_correlation_id=context._parent_correlation_id,
                                iterations=iteration + 1,
                                tool_calls_count=len(all_tool_calls),
                                metadata={"name": self.name},
                            ))

                        # Add output data to span for trace visibility
                        span.set_attribute("output.data", _serialize_span_data(response_text))

                        agent_result = AgentResult(
                            output=response_text,
                            tool_calls=all_tool_calls,
                            context=context,
                        )
                        agent_result = await self._run_after_agent_callback(
                            context,
                            user_message,
                            history,
                            prompt_context,
                            agent_result,
                        )
                        yield agent_result
                        return

                # Max iterations reached
                self.logger.warning(f"Agent reached max iterations ({self.max_iterations})")
                final_output = messages[-1].content if messages else "No output generated"

                if isinstance(context, AgentContext):
                    await context.save_conversation_history(messages)

                # Reset parent to workflow before emitting agent.completed
                context.restore_parent(original_agent_parent)

                # Emit agent.completed checkpoint for journal persistence (with max_iterations flag)
                # Skip if executor already manages lifecycle (to avoid duplicate events)
                if context and not getattr(context, '_executor_managed_lifecycle', False):
                    context.emit(AgentCompleted(
                        name=self.name,
                        correlation_id=agent_correlation_id,
                        parent_correlation_id=context._parent_correlation_id,
                        iterations=self.max_iterations,
                        tool_calls_count=len(all_tool_calls),
                        metadata={"name": self.name},
                    ))

                # Add output data to span for trace visibility
                span.set_attribute("output.data", _serialize_span_data(final_output))

                agent_result = AgentResult(
                    output=final_output,
                    tool_calls=all_tool_calls,
                    context=context,
                )
                agent_result = await self._run_after_agent_callback(
                    context,
                    user_message,
                    history,
                    prompt_context,
                    agent_result,
                )
                yield agent_result

        except Exception as e:
            # Reset parent to workflow before emitting agent.failed
            context.restore_parent(original_agent_parent)

            # Skip if executor already manages lifecycle (to avoid duplicate events)
            if context and not getattr(context, '_executor_managed_lifecycle', False):
                context.emit(AgentFailed(
                    name=self.name,
                    correlation_id=agent_correlation_id,
                    parent_correlation_id=context._parent_correlation_id,
                    error_code=type(e).__name__,
                    error_message=str(e),
                    iterations=0,  # Failed before completing any iterations
                    metadata={"name": self.name},
                ))
            raise
        finally:
            if self.sandbox is not None:
                close = getattr(self.sandbox, "close", None)
                if close is not None:
                    await close()
            from ..context import _current_context
            _current_context.reset(token)

    async def _stream_lm_call(
        self,
        request: GenerateRequest,
        sequence_start: int = 0,
        parent_correlation_id: str = "",
        context: Optional[Context] = None,
        iteration: int = 0,
        messages: Optional[List[Message]] = None,
        tool_definitions: Optional[List[ToolDefinition]] = None,
    ) -> AsyncGenerator[Tuple[Event, int], None]:
        """Stream an LLM call and yield events.

        This method calls the LLM and yields LM events (start, delta, stop).
        The final response (including tool_calls) is yielded as a special
        _StreamedLMResponse event at the end.

        When tools are present, uses generate() with synthetic events since
        streaming doesn't yet support tool calls. When no tools, uses real
        streaming which properly exposes thinking blocks for extended thinking.

        Args:
            request: The generate request with model, messages, tools, etc.
            sequence_start: Starting sequence number for events
            parent_correlation_id: Parent correlation ID for tracing

        Yields:
            Tuple of (Event, next_sequence) or (_StreamedLMResponse, next_sequence)
        """
        from ..lm import LMClient as _LanguageModel

        sequence = sequence_start
        collected_text = ""
        usage_dict = None
        tool_calls = []

        # Tool streaming is opt-in because older/custom LanguageModel
        # implementations may not include complete tool calls in LMCompleted.
        has_tools = bool(request.tools)
        has_built_in_tools = bool(request.config.built_in_tools)
        if self._language_model is not None:
            supports_streaming_tools = bool(
                self._language_model.supports_streaming_tools
            )
        else:
            provider, _model_name = self.model.split('/', 1)
            supports_streaming_tools = (
                _LanguageModel.supports_streaming_tools_for_provider(provider)
            )
        has_model_callbacks = (
            self.callbacks.before_model is not None
            or self.callbacks.after_model is not None
        )

        requires_non_streaming_tools = (
            (has_tools or has_built_in_tools)
            and not supports_streaming_tools
        )
        if requires_non_streaming_tools or has_model_callbacks:
            if context is not None:
                response = await self._generate_with_callbacks(
                    context=context,
                    request=request,
                    iteration=iteration,
                    messages=messages or request.messages,
                    tool_definitions=tool_definitions or request.tools,
                )
            else:
                response = await self._call_model_generate(request)

            # Emit synthetic LM events for compatibility
            lm_correlation_id = generate_cid()
            yield (LMContentBlockStarted(
                name=self.model,
                correlation_id=lm_correlation_id,
                parent_correlation_id=parent_correlation_id,
                block_type="text",
                index=0,
            ), sequence + 1)
            sequence += 1
            if response.text:
                yield (LMContentBlockDelta(
                    name=self.model,
                    correlation_id=lm_correlation_id,
                    parent_correlation_id=parent_correlation_id,
                    content=response.text,
                    block_type="text",
                    index=0,
                ), sequence + 1)
                sequence += 1
            yield (LMContentBlockCompleted(
                name=self.model,
                correlation_id=lm_correlation_id,
                parent_correlation_id=parent_correlation_id,
                block_type="text",
                index=0,
            ), sequence + 1)
            sequence += 1

            collected_text = response.text
            tool_calls = response.tool_calls or []
            if response.usage:
                usage_dict = {
                    "input_tokens": getattr(response.usage, 'input_tokens', getattr(response.usage, 'prompt_tokens', 0)),
                    "output_tokens": getattr(response.usage, 'output_tokens', getattr(response.usage, 'completion_tokens', 0)),
                    "cached_tokens": getattr(response.usage, 'cached_tokens', 0),
                }
        else:
            # Use real streaming - properly exposes thinking blocks
            if self._language_model is not None:
                # Legacy LanguageModel - use stream() method
                async for event in self._language_model.stream(request):
                    if isinstance(event, LMCompleted):
                        # Extract final text and usage from completion event
                        output_data = event.output_data or {}
                        collected_text = output_data.get("text", "") if isinstance(output_data, dict) else ""
                        tool_calls = (
                            output_data.get("tool_calls") or []
                            if isinstance(output_data, dict)
                            else []
                        )
                        usage_dict = {
                            "input_tokens": event.input_tokens,
                            "output_tokens": event.output_tokens,
                            "cached_tokens": getattr(event, "cached_tokens", 0),
                        }
                    else:
                        # Forward LM events (thinking/message start/delta/stop)
                        yield (event, sequence + 1)
                        sequence += 1
                        # Collect text from message deltas (not thinking)
                        if isinstance(event, LMContentBlockDelta):
                            if event.content and event.block_type == "text":
                                collected_text += str(event.content)
            else:
                # New API: model is a string, create internal LM instance
                provider, model_name = self.model.split('/', 1)
                internal_lm = _LanguageModel(provider=provider.lower(), default_model=None)
                async for event in internal_lm.stream(request):
                    if isinstance(event, LMCompleted):
                        # Extract final text and usage from completion event
                        output_data = event.output_data or {}
                        collected_text = output_data.get("text", "") if isinstance(output_data, dict) else ""
                        tool_calls = (
                            output_data.get("tool_calls") or []
                            if isinstance(output_data, dict)
                            else []
                        )
                        usage_dict = {
                            "input_tokens": event.input_tokens,
                            "output_tokens": event.output_tokens,
                            "cached_tokens": getattr(event, "cached_tokens", 0),
                        }
                    else:
                        # Forward LM events (thinking/message start/delta/stop)
                        yield (event, sequence + 1)
                        sequence += 1
                        # Collect text from message deltas (not thinking)
                        if isinstance(event, LMContentBlockDelta):
                            if event.content and event.block_type == "text":
                                collected_text += str(event.content)

        # Yield the final response
        yield (_StreamedLMResponse(
            text=collected_text,
            tool_calls=tool_calls,
            usage=usage_dict,
        ), sequence)

    async def stream(
        self,
        user_message: str,
        context: Optional[Context] = None,
        history: Optional[List[Message]] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Event, None]:
        """Run agent with streaming events.

        This is an async generator that yields Event objects during execution.
        Use `async for event in agent.stream(...)` to process events in real-time.

        Args:
            user_message: User's input message
            context: Optional execution context (auto-created if not provided)
            history: Optional conversation history to include
            prompt_context: Optional context variables for system prompt template

        Yields:
            Event objects during execution:
            - agent.started: When agent begins execution
            - lm.message.start/delta/stop: During LLM generation
            - agent.tool_call.started/completed: During tool execution
            - agent.completed: When agent finishes (contains final output)

        Example:
            ```python
            # Streaming execution
            async for event in agent.stream("Analyze recent tech news"):
                if event.event_type == "lm.content_block.delta":
                    print(event.data, end="", flush=True)  # data is raw content for deltas
                elif event.event_type == "agent.completed":
                    print(f"\\nFinal: {event.data['output']}")

            # Non-streaming (use run instead)
            result = await agent.run("Analyze recent tech news")
            print(result.output)
            ```
        """
        # Track sequence number for events
        sequence = 0

        # Generate correlation ID for the agent run
        run_correlation_id = generate_cid()

        # Yield agent.started event
        yield AgentStarted(
            name=self.name,
            correlation_id=run_correlation_id,
            parent_correlation_id="",
            agent_model=self.model_name,
            tool_names=list(self.tools.keys()),
            max_iterations=self.max_iterations,
            input_data={"task": user_message},
        )
        sequence += 1

        try:
            # Run the streaming core loop - yields LM events, tool events, and final result
            result = None
            async for item in self._run_core(
                user_message=user_message,
                context=context,
                history=history,
                prompt_context=prompt_context,
                sequence_start=sequence,
            ):
                if isinstance(item, AgentResult):
                    # Final result - convert to agent.completed event
                    result = item
                    sequence = getattr(item, '_last_sequence', sequence)
                elif isinstance(item, Event):
                    # Forward LM and tool events
                    yield item
                    sequence = item.sequence + 1 if hasattr(item, 'sequence') else sequence

            # Yield agent.completed event with the result
            if result:
                yield AgentCompleted(
                    name=self.name,
                    correlation_id=run_correlation_id,
                    parent_correlation_id="",
                    iterations=len(result.tool_calls) // 2 + 1 if result.tool_calls else 1,
                    tool_calls_count=len(result.tool_calls) if result.tool_calls else 0,
                    handoff_to=result.handoff_to,
                    output_data={"output": result.output, "tool_calls": result.tool_calls},
                )

        except Exception as e:
            # Yield agent.failed event
            yield AgentFailed(
                name=self.name,
                correlation_id=run_correlation_id,
                parent_correlation_id="",
                iterations=0,
                error_code=type(e).__name__,
                error_message=str(e),
            )
            raise

    async def run(
        self,
        user_message: str,
        context: Optional[Context] = None,
        history: Optional[List[Message]] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Run agent to completion (non-streaming).

        Returns an AgentResult directly. Use `agent.stream(...)` if you need
        per-iteration events.

        Args:
            user_message: User's input message
            context: Optional execution context
            history: Optional conversation history
            prompt_context: Optional context variables

        Returns:
            AgentResult with output and execution details

        Example:
            ```python
            result = await agent.run("Analyze recent tech news")
            print(result.output)
            ```
        """
        result = None
        async for event in self.stream(user_message, context, history, prompt_context):
            if isinstance(event, AgentCompleted):
                # Extract result from the completed event
                output_data = event.output_data or {}
                result = AgentResult(
                    output=output_data.get("output", ""),
                    tool_calls=output_data.get("tool_calls", []),
                    context=context,
                    handoff_to=event.handoff_to,
                )
            elif isinstance(event, AgentFailed):
                # Re-raise the error (it was already raised in run())
                pass

        if result is None:
            # This shouldn't happen, but handle gracefully
            raise RuntimeError("Agent completed without producing a result")

        return result

    async def _run_impl(
        self,
        user_message: str,
        context: Optional[Context] = None,
        history: Optional[List[Message]] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Internal implementation of agent execution.

        This contains the core agent loop logic. Called by both run() and stream().
        """
        self.logger.debug(f" _run_impl called for agent '{self.name}', tools={list(self.tools.keys())}")
        # Create or adapt context
        if context is None:
            # Try to get context from task-local storage (set by workflow/function decorator)
            context = get_current_context()

        # IMPORTANT: Capture workflow context NOW before we replace it with AgentContext
        # This allows LM calls inside the agent to emit workflow checkpoints
        from ..workflow import WorkflowContext
        workflow_ctx = context if isinstance(context, WorkflowContext) else None

        # Generate correlation_id for pairing agent.started ↔ agent.completed/failed
        agent_correlation_id = generate_cid()

        if context is None:
            # Standalone execution - create AgentContext with valid UUID
            import uuid
            run_id = str(uuid.uuid4())
            context = AgentContext(
                run_id=run_id,
                agent_name=self.name,
            )
        elif isinstance(context, AgentContext):
            # Already AgentContext - use as-is
            pass
        elif hasattr(context, '_workflow_entity'):
            # WorkflowContext - create AgentContext that inherits state
            # Auto-detect memory scope based on user_id/session_id/run_id priority
            entity_key, scope = self._detect_memory_scope(context)

            # Extract the ID from entity_key (e.g., "session:abc-123" → "abc-123")
            detected_session_id = entity_key.split(":", 1)[1] if ":" in entity_key else context.run_id

            # Use parent's run_id (valid UUID) for events, session_id for conversation history
            context = AgentContext(
                run_id=context.run_id,  # Use parent's UUID, not compound ID
                agent_name=self.name,
                session_id=detected_session_id,  # Use auto-detected scope
                parent_context=context,
                runtime_context=getattr(context, '_runtime_context', None),  # Inherit trace context
                trace_metadata=getattr(context, '_trace_metadata', None),  # Inherit tenant_id
            )
        else:
            # FunctionContext or other - create new AgentContext
            context = AgentContext(
                run_id=context.run_id,
                agent_name=self.name,
                parent_context=context,
                runtime_context=getattr(context, '_runtime_context', None),
                trace_metadata=getattr(context, '_trace_metadata', None),
            )

        if self.sandbox is not None:
            setattr(context, "sandbox", self.sandbox)

        # Emit agent.started checkpoint for journal persistence
        # Skip if executor already emitted (to avoid duplicate events)
        # Use _parent_correlation_id to link agent to parent step in hierarchy
        if context and not getattr(context, '_executor_managed_lifecycle', False):
            context.emit(AgentStarted(
                name=self.name,
                correlation_id=agent_correlation_id,
                parent_correlation_id=context._parent_correlation_id,
                agent_model=self.model_name,
                tool_names=list(self.tools.keys()),
                max_iterations=self.max_iterations,
                input_data={"task": user_message},
                metadata={"name": self.name},
            ))

        # Set agent as parent for iteration events (using Context-based tracking)
        original_agent_parent = context.set_as_parent(agent_correlation_id)

        # NEW: Check if this is a resume from HITL
        if workflow_ctx and hasattr(workflow_ctx, "_agent_resume_info"):
            resume_info = workflow_ctx._agent_resume_info
            if resume_info["agent_name"] == self.name:
                self.logger.info("Detected HITL resume, calling resume_from_hitl()")

                # Clear resume info to avoid re-entry
                delattr(workflow_ctx, "_agent_resume_info")

                # Resume from checkpoint (context setup happens inside resume_from_hitl)
                return await self.resume_from_hitl(
                    context=workflow_ctx,
                    agent_context=resume_info["agent_context"],
                    user_response=resume_info["user_response"],
                )

        # Set context in task-local storage for automatic propagation to tools and LM calls
        token = set_current_context(context)
        try:
            try:
                # Build conversation messages
                messages: List[Message] = []

                # 1. Start with explicitly provided history (if any)
                if history:
                    messages.extend(history)
                    self.logger.debug(f"Prepended {len(history)} messages from explicit history")

                # 2. Load conversation history from state (if AgentContext)
                if isinstance(context, AgentContext):
                    stored_messages = await context.get_conversation_history()
                    messages.extend(stored_messages)

                # 3. Add new user message
                messages.append(Message.user(user_message))

                # 4. Save updated conversation to context storage
                if isinstance(context, AgentContext):
                    # Only save the stored + new message (not the explicit history)
                    messages_to_save = stored_messages + [Message.user(user_message)] if history else messages
                    await context.save_conversation_history(messages_to_save)

                # Create span for agent execution (uses contextvar for async-safe parent-child linking)
                from ..tracing import create_span

                with create_span(
                    self.name,
                    "agent",
                    context._runtime_context if hasattr(context, "_runtime_context") else None,
                    {
                        "agent.name": self.name,
                        "agent.model": self.model_name,  # Use model_name (always a string)
                        "agent.max_iterations": str(self.max_iterations),
                        "input.data": _serialize_span_data({"message": user_message}),
                    },
                ) as span:
                    all_tool_calls: List[Dict[str, Any]] = []
                    import time as _time

                    # NOTE: agent.started checkpoint is NOT sent here
                    # The caller (run()) yields Event.agent_started which the worker processes

                    # Render system prompt with context variables
                    rendered_instructions = self._compose_system_prompt(prompt_context)
                    if prompt_context:
                        self.logger.debug(f"Rendered system prompt with {len(prompt_context)} context variables")

                    # Reasoning loop
                    for iteration in range(self.max_iterations):
                        iteration_start_time = _time.time()
                        # Generate correlation_id for pairing agent.iteration.started ↔ agent.iteration.completed
                        iteration_correlation_id = generate_cid()

                        # Emit iteration started checkpoint
                        if context:
                            context.emit(AgentIterationStarted(
                                name=self.name,
                                correlation_id=iteration_correlation_id,
                                parent_correlation_id=agent_correlation_id,
                                iteration=iteration + 1,
                                max_iterations=self.max_iterations,
                                input_data={"iteration": iteration + 1, "max_iterations": self.max_iterations},
                                metadata={"name": self.name},
                            ))

                        # Set iteration as parent for lm.call and tool events
                        original_iteration_parent = context.set_as_parent(iteration_correlation_id)

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
                                system_prompt=rendered_instructions,
                                messages=messages,
                                tools=tool_defs if tool_defs else [],
                            )
                            self._apply_generation_config(request)
                            response = await self._language_model.generate(request)

                            # Track cost for this LLM call
                            self._track_llm_cost(response, context)
                        else:
                            # New API: model is a string, create internal LM instance
                            request = GenerateRequest(
                                model=self.model,
                                system_prompt=rendered_instructions,
                                messages=messages,
                                tools=tool_defs if tool_defs else [],
                            )
                            self._apply_generation_config(request)

                            # Create an internal LM instance for generation.
                            from ..lm import LMClient as _LanguageModel
                            provider, model_name = self.model.split('/', 1)
                            internal_lm = _LanguageModel(provider=provider.lower(), default_model=None)
                            response = await internal_lm.generate(request)

                            # Track cost for this LLM call
                            self._track_llm_cost(response, context)

                        # Add assistant response to messages
                        messages.append(Message.assistant(response.text))

                        # Check if LLM wants to use tools
                        self.logger.debug(f" LLM response has tool_calls={response.tool_calls is not None and len(response.tool_calls) > 0}, count={len(response.tool_calls) if response.tool_calls else 0}")
                        if response.tool_calls:
                            self.logger.debug(f" Agent calling {len(response.tool_calls)} tool(s): {[tc.get('name', 'unknown') for tc in response.tool_calls]}")

                            # Store current conversation in context for potential handoffs
                            # Use a simple dict attribute since we don't need full state persistence for this
                            if not hasattr(context, '_agent_data'):
                                context._agent_data = {}
                            context._agent_data["_current_conversation"] = messages

                            # Execute tool calls
                            tool_results = []
                            for tool_idx, tool_call in enumerate(response.tool_calls):
                                tool_name = tool_call["name"]
                                tool_args_str = tool_call["arguments"]
                                tool_call_id = tool_call.get("id", "")

                                # Track tool call
                                all_tool_calls.append(
                                    {
                                        "name": tool_name,
                                        "arguments": tool_args_str,
                                        "iteration": iteration + 1,
                                    }
                                )

                                # Generate correlation ID for this tool call
                                tool_correlation_id = f"tool-{secrets.token_hex(5)}"
                                tool_start_time = _time.time()

                                # Emit tool call started event
                                self.logger.debug(f" Tool call started: {tool_name}, context={context is not None}, correlation_id={tool_correlation_id}")
                                if context:
                                    event = ToolCallStarted(
                                        name=tool_name,
                                        correlation_id=tool_correlation_id,
                                        parent_correlation_id=iteration_correlation_id,
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        input_data={"arguments": tool_args_str},
                                        index=tool_idx,
                                    )
                                    self.logger.debug(f" Emitting ToolCallStarted event: {event.event_type}")
                                    context.emit(event)

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
                                        result = await tool.invoke_with_stable_key(
                                            context,
                                            tool_args,
                                            stable_key=tool_call_id or None,
                                        )

                                        # Check if this was a handoff
                                        if isinstance(result, dict) and result.get("_handoff"):
                                            self.logger.info(
                                                f"Handoff detected to '{result['to_agent']}', "
                                                f"terminating current agent"
                                            )
                                            # Save conversation before returning
                                            if isinstance(context, AgentContext):
                                                await context.save_conversation_history(messages)
                                            # Add output data to span for trace visibility
                                            span.set_attribute("output.data", _serialize_span_data(result["output"]))
                                            # Emit tool call completed event for handoff
                                            if context:
                                                tool_duration_ms = int((_time.time() - tool_start_time) * 1000)
                                                context.emit(ToolCallCompleted(
                                                    name=tool_name,
                                                    correlation_id=tool_correlation_id,
                                                    parent_correlation_id=iteration_correlation_id,
                                                    tool_name=tool_name,
                                                    tool_call_id=tool_call_id,
                                                    output_data={"result": _serialize_tool_result(result["output"])},
                                                    duration_ms=tool_duration_ms,
                                                    index=tool_idx,
                                                ))
                                            # Return immediately with handoff result
                                            return AgentResult(
                                                output=result["output"],
                                                tool_calls=all_tool_calls + result.get("tool_calls", []),
                                                context=context,
                                                handoff_to=result["to_agent"],
                                                handoff_metadata=result,
                                            )

                                        result_text = _serialize_tool_result(result)

                                    tool_results.append(
                                        {"tool": tool_name, "result": result_text, "error": None}
                                    )

                                    # Emit tool call completed event
                                    self.logger.debug(f" Tool call completed: {tool_name}, context={context is not None}")
                                    if context:
                                        tool_duration_ms = int((_time.time() - tool_start_time) * 1000)
                                        event = ToolCallCompleted(
                                            name=tool_name,
                                            correlation_id=tool_correlation_id,
                                            parent_correlation_id=iteration_correlation_id,
                                            tool_name=tool_name,
                                            tool_call_id=tool_call_id,
                                            output_data={"result": result_text},
                                            duration_ms=tool_duration_ms,
                                            index=tool_idx,
                                        )
                                        self.logger.debug(f" Emitting ToolCallCompleted event: {event.event_type}, duration_ms={tool_duration_ms}")
                                        context.emit(event)

                                except WaitingForUserInputException as e:
                                    # HITL PAUSE: Capture agent state and propagate exception
                                    self.logger.info(f"Agent pausing for user input at iteration {iteration}")

                                    # Serialize messages to dict format
                                    messages_dict = [
                                        {"role": msg.role.value, "content": msg.content}
                                        for msg in messages
                                    ]

                                    # Enhance exception with agent execution context
                                    raise WaitingForUserInputException(
                                        question=e.question,
                                        input_type=e.input_type,
                                        options=e.options,
                                        checkpoint_state=e.checkpoint_state,
                                        pause_index=e.pause_index,
                                        step_name=e.step_name,
                                        step_correlation_id=e.step_correlation_id,
                                        allow_custom=e.allow_custom,
                                        skippable=e.skippable,
                                        checkpoint_metadata=e.checkpoint_metadata,
                                        agent_context={
                                            "agent_name": self.name,
                                            "iteration": iteration,
                                            "messages": messages_dict,
                                            "tool_results": tool_results,
                                            "pending_tool_call": {
                                                "name": tool_call["name"],
                                                "arguments": tool_call["arguments"],
                                                "tool_call_index": response.tool_calls.index(tool_call),
                                            },
                                            "all_tool_calls": all_tool_calls,
                                            "model_config": self._model_config_snapshot(),
                                        },
                                    ) from e

                                except Exception as e:
                                    # Regular tool errors - log and continue
                                    self.logger.error(f"Tool execution error: {e}")
                                    tool_results.append(
                                        {"tool": tool_name, "result": None, "error": str(e)}
                                    )

                                    # Emit tool call failed event
                                    self.logger.debug(f" Tool call failed: {tool_name}, error={e}")
                                    if context:
                                        event = ToolCallFailed(
                                            name=tool_name,
                                            correlation_id=tool_correlation_id,
                                            parent_correlation_id=iteration_correlation_id,
                                            tool_name=tool_name,
                                            tool_call_id=tool_call_id,
                                            error_code=type(e).__name__,
                                            error_message=str(e),
                                        )
                                        self.logger.debug(f" Emitting ToolCallFailed event: {event.event_type}")
                                        context.emit(event)

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

                            # Reset parent before emitting iteration.completed
                            context.restore_parent(original_iteration_parent)

                            # Emit iteration completed checkpoint (with tool calls)
                            iteration_duration_ms = int((_time.time() - iteration_start_time) * 1000)
                            if context:
                                context.emit(AgentIterationCompleted(
                                    name=self.name,
                                    correlation_id=iteration_correlation_id,
                                    parent_correlation_id=agent_correlation_id,
                                    iteration=iteration + 1,
                                    duration_ms=iteration_duration_ms,
                                    has_tool_calls=True,
                                    tool_calls_count=len(tool_results),
                                    metadata={"name": self.name},
                                ))

                            # Continue loop for agent to process results

                        else:
                            # No tool calls - agent is done
                            self.logger.debug(f"Agent completed after {iteration + 1} iterations")

                            # Reset parent before emitting iteration.completed
                            context.restore_parent(original_iteration_parent)

                            # Emit iteration completed checkpoint
                            iteration_duration_ms = int((_time.time() - iteration_start_time) * 1000)
                            if context:
                                context.emit(AgentIterationCompleted(
                                    name=self.name,
                                    correlation_id=iteration_correlation_id,
                                    parent_correlation_id=agent_correlation_id,
                                    iteration=iteration + 1,
                                    duration_ms=iteration_duration_ms,
                                    has_tool_calls=False,
                                    tool_calls_count=0,
                                    metadata={"name": self.name},
                                ))

                            # Save conversation before returning
                            if isinstance(context, AgentContext):
                                await context.save_conversation_history(messages)

                            # Reset parent to workflow before emitting agent.completed
                            context.restore_parent(original_agent_parent)

                            # Emit completion checkpoint
                            # Skip if executor already manages lifecycle (to avoid duplicate events)
                            if context and not getattr(context, '_executor_managed_lifecycle', False):
                                context.emit(AgentCompleted(
                                    name=self.name,
                                    correlation_id=agent_correlation_id,
                                    parent_correlation_id=context._parent_correlation_id,
                                    iterations=iteration + 1,
                                    tool_calls_count=len(all_tool_calls),
                                    output_length=len(response.text),
                                    metadata={"name": self.name},
                                ))

                            # Add output data to span for trace visibility
                            span.set_attribute("output.data", _serialize_span_data(response.text))

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
                        await context.save_conversation_history(messages)

                    # Reset parent to workflow before emitting agent.completed
                    context.restore_parent(original_agent_parent)

                    # Emit completion checkpoint (iterations == max_iterations indicates max iterations reached)
                    # Skip if executor already manages lifecycle (to avoid duplicate events)
                    if context and not getattr(context, '_executor_managed_lifecycle', False):
                        context.emit(AgentCompleted(
                            name=self.name,
                            correlation_id=agent_correlation_id,
                            parent_correlation_id=context._parent_correlation_id,
                            iterations=self.max_iterations,
                            tool_calls_count=len(all_tool_calls),
                            output_length=len(final_output),
                            metadata={"name": self.name},
                        ))

                    # Add output data to span for trace visibility
                    span.set_attribute("output.data", _serialize_span_data(final_output))

                    return AgentResult(
                        output=final_output,
                        tool_calls=all_tool_calls,
                        context=context,
                    )
            except Exception as e:
                # Reset parent to workflow before emitting agent.failed
                context.restore_parent(original_agent_parent)

                # Emit error checkpoint for observability
                # Skip if executor already manages lifecycle (to avoid duplicate events)
                if context and not getattr(context, '_executor_managed_lifecycle', False):
                    context.emit(AgentFailed(
                        name=self.name,
                        correlation_id=agent_correlation_id,
                        parent_correlation_id=context._parent_correlation_id,
                        error_code=type(e).__name__,
                        error_message=str(e),
                        iterations=iteration if 'iteration' in locals() else 0,
                        metadata={"name": self.name},
                    ))
                raise
        finally:
            # Always reset context to prevent leakage between agent executions
            if self.sandbox is not None:
                close = getattr(self.sandbox, "close", None)
                if close is not None:
                    await close()
            from ..context import _current_context
            _current_context.reset(token)

    async def resume_from_hitl(
        self,
        context: Context,
        agent_context: Dict,
        user_response: str,
    ) -> AgentResult:
        """
        Resume agent execution after HITL pause.

        This method reconstructs agent state from the checkpoint and injects
        the user's response as the successful tool result, then continues
        the conversation loop.

        Args:
            context: Current execution context (workflow or agent)
            agent_context: Agent state from WaitingForUserInputException.agent_context
            user_response: User's answer to the HITL question

        Returns:
            AgentResult with final output and tool calls
        """
        self.logger.info(f"Resuming agent '{self.name}' from HITL pause")

        # 1. Restore conversation state
        messages = [
            Message(role=lm.MessageRole(msg["role"]), content=msg["content"])
            for msg in agent_context["messages"]
        ]
        iteration = agent_context["iteration"]
        all_tool_calls = agent_context["all_tool_calls"]

        # 2. Restore partial tool results for current iteration
        tool_results = agent_context["tool_results"]

        # 3. Inject user response as successful tool result
        pending_tool = agent_context["pending_tool_call"]
        tool_results.append({
            "tool": pending_tool["name"],
            "result": serialize_to_str(user_response),
            "error": None,
        })

        self.logger.debug(
            f"Injected user response for tool '{pending_tool['name']}': {user_response}"
        )

        # 4. Add tool results to conversation
        results_text = "\n".join([
            f"Tool: {tr['tool']}\nResult: {tr['result']}"
            if tr["error"] is None
            else f"Tool: {tr['tool']}\nError: {tr['error']}"
            for tr in tool_results
        ])
        messages.append(Message.user(
            f"Tool results:\n{results_text}\n\n"
            f"Please provide your final answer based on these results."
        ))

        # 5. Continue agent execution loop from next iteration
        return await self._continue_execution_from_iteration(
            context=context,
            messages=messages,
            iteration=iteration + 1,  # Next iteration
            all_tool_calls=all_tool_calls,
        )

    async def _continue_execution_from_iteration(
        self,
        context: Context,
        messages: List[Message],
        iteration: int,
        all_tool_calls: List[Dict],
    ) -> AgentResult:
        """
        Continue agent execution from a specific iteration.

        This is the core execution loop extracted to support both:
        1. Normal execution (starting from iteration 0)
        2. Resume after HITL (starting from iteration N)

        Args:
            context: Execution context
            messages: Conversation history
            iteration: Starting iteration number
            all_tool_calls: Accumulated tool calls

        Returns:
            AgentResult with output and tool calls
        """
        # Generate correlation_id for pairing agent.started ↔ agent.completed/failed
        agent_correlation_id = generate_cid()

        # Set agent as parent for iteration events (no agent.started emit since this is a continuation)
        original_agent_parent = context.set_as_parent(agent_correlation_id)

        # Prepare tool definitions
        tool_defs = [
            ToolDefinition(
                name=name,
                description=tool.description or f"Tool: {name}",
                parameters=tool.input_schema if hasattr(tool, "input_schema") else {},
            )
            for name, tool in self.tools.items()
        ]

        # Main iteration loop (continue from specified iteration)
        while iteration < self.max_iterations:
            self.logger.debug(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            request = GenerateRequest(
                model=self.model if self._language_model is None else "mock-model",
                system_prompt=self._compose_system_prompt(),
                messages=messages,
                tools=tool_defs if tool_defs else [],
            )
            self._apply_generation_config(request)

            response = await self._generate_with_callbacks(
                context=context,
                request=request,
                iteration=iteration + 1,
                messages=messages,
                tool_definitions=tool_defs,
            )

            # Add assistant response to messages
            messages.append(Message.assistant(response.text))

            # Check if LLM wants to use tools
            if response.tool_calls:
                self.logger.debug(f"Agent calling {len(response.tool_calls)} tool(s)")

                # Store current conversation in context for potential handoffs
                if not hasattr(context, '_agent_data'):
                    context._agent_data = {}
                context._agent_data["_current_conversation"] = messages

                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args_str = tool_call["arguments"]

                    # Track tool call
                    all_tool_calls.append({
                        "name": tool_name,
                        "arguments": tool_args_str,
                        "iteration": iteration + 1,
                    })

                    # Execute tool
                    try:
                        # Parse arguments
                        tool_args = json.loads(tool_args_str)

                        # Get tool
                        tool = self.tools.get(tool_name)
                        if not tool:
                            result_text = f"Error: Tool '{tool_name}' not found"
                        else:
                            result = await self._invoke_tool_with_callbacks(
                                ToolCallbackContext(
                                    agent=self,
                                    context=context,
                                    iteration=iteration + 1,
                                    tool_name=tool_name,
                                    tool_call_id=tool_call.get("id", ""),
                                    tool_call=tool_call,
                                    arguments=tool_args,
                                    tool=tool,
                                )
                            )

                            # Check if this was a handoff
                            if isinstance(result, dict) and result.get("_handoff"):
                                self.logger.info(
                                    f"Handoff detected to '{result['to_agent']}', "
                                    f"terminating current agent"
                                )
                                # Save conversation before returning
                                if isinstance(context, AgentContext):
                                    await context.save_conversation_history(messages)
                                # Return immediately with handoff result
                                return AgentResult(
                                    output=result["output"],
                                    tool_calls=all_tool_calls + result.get("tool_calls", []),
                                    context=context,
                                    handoff_to=result["to_agent"],
                                    handoff_metadata=result,
                                )

                            result_text = _serialize_tool_result(result)

                        tool_results.append(
                            {"tool": tool_name, "result": result_text, "error": None}
                        )

                    except WaitingForUserInputException as e:
                        # HITL PAUSE: Capture agent state and propagate exception
                        self.logger.info(f"Agent pausing for user input at iteration {iteration}")

                        # Serialize messages to dict format
                        messages_dict = [
                            {"role": msg.role.value, "content": msg.content}
                            for msg in messages
                        ]

                        # Enhance exception with agent execution context
                        raise WaitingForUserInputException(
                            question=e.question,
                            input_type=e.input_type,
                            options=e.options,
                            checkpoint_state=e.checkpoint_state,
                            pause_index=e.pause_index,
                            step_name=e.step_name,
                            step_correlation_id=e.step_correlation_id,
                            allow_custom=e.allow_custom,
                            skippable=e.skippable,
                            checkpoint_metadata=e.checkpoint_metadata,
                            agent_context={
                                "agent_name": self.name,
                                "iteration": iteration,
                                "messages": messages_dict,
                                "tool_results": tool_results,
                                "pending_tool_call": {
                                    "name": tool_call["name"],
                                    "arguments": tool_call["arguments"],
                                    "tool_call_index": response.tool_calls.index(tool_call),
                                },
                                "all_tool_calls": all_tool_calls,
                                "model_config": self._model_config_snapshot(),
                            },
                        ) from e

                    except Exception as e:
                        # Regular tool errors - log and continue
                        self.logger.error(f"Tool execution error: {e}")
                        tool_results.append(
                            {"tool": tool_name, "result": None, "error": str(e)}
                        )

                # Add tool results to conversation
                results_text = "\n".join([
                    f"Tool: {tr['tool']}\nResult: {tr['result']}"
                    if tr["error"] is None
                    else f"Tool: {tr['tool']}\nError: {tr['error']}"
                    for tr in tool_results
                ])
                messages.append(Message.user(
                    f"Tool results:\n{results_text}\n\n"
                    f"Please provide your final answer based on these results."
                ))

                # Continue loop for agent to process results

            else:
                # No tool calls - agent is done
                self.logger.debug(f"Agent completed after {iteration + 1} iterations")
                # Save conversation before returning
                if isinstance(context, AgentContext):
                    await context.save_conversation_history(messages)

                # Reset parent to workflow before emitting agent.completed
                context.restore_parent(original_agent_parent)

                # Emit completion checkpoint
                # Skip if executor already manages lifecycle (to avoid duplicate events)
                if context and not getattr(context, '_executor_managed_lifecycle', False):
                    context.emit(AgentCompleted(
                        name=self.name,
                        correlation_id=agent_correlation_id,
                        parent_correlation_id=context._parent_correlation_id,
                        iterations=iteration + 1,
                        tool_calls_count=len(all_tool_calls),
                        output_length=len(response.text),
                        metadata={"name": self.name},
                    ))

                return AgentResult(
                    output=response.text,
                    tool_calls=all_tool_calls,
                    context=context,
                )

            iteration += 1

        # Max iterations reached
        self.logger.warning(f"Agent reached max iterations ({self.max_iterations})")
        final_output = messages[-1].content if messages else "No output generated"
        # Save conversation before returning
        if isinstance(context, AgentContext):
            await context.save_conversation_history(messages)

        # Reset parent to workflow before emitting agent.completed
        context.restore_parent(original_agent_parent)

        # Emit completion checkpoint (iterations == max_iterations indicates max iterations reached)
        # Skip if executor already manages lifecycle (to avoid duplicate events)
        if context and not getattr(context, '_executor_managed_lifecycle', False):
            context.emit(AgentCompleted(
                name=self.name,
                correlation_id=agent_correlation_id,
                parent_correlation_id=context._parent_correlation_id,
                iterations=self.max_iterations,
                tool_calls_count=len(all_tool_calls),
                output_length=len(final_output),
                metadata={"name": self.name},
            ))

        return AgentResult(
            output=final_output,
            tool_calls=all_tool_calls,
            context=context,
        )
