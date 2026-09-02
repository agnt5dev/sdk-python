"""
Tool component for Agent capabilities with automatic schema extraction.

Tools extend what agents can do by providing structured interfaces to functions,
with automatic schema generation from Python type hints and docstrings.
"""

import asyncio
import functools
import inspect
import json
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    get_args,
    get_origin,
)

from docstring_parser import parse as parse_docstring

from ._serialization import deserialize, serialize, serialize_to_str
from ._telemetry import setup_module_logger, truncate_span_attribute_value
from .activation import (
    ActivationKind,
    ActivationRecoveryPolicy,
    _reset_current_activation,
    _set_current_activation,
    activation_request_from_context,
)
from .context import Context, set_current_context
from .exceptions import ConfigurationError

if TYPE_CHECKING:
    from .workflow import WorkflowContext

logger = setup_module_logger(__name__)

T = TypeVar("T")
ToolHandler = Callable[..., Awaitable[T]]


def _coerce_recovery_policy(
    value: ActivationRecoveryPolicy | str,
) -> ActivationRecoveryPolicy:
    if isinstance(value, ActivationRecoveryPolicy):
        return value
    normalized = value.strip().lower()
    mapping = {
        "idempotent_retry": ActivationRecoveryPolicy.IDEMPOTENT_RETRY,
        "durable_steps": ActivationRecoveryPolicy.DURABLE_STEPS,
        "unknown_outcome": ActivationRecoveryPolicy.UNKNOWN_OUTCOME,
        "compensate": ActivationRecoveryPolicy.COMPENSATE,
        "fail": ActivationRecoveryPolicy.FAIL,
    }
    try:
        return mapping[normalized]
    except KeyError as error:
        raise ConfigurationError(f"Unsupported tool recovery policy: {value!r}") from error


def _serialize_for_span(value: Any) -> str:
    """Serialize a value to JSON string for span attributes.

    Handles Pydantic models, dataclasses, and other complex types.

    Args:
        value: The value to serialize

    Returns:
        JSON string representation of the value
    """
    if value is None:
        return "null"

    # Use centralized serialization that handles Pydantic models, dataclasses, etc.
    try:
        serialized = serialize_to_str(value)
    except (TypeError, ValueError):
        serialized = repr(value)
    return truncate_span_attribute_value(serialized)


def _consume_eval_tool_fault(ctx: Context, tool_name: str) -> Optional[str]:
    metadata = getattr(ctx, "_trace_metadata", None) or getattr(ctx, "metadata", None) or {}
    if not isinstance(metadata, dict) or metadata.get("agnt5_eval_role") != "target":
        return None
    raw = metadata.get("agnt5.eval.tool_faults") or metadata.get("agnt5_eval_tool_faults")
    if not raw:
        return None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None

    counts = getattr(ctx, "_eval_tool_fault_counts", None)
    if counts is None:
        counts = {}
        setattr(ctx, "_eval_tool_fault_counts", counts)
    call_count = int(counts.get(tool_name, 0))
    counts[tool_name] = call_count + 1

    for spec in parsed:
        if not isinstance(spec, dict):
            continue
        candidate = str(spec.get("tool") or spec.get("name") or "").strip()
        if candidate != tool_name:
            continue
        after_calls = _nonnegative_int(spec.get("after_calls", spec.get("afterCalls", 0)), 0)
        times = _nonnegative_int(spec.get("times", 1), 1)
        if call_count < after_calls or call_count >= after_calls + times:
            continue
        code = str(
            spec.get("error_code") or spec.get("errorCode") or "AGNT5_EVAL_TOOL_FAULT"
        ).strip()
        message = str(spec.get("message") or f"simulated eval tool fault for {tool_name}").strip()
        return f"{code}: {message}"
    return None


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """
    Convert Python type to JSON Schema.

    Args:
        py_type: Python type annotation

    Returns:
        JSON Schema dict with type and potentially items/properties
    """
    # Handle None/NoneType
    if py_type is None or py_type is type(None):
        return {"type": "null"}

    # Get origin and args for generic types
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle Optional[T] -> unwrap to T
    if origin is type(None.__class__):  # Union type
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])
        # Multiple non-None types -> just use first one
        if non_none_args:
            return _python_type_to_json_schema(non_none_args[0])
        return {"type": "null"}

    # Handle List[T] - need to specify items
    if origin is list or (origin is None and py_type is list):
        if args:
            # List[T] - extract T and create items schema
            item_type = args[0]
            return {"type": "array", "items": _python_type_to_json_schema(item_type)}
        else:
            # Bare list without type parameter
            return {
                "type": "array",
                "items": {"type": "string"},  # Default to string items
            }

    # Handle Dict[K, V] - basic object schema
    if origin is dict or (origin is None and py_type is dict):
        return {"type": "object"}

    # Handle basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        Any: "string",  # Default to string for Any
    }

    # Direct type match
    json_type = type_map.get(py_type, "string")
    return {"type": json_type}


def _extract_schema_from_function(func: Callable) -> Dict[str, Any]:
    """
    Extract JSON schema from function signature and docstring.

    Args:
        func: Function to extract schema from

    Returns:
        Dict containing input_schema and output_schema
    """
    # Parse function signature
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    parsed_doc = parse_docstring(docstring)

    # Build parameter schemas
    properties = {}
    required = []

    # Build mapping from param name to docstring description
    param_descriptions = {}
    if parsed_doc.params:
        for param_doc in parsed_doc.params:
            param_descriptions[param_doc.arg_name] = param_doc.description or ""

    for param_name, param in sig.parameters.items():
        # Skip 'ctx' parameter (Context is auto-injected)
        if param_name == "ctx":
            continue

        # Get type annotation
        param_type = param.annotation
        if param_type == inspect.Parameter.empty:
            param_type = Any

        # Get description from docstring
        description = param_descriptions.get(param_name, "")

        # Build parameter schema using full JSON Schema conversion
        param_schema = _python_type_to_json_schema(param_type)
        param_schema["description"] = description

        properties[param_name] = param_schema

        # Check if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Build input schema
    input_schema = {"type": "object", "properties": properties, "required": required}

    # Extract return type for output schema (optional for basic tool functionality)
    return_type = sig.return_annotation
    output_schema = None
    if return_type != inspect.Parameter.empty:
        output_schema = _python_type_to_json_schema(return_type)

    return {"input_schema": input_schema, "output_schema": output_schema}


class Tool:
    """
    Represents a tool that agents can use.

    Tools wrap functions with automatic schema extraction and provide
    a structured interface for agent invocation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        handler: ToolHandler,
        input_schema: Optional[Dict[str, Any]] = None,
        confirmation: bool = False,
        auto_schema: bool = False,
        recovery_policy: ActivationRecoveryPolicy | str = (
            ActivationRecoveryPolicy.UNKNOWN_OUTCOME
        ),
        durable: bool = True,
    ):
        """
        Initialize a Tool.

        Args:
            name: Tool name
            description: Tool description for agents
            handler: Function that implements the tool
            input_schema: Manual JSON schema for input parameters
            confirmation: Whether tool requires human confirmation before execution
            auto_schema: Whether to automatically extract schema from handler
        """
        self.name = name
        self.description = description
        self.handler = handler
        self.confirmation = confirmation
        self.recovery_policy = _coerce_recovery_policy(recovery_policy)
        self.durable = durable

        # Extract or use provided schema
        if auto_schema:
            schemas = _extract_schema_from_function(handler)
            self.input_schema = schemas["input_schema"]
            self.output_schema = schemas.get("output_schema")
        else:
            self.input_schema = input_schema or {"type": "object", "properties": {}}
            self.output_schema = None

        # Validate handler signature
        self._validate_handler()

        logger.debug(f"Created tool '{name}' with auto_schema={auto_schema}")

    def _validate_handler(self) -> None:
        """Validate that handler has correct signature."""
        sig = inspect.signature(self.handler)
        params = list(sig.parameters.values())

        if not params:
            raise ConfigurationError(
                f"Tool handler '{self.name}' must have at least one parameter (ctx: Context)"
            )

        first_param = params[0]
        if first_param.annotation != Context and first_param.annotation != inspect.Parameter.empty:
            logger.warning(
                f"Tool handler '{self.name}' first parameter should be 'ctx: Context', "
                f"got '{first_param.annotation}'"
            )

    async def invoke(self, ctx: Context, **kwargs) -> Any:
        return await self._invoke(ctx, kwargs, stable_key=None)

    async def invoke_with_stable_key(
        self,
        ctx: Context,
        arguments: Dict[str, Any],
        *,
        stable_key: Optional[str],
        tool_call_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Any:
        """Invoke using a provider/tool-call identity when one is available."""

        return await self._invoke(
            ctx,
            arguments,
            stable_key=stable_key,
            tool_call_id=tool_call_id,
            iteration=iteration,
        )

    def _activation_enabled(self, ctx: Any) -> bool:
        """Whether an invocation on this context runs through a TOOL activation.

        When it does, the runtime journals the activation as the tool_call.*
        record and callers must not emit their own lifecycle events.
        """

        metadata = getattr(ctx, "_trace_metadata", None) or {}
        return (
            self.durable
            and metadata.get("durable_activation_v1") == "true"
            and getattr(ctx, "_activation_client", None) is not None
        )

    async def _invoke(
        self,
        ctx: Context,
        kwargs: Dict[str, Any],
        *,
        stable_key: Optional[str],
        allow_activation: bool = True,
        allow_memo: bool = True,
        tool_call_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Any:
        """
        Invoke the tool with given arguments.

        Args:
            ctx: Execution context
            **kwargs: Tool arguments matching input_schema

        Returns:
            Tool execution result

        Raises:
            ConfigurationError: If tool requires confirmation (not yet implemented)

        Note:
            If memoization is enabled on the context, this method will check
            the journal for cached results before executing and cache results
            after successful execution.
        """
        if allow_activation and self._activation_enabled(ctx):
            return await self._invoke_durable(
                ctx,
                kwargs,
                stable_key=stable_key,
                activation_client=ctx._activation_client,
                tool_call_id=tool_call_id,
                iteration=iteration,
            )

        injected_fault = _consume_eval_tool_fault(ctx, self.name)
        if injected_fault:
            raise RuntimeError(injected_fault)

        # Check for memoization before tool execution
        step_key = None
        content_hash = None
        memo = None

        if allow_memo and hasattr(ctx, "_memo") and ctx._memo:
            memo = ctx._memo
            step_key, content_hash = memo.tool_call_key(self.name, kwargs)

            # Check cache first - skip execution if cached
            found, cached_result = await memo.get_cached_tool_result(step_key, content_hash)
            if found:
                logger.debug(f"Tool call {step_key} served from memoization cache")
                return cached_result

        if self.confirmation:
            logger.warning(
                f"Tool '{self.name}' requires confirmation; no approval handler is configured"
            )

        # Set context in task-local storage for automatic propagation to nested calls
        token = set_current_context(ctx)
        try:
            try:
                # Create span for tool execution (uses contextvar for async-safe parent-child linking)
                from .tracing import create_span

                logger.debug(f"Invoking tool '{self.name}' with args: {list(kwargs.keys())}")

                # Create span with runtime_context for parent-child span linking
                # contextvar handles proper nesting for parallel async operations
                with create_span(
                    self.name,
                    "tool",
                    ctx._runtime_context if hasattr(ctx, "_runtime_context") else None,
                    {
                        "tool.name": self.name,
                        "tool.args": ",".join(kwargs.keys()),
                        "input.data": _serialize_for_span(kwargs),
                    },
                ) as span:
                    # Handler is already async (validated in tool() decorator)
                    result = await self.handler(ctx, **kwargs)

                    # Add output data to span for trace visibility
                    span.set_attribute("output.data", _serialize_for_span(result))

                    logger.debug(f"Tool '{self.name}' completed successfully")

                    # Cache result for replay if memoization is enabled
                    if memo and step_key:
                        await memo.cache_tool_result(step_key, content_hash, result)

                    return result
            except Exception:
                # Don't emit ToolCallFailed here - the agent's _run_core() already
                # emits it with proper correlation_id tracking. Emitting here would
                # create an orphaned failed event with a different correlation_id.
                # Agent calls emit correlated failure events in the agent execution path.
                raise
        finally:
            # Always reset context to prevent leakage
            from .context import _current_context

            _current_context.reset(token)

    async def _invoke_durable(
        self,
        ctx: Context,
        arguments: Dict[str, Any],
        *,
        stable_key: Optional[str],
        activation_client: Any,
        tool_call_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> Any:
        logical_key = (
            f"tool:{self.name}:{stable_key}"
            if stable_key
            else ctx.allocate_activation_key("tool", self.name)
        )
        request = activation_request_from_context(
            ctx,
            kind=ActivationKind.TOOL,
            stable_key=logical_key,
            input_value={"name": self.name, "arguments": arguments},
            recovery_policy=self.recovery_policy,
            display_name=self.name,
            input_data={
                "name": self.name,
                "arguments": arguments,
                "tool_call_id": tool_call_id or "",
                "iteration": iteration,
            },
        )
        activation_token = None
        started_at = time.monotonic()

        def on_admitted(decision):
            nonlocal activation_token
            activation_token = _set_current_activation(decision)

        async def execute():
            return await self._invoke(
                ctx,
                arguments,
                stable_key=stable_key,
                allow_activation=False,
                allow_memo=False,
            )

        retryable = self.recovery_policy in {
            ActivationRecoveryPolicy.IDEMPOTENT_RETRY,
            ActivationRecoveryPolicy.DURABLE_STEPS,
        }
        try:
            result, _receipt = await activation_client.run(
                request,
                execute,
                encode_output=serialize,
                decode_output=deserialize,
                latency_ms=lambda: int((time.monotonic() - started_at) * 1000),
                on_admitted=on_admitted,
                failure_error_code="TOOL_FAILED",
                failure_retryable=retryable,
                failure_external_outcome_certainty="UNKNOWN",
            )
            return result
        finally:
            if activation_token is not None:
                _reset_current_activation(activation_token)

    def get_schema(self) -> Dict[str, Any]:
        """
        Get complete tool schema for agent consumption.

        Returns:
            Dict with name, description, and input_schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "requires_confirmation": self.confirmation,
        }

    async def __call__(self, ctx: Context, **kwargs) -> Any:
        """Make Tool callable - allows using decorated tools as functions.

        This enables the @tool decorator to return a Tool instance while
        still allowing it to be called like a function.

        Example:
            @tool
            async def my_tool(ctx: Context, x: int) -> int:
                return x * 2

            # my_tool is a Tool instance but callable:
            result = await my_tool(ctx, x=5)

        Args:
            ctx: Execution context
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        return await self.invoke(ctx, **kwargs)


class ToolRegistry:
    """Global registry for tools."""

    _tools: Dict[str, Tool] = {}

    @classmethod
    def register(cls, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in cls._tools:
            logger.warning(f"Overwriting existing tool '{tool.name}'")
        cls._tools[tool.name] = tool
        logger.debug(f"Registered tool '{tool.name}'")

    @classmethod
    def get(cls, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return cls._tools.get(name)

    @classmethod
    def all(cls) -> Dict[str, Tool]:
        """Get all registered tools."""
        return cls._tools.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tools (for testing)."""
        cls._tools.clear()
        logger.debug("Cleared tool registry")

    @classmethod
    def list_names(cls) -> List[str]:
        """Get list of all tool names."""
        return list(cls._tools.keys())


def tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    auto_schema: bool = True,
    confirmation: bool = False,
    input_schema: Optional[Dict[str, Any]] = None,
    recovery_policy: ActivationRecoveryPolicy | str = (
        ActivationRecoveryPolicy.UNKNOWN_OUTCOME
    ),
    durable: bool = True,
) -> Callable[..., Any]:
    """
    Decorator to mark a function as a tool with automatic schema extraction.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to first line of docstring)
        auto_schema: Automatically extract schema from type hints and docstring
        confirmation: Whether tool requires confirmation before execution
        input_schema: Manual schema (only if auto_schema=False)
        recovery_policy: Interrupted-work policy. Ordinary tools default to
            ``unknown_outcome``; explicitly idempotent or nested durable tools
            may opt into ``idempotent_retry`` or ``durable_steps``.
        durable: Whether this callable is itself a tool activation. Delegation
            adapters use a child activation instead.

    Returns:
        Decorated function that can be invoked as a tool

    Example:
        ```python
        @tool
        def search_web(ctx: Context, query: str, max_results: int = 10) -> List[Dict]:
            \"\"\"Search the web for information.

            Args:
                query: The search query string
                max_results: Maximum number of results to return

            Returns:
                List of search results
            \"\"\"
            # Implementation
            return results
        ```
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Determine tool name
        tool_name = name or func.__name__

        # Extract description from docstring if not provided
        tool_description = description
        if tool_description is None:
            docstring = inspect.getdoc(func)
            if docstring:
                parsed_doc = parse_docstring(docstring)
                tool_description = parsed_doc.short_description or parsed_doc.long_description or ""
            else:
                tool_description = ""

        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        if not params:
            raise ConfigurationError(
                f"Tool function '{func.__name__}' must have at least one parameter (ctx: Context)"
            )

        first_param = params[0]
        if first_param.annotation != Context and first_param.annotation != inspect.Parameter.empty:
            raise ConfigurationError(
                f"Tool function '{func.__name__}' first parameter must be 'ctx: Context', "
                f"got '{first_param.annotation}'"
            )

        # Convert sync to async if needed
        if not asyncio.iscoroutinefunction(func):
            original_func = func

            @functools.wraps(original_func)
            async def async_wrapper(*args, **kwargs):
                # Run sync function in thread pool to prevent blocking event loop
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: original_func(*args, **kwargs))

            handler_func = async_wrapper
        else:
            handler_func = func

        # Create Tool instance
        tool_instance = Tool(
            name=tool_name,
            description=tool_description,
            handler=handler_func,
            input_schema=input_schema,
            confirmation=confirmation,
            auto_schema=auto_schema,
            recovery_policy=recovery_policy,
            durable=durable,
        )

        # Register tool
        ToolRegistry.register(tool_instance)

        # Copy function metadata to Tool for inspection
        # This allows isinstance(my_tool, Tool) while preserving function identity
        tool_instance.__name__ = func.__name__
        tool_instance.__doc__ = func.__doc__
        tool_instance.__module__ = func.__module__

        return tool_instance

    if _func is None:
        return decorator
    return decorator(_func)


# ============================================================================
# Built-in Human-in-the-Loop Tools
# ============================================================================


class AskUserTool(Tool):
    """
    Built-in tool that agents can use to request text input from users.

    This tool pauses the workflow execution and waits for the user to provide
    a text response. The workflow resumes when the user submits their input.

    Example:
        ```python
        from agnt5 import Agent, workflow, WorkflowContext
        from agnt5.tool import AskUserTool

        @workflow(chat=True)
        async def agent_with_hitl(ctx: WorkflowContext, query: str) -> dict:
            agent = Agent(
                name="research_agent",
                model="openai/gpt-4o-mini",
                instructions="You are a research assistant.",
                tools=[AskUserTool(ctx)]
            )

            result = await agent.run(query, context=ctx)
            return {"response": result.output}
        ```
    """

    def __init__(self, context: Optional["WorkflowContext"] = None):  # type: ignore
        """
        Initialize AskUserTool.

        Args:
            context: Optional workflow context with wait_for_user capability.
                     If not provided, will attempt to get from task-local contextvar.
        """
        # Import here to avoid circular dependency
        from .workflow import WorkflowContext

        if context is not None and not isinstance(context, WorkflowContext):
            raise ConfigurationError(
                "AskUserTool requires a WorkflowContext. "
                "This tool can only be used within workflows."
            )

        super().__init__(
            name="ask_user",
            description="Ask the user a question and wait for their text response",
            handler=self._handler,
            auto_schema=True,
            durable=False,
        )
        self.context = context

    async def _handler(self, ctx: Context, question: str) -> str:
        """
        Ask user a question and wait for their response.

        Args:
            ctx: Execution context (may contain WorkflowContext via contextvar)
            question: Question to ask the user

        Returns:
            User's text response
        """
        # Import here to avoid circular dependency
        from .context import get_current_context
        from .workflow import WorkflowContext

        # Use explicit context if provided during __init__
        workflow_ctx = self.context

        # If not provided, try to get from task-local contextvar
        if workflow_ctx is None:
            current = get_current_context()
            if isinstance(current, WorkflowContext):
                workflow_ctx = current
            elif hasattr(current, "_workflow_entity"):
                # Current context has workflow entity (is WorkflowContext)
                workflow_ctx = current  # type: ignore

        if workflow_ctx is None:
            raise ConfigurationError(
                "AskUserTool requires WorkflowContext. "
                "Either pass context to __init__ or ensure tool is used within a workflow."
            )

        return await workflow_ctx.wait_for_user(question, input_type="text")


class RequestApprovalTool(Tool):
    """
    Built-in tool that agents can use to request approval from users.

    This tool pauses the workflow execution and presents an approval request
    to the user with approve/reject options. The workflow resumes when the
    user makes a decision.

    Example:
        ```python
        from agnt5 import Agent, workflow, WorkflowContext
        from agnt5.tool import RequestApprovalTool

        @workflow(chat=True)
        async def deployment_agent(ctx: WorkflowContext, changes: dict) -> dict:
            agent = Agent(
                name="deploy_agent",
                model="openai/gpt-4o-mini",
                instructions="You help deploy code changes safely.",
                tools=[RequestApprovalTool(ctx)]
            )

            result = await agent.run(
                f"Review and deploy these changes: {changes}",
                context=ctx
            )
            return {"response": result.output}
        ```
    """

    def __init__(self, context: Optional["WorkflowContext"] = None):  # type: ignore
        """
        Initialize RequestApprovalTool.

        Args:
            context: Optional workflow context with wait_for_user capability.
                     If not provided, will attempt to get from task-local contextvar.
        """
        # Import here to avoid circular dependency
        from .workflow import WorkflowContext

        if context is not None and not isinstance(context, WorkflowContext):
            raise ConfigurationError(
                "RequestApprovalTool requires a WorkflowContext. "
                "This tool can only be used within workflows."
            )

        super().__init__(
            name="request_approval",
            description="Request user approval for an action before proceeding",
            handler=self._handler,
            auto_schema=True,
            durable=False,
        )
        self.context = context

    async def _handler(self, ctx: Context, action: str, details: str = "") -> str:
        """
        Request approval from user for an action.

        Args:
            ctx: Execution context (may contain WorkflowContext via contextvar)
            action: The action requiring approval
            details: Additional details about the action

        Returns:
            "approve" or "reject" based on user's decision
        """
        # Import here to avoid circular dependency
        from .context import get_current_context
        from .workflow import WorkflowContext

        # Use explicit context if provided during __init__
        workflow_ctx = self.context

        # If not provided, try to get from task-local contextvar
        if workflow_ctx is None:
            current = get_current_context()
            if isinstance(current, WorkflowContext):
                workflow_ctx = current
            elif hasattr(current, "_workflow_entity"):
                # Current context has workflow entity (is WorkflowContext)
                workflow_ctx = current  # type: ignore

        if workflow_ctx is None:
            raise ConfigurationError(
                "RequestApprovalTool requires WorkflowContext. "
                "Either pass context to __init__ or ensure tool is used within a workflow."
            )

        question = f"Action: {action}"
        if details:
            question += f"\n\nDetails:\n{details}"
        question += "\n\nDo you approve?"

        return await workflow_ctx.wait_for_user(
            question,
            input_type="approval",
            options=[{"id": "approve", "label": "Approve"}, {"id": "reject", "label": "Reject"}],
        )
