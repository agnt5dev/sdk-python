"""
Tests for Agent component.

Tests cover:
- Agent creation and configuration
- Agent execution with mock LLM
- Tool orchestration
- Multi-turn conversations via AgentContext
- Handoff mechanisms
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agnt5 import Agent, tool, Context
from agnt5.agent import AgentResult, AgentContext, AgentRegistry, Handoff, handoff
from agnt5.events import Event
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, Message, MessageRole, TokenUsage
from agnt5.lm.events import LMContentBlockStarted, LMContentBlockDelta, LMContentBlockCompleted, LMCompleted
from agnt5.tool import ToolRegistry


# Mock Language Model for testing
class MockLanguageModel(LanguageModel):
    """Mock LLM for testing."""

    def __init__(self, responses=None, tool_calls=None):
        """Initialize mock LLM.

        Args:
            responses: List of text responses to return
            tool_calls: List of tool calls to simulate
        """
        self.responses = responses or ["Mock response"]
        self.tool_calls_list = tool_calls or []
        self.call_count = 0
        self.requests = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate mock response."""
        self.requests.append(request)

        # Get response for this call
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]

        # Get tool calls if any
        tool_calls = None
        if self.call_count < len(self.tool_calls_list):
            tool_calls = self.tool_calls_list[self.call_count]

        self.call_count += 1

        return GenerateResponse(
            text=response_text,
            tool_calls=tool_calls,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def stream(self, request: GenerateRequest):
        """Mock stream that yields proper Event objects."""
        # Get response for this call (use same logic as generate)
        response_text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1

        # Yield content block start
        yield LMContentBlockStarted(
            name="mock-model",
            correlation_id="mock-corr",
            parent_correlation_id="mock-parent",
            block_type="text",
            index=0,
        )

        # Yield delta with full text
        yield LMContentBlockDelta(
            name="mock-model",
            correlation_id="mock-corr",
            parent_correlation_id="mock-parent",
            content=response_text,
            block_type="text",
            index=0,
        )

        # Yield content block stop
        yield LMContentBlockCompleted(
            name="mock-model",
            correlation_id="mock-corr",
            parent_correlation_id="mock-parent",
            block_type="text",
            index=0,
        )

        # Yield completion event
        yield LMCompleted(
            name="mock-model",
            correlation_id="mock-corr",
            parent_correlation_id="mock-parent",
            model="mock-model",
            provider="mock",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            output_data={"text": response_text},
        )


# Test fixtures
@pytest.fixture(autouse=True)
def clear_registries():
    """Clear registries before each test."""
    ToolRegistry.clear()
    AgentRegistry.clear()
    yield
    ToolRegistry.clear()
    AgentRegistry.clear()


@pytest.fixture
def mock_lm():
    """Create mock language model."""
    return MockLanguageModel()


@pytest.fixture
def sample_tool():
    """Create sample tool for testing."""

    @tool
    async def test_tool(ctx: Context, value: int) -> int:
        """Test tool that doubles a value."""
        return value * 2

    return test_tool


# Test Agent Creation


def test_agent_creation(mock_lm):
    """Test basic agent creation."""
    agent = Agent(
        name="test_agent",
        model=mock_lm,
        instructions="You are a test agent",
    )

    assert agent.name == "test_agent"
    assert agent.instructions == "You are a test agent"
    assert agent._language_model is mock_lm  # LanguageModel stored in _language_model
    assert len(agent.tools) == 0


def test_agent_creation_with_string_model():
    """Test agent creation with string model (new API)."""
    agent = Agent(
        name="string_model_agent",
        model="openai/gpt-4o-mini",
        instructions="Test agent with string model",
    )

    assert agent.name == "string_model_agent"
    assert agent.model == "openai/gpt-4o-mini"
    assert agent.model_name == "openai/gpt-4o-mini"


def test_agent_with_tools(mock_lm, sample_tool):
    """Test agent creation with tools."""
    agent = Agent(
        name="tool_agent",
        model=mock_lm,
        instructions="Agent with tools",
        tools=[sample_tool],
    )

    assert len(agent.tools) == 1
    assert "test_tool" in agent.tools


def test_agent_configuration(mock_lm):
    """Test agent configuration options."""
    agent = Agent(
        name="configured_agent",
        model=mock_lm,
        instructions="Configured agent",
        model_name="gpt-4",
        temperature=0.5,
        max_iterations=20,
    )

    assert agent.model_name == "gpt-4"
    assert agent.temperature == 0.5
    assert agent.max_iterations == 20


# Test Agent Execution (with new string-based model API)


class MockAsyncStreamIterator:
    """Mock async iterator for stream_iter that yields chunks."""
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


@pytest.mark.asyncio
async def test_agent_run_simple():
    """Test simple agent run without tools using string model."""
    # Mock the Rust LLM with both generate and stream responses
    mock_response = MagicMock()
    mock_response.text = "Hello! How can I help you?"
    mock_response.content = "Hello! How can I help you?"
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    mock_response.tool_calls = None
    mock_response.object = None

    # Create mock stream chunks (Rust stream_iter returns an async iterator)
    class MockChunk:
        def __init__(self, chunk_type, text="", block_type=None, index=0):
            self.chunk_type = chunk_type
            self.text = text
            self.block_type = block_type
            self.index = index
            self.model = "mock"
            self.finish_reason = "stop" if chunk_type == "completed" else None
            self.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30) if chunk_type == "completed" else None

    mock_chunks = [
        MockChunk("content_block_start", block_type="text", index=0),
        MockChunk("delta", text="Hello! How can I help you?", block_type="text", index=0),
        MockChunk("content_block_stop", index=0),
        MockChunk("completed", text="Hello! How can I help you?"),
    ]

    with patch('agnt5.lm.client.RustLanguageModel') as mock_rust_class:
        mock_instance = MagicMock()
        mock_instance.generate = AsyncMock(return_value=mock_response)
        # stream_iter returns an async iterator, not a coroutine
        mock_instance.stream_iter = MagicMock(return_value=MockAsyncStreamIterator(mock_chunks))
        mock_rust_class.return_value = mock_instance

        agent = Agent(
            name="simple_agent",
            model="openai/gpt-4o-mini",
            instructions="Be helpful",
        )

        result = await agent.run_sync("Hi there")

        assert isinstance(result, AgentResult)
        assert result.output == "Hello! How can I help you?"
        assert len(result.tool_calls) == 0


@pytest.mark.asyncio
async def test_agent_run_with_mock_lm(mock_lm):
    """Test agent run with legacy mock LLM (backward compatibility)."""
    mock_lm.responses = ["Hello! How can I help you?"]

    agent = Agent(
        name="simple_agent",
        model=mock_lm,
        instructions="Be helpful",
    )

    result = await agent.run_sync("Hi there")

    assert isinstance(result, AgentResult)
    assert result.output == "Hello! How can I help you?"
    assert len(result.tool_calls) == 0
    assert mock_lm.call_count == 1


@pytest.mark.asyncio
async def test_agent_run_with_agent_context(mock_lm):
    """Test agent run with AgentContext for conversation history."""
    from agnt5.entity import create_entity_context

    # Create entity context for state management
    manager, token = create_entity_context()

    try:
        mock_lm.responses = [
            "Hello! I'm here to help.",
            "Sure, I remember our conversation.",
        ]

        agent = Agent(
            name="ctx_agent",
            model=mock_lm,
            instructions="Be conversational",
        )

        # First message
        ctx1 = AgentContext(run_id="session-1", agent_name="ctx_agent")
        result1 = await agent.run_sync("Hi there", context=ctx1)
        assert "Hello" in result1.output

        # Second message (should have conversation history)
        ctx2 = AgentContext(run_id="session-1", agent_name="ctx_agent")
        result2 = await agent.run_sync("Do you remember?", context=ctx2)

        # Check that conversation history was loaded
        history = await ctx2.get_conversation_history()
        assert len(history) >= 2  # At least user + assistant from first turn

    finally:
        from agnt5.entity import _entity_state_adapter_ctx
        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


@pytest.mark.asyncio
async def test_agent_with_tool_execution(sample_tool):
    """Test agent executing a tool."""
    # Mock LLM that calls a tool, then provides final response
    mock_lm = MockLanguageModel(
        responses=[
            "I'll use the tool",
            "The doubled value is 20",
        ],
        tool_calls=[
            [{"id": "call_1", "name": "test_tool", "arguments": '{"value": 10}'}],
            None,  # No tool calls in second response
        ],
    )

    agent = Agent(
        name="tool_agent",
        model=mock_lm,
        instructions="Use tools when needed",
        tools=[sample_tool],
    )

    result = await agent.run_sync("Double the number 10")

    assert "The doubled value is 20" in result.output
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "test_tool"
    assert mock_lm.call_count == 2  # Initial call + after tool execution


@pytest.mark.asyncio
async def test_agent_multiple_tool_calls():
    """Test agent making multiple tool calls."""

    @tool
    async def add(ctx: Context, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool
    async def multiply(ctx: Context, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    mock_lm = MockLanguageModel(
        responses=[
            "Let me calculate",
            "Let me multiply",
            "The result is 50",
        ],
        tool_calls=[
            [{"id": "call_1", "name": "add", "arguments": '{"a": 10, "b": 15}'}],
            [{"id": "call_2", "name": "multiply", "arguments": '{"a": 25, "b": 2}'}],
            None,
        ],
    )

    agent = Agent(
        name="calc_agent",
        model=mock_lm,
        instructions="Perform calculations",
        tools=[add, multiply],
    )

    result = await agent.run_sync("Add 10 and 15, then multiply by 2")

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "add"
    assert result.tool_calls[1]["name"] == "multiply"


@pytest.mark.asyncio
async def test_agent_tool_error_handling():
    """Test agent handling tool execution errors."""

    @tool
    async def failing_tool(ctx: Context) -> str:
        """Tool that always fails."""
        raise ValueError("Tool execution failed")

    mock_lm = MockLanguageModel(
        responses=[
            "Using the tool",
            "I encountered an error with the tool",
        ],
        tool_calls=[
            [{"id": "call_1", "name": "failing_tool", "arguments": "{}"}],
            None,
        ],
    )

    agent = Agent(
        name="error_agent",
        model=mock_lm,
        instructions="Handle errors gracefully",
        tools=[failing_tool],
    )

    result = await agent.run_sync("Use the failing tool")

    # Agent should complete despite tool error
    assert result.output
    assert len(result.tool_calls) == 1


@pytest.mark.asyncio
async def test_agent_max_iterations():
    """Test agent respecting max iterations."""
    # Mock LLM that always wants to call tools
    mock_lm = MockLanguageModel(
        responses=["Calling tool"] * 20,
        tool_calls=[[{"id": f"call_{i}", "name": "test_tool", "arguments": '{"value": 1}'}] for i in range(20)],
    )

    @tool
    async def test_tool(ctx: Context, value: int) -> int:
        return value

    agent = Agent(
        name="loop_agent",
        model=mock_lm,
        instructions="Test",
        tools=[test_tool],
        max_iterations=3,
    )

    result = await agent.run_sync("Keep calling tools")

    # Should stop at max_iterations
    assert len(result.tool_calls) <= 3


# Test Agent as Tool


@pytest.mark.asyncio
async def test_agent_as_tool():
    """Test agent-as-tool pattern."""
    specialist_lm = MockLanguageModel(responses=["Specialist response"])

    specialist = Agent(
        name="specialist",
        model=specialist_lm,
        instructions="I am a specialist",
    )

    # Convert agent to tool
    specialist_tool = specialist.to_tool()

    assert specialist_tool.name == "ask_specialist"  # Tool name is prefixed with 'ask_'
    assert "specialist" in specialist_tool.description.lower()

    # Use specialist as a tool
    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")
    result = await specialist_tool.invoke(ctx, message="Test query")  # Parameter is 'message'

    assert result == "Specialist response"


@pytest.mark.asyncio
async def test_agent_with_agent_tools():
    """Test coordinator agent using another agent as a tool."""
    specialist_lm = MockLanguageModel(responses=["Specialist answer"])
    coordinator_lm = MockLanguageModel(
        responses=["Using specialist", "Final answer"],
        tool_calls=[
            [{"id": "call_1", "name": "ask_specialist", "arguments": '{"message": "Help with this"}'}],
            None,
        ],
    )

    specialist = Agent(
        name="specialist",
        model=specialist_lm,
        instructions="I am a specialist",
    )

    coordinator = Agent(
        name="coordinator",
        model=coordinator_lm,
        instructions="Coordinate using specialists",
        tools=[specialist],  # Pass agent directly as tool
    )

    result = await coordinator.run_sync("Need specialist help")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "ask_specialist"  # Tool name is prefixed with 'ask_'


# Test Handoff Mechanism


@pytest.mark.asyncio
async def test_agent_handoff():
    """Test agent handoff mechanism."""
    target_lm = MockLanguageModel(responses=["I am the target agent"])
    source_lm = MockLanguageModel(
        responses=["Transferring to target"],
        tool_calls=[
            [{"id": "call_1", "name": "transfer_to_target", "arguments": '{"message": "Handle this please"}'}],
        ],
    )

    target_agent = Agent(
        name="target",
        model=target_lm,
        instructions="I handle specialized tasks",
    )

    source_agent = Agent(
        name="source",
        model=source_lm,
        instructions="I coordinate tasks",
        handoffs=[handoff(target_agent, "Transfer to target agent")],
    )

    result = await source_agent.run_sync("Need specialized help")

    # Should have handed off to target
    assert result.handoff_to == "target"
    assert result.output == "I am the target agent"


def test_handoff_configuration():
    """Test handoff configuration creation."""
    mock_lm = MockLanguageModel()

    target = Agent(name="target", model=mock_lm, instructions="Target agent")

    # Create handoff using helper function
    h = handoff(target, "Transfer to target")

    assert isinstance(h, Handoff)
    assert h.agent is target
    assert h.description == "Transfer to target"
    assert h.tool_name == "transfer_to_target"
    assert h.pass_full_history is True


def test_handoff_agent_direct_pass():
    """Test that agents can be passed directly to handoffs parameter (auto-wrapped)."""
    mock_lm = MockLanguageModel()

    technical = Agent(name="technical", model=mock_lm, instructions="Technical specialist")
    business = Agent(name="business", model=mock_lm, instructions="Business specialist")

    # Pass agents directly - should be auto-wrapped in Handoff
    coordinator = Agent(
        name="coordinator",
        model=mock_lm,
        instructions="Route requests",
        handoffs=[technical, business]  # Pass Agent instances directly
    )

    # Verify handoffs were created and normalized
    assert len(coordinator.handoffs) == 2
    assert all(isinstance(h, Handoff) for h in coordinator.handoffs)

    # Check first handoff (technical)
    assert coordinator.handoffs[0].agent is technical
    assert coordinator.handoffs[0].tool_name == "transfer_to_technical"
    assert coordinator.handoffs[0].description == "Technical specialist"  # Uses agent instructions
    assert coordinator.handoffs[0].pass_full_history is True  # Default

    # Check second handoff (business)
    assert coordinator.handoffs[1].agent is business
    assert coordinator.handoffs[1].tool_name == "transfer_to_business"

    # Verify handoff tools were created
    assert "transfer_to_technical" in coordinator.tools
    assert "transfer_to_business" in coordinator.tools


def test_handoff_mixed_agent_and_handoff_instances():
    """Test that handoffs parameter can accept mix of Agent and Handoff instances."""
    mock_lm = MockLanguageModel()

    technical = Agent(name="technical", model=mock_lm, instructions="Technical specialist")
    business = Agent(name="business", model=mock_lm, instructions="Business specialist")

    # Mix direct agents and Handoff configurations
    coordinator = Agent(
        name="coordinator",
        model=mock_lm,
        instructions="Route requests",
        handoffs=[
            technical,  # Direct agent - uses defaults
            Handoff(  # Custom handoff configuration
                agent=business,
                description="Custom business description",
                tool_name="escalate_to_business",
                pass_full_history=False
            )
        ]
    )

    # Verify both handoffs exist
    assert len(coordinator.handoffs) == 2

    # First handoff (auto-wrapped)
    assert coordinator.handoffs[0].agent is technical
    assert coordinator.handoffs[0].tool_name == "transfer_to_technical"
    assert coordinator.handoffs[0].pass_full_history is True

    # Second handoff (custom config)
    assert coordinator.handoffs[1].agent is business
    assert coordinator.handoffs[1].tool_name == "escalate_to_business"
    assert coordinator.handoffs[1].description == "Custom business description"
    assert coordinator.handoffs[1].pass_full_history is False

    # Verify tools
    assert "transfer_to_technical" in coordinator.tools
    assert "escalate_to_business" in coordinator.tools


# Test Agent Registry


def test_agent_registry_operations():
    """Test AgentRegistry operations."""
    mock_lm = MockLanguageModel()

    agent1 = Agent(name="agent1", model=mock_lm, instructions="First")
    agent2 = Agent(name="agent2", model=mock_lm, instructions="Second")

    AgentRegistry.register(agent1)
    AgentRegistry.register(agent2)

    # Test get
    retrieved = AgentRegistry.get("agent1")
    assert retrieved is agent1

    # Test all
    all_agents = AgentRegistry.all()
    assert len(all_agents) == 2
    assert "agent1" in all_agents
    assert "agent2" in all_agents

    # Test clear
    AgentRegistry.clear()
    assert len(AgentRegistry.all()) == 0


# Test AgentContext


def test_agent_context_creation():
    """Test AgentContext creation."""
    from agnt5.entity import create_entity_context

    manager, token = create_entity_context()

    try:
        ctx = AgentContext(run_id="test-123", agent_name="test_agent")

        assert ctx.run_id == "test-123"
        assert ctx.session_id == "test-123"  # Defaults to run_id
        assert hasattr(ctx, 'state')

    finally:
        from agnt5.entity import _entity_state_adapter_ctx
        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


def test_agent_context_with_session_id():
    """Test AgentContext with custom session_id."""
    from agnt5.entity import create_entity_context

    manager, token = create_entity_context()

    try:
        ctx = AgentContext(
            run_id="run-456",
            agent_name="test_agent",
            session_id="custom-session"
        )

        assert ctx.run_id == "run-456"
        assert ctx.session_id == "custom-session"

    finally:
        from agnt5.entity import _entity_state_adapter_ctx
        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


async def test_agent_context_conversation_history():
    """Test AgentContext conversation history management."""
    from agnt5.entity import create_entity_context

    manager, token = create_entity_context()

    try:
        ctx = AgentContext(run_id="test-123", agent_name="test_agent")

        # Initially empty
        history = await ctx.get_conversation_history()
        assert len(history) == 0

        # Save some messages
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]
        await ctx.save_conversation_history(messages)

        # Retrieve
        retrieved = await ctx.get_conversation_history()
        assert len(retrieved) == 2
        assert retrieved[0].content == "Hello"
        assert retrieved[1].content == "Hi there!"

    finally:
        from agnt5.entity import _entity_state_adapter_ctx
        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()


# Test Tool Not Found


@pytest.mark.asyncio
async def test_agent_tool_not_found():
    """Test agent handling missing tool.

    When the LLM returns a tool call for a tool that doesn't exist,
    the agent should handle it gracefully and continue.
    """
    # Create a dummy tool so the agent uses generate() instead of stream()
    # (streaming doesn't support tool calls)
    @tool
    async def dummy_tool(ctx: Context, x: int) -> int:
        """A dummy tool that's never called."""
        return x

    mock_lm = MockLanguageModel(
        responses=[
            "Calling tool",
            "Tool not found, continuing anyway",
        ],
        tool_calls=[
            [{"id": "call_1", "name": "nonexistent_tool", "arguments": "{}"}],
            None,
        ],
    )

    agent = Agent(
        name="missing_tool_agent",
        model=mock_lm,
        instructions="Test",
        tools=[dummy_tool],  # Need at least one tool to use generate() path
    )

    result = await agent.run_sync("Use a missing tool")

    # Should complete despite missing tool
    assert result.output
    assert len(result.tool_calls) == 1


# Test Agent Result


def test_agent_result_creation():
    """Test AgentResult creation."""
    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")

    result = AgentResult(
        output="Test output",
        tool_calls=[{"name": "test", "arguments": "{}"}],
        context=ctx,
    )

    assert result.output == "Test output"
    assert len(result.tool_calls) == 1
    assert result.context is ctx
    assert result.handoff_to is None


def test_agent_result_with_handoff():
    """Test AgentResult with handoff metadata."""
    ctx = Context(run_id="test-123", correlation_id="corr-123", parent_correlation_id="parent-123")

    result = AgentResult(
        output="Handed off",
        tool_calls=[],
        context=ctx,
        handoff_to="target_agent",
        handoff_metadata={"from": "source", "to": "target"},
    )

    assert result.handoff_to == "target_agent"
    assert result.handoff_metadata["from"] == "source"


# Test Streaming API


@pytest.mark.asyncio
async def test_agent_run_returns_async_generator(mock_lm):
    """Test that agent.run() returns an async generator."""
    mock_lm.responses = ["Hello! How can I help you?"]

    agent = Agent(
        name="streaming_agent",
        model=mock_lm,
        instructions="Be helpful",
    )

    # run() should return an async generator
    result = agent.run("Hi there")
    import inspect
    assert inspect.isasyncgen(result)

    # Clean up - consume the generator
    async for _ in result:
        pass


@pytest.mark.asyncio
async def test_agent_run_streaming_events(mock_lm):
    """Test that agent.run() yields correct events."""
    mock_lm.responses = ["Hello! How can I help you?"]

    agent = Agent(
        name="streaming_agent",
        model=mock_lm,
        instructions="Be helpful",
    )

    events = []
    async for event in agent.run("Hi there"):
        events.append(event)

    # Should have at least started and completed events
    assert len(events) >= 2

    # First event should be agent.started
    assert events[0].event_type == "agent.started"
    assert events[0].name == "streaming_agent"

    # Last event should be agent.completed
    assert events[-1].event_type == "agent.completed"
    assert events[-1].output_data["output"] == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_agent_run_streaming_with_error(mock_lm):
    """Test that agent.run() yields agent.failed event on error."""
    mock_lm.responses = ["Starting..."]

    # Create an agent that will fail due to a mock exception
    agent = Agent(
        name="failing_agent",
        model=mock_lm,
        instructions="Test",
    )

    # Override both generate and stream to raise an exception
    # (agent uses stream() when there are no tools)
    async def failing_generate(request):
        raise ValueError("Test error")

    async def failing_stream(request):
        raise ValueError("Test error")
        yield  # Make it a generator (never reached)

    mock_lm.generate = failing_generate
    mock_lm.stream = failing_stream

    events = []
    with pytest.raises(ValueError):
        async for event in agent.run("Hi"):
            events.append(event)

    # Should have agent.started event
    assert len(events) >= 1
    assert events[0].event_type == "agent.started"


@pytest.mark.asyncio
async def test_agent_run_sync_equivalent_to_streaming(mock_lm):
    """Test that run_sync() produces same output as consuming run()."""
    mock_lm.responses = ["Hello! How can I help you?"]

    agent = Agent(
        name="test_agent",
        model=mock_lm,
        instructions="Be helpful",
    )

    # Get result via run_sync
    result_sync = await agent.run_sync("Hi there")

    # Reset mock for second call
    mock_lm.call_count = 0

    # Get result by consuming run() generator
    final_event = None
    async for event in agent.run("Hi there"):
        if event.event_type == "agent.completed":
            final_event = event

    # Both should produce same output
    assert final_event is not None
    assert result_sync.output == final_event.output_data["output"]
