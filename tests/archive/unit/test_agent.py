"""
Tests for Agent component.

Tests cover:
- Agent creation and configuration
- Agent execution with mock LLM
- Tool orchestration
- Multi-turn conversations
- State management
"""

import pytest

from agnt5 import Agent, AgentResult, Context, tool
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, Message, MessageRole, TokenUsage


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
        """Mock stream (not implemented)."""
        yield "Mock"
        yield " stream"


# Test fixtures
@pytest.fixture
def mock_lm():
    """Create mock language model."""
    return MockLanguageModel()


@pytest.fixture
def sample_tool():
    """Create sample tool for testing."""

    @tool(auto_schema=True)
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
    assert agent.model is mock_lm
    assert len(agent.tools) == 0


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


# Test Agent Execution


@pytest.mark.asyncio
async def test_agent_run_simple(mock_lm):
    """Test simple agent run without tools."""
    mock_lm.responses = ["Hello! How can I help you?"]

    agent = Agent(
        name="simple_agent",
        model=mock_lm,
        instructions="Be helpful",
    )

    result = await agent.run("Hi there")

    assert isinstance(result, AgentResult)
    assert result.output == "Hello! How can I help you?"
    assert len(result.tool_calls) == 0
    assert mock_lm.call_count == 1


@pytest.mark.asyncio
async def test_agent_run_with_context():
    """Test agent run with provided context."""
    mock_lm = MockLanguageModel(responses=["Response"])

    agent = Agent(
        name="ctx_agent",
        model=mock_lm,
        instructions="Test",
    )

    ctx = Context(run_id="test-123")
    result = await agent.run("Test message", context=ctx)

    assert result.context is ctx
    assert result.context.run_id == "test-123"


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

    result = await agent.run("Double the number 10")

    assert "The doubled value is 20" in result.output
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "test_tool"
    assert mock_lm.call_count == 2  # Initial call + after tool execution


@pytest.mark.asyncio
async def test_agent_multiple_tool_calls():
    """Test agent making multiple tool calls."""

    @tool(auto_schema=True)
    async def add(ctx: Context, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tool(auto_schema=True)
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

    result = await agent.run("Add 10 and 15, then multiply by 2")

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "add"
    assert result.tool_calls[1]["name"] == "multiply"


@pytest.mark.asyncio
async def test_agent_tool_error_handling():
    """Test agent handling tool execution errors."""

    @tool(auto_schema=True)
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

    result = await agent.run("Use the failing tool")

    # Agent should complete despite tool error
    assert "error" in result.output.lower() or result.output
    assert len(result.tool_calls) == 1


@pytest.mark.asyncio
async def test_agent_max_iterations():
    """Test agent respecting max iterations."""
    # Mock LLM that always wants to call tools
    mock_lm = MockLanguageModel(
        responses=["Calling tool"] * 20,
        tool_calls=[[{"id": f"call_{i}", "name": "test_tool", "arguments": '{"value": 1}'}] for i in range(20)],
    )

    @tool(auto_schema=True)
    async def test_tool(ctx: Context, value: int) -> int:
        return value

    agent = Agent(
        name="loop_agent",
        model=mock_lm,
        instructions="Test",
        tools=[test_tool],
        max_iterations=3,
    )

    result = await agent.run("Keep calling tools")

    # Should stop at max_iterations
    assert len(result.tool_calls) <= 3


# Test Multi-Turn Chat


@pytest.mark.asyncio
async def test_agent_chat():
    """Test multi-turn chat."""
    mock_lm = MockLanguageModel(
        responses=[
            "Hello! I'm here to help.",
            "Sure, I can explain that.",
        ]
    )

    agent = Agent(
        name="chat_agent",
        model=mock_lm,
        instructions="Be conversational",
    )

    messages = []

    # First turn
    response1, messages = await agent.chat("Hi", messages)
    assert response1 == "Hello! I'm here to help."
    assert len(messages) == 2  # User + assistant

    # Second turn
    response2, messages = await agent.chat("Explain something", messages)
    assert response2 == "Sure, I can explain that."
    assert len(messages) == 4  # 2 previous + 2 new


@pytest.mark.asyncio
async def test_agent_chat_with_context():
    """Test chat with context preservation."""
    mock_lm = MockLanguageModel(responses=["Response"])

    agent = Agent(
        name="ctx_chat_agent",
        model=mock_lm,
        instructions="Test",
    )

    ctx = Context(run_id="chat-session")
    messages = []

    response, messages = await agent.chat("Test", messages, context=ctx)

    # Context should be passed through
    assert response


# Test Agent with State


@pytest.mark.asyncio
async def test_agent_state_management():
    """Test agent using context state through tools."""

    @tool(auto_schema=True)
    async def save_value(ctx: Context, value: str) -> str:
        """Save a value to state."""
        ctx.set("saved_value", value)
        return f"Saved: {value}"

    @tool(auto_schema=True)
    async def get_value(ctx: Context) -> str:
        """Get saved value."""
        return ctx.get("saved_value", "No value")

    mock_lm = MockLanguageModel(
        responses=[
            "Saving value",
            "Value saved",
            "Getting value",
            "The value is test123",
        ],
        tool_calls=[
            [{"id": "call_1", "name": "save_value", "arguments": '{"value": "test123"}'}],
            None,
            [{"id": "call_2", "name": "get_value", "arguments": "{}"}],
            None,
        ],
    )

    agent = Agent(
        name="state_agent",
        model=mock_lm,
        instructions="Manage state",
        tools=[save_value, get_value],
    )

    ctx = Context(run_id="state-test")

    # Save value
    result1 = await agent.run("Save test123", context=ctx)
    assert len(result1.tool_calls) == 1

    # Retrieve value
    result2 = await agent.run("Get the value", context=ctx)
    assert len(result2.tool_calls) == 1

    # Check state persisted
    assert ctx.get("saved_value") == "test123"


# Test LLM Request Building


@pytest.mark.asyncio
async def test_agent_builds_correct_request(sample_tool):
    """Test that agent builds correct LLM request."""
    mock_lm = MockLanguageModel(responses=["Done"])

    agent = Agent(
        name="request_agent",
        model=mock_lm,
        instructions="Test instructions",
        tools=[sample_tool],
        model_name="gpt-4",
        temperature=0.5,
    )

    await agent.run("Test message")

    # Check first request (before response is added to conversation)
    assert len(mock_lm.requests) >= 1
    request = mock_lm.requests[0]

    assert request.model == "gpt-4"
    assert request.system_prompt == "Test instructions"
    # First message should be user message
    assert request.messages[0].content == "Test message"
    assert request.messages[0].role == MessageRole.USER
    assert len(request.tools) == 1
    assert request.tools[0].name == "test_tool"
    assert request.config.temperature == 0.5


# Test Tool Not Found


@pytest.mark.asyncio
async def test_agent_tool_not_found():
    """Test agent handling missing tool."""
    mock_lm = MockLanguageModel(
        responses=[
            "Calling tool",
            "Tool not found",
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
        tools=[],
    )

    result = await agent.run("Use a missing tool")

    # Should complete despite missing tool
    assert result.output
    assert len(result.tool_calls) == 1
