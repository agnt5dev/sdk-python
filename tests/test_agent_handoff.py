"""
Tests for Agent handoffs and agents-as-tools functionality.

Tests cover:
- Agents as tools (implicit invocation)
- Explicit handoffs with transfer_to tools
- Context and state passing between agents
- Conversation history passing
- Handoff metadata tracking
"""

import pytest

from agnt5 import Agent, AgentResult, Context, Handoff, handoff, tool
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


# Tests for Agents as Tools


@pytest.mark.asyncio
async def test_agent_as_tool_basic():
    """Test using one agent as a tool for another agent."""
    # Create specialist agent
    specialist_lm = MockLanguageModel(responses=["Specialist response"])

    specialist = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="You are a specialist",
    )

    # Convert to tool
    specialist_tool = specialist.to_tool()

    assert specialist_tool.name == "specialist"
    assert "specialist" in specialist_tool.description.lower()

    # Test invoking as tool
    ctx = Context(run_id="test-1")
    result = await specialist_tool.invoke(ctx, user_message="Test question")

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_agent_auto_converts_to_tool():
    """Test that Agent instances in tools list are auto-converted."""
    specialist = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="Specialist agent",
    )

    # Create coordinator with specialist passed directly
    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="Coordinator agent",
        tools=[specialist],  # Pass agent directly
    )

    # Check that specialist was converted to a tool
    assert "specialist" in coordinator.tools
    assert coordinator.tools["specialist"].name == "specialist"


@pytest.mark.asyncio
async def test_agent_using_another_agent_as_tool():
    """Test full flow of one agent using another as a tool."""

    # Specialist that returns a fixed response
    specialist_lm = MockLanguageModel(responses=["Specialist analyzed the data"])

    specialist = Agent(
        name="analyst",
        model="openai/gpt-4o-mini",
        instructions="You are a data analyst",
    )

    # Coordinator that calls the specialist
    coordinator_lm = MockLanguageModel(
        responses=[
            "I'll use the analyst",
            "Based on the analysis, here's my recommendation",
        ],
        tool_calls=[
            [{"id": "call_1", "name": "analyst", "arguments": '{"user_message": "Analyze this data"}'}],
            None,
        ],
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="You coordinate with specialists",
        tools=[specialist],
    )

    result = await coordinator.run("Please analyze some data")

    # Should have called the analyst tool
    assert len(result.tool_calls) >= 1
    assert result.tool_calls[0]["name"] == "analyst"


# Tests for Handoff class


def test_handoff_creation():
    """Test creating a Handoff configuration."""
    target = Agent(
        name="target",
        model="openai/gpt-4o-mini",
        instructions="Target agent",
    )

    h = Handoff(
        agent=target,
        description="Transfer to target",
        tool_name="custom_transfer",
        pass_full_history=False,
    )

    assert h.agent is target
    assert h.description == "Transfer to target"
    assert h.tool_name == "custom_transfer"
    assert h.pass_full_history is False


def test_handoff_helper_function():
    """Test handoff() helper function."""
    target = Agent(
        name="target",
        model="openai/gpt-4o-mini",
        instructions="Target agent",
    )

    h = handoff(target, description="Custom description")

    assert isinstance(h, Handoff)
    assert h.agent is target
    assert h.description == "Custom description"
    assert h.tool_name == "transfer_to_target"


def test_handoff_default_tool_name():
    """Test handoff tool name defaults to transfer_to_{agent_name}."""
    target = Agent(name="specialist", model="openai/gpt-4o-mini", instructions="Test")

    h = handoff(target)

    assert h.tool_name == "transfer_to_specialist"


# Tests for Agent with Handoffs


def test_agent_with_handoffs_creates_tools():
    """Test that handoffs are converted to tools."""
    target = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="Specialist",
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="Coordinator",
        handoffs=[handoff(target)],
    )

    # Check handoff tool was created
    assert "transfer_to_specialist" in coordinator.tools
    handoff_tool = coordinator.tools["transfer_to_specialist"]
    assert "specialist" in handoff_tool.description.lower()


@pytest.mark.asyncio
async def test_handoff_execution():
    """Test executing a handoff."""
    # Target agent that will receive the handoff
    target_lm = MockLanguageModel(responses=["Target agent response"])

    target = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="Specialist agent",
    )

    # Coordinator with handoff
    coordinator_lm = MockLanguageModel(
        responses=[
            "Transferring to specialist",
            "This should not be reached",
        ],
        tool_calls=[
            [
                {
                    "id": "call_1",
                    "name": "transfer_to_specialist",
                    "arguments": '{"message": "Please handle this"}',
                }
            ],
            None,
        ],
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="Coordinator with handoff",
        handoffs=[handoff(target)],
    )

    result = await coordinator.run("Transfer this request")

    # Handoff should have occurred
    assert result.handoff_to == "specialist"
    assert result.handoff_metadata is not None
    assert result.handoff_metadata.get("_handoff") is True


@pytest.mark.asyncio
async def test_handoff_stops_current_agent():
    """Test that handoff stops the current agent's execution."""
    target_lm = MockLanguageModel(responses=["Final answer from target"])

    target = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="Specialist",
    )

    # Coordinator that tries to continue after handoff (shouldn't happen)
    coordinator_lm = MockLanguageModel(
        responses=[
            "Transferring",
            "This response should NOT appear",  # Should not be reached
        ],
        tool_calls=[
            [
                {
                    "id": "call_1",
                    "name": "transfer_to_specialist",
                    "arguments": '{"message": "Handle this"}',
                }
            ],
            None,
        ],
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="Coordinator",
        handoffs=[handoff(target)],
    )

    result = await coordinator.run("Test request")

    # Output should be from target, not coordinator
    assert result.handoff_to == "specialist"
    # Coordinator should only have made one iteration (the handoff)
    assert coordinator_lm.call_count == 1


# Tests for Context Passing


@pytest.mark.asyncio
async def test_handoff_passes_context():
    """Test that context is passed during handoff."""

    @tool(auto_schema=True)
    async def save_data(ctx: Context, key: str, value: str) -> str:
        """Save data to context."""
        ctx.set(key, value)
        return f"Saved {key}={value}"

    # Agent that saves to context before handoff
    source_lm = MockLanguageModel(
        responses=["Saving data", "Transferring"],
        tool_calls=[
            [{"id": "call_1", "name": "save_data", "arguments": '{"key": "user_id", "value": "12345"}'}],
            [{"id": "call_2", "name": "transfer_to_target", "arguments": '{"message": "Continue"}'}],
        ],
    )

    target_lm = MockLanguageModel(responses=["Received context"])

    target = Agent(
        name="target",
        model="openai/gpt-4o-mini",
        instructions="Target agent",
    )

    source = Agent(
        name="source",
        model="openai/gpt-4o-mini",
        instructions="Source agent",
        tools=[save_data],
        handoffs=[handoff(target)],
    )

    ctx = Context(run_id="shared-context")
    result = await source.run("Save user_id then transfer", context=ctx)

    # Context should be shared
    assert ctx.get("user_id") == "12345"
    assert result.handoff_to == "target"


@pytest.mark.asyncio
async def test_handoff_conversation_history():
    """Test that conversation history is stored for handoffs."""
    target_lm = MockLanguageModel(responses=["Target response"])

    target = Agent(
        name="target",
        model="openai/gpt-4o-mini",
        instructions="Target",
    )

    source_lm = MockLanguageModel(
        responses=["Transferring"],
        tool_calls=[
            [{"id": "call_1", "name": "transfer_to_target", "arguments": '{"message": "Continue"}'}],
        ],
    )

    source = Agent(
        name="source",
        model="openai/gpt-4o-mini",
        instructions="Source",
        handoffs=[handoff(target, pass_full_history=True)],
    )

    ctx = Context(run_id="test-history")
    result = await source.run("Test message", context=ctx)

    # History should be stored in context
    conversation = ctx.get("_current_conversation")
    assert conversation is not None
    assert len(conversation) > 0


# Tests for AgentResult with Handoff Metadata


@pytest.mark.asyncio
async def test_agent_result_handoff_metadata():
    """Test AgentResult includes handoff metadata."""
    target = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="Specialist",
    )

    source = Agent(
        name="triage",
        model="openai/gpt-4o-mini",
        instructions="Triage",
        handoffs=[handoff(target)],
    )

    result = AgentResult(
        output="Test output",
        tool_calls=[],
        context=Context(run_id="test"),
        handoff_to="specialist",
        handoff_metadata={"test": "data"},
    )

    assert result.handoff_to == "specialist"
    assert result.handoff_metadata == {"test": "data"}


# Tests for Complex Workflows


@pytest.mark.asyncio
async def test_mixed_tools_and_handoffs():
    """Test agent with both regular tools and handoffs."""

    @tool(auto_schema=True)
    async def regular_tool(ctx: Context, value: int) -> int:
        """Regular tool."""
        return value * 2

    specialist = Agent(
        name="specialist",
        model="openai/gpt-4o-mini",
        instructions="Specialist",
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="Coordinator with mixed tools",
        tools=[regular_tool],
        handoffs=[handoff(specialist)],
    )

    # Should have both regular tool and handoff tool
    assert "regular_tool" in coordinator.tools
    assert "transfer_to_specialist" in coordinator.tools
    assert len(coordinator.tools) == 2


@pytest.mark.asyncio
async def test_agent_as_tool_plus_handoff():
    """Test using an agent as a tool AND having handoffs to other agents."""

    tool_agent = Agent(
        name="tool_agent",
        model="openai/gpt-4o-mini",
        instructions="Tool agent",
    )

    handoff_agent = Agent(
        name="handoff_agent",
        model="openai/gpt-4o-mini",
        instructions="Handoff agent",
    )

    coordinator = Agent(
        name="coordinator",
        model="openai/gpt-4o-mini",
        instructions="Coordinator",
        tools=[tool_agent],  # Agent as tool
        handoffs=[handoff(handoff_agent)],  # Explicit handoff
    )

    # Should have both agent-as-tool and handoff tool
    assert "tool_agent" in coordinator.tools
    assert "transfer_to_handoff_agent" in coordinator.tools
