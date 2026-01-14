"""
Agent fixtures for integration tests.

This module provides reusable agent backends for testing various scenarios:
- Happy path: Basic agent operations with mock LLMs
- Error scenarios: Agents that test failure modes
- Edge cases: Multi-turn, streaming, tool-using agents
- LLM failures: Agents for testing LLM error handling

These fixtures use mock LLMs by default to avoid API key dependencies in tests.
For tests requiring real LLM integration, see tests/api_compat/.

Note: Many of these fixtures return functions that create agents, not agents directly,
to allow tests to configure them (e.g., provide mock LLMs).
"""

import asyncio
from typing import Any, Optional, List

from agnt5 import Agent, Context, FunctionContext, function, tool


# =============================================================================
# MOCK LLM FOR TESTING
# =============================================================================


class MockLLM:
    """
    Mock language model for testing without API calls.

    Provides deterministic responses for testing agent logic without
    requiring real LLM API keys.
    """

    def __init__(self, responses: List[str] = None, should_fail: bool = False):
        """
        Initialize mock LLM.

        Args:
            responses: List of canned responses to return
            should_fail: Whether to simulate LLM failures
        """
        self.responses = responses or ["Mock LLM response"]
        self.call_count = 0
        self.should_fail = should_fail

    async def generate(self, messages: List[dict], **kwargs) -> dict:
        """Generate mock response."""
        if self.should_fail:
            raise RuntimeError("Mock LLM failure")

        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        return {
            "content": response,
            "role": "assistant",
            "model": "mock-llm",
        }


# =============================================================================
# HAPPY PATH AGENT FIXTURES
# =============================================================================


def create_basic_agent(name: str = "basic_agent", mock_llm: MockLLM = None) -> Agent:
    """
    Create a basic agent for simple tests.

    Args:
        name: Agent name
        mock_llm: Optional mock LLM (creates default if not provided)

    Returns:
        Configured Agent instance
    """
    if mock_llm is None:
        mock_llm = MockLLM(responses=["Hello! I'm a test agent."])

    return Agent(
        name=name,
        model="mock/test-model",
        instructions="You are a helpful test agent.",
        temperature=0.7,
        max_iterations=5,
        # In real SDK, would pass llm instance here
    )


@function
async def basic_agent(ctx: FunctionContext, message: str) -> dict:
    """
    Basic agent function for integration tests.

    Args:
        message: Input message

    Returns:
        Dict with agent response
    """
    ctx.logger.info(f"[basic_agent] Processing: {message}")

    # In tests, this would use a mock LLM
    # For now, return deterministic response
    return {
        "message": message,
        "response": f"Mock response to: {message}",
        "model": "mock/test-model",
    }


@function
async def agent_with_tools(ctx: FunctionContext, message: str, tools_to_use: List[str] = None) -> dict:
    """
    Agent with tool calling capabilities.

    Args:
        message: Input message
        tools_to_use: List of tool names to provide

    Returns:
        Dict with agent response and tool usage info
    """
    ctx.logger.info(f"[agent_with_tools] Processing: {message}")

    # Mock tool usage
    tools_used = tools_to_use or []

    return {
        "message": message,
        "response": f"Used tools: {', '.join(tools_used)}",
        "tools_used": tools_used,
        "model": "mock/test-model",
    }


@function
async def multi_turn_agent(
    ctx: FunctionContext,
    messages: List[dict],
    session_id: Optional[str] = None
) -> dict:
    """
    Agent with multi-turn conversation support.

    Args:
        messages: List of conversation messages
        session_id: Optional session identifier

    Returns:
        Dict with agent response and conversation metadata
    """
    ctx.logger.info(f"[multi_turn_agent] Processing {len(messages)} messages")

    # Mock multi-turn response
    last_message = messages[-1] if messages else {"content": ""}

    return {
        "session_id": session_id or "default",
        "message_count": len(messages),
        "last_message": last_message,
        "response": f"Multi-turn response to: {last_message.get('content', '')}",
        "model": "mock/test-model",
    }


@function
async def streaming_agent(ctx: FunctionContext, message: str) -> dict:
    """
    Agent with streaming response support.

    Args:
        message: Input message

    Returns:
        Dict with streaming metadata
    """
    ctx.logger.info(f"[streaming_agent] Processing: {message}")

    # Mock streaming
    chunks = ["This ", "is ", "a ", "streaming ", "response."]

    return {
        "message": message,
        "response": "".join(chunks),
        "chunks": chunks,
        "streaming": True,
        "model": "mock/test-model",
    }


# =============================================================================
# TOOL DEFINITIONS FOR AGENT TESTS
# =============================================================================


@tool
async def calculator_tool(ctx: Context, operation: str, a: float, b: float) -> float:
    """
    Calculator tool for testing agent tool usage.

    Args:
        ctx: Execution context
        operation: Operation to perform (+, -, *, /)
        a: First operand
        b: Second operand

    Returns:
        Result of operation
    """
    ctx.logger.info(f"Calculator: {a} {operation} {b}")

    if operation == "+":
        return a + b
    elif operation == "-":
        return a - b
    elif operation == "*":
        return a * b
    elif operation == "/":
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


@tool
async def search_tool(ctx: Context, query: str, limit: int = 10) -> dict:
    """
    Mock search tool for testing.

    Args:
        ctx: Execution context
        query: Search query
        limit: Maximum results

    Returns:
        Mock search results
    """
    ctx.logger.info(f"Search: {query} (limit: {limit})")

    return {
        "query": query,
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}"}
            for i in range(min(limit, 5))
        ],
        "total": min(limit, 5),
    }


@tool
async def weather_tool(ctx: Context, location: str) -> dict:
    """
    Mock weather tool for testing.

    Args:
        ctx: Execution context
        location: Location to get weather for

    Returns:
        Mock weather data
    """
    ctx.logger.info(f"Weather for: {location}")

    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny",
        "humidity": 65,
    }


# =============================================================================
# ERROR SCENARIO AGENT FIXTURES
# =============================================================================


@function
async def failing_agent(ctx: FunctionContext, message: str, error_type: str = "RuntimeError") -> dict:
    """
    Agent that always fails for error testing.

    Args:
        message: Input message
        error_type: Type of error to raise

    Returns:
        Never returns (always raises)

    Raises:
        Various error types based on error_type parameter
    """
    ctx.logger.info(f"[failing_agent] Will raise {error_type}")

    if error_type == "ValueError":
        raise ValueError("Agent failed with ValueError")
    elif error_type == "TypeError":
        raise TypeError("Agent failed with TypeError")
    elif error_type == "RuntimeError":
        raise RuntimeError("Agent failed with RuntimeError")
    else:
        raise Exception(f"Agent failed with {error_type}")


@function
async def timeout_agent(ctx: FunctionContext, message: str, duration: float = 30.0) -> dict:
    """
    Agent that times out for timeout testing.

    Args:
        message: Input message
        duration: How long to sleep before responding

    Returns:
        Response if duration is short enough
    """
    ctx.logger.info(f"[timeout_agent] Sleeping for {duration}s")

    await asyncio.sleep(duration)

    return {
        "message": message,
        "response": f"Response after {duration}s",
        "duration": duration,
    }


@function
async def agent_with_broken_tools(ctx: FunctionContext, message: str) -> dict:
    """
    Agent with broken tools for error testing.

    Args:
        message: Input message

    Returns:
        Dict indicating tool failure
    """
    ctx.logger.info(f"[agent_with_broken_tools] Processing: {message}")

    # Simulate tool execution failure
    raise RuntimeError("Tool execution failed: calculator_tool not found")


@function
async def llm_error_agent(ctx: FunctionContext, message: str, error_type: str = "rate_limit") -> dict:
    """
    Agent that simulates LLM API errors.

    Args:
        message: Input message
        error_type: Type of LLM error (rate_limit, timeout, auth, server_error)

    Returns:
        Never returns (always raises)

    Raises:
        Various LLM-related errors
    """
    ctx.logger.info(f"[llm_error_agent] Simulating {error_type}")

    if error_type == "rate_limit":
        raise RuntimeError("Rate limit exceeded (429)")
    elif error_type == "timeout":
        raise RuntimeError("LLM request timeout")
    elif error_type == "auth":
        raise RuntimeError("Invalid API key")
    elif error_type == "server_error":
        raise RuntimeError("LLM server error (500)")
    elif error_type == "context_length":
        raise RuntimeError("Context length exceeded")
    else:
        raise RuntimeError(f"LLM error: {error_type}")


@function
async def sometimes_failing_agent(
    ctx: FunctionContext,
    message: str,
    fail_probability: float = 0.5
) -> dict:
    """
    Agent that randomly fails.

    Args:
        message: Input message
        fail_probability: Probability of failure (0.0 to 1.0)

    Returns:
        Success dict or raises error

    Raises:
        RuntimeError: Random failure based on probability
    """
    import random

    ctx.logger.info(f"[sometimes_failing_agent] fail_probability={fail_probability}")

    if random.random() < fail_probability:
        raise RuntimeError(f"Random agent failure (probability: {fail_probability})")

    return {
        "message": message,
        "response": "Success after random check",
        "model": "mock/test-model",
    }


# =============================================================================
# EDGE CASE AGENT FIXTURES
# =============================================================================


@function
async def empty_response_agent(ctx: FunctionContext, message: str) -> dict:
    """
    Agent that returns empty response.

    Args:
        message: Input message

    Returns:
        Dict with empty response
    """
    ctx.logger.info(f"[empty_response_agent] Processing: {message}")

    return {
        "message": message,
        "response": "",
        "model": "mock/test-model",
    }


@function
async def large_response_agent(ctx: FunctionContext, message: str, size_kb: int = 100) -> dict:
    """
    Agent that returns large response.

    Args:
        message: Input message
        size_kb: Size of response in KB

    Returns:
        Dict with large response
    """
    ctx.logger.info(f"[large_response_agent] Generating {size_kb}KB response")

    large_response = "x" * (size_kb * 1024)

    return {
        "message": message,
        "response": large_response,
        "size_kb": size_kb,
        "model": "mock/test-model",
    }


@function
async def slow_agent(ctx: FunctionContext, message: str, delay: float = 1.0) -> dict:
    """
    Agent with artificial delay.

    Args:
        message: Input message
        delay: Delay in seconds

    Returns:
        Dict with response and timing info
    """
    ctx.logger.info(f"[slow_agent] Delaying {delay}s")

    await asyncio.sleep(delay)

    return {
        "message": message,
        "response": f"Response after {delay}s delay",
        "delay": delay,
        "model": "mock/test-model",
    }


@function
async def unicode_agent(ctx: FunctionContext, message: str) -> dict:
    """
    Agent for testing Unicode handling.

    Args:
        message: Input message (may contain Unicode)

    Returns:
        Dict with Unicode response
    """
    ctx.logger.info(f"[unicode_agent] Processing: {message[:50]}")

    # Return response with various Unicode characters
    unicode_response = f"你好！Response to: {message} 🎉"

    return {
        "message": message,
        "response": unicode_response,
        "byte_length": len(unicode_response.encode("utf-8")),
        "model": "mock/test-model",
    }


@function
async def max_iterations_agent(
    ctx: FunctionContext,
    message: str,
    max_iterations: int = 3
) -> dict:
    """
    Agent for testing iteration limits.

    Args:
        message: Input message
        max_iterations: Maximum iterations allowed

    Returns:
        Dict with iteration info
    """
    ctx.logger.info(f"[max_iterations_agent] max_iterations={max_iterations}")

    # Mock iteration behavior
    iterations_used = min(max_iterations, 2)  # Simulate 2 iterations

    return {
        "message": message,
        "response": f"Completed in {iterations_used} iterations",
        "iterations_used": iterations_used,
        "max_iterations": max_iterations,
        "model": "mock/test-model",
    }


@function
async def context_aware_agent(ctx: FunctionContext, message: str, context_data: dict = None) -> dict:
    """
    Agent that uses execution context.

    Args:
        message: Input message
        context_data: Additional context data

    Returns:
        Dict with context-aware response
    """
    ctx.logger.info(f"[context_aware_agent] run_id={ctx.run_id}")

    return {
        "message": message,
        "response": f"Context-aware response",
        "run_id": ctx.run_id,
        "correlation_id": ctx.correlation_id,
        "context_data": context_data or {},
        "model": "mock/test-model",
    }
