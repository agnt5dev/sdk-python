# AGNT5 Python SDK Examples

This directory contains comprehensive examples demonstrating the AGNT5 Python SDK capabilities.

## Directory Structure

```
examples/
├── app.py                           # Worker entry point (registers all examples)
├── ex_01_functions.py               # Durable functions with retry policies
├── ex_02_entities.py                # Stateful entities (Counter, Memory, KV, BankAccount)
├── ex_03_workflows.py               # Multi-step workflow orchestration
├── ex_04_lm_functions_openai.py     # OpenAI language model integration
├── ex_05_lm_functions_anthropic.py  # Anthropic Claude integration
├── ex_06_agents.py                  # Basic AI agents
├── ex_07_agents_with_tools.py       # Agents with tools + multi-agent patterns
├── ex_08_hitl.py                    # Human-in-the-loop workflows
├── ex_09_streaming.py               # Function streaming (text, progress, JSON)
├── ex_10_structured_output.py       # Structured LLM output (Pydantic, dataclass)
├── ex_11_agent_streaming.py         # Agent streaming with events
└── ex_12_workflow_streaming.py      # Workflow streaming
```

## Running Examples

### Standalone Mode

Each example can run independently without the platform:

```bash
cd sdk-python

# Run individual examples
uv run python examples/ex_01_functions.py
uv run python examples/ex_02_entities.py
uv run python examples/ex_06_agents.py
```

### With Platform (via Worker)

Run all examples as a registered worker:

```bash
# Start the platform dev server + worker
just platform standalone python

# Or run the worker directly
uv run python examples/app.py
```

## Examples Overview

### ex_01_functions.py - Durable Functions

**Components**: `greet`, `add`, `process_data`, `failing_function`

Demonstrates:
- Simple function definitions with `@function` decorator
- Type-safe parameters and return values
- Error handling and automatic retries
- Retry policies with exponential backoff

```python
from agnt5 import function

@function
async def greet(name: str) -> str:
    return f"Hello, {name}!"
```

### ex_02_entities.py - Stateful Entities

**Components**: `Counter`, `ConversationMemory`, `KeyValueStore`, `BankAccount`

Demonstrates:
- Typed state with Pydantic models (`Entity[StateModel]`)
- Direct state attribute access (`self.state.count`)
- Automatic mutation detection (reads skip persistence)
- `@query` decorator for read-only methods
- Single-writer consistency guarantees

```python
from agnt5 import Entity
from pydantic import BaseModel

class CounterState(BaseModel):
    count: int = 0

class Counter(Entity[CounterState]):
    async def increment(self, amount: int = 1) -> int:
        self.state.count += amount
        return self.state.count
```

### ex_03_workflows.py - Workflow Orchestration

**Components**: `data_pipeline`, `order_fulfillment`, `etl_workflow`, `approval_workflow`

Demonstrates:
- Multi-step workflow definitions
- Step-by-step execution with checkpointing
- Parallel step execution
- Error handling and recovery
- Stateful workflows with context

### ex_04_lm_functions_openai.py - OpenAI Integration

**Components**: `lm_generate`, `lm_stream`, `lm_summarize`, `lm_classify`

Demonstrates:
- OpenAI model integration via `lm.generate()`
- Streaming responses with `lm.stream()`
- Text summarization and classification
- Temperature and parameter configuration

```python
from agnt5 import lm

response = await lm.generate(
    model="openai/gpt-4o-mini",
    prompt="Explain quantum computing in simple terms"
)
```

### ex_05_lm_functions_anthropic.py - Anthropic Integration

**Components**: `claude_generate`, `claude_stream`, `claude_analyze`, `claude_translate`

Demonstrates:
- Anthropic Claude integration
- Multi-turn conversations
- Code analysis and translation
- Streaming with Claude models

### ex_06_agents.py - Basic Agents

**Components**: `simple_agent`, `claude_agent`, `code_assistant`, `creative_writer`, `conversation_agent`

Demonstrates:
- AI agent creation with `Agent` class
- System prompts and instructions
- Conversation context management
- Different model configurations

```python
from agnt5 import Agent

agent = Agent(
    name="assistant",
    model="openai/gpt-4o-mini",
    instructions="You are a helpful assistant."
)

result = await agent.run("What is Python?")
```

### ex_07_agents_with_tools.py - Agents with Tools

**Components**: `calculator_agent`, `weather_assistant`, `multi_tool_agent`, `research_coordinator`, `support_router`

**Tools**: `calculate`, `get_weather`, `search_database`, `send_email`

Demonstrates:
- Tool definitions with `@tool` decorator
- Agent tool integration
- Multi-tool agents
- Agents as tools for other agents
- Explicit handoffs between agents

```python
from agnt5 import Agent, tool

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

agent = Agent(
    name="calculator",
    model="openai/gpt-4o-mini",
    tools=[calculate]
)
```

### ex_08_hitl.py - Human-in-the-Loop

**Components**: `hitl_text_input`, `hitl_approval`, `hitl_choice`, `hitl_onboarding`, `hitl_agent_questions`, `hitl_conditional_approval`

Demonstrates:
- Requesting human input during workflows
- Approval gates for sensitive operations
- Choice selection from options
- Multi-step onboarding flows
- Conditional approval based on thresholds

### ex_09_streaming.py - Function Streaming

**Components**: `stream_text`, `stream_story`, `stream_counter`, `stream_progress`, `stream_json_chunks`

Demonstrates:
- Streaming text output from functions
- Progress reporting with structured events
- JSON chunk streaming for large data
- Real-time counter updates

### ex_10_structured_output.py - Structured LLM Output

Demonstrates:
- Dataclass-based structured output
- Pydantic model validation
- Nested complex structures
- Raw JSON schema definitions
- Accessing output via `.structured_output`

```python
from pydantic import BaseModel
from agnt5 import lm

class SecurityAnalysis(BaseModel):
    vulnerabilities: list[str]
    severity: str
    safe_to_deploy: bool

response = await lm.generate(
    model="openai/gpt-4o",
    prompt="Analyze this code for security issues...",
    response_format=SecurityAnalysis
)
analysis = response.structured_output
```

### ex_11_agent_streaming.py - Agent Streaming

**Components**: `stream_agent_chat`, `stream_agent_with_tools`, `stream_agent_simple`

Demonstrates:
- Real-time agent response streaming
- Tool call events during streaming
- Event-based agent interaction
- Streaming with tool-equipped agents

### ex_12_workflow_streaming.py - Workflow Streaming

**Components**: `research_workflow`, `mixed_workflow`, `simple_agent_workflow`

Demonstrates:
- Streaming workflow step execution
- Mixed streaming (functions + agents)
- Research workflow with multiple steps
- Real-time workflow progress updates

## Core Concepts

### Context

The `Context` object (available in workflows) provides:
- **State Management**: `get()`, `set()`, `delete()`
- **Checkpointing**: `step()` / `run()` for expensive operations
- **Metadata**: `run_id`, `attempt`, `component_type`
- **Logging**: `ctx.logger` for structured logs

### Entities

Entities provide:
- **Typed State**: Use Pydantic models for state validation
- **Durability**: State persists automatically on mutations
- **Single-Writer**: Guaranteed consistency (no race conditions)
- **Query Methods**: Use `@query` for read-only operations (skip persistence)

### Agents

Agents provide:
- **LLM Integration**: Built-in support for OpenAI and Anthropic
- **Tool Calling**: Define tools with `@tool` decorator
- **Multi-Agent**: Use agents as tools or explicit handoffs
- **Streaming**: Real-time response streaming

### Streaming

Streaming options:
- **Function Streaming**: Yield values with `stream=True`
- **LM Streaming**: Use `lm.stream()` for token-by-token output
- **Agent Streaming**: `agent.stream()` for real-time responses
- **Workflow Streaming**: Stream workflow step events

## Integration Tests

Tests for these examples are in `tests/integration/`:

| Example | Test File |
|---------|-----------|
| ex_01 | `test_ex_100_run_functions.py`, `test_ex_110_submit_functions.py` |
| ex_02 | `test_ex_200_entities.py`, `test_ex_210_entities_durability.py` |
| ex_03 | `test_ex_300_workflows.py`, `test_ex_310_stream_workflows.py` |
| ex_04 | `test_ex_400_lm_functions_openai.py` |
| ex_05 | `test_ex_410_lm_functions_anthropic.py` |
| ex_06 | `test_ex_500_agents.py` |
| ex_07 | `test_ex_510_agents_with_tools.py` |
| ex_08 | `test_ex_600_hitl.py` |
| ex_09 | `test_ex_120_stream_functions.py` |
| ex_11 | `test_ex_520_stream_agents.py` |
| ex_12 | `test_ex_310_stream_workflows.py` |

Run integration tests:

```bash
# All integration tests
uv run pytest tests/integration/ -v

# Specific example tests
uv run pytest tests/integration/test_ex_200_entities.py -v
```

## Next Steps

1. Start with `ex_01_functions.py` to understand basic durable functions
2. Explore `ex_02_entities.py` for stateful applications
3. Try `ex_06_agents.py` for AI agent development
4. Combine patterns in `ex_07_agents_with_tools.py` for advanced use cases

For more information, see the [SDK documentation](../README.md).
