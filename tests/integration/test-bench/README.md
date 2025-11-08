# AGNT5 Test Bench

Comprehensive integration test suite for the AGNT5 Python SDK. This test bench serves as the primary development and testing environment for SDK features including functions, workflows, entities, agents, and tools.

## Overview

The test bench is designed to:
- **Integration Testing**: Validate SDK functionality against the AGNT5 platform
- **Development Testing**: Test new features during SDK development
- **Regression Testing**: Ensure existing features continue to work
- **Multi-Provider Testing**: Test LLM integrations across OpenAI, Anthropic, and OpenRouter

## Project Structure

```
test-bench/
├── src/agnt5_test_bench/
│   ├── functions/               # Test functions
│   │   ├── error_functions.py   # Error injection functions
│   │   ├── simple_functions.py  # Basic test functions
│   │   └── lm_functions.py      # LLM integration functions (organized by capability)
│   ├── workflows/               # Test workflows
│   │   ├── simple_workflows.py          # Basic workflows (order processing, pipelines, agent workflows)
│   │   ├── agent_basic_workflows.py     # Basic agent tests
│   │   ├── agent_tools_workflows.py     # Agent tool integration
│   │   ├── agent_handoff_workflows.py   # Agent handoff patterns
│   │   ├── agent_collaboration_workflows.py  # Multi-agent collaboration
│   │   ├── agent_hitl_workflows.py      # Human-in-the-loop workflows
│   │   ├── memory_test_workflows.py     # Session and user memory tests
│   │   ├── replay_test_workflows.py     # Workflow replay testing
│   │   └── comprehensive_integration_test.py  # End-to-end scenarios
│   ├── entities.py              # Stateful entity definitions
│   ├── tools.py                 # Common tool definitions
│   └── agents.py                # Reusable agent definitions
├── app.py                       # Main worker application
├── pyproject.toml               # Python project configuration
├── .env.example                 # Example environment configuration
└── README.md                    # This file
```

## Components

### Functions

#### Simple Functions (`functions/simple_functions.py`)
- `greet`: Basic greeting function
- `long_task`: Simulates long-running tasks
- `flaky_function`: Tests retry logic
- `failing_function`: Tests error handling
- `generate_text`: Placeholder for text generation

#### LLM Functions (`functions/lm_functions.py`)

Organized by capability (not provider):

**Simple Generation**
- `generate_greeting`, `generate_joke` (OpenAI)
- `generate_text_anthropic`, `summarize_with_anthropic` (Anthropic)
- `generate_text_openrouter`, `compare_models_openrouter` (OpenRouter)

**Structured Output**
- `analyze_sentiment` - Returns structured `SentimentAnalysis` object

**Streaming**
- `generate_story` - Streaming text generation

**Multi-turn Conversations**
- `chat_with_context` - Chat with conversation history

**Error Handling**
- `generate_with_invalid_model` - Tests error propagation

#### Error Functions (`functions/error_functions.py`)
- `intermittent_error`: Random error injection for testing resilience

### Workflows

#### Simple Workflows (`workflows/simple_workflows.py`)
- `test_workflow`: Basic workflow testing
- `test_workflow_with_state`: Workflow state management and agent consolidation
- `order_fulfillment`: Multi-step order processing
- `data_pipeline`: Data transformation pipeline
- `tool_orchestrated_workflow`: Direct tool invocation
- `agent_research_workflow`: Agent with tool orchestration
- `agent_multi_step_workflow`: Complex multi-agent workflow

#### Agent Workflows
- **Basic**: `test_agent_basic` - Simple agent execution
- **Tools**: `test_agent_with_simple_tool`, `test_agent_with_multiple_tools`
- **Handoffs**: `test_handoff_simple`, `test_handoff_with_context`, `test_handoff_complex`
- **Collaboration**: Multi-agent coordination patterns
- **HITL**: `test_agent_with_hitl` - Human-in-the-loop workflows
- **Memory**: Session memory, user memory, multi-agent sessions

### Entities

Three entity types for testing state persistence:

- **`ShoppingCart`**: Tests state persistence and isolation
  - Methods: `add_item`, `get_total`, `get_items`, `clear`

- **`Counter`**: Tests concurrency and single-writer guarantees
  - Methods: `increment`, `get_count`, `reset`

- **`BankAccount`**: Tests state durability and transaction history
  - Methods: `deposit`, `withdraw`, `get_balance`, `get_transactions`

### Tools

Common tools for agent testing:

- **`calculate_total`**: Statistical operations (sum, average, min, max)
- **`search_database`**: Mock database search
- **`format_report`**: Report formatting (summary, detailed, compact)
- **`validate_data`**: Field validation

### Agents

Reusable agents (test-specific agents are defined inline in workflows):

- **`chat_agent`**: General chat agent for session testing

## Setup

### Prerequisites

- Python 3.11+
- AGNT5 platform running locally
- API keys for LLM providers (optional, for LLM tests)

### Installation

```bash
# From the test-bench directory
cd sdk/sdk-python/tests/integration/test-bench

# Install dependencies
uv sync

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys (optional)
vim .env
```

### Environment Configuration

See `.env.example` for all available configuration options. Required API keys:

- **OPENAI_API_KEY**: Required for OpenAI LLM functions and agent tests
- **ANTHROPIC_API_KEY**: Required for Anthropic/Claude LLM functions
- **OPENROUTER_API_KEY**: Required for OpenRouter LLM functions

Platform configuration:
- **AGNT5_COORDINATOR_ENDPOINT**: Worker coordinator URL (default: http://localhost:34186)

## Running the Test Bench

### Development Server

The recommended way to run the test bench during development:

```bash
# From the agnt5 root directory
just platform dev-server python
```

This will:
1. Build the Python SDK with maturin
2. Start the platform backend services
3. Launch the test-bench worker
4. Connect to the local coordinator

### Standalone Mode

Run the test bench directly:

```bash
# From the test-bench directory
python app.py
```

### Restarting the Worker

When developing, restart just the Python worker:

```bash
pm2 restart agnt5-python-test-bench
```

## Testing

### Component Registration

The test bench uses `auto_register=True` to automatically discover and register all components from the package. This includes:
- All functions decorated with `@function`
- All workflows decorated with `@workflow`
- All entities extending `Entity`
- All agents decorated with `@agent`
- All tools decorated with `@tool`

### Testing LLM Functions

LLM functions require API keys. Functions will gracefully skip if keys are not available:

```python
# Functions check for API keys
if not os.getenv("OPENAI_API_KEY"):
    ctx.logger.warning("OPENAI_API_KEY not set, skipping agent execution")
    return {"status": "skipped", "reason": "OPENAI_API_KEY not configured"}
```

### Testing Workflows

Workflows can be invoked via the AGNT5 client:

```python
from agnt5 import Client

client = Client(coordinator_endpoint="http://localhost:34186")

# Invoke workflow
result = client.workflow("order_fulfillment").run(
    order_id="ORDER-123",
    items=[{"id": "item-1", "qty": 2}]
)
```

### Testing Entities

Entities maintain state across invocations:

```python
# Add item to cart
client.entity("ShoppingCart", "user-123").add_item(
    item_id="widget-1",
    quantity=2,
    price=29.99
)

# Get total (state persists)
total = client.entity("ShoppingCart", "user-123").get_total()
```

## Development Workflow

### Adding New Components

1. **Add component**: Create in appropriate module (functions/, workflows/, entities.py, tools.py, agents.py)
2. **Auto-registration**: Components are automatically discovered via `auto_register=True`
3. **Test**: Restart worker and invoke via client or dev-server
4. **Iterate**: Make changes and restart worker

### Organization Principles

- **Functions**: Organized by capability (simple, LLM, error injection)
- **Workflows**: Organized by test category (simple, agent, memory, replay)
- **Entities**: All in `entities.py` (small number of entities)
- **Tools**: Common tools in `tools.py`, test-specific tools inline in workflows
- **Agents**: Reusable agents in `agents.py`, test-specific agents inline in workflows

## Differences from Other Test Directories

### test-bench vs test-service

**test-bench** (this directory) has replaced **test-service** as the primary integration test suite:
- Comprehensive agent workflow testing (20+ workflows)
- Memory, handoff, and HITL testing
- MCP verification infrastructure
- Used with `just platform dev-server python`

**test-service** has been deprecated and its components migrated here.

### test-bench vs sdk-python-benchmark

**test-bench**: Integration and functional testing
- Focus: Correctness, feature coverage, edge cases
- Purpose: SDK development and validation
- Audience: SDK developers

**sdk-python-benchmark**: Load and performance testing
- Focus: Performance, throughput, latency
- Purpose: Benchmarking and optimization
- Audience: Performance engineers, users evaluating AGNT5

## Logs and Observability

The test bench logs to stdout and integrates with the AGNT5 platform's observability:

- **Logs**: Available via platform logs API and UI
- **Traces**: OpenTelemetry traces for all operations
- **Metrics**: Performance metrics and health checks
- **MCP**: Use `verify_mcp.py` to validate observability

## Troubleshooting

### Worker Not Registering

```bash
# Check coordinator is running
curl http://localhost:34186/health

# Check worker logs
pm2 logs agnt5-python-test-bench

# Restart worker
pm2 restart agnt5-python-test-bench
```

### LLM Functions Failing

```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $OPENROUTER_API_KEY

# Check .env file
cat .env
```

### Import Errors

```bash
# Rebuild SDK
just sdk release-develop

# Reinstall test-bench
cd sdk/sdk-python/tests/integration/test-bench
uv sync
```

## References

- **Platform Documentation**: `/CLAUDE.md`
- **SDK Documentation**: `/sdk/CLAUDE.md`
- **Python SDK Documentation**: `/sdk/sdk-python/CLAUDE.md`
- **Development Server**: `just platform dev-server --help`
