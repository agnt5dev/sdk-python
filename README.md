# AGNT5 Python SDK

Build durable, resilient agent-first applications with the AGNT5 Python SDK.

**Version:** 0.2.0
**Status:** Phase 1 - Core Components Implemented (In Development)

## Installation

```bash
# Using uv (recommended)
uv add agnt5

# Using pip
pip install agnt5
```

## Quick Start

### Basic Function

```python
from agnt5 import Context, function

@function
async def greet(ctx: Context, name: str) -> str:
    """Simple greeting function."""
    ctx.logger.info(f"Greeting user: {name}")
    return f"Hello, {name}!"

# Usage
ctx = Context(run_id="example-1")
result = await greet(ctx, "Alice")
print(result)  # "Hello, Alice!"
```

### State Management

```python
@function
async def process_data(ctx: Context, data: str) -> dict:
    """Function with state management."""
    # Store in state
    ctx.set("input_data", data)

    # Retrieve from state
    stored = await ctx.get("input_data")

    return {"processed": data.upper(), "length": len(data)}
```

### Retry Policies

```python
from agnt5 import RetryPolicy, BackoffPolicy, BackoffType

@function(
    retries=RetryPolicy(max_attempts=5, initial_interval_ms=500),
    backoff=BackoffPolicy(type=BackoffType.EXPONENTIAL, multiplier=2.0)
)
async def api_call(ctx: Context, endpoint: str) -> dict:
    """API call with custom retry policy."""
    # Your API logic here
    return {"status": "success"}
```

### Checkpointing

```python
@function
async def expensive_pipeline(ctx: Context, dataset_id: str) -> dict:
    """Function with checkpointing for expensive operations."""

    # Step 1: Load data (checkpointed)
    data = await ctx.step("load_data", lambda: load_from_storage(dataset_id))

    # Step 2: Transform data (checkpointed)
    transformed = await ctx.step("transform", lambda: apply_transformations(data))

    # Step 3: Validate (checkpointed)
    valid = await ctx.step("validate", lambda: validate_results(transformed))

    return {"dataset_id": dataset_id, "result": transformed, "valid": valid}
```

### Agent with Tools

```python
from agnt5 import Agent, Context, tool
from agnt5.lm import OpenAILanguageModel

@tool(auto_schema=True)
async def search_database(ctx: Context, query: str) -> list:
    """Search internal database."""
    ctx.logger.info(f"Searching for: {query}")
    return [{"id": 1, "title": "Result", "content": "..."}]

# Create agent
lm = OpenAILanguageModel()
agent = Agent(
    name="assistant",
    model=lm,
    instructions="You are a helpful assistant. Use tools when needed.",
    tools=[search_database],
    model_name="gpt-4o-mini",
)

# Run agent
result = await agent.run("Find information about Python async")
print(result.output)
```

## Core Components

### Context

The execution environment for all AGNT5 components:

**Implemented (Phase 1):**
- ✅ **State Management**: `get()`, `set()`, `delete()`
- ✅ **Checkpointing**: `step()` / `run()` for expensive ops
- ✅ **Metadata**: `run_id`, `attempt`, `component_type`
- ✅ **Logging**: `ctx.logger` for structured logs
- ✅ **Orchestration**: `task()`, `parallel()`, `gather()`
- ✅ **Coordination**: `signal()`, `timer()`, `sleep()`

**Coming in Phase 2:**
- 🔄 **Platform Integration**: `spawn()` for durable task execution
- 🔄 **Advanced Features**: `metrics()`, `secrets()`

### Functions

Durable, stateless operations with automatic retry:

**Features:**
- ✅ Automatic retry with configurable policies
- ✅ Multiple backoff strategies (exponential, linear, constant)
- ✅ Checkpoint support for expensive operations
- ✅ Function registry for discovery
- ✅ Sync/async function support
- ✅ Context injection

**Configuration:**
```python
@function(
    name="custom_name",                    # Optional: custom function name
    retries=RetryPolicy(                   # Optional: retry configuration
        max_attempts=5,
        initial_interval_ms=1000,
        max_interval_ms=30000
    ),
    backoff=BackoffPolicy(                 # Optional: backoff strategy
        type=BackoffType.EXPONENTIAL,
        multiplier=2.0
    )
)
async def my_function(ctx: Context, ...) -> ...:
    pass
```

### Entity

Stateful components with single-writer consistency:

**Features:**
- ✅ Type-safe entity definitions with `@entity` decorator
- ✅ Single-writer consistency (operations serialized per entity key)
- ✅ State persistence across method calls
- ✅ Parallel execution for different entity keys
- ✅ Method registration and invocation
- ✅ Context integration for logging and orchestration

**Example:**
```python
from agnt5 import Context, entity

@entity
class Counter:
    pass

@Counter.method
async def increment(ctx: Context, amount: int = 1) -> int:
    current = ctx.get("count", 0)
    new_count = current + amount
    ctx.set("count", new_count)
    return new_count

# Usage
counter = Counter.instance("counter-1")
result = await counter.increment(ctx, amount=5)
```

### Tool

LLM-callable functions with automatic schema generation:

**Features:**
- ✅ Automatic JSON schema generation from type hints
- ✅ Support for primitive types, lists, dicts, and unions
- ✅ Optional parameters with defaults
- ✅ Manual schema override support
- ✅ Tool registry for discovery
- ✅ Context injection for state and logging

**Example:**
```python
from agnt5 import Context, tool
from typing import List

@tool(auto_schema=True)
async def search_web(ctx: Context, query: str, max_results: int = 5) -> List[dict]:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    ctx.logger.info(f"Searching for: {query}")
    # Implementation...
    return [{"title": "Result", "url": "https://..."}]

# Get JSON schema for LLM
schema = search_web.get_schema()
```

### Workflow

Orchestration primitives for complex task coordination:

**Features:**
- ✅ State management and checkpointing
- ✅ Parallel task execution with `ctx.parallel()`
- ✅ Named result gathering with `ctx.gather()`
- ✅ Signal-based coordination with `ctx.signal()`
- ✅ Timer scheduling with `ctx.timer()`
- ✅ Sleep delays with `ctx.sleep()`
- ✅ Conditional logic and loops
- ✅ Workflow registry

**Example:**
```python
from agnt5 import Context, workflow

@workflow
async def process_pipeline(ctx: Context, dataset_id: str) -> dict:
    """Multi-stage data processing pipeline."""

    # Parallel task execution
    results = await ctx.parallel(
        extract_task(ctx, dataset_id),
        validate_task(ctx, dataset_id),
    )

    # Wait for signal
    signal_data = await ctx.signal("approval_signal", timeout_ms=60000)

    # Sleep before next step
    await ctx.sleep(5000)

    return {"status": "complete", "results": results}
```

### Agent

LLM-driven autonomous agents with tool orchestration:

**Features:**
- ✅ LLM integration (OpenAI, matches Rust sdk-core interface)
- ✅ Tool selection and execution
- ✅ Multi-turn reasoning loop
- ✅ Context and state management
- ✅ Multi-turn conversations with `agent.chat()`
- ✅ Configurable iteration limits and temperature
- ✅ Tool error handling

**Example:**
```python
from agnt5 import Agent, Context, tool
from agnt5.lm import OpenAILanguageModel

@tool(auto_schema=True)
async def search_web(ctx: Context, query: str) -> list:
    # Search implementation
    return [{"title": "Result", "url": "..."}]

lm = OpenAILanguageModel()
agent = Agent(
    name="researcher",
    model=lm,
    instructions="You are a research assistant.",
    tools=[search_web],
    model_name="gpt-4o-mini",
    max_iterations=10,
)

result = await agent.run("What are the latest AI trends?")
print(result.output)
```

### Types

**Retry Configuration:**
- `RetryPolicy`: Configure retry behavior
  - `max_attempts`: Maximum retry count (default: 3)
  - `initial_interval_ms`: First retry delay (default: 1000ms)
  - `max_interval_ms`: Maximum retry delay (default: 60000ms)

- `BackoffPolicy`: Configure backoff strategy
  - `type`: CONSTANT, LINEAR, or EXPONENTIAL (default: EXPONENTIAL)
  - `multiplier`: Backoff multiplier (default: 2.0)

**Exceptions:**
- `AGNT5Error`: Base exception
- `ConfigurationError`: Invalid configuration
- `ExecutionError`: Function execution failure
- `RetryError`: Exceeded max retry attempts
- `StateError`: State operation failure
- `CheckpointError`: Checkpoint operation failure

## Examples

See the [examples/](examples/) directory for complete working examples:

1. **01_basic_function.py** - Basic function usage, state management, checkpointing
2. **02_retry_policies.py** - Retry policies, error handling, backoff strategies
3. **03_function_registry.py** - Function registry and discovery
4. **04_entity_basic.py** - Entity component for stateful operations
5. **05_entity_coordination.py** - Entity coordination and advanced patterns
6. **06_workflow_basic.py** - Workflow orchestration and patterns
7. **07_workflow_advanced.py** - Advanced workflow features
8. **08_tool_basic.py** - Tool implementation for LLM integration
9. **09_tool_advanced.py** - Advanced tool patterns and schemas
10. **10_agent_basic.py** - Agent implementation with LLM and tool orchestration

Run examples:
```bash
uv run python examples/01_basic_function.py
uv run python examples/04_entity_basic.py
uv run python examples/06_workflow_basic.py
uv run python examples/08_tool_basic.py
# Set OPENAI_API_KEY for agent examples
export OPENAI_API_KEY=your-key
uv run python examples/10_agent_basic.py
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=agnt5

# Run specific test file
uv run pytest tests/test_agent.py -v
```

Current test coverage: **88%** (140 tests passing)

## Development Roadmap

### Phase 1: Core SDK Components (In Progress)
- ✅ Core contracts (Context, Function)
- ✅ Retry policies and error handling
- ✅ State management and checkpointing
- ✅ Function registry
- ✅ Entity component (stateful operations with single-writer consistency)
- ✅ Tool component (LLM function integration with auto-schema)
- ✅ Workflow component (orchestration with signals and coordination)
- ✅ Agent component (LLM-driven autonomous agents with tool orchestration)
- ✅ Language Model interface (OpenAI integration, matches Rust sdk-core)
- ✅ Orchestration APIs (task, parallel, gather, sleep, timer, signal)
- ✅ Comprehensive tests (140 tests, 88% coverage) and examples

**Next in Phase 1:**
- 🔄 Durability and persistence layer
- 🔄 Event sourcing integration
- 🔄 Platform communication
- 🔄 Additional LM providers
- 🔄 Enhanced error handling and recovery

### Phase 2: Production Features
- 🔄 PyO3 bindings to Rust sdk-core for performance
- 🔄 Full LM provider support (Anthropic, Azure, Bedrock, Groq, OpenRouter)
- 🔄 Multi-agent coordination primitives
- 🔄 Observability (metrics, tracing with OpenTelemetry)
- 🔄 Secrets management integration
- 🔄 Human-in-the-loop workflows
- 🔄 Streaming responses for LLM and workflows

## API Reference

### Context

```python
class Context:
    # Metadata
    run_id: str
    step_id: Optional[str]
    attempt: int
    component_type: str

    # State Management
    async def get(key: str, default: Any = None) -> Any
    def set(key: str, value: Any) -> None
    def delete(key: str) -> None

    # Checkpointing
    async def step(name: str, func: Callable) -> T
    async def run(name: str, func: Callable) -> T  # Alias for step()

    # Logging
    def log() -> Logger
    logger: Logger  # Property alias
```

### Function Decorator

```python
def function(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    retries: Optional[RetryPolicy] = None,
    backoff: Optional[BackoffPolicy] = None
) -> Callable
```

### Function Registry

```python
class FunctionRegistry:
    @staticmethod
    def register(config: FunctionConfig) -> None

    @staticmethod
    def get(name: str) -> Optional[FunctionConfig]

    @staticmethod
    def all() -> Dict[str, FunctionConfig]

    @staticmethod
    def clear() -> None
```

## Design Philosophy

**Phase 1 (Current):**
- Pure Python implementation
- Clean API surface matching the spec
- Immediate developer value
- Foundation for future integration

**Phase 2+ (Future):**
- Rust core for performance-critical operations
- Platform integration via gRPC
- Event sourcing and durability guarantees
- Full orchestration capabilities

## Contributing

The SDK is in active development. For contributions:

1. Follow existing code style and patterns
2. Add tests for new features
3. Update documentation
4. Run tests before submitting: `uv run pytest`

## Documentation

- [Context API Spec](docs/context.md)
- [Function Component Spec](docs/function.md)
- [Entity Component Spec](docs/entity.md) *(Phase 2)*
- [Workflow Component Spec](docs/workflow.md) *(Phase 2)*

## License

Apache 2.0

## Support

- **Documentation**: https://docs.agnt5.com
- **Issues**: https://github.com/agnt5/agnt5/issues
- **Homepage**: https://agnt5.com
