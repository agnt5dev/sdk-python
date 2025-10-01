# AGNT5 Python SDK Examples

This directory contains examples demonstrating the AGNT5 Python SDK capabilities.

## Running Examples

```bash
# From the sdk-python directory
cd /path/to/sdk-python

# Run individual examples
python examples/01_basic_function.py
python examples/02_retry_policies.py
python examples/03_function_registry.py
```

## Examples Overview

### 01_basic_function.py
**Basic Function Usage**

Demonstrates:
- Simple function definition with `@function` decorator
- Context parameter usage
- State management (`ctx.get()`, `ctx.set()`)
- Checkpointing with `ctx.step()`

Key concepts:
- Every function receives `ctx: Context` as first parameter
- Use state for storing data within execution
- Checkpoint expensive operations to avoid re-execution on retry

### 02_retry_policies.py
**Retry Policies and Error Handling**

Demonstrates:
- Configuring custom retry policies
- Different backoff strategies (exponential, linear, constant)
- Handling `RetryError` exceptions
- Retry attempt tracking via `ctx.attempt`

Key concepts:
- Default: 3 retries with exponential backoff
- Customize with `RetryPolicy` and `BackoffPolicy`
- Access attempt number via `ctx.attempt`

### 03_function_registry.py
**Function Registry and Discovery**

Demonstrates:
- Automatic function registration
- Custom function names
- Function discovery via `FunctionRegistry`
- Calling functions by name

Key concepts:
- Functions auto-register on decoration
- Use `FunctionRegistry.get(name)` for discovery
- Custom names with `@function(name="custom_name")`

## Core Concepts

### Context

The `Context` object provides:
- **State Management**: `get()`, `set()`, `delete()`
- **Checkpointing**: `step()` / `run()` for expensive ops
- **Metadata**: `run_id`, `attempt`, `component_type`
- **Logging**: `ctx.logger` for structured logs

### Functions

Functions are:
- **Durable**: Automatic retries on failure
- **Stateless**: Use Context for state (or Entities in Phase 2)
- **Isolated**: Each invocation is independent
- **Observable**: Built-in logging and tracing

### Retry Policies

Configure retries with:
- `max_attempts`: Maximum retry count
- `initial_interval_ms`: First retry delay
- `max_interval_ms`: Maximum retry delay
- `BackoffPolicy`: Strategy (constant/linear/exponential)

## What's Not Yet Implemented (Phase 2+)

The following Context APIs will raise `NotImplementedError`:

- **Orchestration**: `ctx.task()`, `ctx.parallel()`, `ctx.gather()`, `ctx.spawn()`
- **Coordination**: `ctx.signal()`, `ctx.timer()`, `ctx.sleep()`
- **AI Integration**: `ctx.llm`, `ctx.tools`
- **Advanced Features**: `ctx.entity()`, `ctx.metrics()`, `ctx.secrets()`

These will be implemented when:
- Phase 2: Rust core integration + Platform integration
- Phase 3: Entity & Workflow components
- Phase 4: Full runtime with all features

## Next Steps

1. Try modifying the examples
2. Create your own functions
3. Experiment with retry policies
4. Build a simple application using functions

For more information, see the [SDK documentation](../README.md).
