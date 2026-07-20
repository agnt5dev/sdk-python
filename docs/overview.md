# AGNT5 Python SDK Overview

## Introduction

The AGNT5 Python SDK provides a comprehensive framework for building durable, fault-tolerant AI applications and workflows. The SDK is designed with a layered architecture where each component builds upon the ones below it, providing progressively higher-level abstractions while maintaining strong durability guarantees.

## Architecture Layers

The SDK is organized into three distinct layers, each serving a specific purpose:

### Layer 1: Foundation

**Context** - The execution environment that provides APIs for all other components.

The Context component is the foundation of all AGNT5 applications. It provides:
- State management across retries and failures
- Checkpointing for expensive operations
- Orchestration APIs for complex workflows
- Observability through logging and metrics
- Access to platform services (LLM, secrets, configuration)

Every component in the SDK receives a Context instance, making it the universal interface for interacting with the AGNT5 platform.

### Layer 2: Durability Primitives

These components provide **durability guarantees** - they survive failures, automatically retry operations, and maintain state across restarts:

**Function** - Durable stateless operations
- Automatic retry with configurable policies
- Multiple backoff strategies (exponential, linear, constant)
- State management through Context
- Checkpointing for idempotent operations
- Function registry for discovery

**Entity** - Durable stateful components
- Persistent state with single-writer consistency
- Method-based API for state operations
- Automatic state persistence and recovery
- Support for both exclusive and shared operations
- Session pattern implementation (conversations, multi-agent coordination)

**Workflow** - Durable multi-step orchestration
- Coordinate multiple functions and entities
- Sequential and parallel execution patterns
- Signal-based coordination
- Human-in-the-loop workflows
- Timer-based scheduling

All Layer 2 components are built on top of Context and provide the core reliability guarantees that make AGNT5 applications production-ready.

### Layer 3: High-Level Abstractions

These components leverage the durability primitives to provide specialized functionality:

**Tool** - Agent capabilities (built on Function)
- Automatic schema extraction from Python code
- Multiple tool types (Function, Hosted, MCP, OpenAPI)
- Confirmation policies for dangerous operations
- Rich metadata with examples and constraints
- Inherits durability from Function primitive

**Agent** - LLM-driven autonomous agents (built on Tool + Entity + Workflow)
- LLM-powered reasoning and planning
- Dynamic tool selection and execution
- Memory integration for long-term knowledge
- Session-aware for conversation context
- Multi-agent coordination patterns
- Streaming support for real-time responses

Layer 3 components don't add new durability guarantees - they build on the reliability of Layer 2 while providing higher-level developer experiences.

## Component Dependencies

```
┌─────────────────────────────────────────────┐
│  Layer 3: High-Level Abstractions           │
│  ┌─────────┐  ┌──────────────────────────┐  │
│  │  Tool   │  │       Agent              │  │
│  │         │◄─┤  (Tool + Entity +       │  │
│  └────┬────┘  │   Workflow)              │  │
│       │       └──────────────────────────┘  │
└───────┼──────────────────────────────────────┘
        │
┌───────┼──────────────────────────────────────┐
│  Layer 2: Durability Primitives              │
│  ┌────▼──────┐  ┌─────────┐  ┌──────────┐   │
│  │ Function  │  │ Entity  │  │ Workflow │   │
│  └────┬──────┘  └────┬────┘  └────┬─────┘   │
│       │              │            │          │
└───────┼──────────────┼────────────┼──────────┘
        │              │            │
        ▼              ▼            ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Foundation                         │
│  ┌──────────────────────────────────────┐   │
│  │           Context                    │   │
│  │  (State, Orchestration, LLM, Obs.)   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Component Overview

### Context (Foundation)

**Purpose**: Execution environment providing APIs for all components

**Key Features**:
- State management (`get`, `set`, `delete`)
- Checkpointing (`step`, `run`)
- Orchestration (`task`, `parallel`, `gather`, `spawn`)
- Coordination (`signal`, `timer`, `sleep`)
- AI Integration (`llm.generate`, `llm.stream`)
- Observability (`logger`, `metrics`, `trace_span`)
- Configuration (`secrets`, `config`, `headers`)
- Messaging (`send_to`, `subscribe`)

**Dependencies**: None (foundation layer)

**Status**: Partial implementation (state, checkpointing, logging complete; orchestration/LLM pending)

### Function (Durability Primitive)

**Purpose**: Durable stateless operations with automatic retry

**Key Features**:
- `@function` decorator for defining handlers
- Automatic retry with exponential/linear/constant backoff
- Context injection for state and metadata
- Checkpointing via `ctx.step()` for idempotency
- Function registry for discovery
- Sync and async function support

**Dependencies**: Context

**Status**: Implemented

**Example**:
```python
from agnt5 import function, Context

@function(retries={"max_attempts": 3}, backoff={"type": "exponential"})
async def process_data(ctx: Context, data: dict) -> dict:
    # State persists across retries
    ctx.set("attempt_count", ctx.attempt)

    # Checkpoint expensive operations
    result = await ctx.step("fetch", lambda: fetch_external_data(data))

    return {"processed": result}
```

### Entity (Durability Primitive)

**Purpose**: Durable stateful components with persistent state

**Key Features**:
- `entity()` function for defining stateful components
- `@entity.method` decorator for operations
- Single-writer consistency per entity instance
- Automatic state persistence and recovery
- Session pattern support (conversations, multi-agent)

**Dependencies**: Context, Function patterns

**Status**: Not implemented

**Example Pattern**:
```python
from agnt5 import entity, Context

UserAccount = entity("UserAccount")

@UserAccount.method
async def deposit(ctx: Context, amount: float) -> dict:
    balance = ctx.get("balance", 0.0)
    new_balance = balance + amount
    ctx.set("balance", new_balance)
    return {"balance": new_balance}

# Usage: Each user_id gets isolated state
account = UserAccount(key="user_123")
result = await account.deposit(amount=100.0)
```

### Workflow (Durability Primitive)

**Purpose**: Durable multi-step orchestration with coordination

**Key Features**:
- `@workflow` decorator for defining workflows
- Sequential and parallel task execution
- Conditional branching and loops
- Signal-based coordination between workflows
- Human-in-the-loop approvals
- Timer-based scheduling

**Dependencies**: Context, Function, Entity

**Status**: Not implemented

**Example Pattern**:
```python
from agnt5 import workflow, Context

@workflow
async def order_fulfillment(ctx: Context, order_id: str) -> dict:
    # Sequential steps
    payment = await ctx.task(process_payment, order_id)

    # Parallel execution
    shipping, notification = await ctx.parallel(
        ctx.task(schedule_shipping, order_id),
        ctx.task(send_confirmation, order_id)
    )

    # Wait for external signal
    await ctx.signal.wait("order_shipped")

    return {"status": "completed", "tracking": shipping}
```

### Tool (High-Level Abstraction)

**Purpose**: Agent capabilities with automatic schema extraction

**Key Features**:
- `@tool()` decorator with auto-schema from docstrings
- Multiple tool types (Function, Hosted, MCP, OpenAPI)
- Confirmation policies for dangerous operations
- Type hints and docstring parsing
- Tool composition patterns
- Context access for advanced operations

**Dependencies**: Function (inherits durability)

**Status**: Not implemented

**Example Pattern**:
```python
from agnt5 import tool

@tool(confirmation=True)
def delete_database(database_name: str) -> dict:
    """Delete a database permanently.

    Args:
        database_name: Name of the database to delete

    Returns:
        Status of deletion operation

    Warning:
        This operation is irreversible.
    """
    # Schema extracted automatically from docstring and type hints
    # Confirmation required before execution
    return {"status": "deleted", "database": database_name}
```

### Agent (High-Level Abstraction)

**Purpose**: LLM-driven autonomous agents with reasoning capabilities

**Key Features**:
- LLM-powered reasoning and planning
- Dynamic tool selection and orchestration
- Memory integration for long-term knowledge
- Session-aware for conversation context
- Multi-agent coordination patterns
- Streaming responses for real-time UX
- Agent handoff patterns

**Dependencies**: Tool, Entity (for sessions), Workflow (for orchestration)

**Status**: Not implemented

**Example Pattern**:
```python
from agnt5 import Agent, LanguageModel, Session, Memory

session = Session(id="chat-123", user_id="user-456")
memory = Memory(service=VectorMemoryService())
lm = LanguageModel()

agent = Agent(
    name="research_assistant",
    model=lm,
    instructions="You are a helpful research assistant.",
    tools=[search_papers, analyze_paper, generate_summary],
    session=session,
    memory=memory
)

# Agent autonomously selects and executes tools
result = await agent.run("Summarize recent work on transformers")
```

## Session Pattern (Entity-Based)

**Important Note**: Session is not a separate component - it's a pattern implemented using Entity.

A Session is simply an Entity with conversation-oriented methods:
- `add_message()` - Add to conversation history
- `get_history()` - Retrieve conversation
- `set_state()` - Store session state
- Multi-agent coordination through shared entity state

Example:
```python
# Session is an Entity pattern
ConversationSession = entity("Session")

@ConversationSession.method
async def add_message(ctx: Context, role: str, content: str):
    history = ctx.get("messages", [])
    history.append({"role": role, "content": content})
    ctx.set("messages", history)

session = ConversationSession(key="chat-123")
await session.add_message(role="user", content="Hello!")
```

## Capability overview

### Core contracts

**Status**: Released as v0.2.0

**What's Available**:
- Context: State management, checkpointing, logging
- Function: Complete implementation with retry, backoff, registry
- 91% test coverage (56 tests)
- Working examples and documentation

**What Developers Can Build**:
- Durable functions with automatic retry
- State management across failures
- Checkpointed operations for idempotency
- Function discovery and registry

**Limitations**:
- In-memory state (not persisted across process restarts)
- In-memory checkpoints
- No platform integration
- No orchestration, LLM, or coordination features

### Platform integration

**Focus**: Connect to AGNT5 platform services

**Features**:
- Rust core integration for performance
- gRPC communication with Gateway and Execution Engine
- Event sourcing with Redpanda
- State projections with CockroachDB
- Orchestration APIs (`task`, `parallel`, `gather`, `spawn`)
- LLM integration (`ctx.llm`)
- Signals and timers
- Workflow component
- Tool component (built on durable Functions)
- Agent component (built on Tools + Entity patterns)
- Durable HTTP client
- Metrics and distributed tracing

**Benefits**:
- True durability (state survives process restarts)
- Distributed execution
- Exactly-once semantics
- Production-grade observability

### Advanced features

**Focus**: Entity component and advanced patterns

**Features**:
- Entity component with persistent state
- Single-writer consistency guarantees
- Advanced session patterns
- Production hardening
- Secrets management
- Configuration service integration
- Advanced messaging patterns
- Human-in-the-loop workflows

## Development Principles

### 1. Progressive Disclosure

Start simple, reveal complexity only when needed:
- Basic: `@function` decorator with defaults
- Intermediate: Custom retry policies and backoff
- Advanced: Checkpointing, state management, orchestration

### 2. Type Safety

Leverage Python type hints for:
- Automatic schema extraction (Tools)
- IDE autocomplete and validation
- Runtime type checking
- Documentation generation

### 3. Developer Experience

Prioritize ease of use:
- Intuitive decorators (`@function`, `@tool`, `@workflow`)
- Clear error messages
- Comprehensive examples
- Detailed documentation

### 4. Reliability First

Durability primitives provide strong guarantees:
- Automatic retry on transient failures
- State persistence across restarts
- Exactly-once execution semantics
- Fault tolerance by default

### 5. Composability

Components work together seamlessly:
- Context is universal interface
- Tools built on Functions
- Agents compose Tools, Entities, Workflows
- Clear dependency hierarchy

## Migration Path

### Adopting platform integration

**No Breaking Changes**:
- Existing `@function` decorators work as-is
- State and checkpointing APIs remain the same
- Code continues to work locally

**Opt-In Platform Features**:
- Connect to platform with configuration
- Orchestration APIs become available
- State automatically becomes durable
- Functions can call remote functions

**Incremental Adoption**:
- Mix local and platform-integrated code
- Migrate functions one at a time
- Test with local platform instance

### Adopting advanced features

**Additive Changes**:
- Entity component becomes available
- Advanced patterns enabled
- New APIs added to Context
- Existing code continues to work

## Getting Started

### Installation

```bash
pip install agnt5
```

### Hello World

```python
from agnt5 import function, Context

@function
async def greet(ctx: Context, name: str) -> str:
    ctx.logger.info(f"Greeting {name}")
    return f"Hello, {name}!"

# Run locally
result = await greet(name="World")
print(result)  # "Hello, World!"
```

### With Retry and State

```python
from agnt5 import function, Context

@function(
    retries={"max_attempts": 3, "initial_interval_ms": 1000},
    backoff={"type": "exponential", "multiplier": 2.0}
)
async def fetch_data(ctx: Context, url: str) -> dict:
    # Track attempts
    attempt = ctx.attempt
    ctx.set("last_attempt", attempt)

    # Checkpoint expensive operations
    data = await ctx.step("fetch", lambda: http_get(url))
    parsed = await ctx.step("parse", lambda: parse_json(data))

    return parsed
```

## Next Steps

1. **Read Component Docs**: Detailed documentation for each component
   - [Context API](context.md)
   - [Function Component](function.md)
   - [Entity Component](entity.md)
   - [Workflow Component](workflow.md)
   - [Tool Component](tool.md)
   - [Agent Component](agent.md)
   - [Batch Evaluation](batch_eval.md) - Evaluate components with multiple inputs and scoring

2. **Explore Examples**: Working code in `examples/` directory
   - Basic functions
   - Retry policies
   - Function registry


3. **Check Status**: Track implementation progress
   - [SDK Status](../../docs/status/sdk-python-status.md)

4. **Join Community**: Get help and share feedback
   - GitHub Issues
   - Documentation
   - Examples

## Summary

The AGNT5 Python SDK provides a layered architecture for building reliable AI applications:

- **Layer 1 (Foundation)**: Context provides universal APIs
- **Layer 2 (Durability)**: Function, Entity, Workflow provide fault tolerance
- **Layer 3 (High-Level)**: Tool and Agent provide specialized abstractions

Each layer builds on the one below, with clear dependencies and separation of concerns. Start with Functions, then add platform integration and advanced Entity patterns as needed.

The SDK is designed for progressive disclosure - simple things are simple, complex things are possible.
