# Entity Component

## What is an Entity?

An **Entity** in AGNT5 is a stateful component identified by a unique key. Entities represent stateful things in your application - AI agents with conversation memory, workflow orchestrators, or any business entity that needs to maintain state across interactions.

**Key Characteristics:**
- **Unique Key**: Each instance has a unique identifier (e.g., `agent-conv-123`)
- **Private State**: Built-in key-value storage per instance
- **Single-Writer**: Only one write operation per key at a time (no race conditions)
- **Durable**: State survives crashes and restarts
- **Scalable**: Different keys execute in parallel

## Why are Entities Needed?

### 1. Automatic Consistency

Entities provide single-writer consistency automatically - no locks or coordination code needed:

```python
# Two concurrent calls to increment("counter-1") execute serially
# Final count will be 2, never 1 (no lost updates)
```

### 2. AI Agent & Workflow Modeling

Entities naturally model stateful AI components:

| Entity Type | Key Pattern | Use Case |
|-------------|-------------|----------|
| AI Agent | `agent-{conversation_id}` | Chat history, context, memory |
| Workflow | `workflow-{run_id}` | Step progress, results |
| User Context | `context-{user_id}` | Preferences, personalization |

### 3. State Management & Scalability

Built-in KV storage with horizontal scaling - different keys run in parallel:

```python
# Different keys = parallel execution
await ctx.entity("agent", "conv-1").send_message(msg)  # Parallel
await ctx.entity("agent", "conv-2").send_message(msg)  # Parallel

# Same key = serial execution (consistency guaranteed)
await ctx.entity("agent", "conv-1").send_message(msg1)  # Serial
await ctx.entity("agent", "conv-1").send_message(msg2)  # Serial
```

## How to Use Entities

> **Note**: This page describes the Entity API. Check the current SDK release notes for availability.

### Basic Entity Definition

```python
from agnt5 import entity

# Create entity type
agent = entity("ConversationAgent")

# Write method (exclusive access per key)
@agent.write
async def send_message(ctx, message: str) -> dict:
    history = await ctx.get("history", [])
    history.append({"role": "user", "content": message})

    response = await call_llm(history)
    history.append({"role": "assistant", "content": response})

    ctx.set("history", history)
    return {"response": response}

# Shared method (read-only, concurrent)
@agent.shared
async def get_history(ctx) -> list:
    return await ctx.get("history", [])

# Usage from a function
@function
async def chat(ctx, conv_id: str, msg: str):
    return await ctx.entity("ConversationAgent", conv_id).send_message(msg)
```

**Key APIs:**
- `entity("name")`: Create entity type
- `@agent.write`: Write method (exclusive per key)
- `@agent.shared`: Shared method (read-only, concurrent)
- `ctx.get(key, default)` / `ctx.set(key, value)` / `ctx.delete(key)`: State operations
- `ctx.entity(type, key).method()`: Call entity from function

### Example: Research Agent

```python
research_agent = entity("ResearchAgent")

@research_agent.write
async def start_research(ctx, topic: str) -> dict:
    ctx.set("topic", topic)
    ctx.set("findings", [])
    ctx.set("status", "in_progress")
    return {"status": "started"}

@research_agent.write
async def add_finding(ctx, finding: str, source: str) -> dict:
    findings = await ctx.get("findings", [])
    findings.append({"content": finding, "source": source})
    ctx.set("findings", findings)
    return {"count": len(findings)}

@research_agent.write
async def synthesize(ctx) -> dict:
    findings = await ctx.get("findings", [])
    summary = await generate_summary(findings)
    ctx.set("summary", summary)
    ctx.set("status", "completed")
    return {"summary": summary}

@research_agent.shared
async def get_progress(ctx) -> dict:
    return {
        "status": await ctx.get("status"),
        "findings": len(await ctx.get("findings", []))
    }
```

## Common Patterns

### Conversational AI Agent

```python
agent = entity("ChatAgent")

@agent.write
async def send_message(ctx, message: str) -> dict:
    history = await ctx.get("history", [])
    history.append({"role": "user", "content": message})

    response = await ctx.llm.generate(prompt=history, model="gpt-4")
    history.append({"role": "assistant", "content": response.text})

    # Keep last 20 messages
    if len(history) > 20:
        history = history[-20:]

    ctx.set("history", history)
    return {"response": response.text}

@agent.shared
async def get_history(ctx) -> list:
    return await ctx.get("history", [])
```

### Workflow Orchestrator

```python
workflow = entity("WorkflowOrchestrator")

@workflow.write
async def start(ctx, steps: list) -> dict:
    ctx.set("steps", steps)
    ctx.set("current_step", 0)
    ctx.set("results", [])
    return {"status": "started"}

@workflow.write
async def complete_step(ctx, result: dict) -> dict:
    results = await ctx.get("results", [])
    results.append(result)
    ctx.set("results", results)
    ctx.set("current_step", len(results))
    return {"completed": len(results)}

@workflow.shared
async def get_progress(ctx) -> dict:
    return {
        "current_step": await ctx.get("current_step", 0),
        "total_steps": len(await ctx.get("steps", []))
    }
```

## Best Practices

### 1. Choose Stable, Meaningful Keys

Use unique, stable keys that identify your entity:

```python
#  Good
"agent-conv-{conv_id}"
"workflow-{run_id}"
"user-{user_id}"

# L Avoid
"abc123"  # Not descriptive
"user-{timestamp}"  # Changes every time
```

### 2. Use Shared for Reads

For read-only operations, use `@entity.shared` to enable concurrent access:

```python
@agent.shared
async def get_history(ctx) -> list:
    return await ctx.get("history", [])  # Multiple reads can run in parallel
```

### 3. Design for Concurrency

Different keys run in parallel, same keys serialize:

```python
# Different keys = parallel (scales)
await ctx.entity("agent", "conv-1").send_message(msg)  # Parallel
await ctx.entity("agent", "conv-2").send_message(msg)  # Parallel

# Same key = serial (consistent)
await ctx.entity("agent", "conv-1").send_message(msg1)  # Serial
await ctx.entity("agent", "conv-1").send_message(msg2)  # Serial

# Choose granularity wisely - one entity per conversation, not global
```


## Architecture

Entities use event sourcing for durability and single-writer consistency:

1. Each `ctx.set()` generates an event logged to Redpanda
2. State projected to CockroachDB for querying
3. Runtime serializes write handlers per key (queue per key)
4. Shared handlers can run concurrently for the same key
5. Different keys execute in parallel across workers

## Comparison with Functions

| Aspect | Functions | Entities |
|--------|-----------|----------|
| State | Stateless | Stateful (KV store) |
| Identity | No identity | Unique key per instance |
| Concurrency | Parallel by default | Serial per key, parallel across keys |
| Consistency | No consistency needed | Single-writer guarantee |
| Use Case | Transformations, API calls | Stateful AI agents, workflow state |
| Example | `process_payment()` | Conversation agent, research task |

**When to use Functions:**
- Stateless operations
- Independent requests
- Transformations, ETL

**When to use Entities:**
- Stateful AI agents
- Workflow orchestrators
- User context and personalization

## See Also

- [Function Component](function.md) - Stateless operations
- [Workflow Component](workflow.md) - Multi-step orchestration
- [Context API](context.md) - Entity context and state operations
- [Architecture: Entity Persistence (ADR-003)](../architecture/decisions/003-virtual-object-persistence.md)