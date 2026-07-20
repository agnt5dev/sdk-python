# AGNT5 Python SDK

[![CI](https://github.com/agnt5dev/sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/agnt5dev/sdk-python/actions/workflows/ci.yml)

Build durable AI agents and workflows with automatic retries, checkpointing, and state management.

## Documentation

See the [online documentation](https://agnt5.com/sdk/python/) for example applications, user guides, and detailed API reference.

## Installation

```bash
pip install agnt5
```

## Quick Start

```python
from agnt5 import function, Context

@function
async def analyze_document(ctx: Context, document_id: str) -> dict:
    """Analyze document with automatic retries."""
    ctx.logger.info(f"Analyzing document: {document_id}")

    # Checkpoint expensive operations
    embeddings = await ctx.step("embeddings", lambda: generate_embeddings(document_id))

    # State management
    ctx.set("embedding_id", embeddings["id"])

    return {"document_id": document_id, "status": "completed"}
```

Functions automatically retry on failure. Use `ctx.step()` to checkpoint progress.

## Client Authentication

The AGNT5 client supports two authentication methods:

### Subdomain URLs (Recommended)

Access your deployment directly via its unique subdomain:

```python
from agnt5 import Client

# Production environment
client = Client("https://myproject-prod.run.agnt5.com")

# Staging environment
client = Client("https://myproject-stg.run.agnt5.com")

# Run a workflow
result = client.run("my_workflow", {"input": "data"})
```

### API Keys

For programmatic access, use service keys:

```python
from agnt5 import Client

# Option 1: Pass api_key directly
client = Client(
    gateway_url="https://api.agnt5.com",
    api_key="agnt5_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

# Option 2: Use AGNT5_API_KEY environment variable
# export AGNT5_API_KEY=agnt5_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
client = Client(gateway_url="https://api.agnt5.com")

result = client.run("my_workflow", {"input": "data"})
```

See the [Authentication Guide](https://agnt5.com/docs/guide/authentication) for more details.

## Core Components

### Functions

Stateless operations with automatic retry and checkpointing.

```python
from agnt5 import function, RetryPolicy, BackoffPolicy, BackoffType

@function(
    retries=RetryPolicy(max_attempts=5, initial_interval_ms=1000),
    backoff=BackoffPolicy(type=BackoffType.EXPONENTIAL, multiplier=2.0)
)
async def generate_summary(ctx: Context, text: str) -> dict:
    return await llm.generate(prompt=f"Summarize: {text}")
```

### Entities

Stateful components where each instance maintains its own state.

```python
from agnt5 import entity

@entity
class Conversation:
    pass

@Conversation.method
async def add_message(ctx: Context, role: str, content: str) -> list:
    messages = ctx.get("messages", [])
    messages.append({"role": role, "content": content})
    ctx.set("messages", messages)
    return messages

# Each conversation has independent message history
convo = Conversation.instance("user-123")
await convo.add_message(ctx, role="user", content="Hello!")
```

### Workflows

Orchestrate multi-step processes with parallel execution and coordination.

```python
from agnt5 import workflow

@workflow
async def research_workflow(ctx: Context, topic: str) -> dict:
    # Run searches in parallel
    results = await ctx.parallel(
        search_papers(ctx, topic),
        search_web(ctx, topic)
    )

    # Wait for human review
    reviewed = await ctx.signal("review_complete", timeout_ms=60000)

    return {"topic": topic, "results": results, "reviewed": reviewed}
```

### Tools

Functions callable by LLMs with automatic schema generation.

```python
from agnt5 import tool

@tool
async def search_knowledge_base(ctx: Context, query: str, limit: int = 10) -> list:
    """Search the knowledge base using semantic search.

    Args:
        query: Search query
        limit: Max results to return
    """
    embeddings = await generate_embeddings(query)
    return await vector_db.search(embeddings, limit=limit)

# Schema generated from type hints and docstring
schema = search_knowledge_base.get_schema()
```

### Agents

LLM-powered agents with tool orchestration.

```python
from agnt5 import Agent

agent = Agent(
    name="research_assistant",
    instructions="You are a research assistant. Use tools to find information.",
    tools=[search_knowledge_base, search_papers, summarize_text],
    model_name="gpt-4o"
)

result = await agent.run("Find recent papers on transformer architectures")
```

## Context API

```python
@function
async def example(ctx: Context, data: dict) -> dict:
    # Metadata
    ctx.run_id, ctx.attempt, ctx.logger

    # State management
    ctx.set("key", value)
    ctx.get("key", default=None)
    ctx.delete("key")

    # Checkpointing
    result = await ctx.step("step_name", lambda: expensive_op())

    # Coordination
    await ctx.sleep(5)
    await ctx.timer(delay_ms=1000)
    await ctx.signal("name", timeout_ms=5000)

    # Parallel execution
    await ctx.parallel(task1(), task2())
    await ctx.gather(db=fetch_db(), api=fetch_api())

    return {"result": result}
```

## Error Handling

```python
from agnt5.exceptions import RetryError, StateError

@function
async def robust_function(ctx: Context, data: str) -> dict:
    try:
        return await risky_operation(data)
    except RetryError as e:
        ctx.logger.error(f"Failed after {e.attempts} attempts")
        return {"error": "max_retries_exceeded"}
```

## Examples

See [examples/](examples/) directory for working examples.

## License

Apache 2.0
