# Local Development with AGNT5 Entities

## Overview

This guide explains how to use AGNT5 entities in local scripts, CLI tools, and tests without running the full AGNT5 platform infrastructure.

**TL;DR:** Use `@with_entity_context` decorator for local development:

```python
from agnt5 import with_entity_context

@with_entity_context
async def main():
    session = MyEntity('key')
    await session.do_something()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Quick Start

### Simple Script

```python
from agnt5 import with_entity_context, Entity

class Counter(Entity):
    async def increment(self):
        count = self.state.get("count", 0)
        self.state.set("count", count + 1)
        return count + 1

@with_entity_context  # ← Add this for local development
async def main():
    counter = Counter(key="my-counter")
    result = await counter.increment()
    print(f"Count: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Run it:
```bash
python my_script.py
# Output: Count: 1
```

### Interactive CLI

```python
from agnt5 import with_entity_context

@with_entity_context  # ← Decorator keeps context active for entire CLI
async def interactive_cli():
    session_id = "session-1"

    while True:
        user_input = input("Query: ").strip()
        if user_input == "quit":
            break

        # Create entity - state persists within this CLI session
        session = ResearchSession(key=session_id)
        await session.process(user_input)
        print(f"Response: {await session.get_response()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(interactive_cli())
```

## The Problem: Entity Context Requirement

### Why Do I Need a Decorator?

AGNT5 entities store state through an `EntityStateAdapter`. This adapter needs to be "in context" before you can use entities.

**Without the decorator, you'll see:**
```python
session = ResearchSession('key')
await session.initialize()
# ❌ RuntimeError: Entity requires state adapter context.
```

**Why?** The entity tries to access state but there's no adapter set up.

### How Platform vs Local Works

```
┌─────────────────────────────────────────────────────────┐
│               ON AGNT5 PLATFORM                         │
│                                                          │
│  Worker automatically provides EntityStateAdapter       │
│                            ↓                             │
│  @workflow                                               │
│  async def my_workflow(ctx: Context):                    │
│      session = MyEntity('key')  # ✓ Just works!         │
│      await session.initialize()                          │
│                                                          │
│  State: Persisted to database                           │
│  Lifecycle: Managed by platform                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              LOCAL DEVELOPMENT                           │
│                                                          │
│  You provide EntityStateAdapter via decorator           │
│                            ↓                             │
│  @with_entity_context  # ← YOU add this                 │
│  async def main():                                       │
│      session = MyEntity('key')  # ✓ Works!              │
│      await session.initialize()                          │
│                                                          │
│  State: In-memory (lost on exit)                        │
│  Lifecycle: Decorator handles setup/cleanup             │
└─────────────────────────────────────────────────────────┘
```

## Available Patterns

### Pattern 1: Decorator (Recommended)

**Best for:** Scripts, CLI tools, interactive applications

```python
from agnt5 import with_entity_context

@with_entity_context
async def main():
    # Your entity code here
    session = MyEntity('key')
    await session.do_something()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Pros:**
- ✅ Simple - one decorator, zero cleanup
- ✅ Context active for entire function duration
- ✅ Automatic cleanup when function exits

**Cons:**
- ⚠️ Easy to forget when starting new scripts
- ⚠️ Must remove when deploying to platform (safe to leave, but unnecessary)

### Pattern 2: Context Manager

**Best for:** Explicit control over context lifecycle

```python
from agnt5.entity import create_entity_context

async def main():
    async with create_entity_context():
        # Context active only in this block
        session = MyEntity('key')
        await session.do_something()
        # Context automatically cleaned up here

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Pros:**
- ✅ Explicit scope - clear when entities are usable
- ✅ Pythonic context manager pattern

**Cons:**
- ⚠️ Extra indentation
- ⚠️ More verbose than decorator

### Pattern 3: Manual Management (Advanced)

**Best for:** Custom lifecycle control, testing frameworks

```python
from agnt5.entity import create_entity_context, _entity_state_adapter_ctx

async def main():
    # Create context manually
    manager, token = create_entity_context()

    try:
        # Your entity code
        session = MyEntity('key')
        await session.do_something()
    finally:
        # Clean up manually
        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Pros:**
- ✅ Full control over setup/teardown
- ✅ Can integrate with custom frameworks

**Cons:**
- ⚠️ Verbose - most code for simple cases
- ⚠️ Easy to forget cleanup
- ⚠️ Not recommended for typical usage

### Pattern 4: Pytest Fixtures

**Best for:** Integration tests

```python
import pytest
from agnt5.entity import create_entity_context, _entity_state_adapter_ctx

@pytest.fixture
async def entity_context():
    """Provide entity context for tests."""
    manager, token = create_entity_context()
    try:
        yield
    finally:
        _entity_state_adapter_ctx.reset(token)
        manager.clear_all()

# Use in tests
async def test_my_entity(entity_context):
    session = MyEntity('test-key')
    result = await session.do_something()
    assert result == expected_value
```

**Pros:**
- ✅ Automatic setup/teardown per test
- ✅ Isolated state between tests
- ✅ Standard pytest pattern

## When to Use Each Pattern

| Use Case | Recommended Pattern | Example |
|----------|---------------------|---------|
| Simple script | `@with_entity_context` | One-off data processing |
| Interactive CLI | `@with_entity_context` | Chat bot, research tool |
| Jupyter notebook | `@with_entity_context` | Data analysis, exploration |
| Pytest tests | Fixture | Unit/integration tests |
| Custom test framework | Manual management | Special test runners |
| Library code | Don't use any! | Let caller provide context |

**Important:** For library code that will be used by others, don't add the decorator. Let the caller decide how to provide context.

## Real-World Example: Deep Research

The Deep Research blueprint demonstrates production usage:

```python
# blueprints/agnt5_deep_research/src/deep_research/main.py

from agnt5 import with_entity_context
from deep_research.workflows.research_flow import research_workflow

@with_entity_context  # ← Local development mode
async def interactive_cli(initial_query: str = None):
    """Interactive CLI for research queries.

    Local Mode: State persists in-memory for this session
    Platform Mode: State persists to database across sessions
    """
    session_id = str(uuid.uuid4())

    while True:
        user_input = input("What would you like to research? ").strip()

        if user_input.lower() == 'quit':
            break

        # Call workflow with entities
        result = await research_workflow(
            session_id=session_id,
            user_query=user_input
        )

        display_results(result)

if __name__ == "__main__":
    asyncio.run(interactive_cli())
```

**Usage:**
```bash
# Local development - no platform needed
cd blueprints/agnt5_deep_research
python -m deep_research.main "What is quantum computing?"

# Platform deployment - remove decorator
agnt5 deploy
```

## State Behavior

### Local Mode (with decorator)

**State lifecycle:**
- Created when function starts
- Stored in-memory
- Persists within function duration
- Lost when function exits

**Example:**
```python
@with_entity_context
async def demo():
    counter = Counter('my-counter')
    await counter.increment()  # count = 1
    await counter.increment()  # count = 2
    print(await counter.get_count())  # Output: 2

# After function exits, state is LOST

asyncio.run(demo())
asyncio.run(demo())  # Starts fresh, count = 1 again
```

### Platform Mode

**State lifecycle:**
- Managed by Worker
- Persisted to database
- Survives restarts
- Shared across workers
- Available until explicitly deleted

**Example:**
```python
# Workflow execution 1
@workflow
async def my_workflow(ctx: Context):
    counter = Counter('my-counter')
    await counter.increment()  # count = 1, saved to DB

# Workflow execution 2 (hours later, different worker)
@workflow
async def my_workflow(ctx: Context):
    counter = Counter('my-counter')
    await counter.increment()  # count = 2, loads from DB
```

## Common Patterns

### Pattern: Stateless Script with Entity

```python
from agnt5 import with_entity_context

class DataProcessor(Entity):
    async def process_batch(self, data: list) -> dict:
        results = self.state.get("results", [])
        results.extend(process_data(data))
        self.state.set("results", results)
        return {"processed": len(results)}

@with_entity_context
async def process_file(filepath: str):
    processor = DataProcessor(key="batch-processor")

    with open(filepath) as f:
        for batch in read_batches(f):
            await processor.process_batch(batch)

    # Get final results
    stats = await processor.get_stats()
    print(f"Processed {stats['total']} records")

if __name__ == "__main__":
    import sys
    asyncio.run(process_file(sys.argv[1]))
```

### Pattern: Multi-Session CLI

```python
@with_entity_context
async def chat_cli():
    sessions = {}  # Track multiple sessions

    while True:
        cmd = input("Command: ").strip()

        if cmd.startswith("new"):
            session_id = str(uuid.uuid4())[:8]
            sessions[session_id] = ChatSession(key=session_id)
            print(f"Created session: {session_id}")

        elif cmd.startswith("use "):
            session_id = cmd.split()[1]
            if session_id not in sessions:
                sessions[session_id] = ChatSession(key=session_id)
            current_session = sessions[session_id]

            # Chat loop for this session
            while True:
                msg = input(f"[{session_id}] > ").strip()
                if msg == "back":
                    break
                response = await current_session.send(msg)
                print(f"Bot: {response}")
```

### Pattern: Jupyter Notebook

```python
# Cell 1: Setup
from agnt5 import with_entity_context
import asyncio

# For Jupyter, wrap in decorator
@with_entity_context
async def notebook_main():
    global session  # Make available to other cells
    session = ResearchSession('notebook-session')
    await session.initialize("Climate change research")

asyncio.run(notebook_main())

# Cell 2: Continue using session
@with_entity_context
async def analyze():
    await session.add_data(new_findings)
    results = await session.get_analysis()
    return results

results = asyncio.run(analyze())
display(results)
```

## Troubleshooting

### Error: "Entity requires state adapter context"

**Cause:** Forgot to add decorator

**Fix:**
```python
# Before (broken)
async def main():
    session = MyEntity('key')

# After (works)
@with_entity_context  # ← Add this
async def main():
    session = MyEntity('key')
```

### Error: "Cannot find 'with_entity_context'"

**Cause:** Import issue

**Fix:**
```python
# Correct import
from agnt5 import with_entity_context

# Alternative
from agnt5.entity import with_entity_context
```

### State Not Persisting Between Script Runs

**Expected behavior!** Local mode uses in-memory storage.

**Options:**
1. Keep your script/CLI running for multi-turn interactions
2. Deploy to platform for persistent state
3. Implement custom file-based persistence (advanced)

**Example of persistent CLI:**
```python
@with_entity_context
async def persistent_cli():
    # State persists as long as this function runs
    session = MyEntity('session-1')

    while True:  # Keep running
        cmd = input("> ")
        if cmd == "quit":
            break
        await session.process(cmd)  # State persists across loop iterations
```

### Decorator on Platform Deployment

**Q:** Do I need to remove `@with_entity_context` when deploying?

**A:** No, it's safe to leave it! The decorator detects if platform context already exists and uses that instead.

**However**, it's cleaner to remove it for production code since it's unnecessary:

```python
# Local development
@with_entity_context
async def my_workflow(ctx: Context):
    ...

# Production (decorator removed, Worker provides context)
async def my_workflow(ctx: Context):
    ...
```

### Type Errors with EntityState

**Problem:**
```python
self.state.set("count", 5)  # Type error: 'dict' has no attribute 'set'
```

**Cause:** `self.state` is a plain dict, not an object with methods

**Fix:**
```python
# Use dict syntax
self.state["count"] = 5  # ✓ Correct
count = self.state.get("count", 0)  # ✓ Correct
```

## Migration: Local → Platform

### Step 1: Develop Locally

```python
# my_workflow.py
from agnt5 import with_entity_context, workflow, Context

@with_entity_context  # ← For local testing
@workflow
async def research_flow(ctx: Context, query: str):
    session = ResearchSession(key="session-1")
    await session.initialize(query)
    result = await session.research()
    return result

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(research_flow(query="quantum computing"))
    print(result)
```

Run locally:
```bash
python my_workflow.py
```

### Step 2: Deploy to Platform

Remove decorator and local test code:

```python
# my_workflow.py (production)
from agnt5 import workflow, Context

@workflow  # ← Decorator removed
async def research_flow(ctx: Context, query: str):
    session = ResearchSession(key="session-1")
    await session.initialize(query)
    result = await session.research()
    return result

# Remove local testing code
```

Deploy:
```bash
agnt5 deploy
```

### Step 3: Invoke via Platform

```python
from agnt5 import Client

client = Client()
result = client.workflow("research_flow").run(query="quantum computing")
print(result)
```

## Best Practices

### ✅ DO

- **Use decorator for local development** - Simplest approach
- **Remove decorator for production** - Cleaner code
- **Use meaningful entity keys** - Helps debugging
- **Document local vs platform behavior** - Help future maintainers

```python
@with_entity_context  # Local-only decorator; remove for platform deployment
async def main():
    """
    Local: State in-memory, lost on exit
    Platform: State persisted to database
    """
    pass
```

### ❌ DON'T

- **Don't rely on state persistence locally** - It's in-memory only
- **Don't use decorator in library code** - Let caller provide context
- **Don't forget cleanup** - Decorator handles it, manual doesn't
- **Don't mix patterns** - Pick one and stick with it

## Additional Resources

- [Entity Guide](entity.md) - Complete entity documentation
- [Testing Guide](../tests/integration/README.md) - Integration testing patterns
- [Deep Research Example](../../blueprints/agnt5_deep_research/) - Real-world usage
- [Workflow Guide](workflow.md) - Using entities in workflows

## Summary

**For local development:**
```python
from agnt5 import with_entity_context

@with_entity_context  # ← Add this one line
async def main():
    # Use entities freely
    session = MyEntity('key')
    await session.do_something()
```

**For platform deployment:**
```python
@workflow  # ← No @with_entity_context needed
async def my_workflow(ctx: Context):
    # Worker provides context automatically
    session = MyEntity('key')
    await session.do_something()
```

The decorator bridges local development and platform deployment, giving you the best of both worlds: easy local testing with production-grade state management on the platform.
