# Workflow Component

## What is a Workflow?

A **Workflow** in AGNT5 is a durable, multi-step orchestration that coordinates functions, entities, and external signals. Workflows are written as async Python functions using natural control flow - the platform handles durability and fault tolerance automatically.

**Key Characteristics:**
- **Natural Code**: Write workflows like regular async functions
- **Durable**: Survives failures and resumes from the last completed step
- **Flexible**: Use if/else, loops, and any Python logic
- **Stateful**: Maintains workflow state across steps and restarts
- **Coordinated**: Built-in parallel, sequential, and signal primitives

## Why are Workflows Needed?

### 1. Multi-Step Coordination

Coordinate multiple operations with automatic fault tolerance:

```python
# If any step fails, workflow automatically retries
# State is preserved between steps
# No manual progress tracking needed
```

### 2. Long-Running Processes

Durable execution for processes that take time:

| Use Case | Example Steps | Duration |
|----------|---------------|----------|
| AI Research | Search → Analyze → Synthesize | Minutes to hours |
| Order Processing | Validate → Payment → Fulfillment | Hours to days |
| Data Pipeline | Extract → Transform → Load | Minutes to hours |

### 3. Human-in-the-Loop

Wait for external events without losing state:

```python
# Workflow pauses for approval signal
# State preserved while waiting (hours or days)
# Automatically resumes when signal received
```

## How to Use Workflows

> **Note**: Workflows are currently in active development. The API shown represents the planned design.

### Basic Workflow Definition

```python
from agnt5 import workflow

@workflow
async def process_order(ctx, order_id: str):
    # Validate order
    order = await ctx.step(validate, input={"order_id": order_id})

    # Process payment
    payment = await ctx.step(charge, input={"amount": order["total"]})

    # Fulfill order
    await ctx.step(ship, input={"order_id": order_id})

    return {"status": "completed"}
```

**Key Components:**
- `@workflow`: Decorator to register workflow
- `ctx.step()`: Run a function as a durable step (see [Context API](context.md#ctxstep---run-a-step))
- `await`: Sequential execution (waits for completion)
- Regular Python: Use if/else, loops, variables

**Context APIs for Workflows:**

Workflows use Context APIs for orchestration. For complete API documentation, see [Context API](context.md).

- `ctx.step()` - Run a function as a durable step
- `ctx.parallel()` - Run tasks concurrently
- `ctx.gather()` - Parallel with named results
- `ctx.signal()` - Wait for external events
- `ctx.timer()` - Delays and scheduling

### Example: AI Research Workflow

```python
@workflow
async def ai_research(ctx, topic: str):
    # Initialize research
    await ctx.step(start_research, input={"topic": topic})

    # Search in parallel
    papers, web = await ctx.parallel(
        ctx.step(search_academic),
        ctx.step(search_web)
    )

    # Synthesize results
    summary = await ctx.step(synthesize,
        input={"papers": papers, "web": web}
    )

    # Wait for human review
    approved = await ctx.signal(
        "research_approved",
        timeout_ms=3600000  # 1 hour
    )

    if approved.get("approved"):
        # Publish results
        result = await ctx.step(publish, input={"summary": summary})
        return {"status": "published", "result": result}
    else:
        return {"status": "cancelled"}
```

## Common Patterns

### Sequential Processing

```python
@workflow
async def data_pipeline(ctx, dataset: str):
    # Each step runs after the previous completes
    data = await ctx.step(extract, input={"dataset": dataset})
    transformed = await ctx.step(transform, input={"data": data})
    result = await ctx.step(load, input={"data": transformed})
    return result
```

### Parallel Execution

```python
@workflow
async def multi_source_analysis(ctx, query: str):
    # Initialize
    await ctx.step(setup, input={"query": query})

    # Analyze multiple sources in parallel
    results = await ctx.gather(
        db=ctx.step(analyze_db),
        logs=ctx.step(analyze_logs),
        api=ctx.step(analyze_api)
    )

    # Aggregate results
    final = await ctx.step(combine, input=results)
    return final
```

### Conditional Logic

```python
@workflow
async def conditional_deployment(ctx, version: str):
    # Build and test
    build_result = await ctx.step(build, input={"version": version})
    tests = await ctx.step(test, input={"build_id": build_result["id"]})

    if tests["passed"]:
        # Deploy to production
        result = await ctx.step(deploy, input={"build_id": build_result["id"]})
        return {"status": "deployed", "result": result}
    else:
        # Rollback
        await ctx.step(rollback, input={"build_id": build_result["id"]})
        return {"status": "failed", "reason": "tests failed"}
```

### Retry with Backoff

```python
@workflow
async def process_with_retry(ctx, job_id: str):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = await ctx.step(process, input={"job_id": job_id})
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                delay = 1000 * (2 ** attempt)
                await ctx.timer(delay_ms=delay)
            else:
                return {"status": "failed", "error": str(e)}
```

### Human Approval with Timeout

```python
@workflow
async def deploy_with_approval(ctx, version: str):
    # Build and test
    build_result = await ctx.step(build, input={"version": version})
    tests = await ctx.step(test, input={"build_id": build_result["id"]})

    if not tests["passed"]:
        return {"status": "failed", "reason": "tests failed"}

    # Wait for approval with 30-minute timeout
    approval = await ctx.signal(
        "deploy_approved",
        timeout_ms=1800000,
        default={"approved": False}
    )

    if approval["approved"]:
        result = await ctx.step(deploy, input={"build_id": build_result["id"]})
        return {"status": "deployed", "result": result}
    else:
        return {"status": "cancelled", "reason": "no approval"}
```

## Best Practices

### 1. Keep Workflows Simple

Workflows orchestrate - complex logic belongs in functions:

```python
#  Good - workflow orchestrates
@workflow
async def process_data(ctx, data_id: str):
    data = await ctx.step(extract, input={"id": data_id})
    result = await ctx.step(transform, input={"data": data})
    return result

# L Avoid - complex logic in workflow
@workflow
async def process_data(ctx, data_id: str):
    data = await ctx.step(extract, input={"id": data_id})
    # Don't do heavy computation here
    transformed = [complex_calculation(x) for x in data]
    return transformed
```

### 2. Use Parallel for Independent Tasks

Run independent tasks concurrently:

```python
# These tasks don't depend on each other - run in parallel
results = await ctx.parallel(
    ctx.step(analyze_data),
    ctx.step(fetch_metadata),
    ctx.step(validate_schema)
)
```

### 3. Handle Errors Appropriately

Use try/except for error handling:

```python
@workflow
async def safe_processing(ctx, job_id: str):
    try:
        result = await ctx.step(process, input={"job_id": job_id})
        return {"status": "success", "result": result}
    except Exception as e:
        # Log error and return gracefully
        await ctx.step(log_error, input={"error": str(e)})
        return {"status": "error", "message": str(e)}
```

### 4. Pass Data Between Steps

Use return values to pass data:

```python
@workflow
async def data_flow(ctx, input_data: dict):
    # Step 1 produces data
    step1_result = await ctx.step(step1, input=input_data)

    # Step 2 uses step1's output
    step2_result = await ctx.step(step2, input=step1_result)

    # Step 3 uses step2's output
    final = await ctx.step(step3, input=step2_result)

    return final
```

## Architecture

Workflows are executed by the Orchestration Plane:

1. Workflow registered with `@workflow` decorator
2. Triggered via Gateway API with input parameters
3. Orchestrator executes workflow step-by-step
4. Each `await` checkpoints state to the platform event store
5. On failure, workflow resumes from last checkpoint
6. Parallel tasks distributed across workers
7. Signals and timers managed by orchestrator

## Comparison with Functions and Entities

| Aspect | Functions | Entities | Workflows |
|--------|-----------|----------|-----------|
| Purpose | Single operation | Stateful object | Multi-step orchestration |
| State | Stateless | Keyed state | Workflow state |
| Execution | One invocation | Method calls | Multiple coordinated steps |
| Duration | Seconds | Long-lived | Minutes to days |
| Control Flow | Linear | Event-driven | Sequential + parallel |
| Use Case | Transform data | Chat agent | Order processing pipeline |

**When to use Functions:**
- Single, focused operation
- Quick execution (< 1 minute)
- Stateless transformations

**When to use Entities:**
- Stateful business object
- Multiple operations on same state
- Single-writer consistency needed

**When to use Workflows:**
- Multi-step processes
- Coordination across services
- Long-running operations (minutes to days)
- Human-in-the-loop scenarios

## See Also

- [Function Component](function.md) - Building blocks for workflow steps
- [Entity Component](entity.md) - Stateful components in workflows
- [Context API](context.md) - Workflow execution context