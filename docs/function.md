# Function Component

## What is a Function?

A **Function** is the most fundamental building block in AGNT5 - a durable, stateless operation that can be invoked remotely and survives failures. Unlike traditional serverless functions that lose state on crashes, AGNT5 functions are designed for resilience with automatic retries, checkpointing, and replay capabilities.

Functions in AGNT5 are:
- **Durable**: Automatically retried on failures with configurable retry policies
- **Stateless**: No persistent state between invocations (use Entities for stateful operations)
- **Isolated**: Each invocation is independent and can be executed concurrently
- **Observable**: Integrated with execution context for tracing and debugging

## Why are Functions Needed?

Functions serve as the atomic units of work in AGNT5, providing several key benefits:

### 1. Fault Tolerance
Functions automatically handle transient failures through built-in retry mechanisms. If a function fails due to network issues, resource unavailability, or temporary errors, AGNT5 will retry the operation according to the configured retry policy.

### 2. Distributed Execution
Functions can be invoked from anywhere - workflows, other functions, or external systems - and executed on any available worker in your infrastructure, enabling true distributed computing.

### 3. Progressive Disclosure
Start with simple functions for basic operations, then compose them into more complex workflows and agents as your application grows. Functions are the foundation that scales from simple tasks to sophisticated multi-step processes.

### 4. Observability
Every function execution is tracked through the execution context, providing built-in tracing, logging, and monitoring capabilities without additional instrumentation.

## How to Use Functions

### Basic Function Definition

The simplest way to define a function is using the `@function` decorator:

```python
from agnt5 import function
from agnt5.context import Context

@function
async def greet(ctx: Context, name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"
```

**Key Points:**
- Functions can be `async` (recommended) or synchronous
- First parameter is always `ctx: Context` - the execution context
- Remaining parameters are your function's inputs
- Return type should be JSON-serializable

### Custom Function Name

By default, the function name matches the Python function name. You can override this:

```python
@function(name="add_numbers")
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

### Configuring Retry Policies

Functions support sophisticated retry configurations:

```python
from agnt5.durable import RetryPolicy, BackoffPolicy

@function(
    retries=RetryPolicy(
        max_attempts=5,
        initial_interval_ms=1000,
        max_interval_ms=30000
    ),
    backoff=BackoffPolicy(
        type="exponential",
        multiplier=2.0
    )
)
async def process_payment(ctx: Context, amount: float, currency: str) -> dict:
    """Process a payment with retry logic for transient failures."""
    # Payment processing logic here
    return {"status": "completed", "amount": amount, "currency": currency}
```

**Retry Policy Options:**
- `max_attempts`: Maximum number of retry attempts (default: 3)
- `initial_interval_ms`: Initial delay before first retry (default: 1000ms)
- `max_interval_ms`: Maximum delay between retries (default: 60000ms)

**Backoff Policy Types:**
- `constant`: Fixed delay between retries
- `linear`: Linearly increasing delay
- `exponential`: Exponentially increasing delay (recommended for most cases)

### Using the Context

The `Context` object provides powerful capabilities for durable execution:

#### Checkpointing Steps

For long-running functions, you can checkpoint intermediate results:

```python
@function
async def process_data_pipeline(ctx: Context, dataset_id: str) -> dict:
    """Process a multi-step data pipeline with checkpoints."""

    # Step 1: Load data (checkpointed)
    data = await ctx.run("load_data", lambda: load_from_storage(dataset_id))

    # Step 2: Transform data (checkpointed)
    transformed = await ctx.run("transform", lambda: apply_transformations(data))

    # Step 3: Validate (checkpointed)
    validation = await ctx.run("validate", lambda: validate_results(transformed))

    return {
        "dataset_id": dataset_id,
        "records_processed": len(transformed),
        "validation": validation
    }
```

**Why Checkpoints Matter:**
- If the function crashes after step 2, it will resume from step 3 on retry
- No need to reprocess expensive operations
- Ensures exactly-once execution semantics for each step

#### Making HTTP Requests

The context provides a durable HTTP client:

```python
@function
async def fetch_weather(ctx: Context, city: str) -> dict:
    """Fetch weather data from external API."""

    response = await ctx.http.get(
        url=f"https://api.weather.com/v1/current?city={city}",
        headers={"Authorization": "Bearer YOUR_TOKEN"}
    )

    return response
```

#### Using LLM Clients

Functions can leverage language models through the context:

```python
@function
async def analyze_sentiment(ctx: Context, text: str) -> str:
    """Analyze sentiment using LLM."""

    response = await ctx.llm.generate(
        prompt=[
            {"role": "system", "content": "You are a sentiment analysis expert."},
            {"role": "user", "content": f"Analyze the sentiment: {text}"}
        ],
        model="gpt-4"
    )

    return response.text
```

#### Spawning Child Invocations

Functions can spawn other functions asynchronously:

```python
@function
async def process_batch(ctx: Context, items: list) -> dict:
    """Process items in parallel."""

    # Spawn child invocations for each item
    child_ids = []
    for item in items:
        child_id = await ctx.spawn(
            handler="process_single_item",
            input_data={"item": item}
        )
        child_ids.append(child_id)

    return {"spawned_count": len(child_ids), "child_ids": child_ids}
```

### Complex Input and Output Types

Functions support rich data types:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Order:
    order_id: str
    items: List[dict]
    total: float
    customer_email: str

@function
async def process_order(ctx: Context, order: dict) -> dict:
    """Process an order with structured data."""

    # Access nested data
    order_id = order["order_id"]
    items = order["items"]
    total = order["total"]

    # Process the order
    confirmation = await ctx.run(
        "generate_confirmation",
        lambda: create_confirmation(order_id, items)
    )

    # Send notification
    await ctx.run(
        "send_email",
        lambda: send_order_confirmation(
            to=order["customer_email"],
            confirmation=confirmation
        )
    )

    return {
        "order_id": order_id,
        "status": "completed",
        "confirmation_code": confirmation["code"]
    }
```

### Error Handling

Functions should handle errors appropriately:

```python
@function
async def safe_api_call(ctx: Context, endpoint: str) -> dict:
    """Make an API call with proper error handling."""

    try:
        response = await ctx.http.get(url=endpoint)
        return {"success": True, "data": response}
    except Exception as e:
        # Log the error (automatically tracked by context)
        ctx.logger.error(f"API call failed: {e}")

        # Return error response or re-raise for retry
        return {
            "success": False,
            "error": str(e)
        }
```

## Registration and Discovery

Functions are automatically registered when your worker starts:

```python
# worker.py
from agnt5 import Worker, function

@function
async def my_function(ctx: Context, data: str) -> str:
    return f"Processed: {data}"

# Create and start worker
worker = Worker("my-service", runtime="standalone")
await worker.run()
```

The worker discovers all decorated functions and registers them with the AGNT5 platform, making them available for invocation.

## Best Practices

### 1. Keep Functions Focused
Each function should do one thing well. Break complex operations into multiple functions composed through workflows.

```python
# Good: Focused functions
@function
async def validate_input(ctx: Context, data: dict) -> bool:
    return is_valid(data)

@function
async def process_data(ctx: Context, data: dict) -> dict:
    return transform(data)

# Less ideal: Doing too much in one function
@function
async def validate_and_process(ctx: Context, data: dict) -> dict:
    if not is_valid(data):
        raise ValueError("Invalid data")
    return transform(data)
```

### 2. Use Checkpoints for Expensive Operations
Always checkpoint expensive or non-idempotent operations:

```python
@function
async def process_large_file(ctx: Context, file_url: str) -> dict:
    # Checkpoint expensive download
    content = await ctx.run("download", lambda: download_file(file_url))

    # Checkpoint expensive processing
    result = await ctx.run("process", lambda: process_content(content))

    return result
```

### 3. Design for Idempotency
Functions may be retried, so ensure they're idempotent:

```python
@function
async def update_record(ctx: Context, record_id: str, data: dict) -> dict:
    """Idempotent update using upsert."""
    # Use upsert instead of insert to handle retries
    return await ctx.run(
        "upsert_record",
        lambda: database.upsert(record_id, data)
    )
```

### 4. Leverage Async for I/O Operations
Use async functions for I/O-bound operations:

```python
@function
async def fetch_multiple_sources(ctx: Context, sources: list) -> dict:
    """Efficiently fetch from multiple sources."""
    results = await asyncio.gather(*[
        ctx.http.get(url=source) for source in sources
    ])
    return {"sources": len(sources), "results": results}
```

### 5. Configure Appropriate Retry Policies
Match retry configuration to operation characteristics:

```python
# Short-lived, quick retries for transient network issues
@function(retries=RetryPolicy(max_attempts=5, initial_interval_ms=500))
async def quick_api_call(ctx: Context, data: str) -> dict:
    return await ctx.http.get(url=f"https://api.example.com/{data}")

# Longer backoff for rate-limited APIs
@function(
    retries=RetryPolicy(max_attempts=3, initial_interval_ms=5000),
    backoff=BackoffPolicy(type="exponential", multiplier=3.0)
)
async def rate_limited_call(ctx: Context, query: str) -> dict:
    return await ctx.http.get(url=f"https://rate-limited-api.com/{query}")
```

## Function Lifecycle

### 1. Registration
When your worker starts, the `@function` decorator registers the function in the global registry.

### 2. Invocation
Functions can be invoked through:
- **Gateway HTTP/gRPC API**: External invocations
- **Workflows**: Orchestrated execution
- **Other functions**: Function composition
- **Context.spawn()**: Async child invocations

### 3. Execution
The function runs on an available worker with:
- Automatic input deserialization
- Context injection
- Checkpoint replay
- Output serialization

### 4. Completion or Retry
On success, results are returned to the caller. On failure, the retry policy determines whether to retry or fail permanently.

## Advanced Patterns

### Function Composition

```python
@function
async def extract_data(ctx: Context, source: str) -> dict:
    return {"data": "extracted"}

@function
async def transform_data(ctx: Context, data: dict) -> dict:
    return {"data": "transformed"}

@function
async def load_data(ctx: Context, data: dict) -> bool:
    return True

@function
async def etl_pipeline(ctx: Context, source: str) -> dict:
    """Compose functions into an ETL pipeline."""
    extracted = await ctx.run("extract", lambda: extract_data(ctx, source))
    transformed = await ctx.run("transform", lambda: transform_data(ctx, extracted))
    loaded = await ctx.run("load", lambda: load_data(ctx, transformed))
    return {"success": loaded}
```

### Conditional Execution

```python
@function
async def conditional_processing(ctx: Context, data: dict, mode: str) -> dict:
    """Execute different logic based on mode."""

    if mode == "fast":
        result = await ctx.run("fast_process", lambda: quick_process(data))
    elif mode == "thorough":
        result = await ctx.run("thorough_process", lambda: detailed_process(data))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result
```

## Debugging and Monitoring

### Logging

```python
@function
async def monitored_function(ctx: Context, data: str) -> dict:
    """Function with detailed logging."""
    ctx.logger.info(f"Processing started for data: {data}")

    result = await ctx.run("process", lambda: process_data(data))

    ctx.logger.info(f"Processing completed: {result}")
    return result
```

### Accessing Invocation Metadata

```python
@function
async def introspective_function(ctx: Context, data: str) -> dict:
    """Access execution metadata."""
    return {
        "invocation_id": ctx.invocation_id,
        "run_id": ctx.run_id,
        "attempt": ctx.attempt,
        "service_name": ctx.service_name,
        "result": process(data)
    }
```

## See Also

- [Entity Component](entity.md) - Stateful components with persistent state
- [Workflow Component](workflow.md) - Multi-step orchestration
- [Context API](context.md) - Detailed context capabilities
- [SDK Reference](../sdk/python/reference/decorators.md) - Full API documentation