# Function Component

## What is a Function?

A **Function** is the most fundamental building block in AGNT5 - a durable, stateless operation that can be invoked remotely and survives failures. Unlike traditional serverless functions that lose state on crashes, AGNT5 functions are designed for resilience with automatic retries, checkpointing, and replay capabilities.

Functions in AGNT5 are:
- **Durable**: Automatically retried on failures with configurable retry policies
- **Stateless**: No persistent state between invocations (use Entities for stateful operations)
- **Isolated**: Each invocation is independent and can be executed concurrently
- **Observable**: Integrated with execution context for tracing and debugging
- **Type-Safe**: Full support for type hints and Pydantic models

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
from agnt5 import function, Context

@function
async def greet(ctx: Context, name: str) -> str:
    """Greet a user by name."""
    ctx.logger.info(f"Greeting {name}")
    return f"Hello, {name}!"
```

**Key Points:**
- Functions can be `async` (recommended) or synchronous
- First parameter is `ctx: Context` - the execution context (optional, see below)
- Remaining parameters are your function's inputs
- Return type should be JSON-serializable
- Type hints are automatically extracted for schema generation

### Functions Without Context (Optional)

If your function doesn't need logging, state, or checkpointing, you can omit the context parameter:

```python
@function
async def add(a: int, b: int) -> int:
    """Simple function without context."""
    return a + b

@function
async def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y
```

**When to use context:**
- ✅ Need logging: `ctx.logger.info(...)`
- ✅ Need checkpointing: `await ctx.step(...)`
- ✅ Need to call other functions: `await ctx.task(...)`
- ✅ Need state management: `await ctx.get(...)` / `await ctx.set(...)`
- ❌ Simple pure functions: Omit context for cleaner code

### Custom Function Name

By default, the function name matches the Python function name. You can override this:

```python
@function(name="add_numbers")
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

### Configuring Retry Policies (Simplified)

Functions support flexible retry configurations with simple syntax:

```python
# Simple: Just specify max attempts
@function(retries=5)
async def fetch_data(url: str) -> dict:
    """Fetch data with 5 retry attempts."""
    # ... fetch logic
    return {"data": "..."}

# Simple: Specify backoff type
@function(retries=3, backoff="exponential")
async def process_payment(amount: float, currency: str) -> dict:
    """Process payment with exponential backoff."""
    # ... payment logic
    return {"status": "completed"}

# Advanced: Dict configuration
@function(
    retries={"max_attempts": 5, "initial_interval_ms": 1000, "max_interval_ms": 30000},
    backoff={"type": "exponential", "multiplier": 2.0}
)
async def external_api_call(endpoint: str) -> dict:
    """Call external API with detailed retry configuration."""
    # ... API call logic
    return {"result": "..."}
```

**Retry Configuration Options:**

**Simple Forms:**
- `retries=5` - Just specify max attempts (uses defaults for intervals)
- `backoff="exponential"` - Simple string for backoff type: "constant", "linear", or "exponential"

**Dict Forms:**
- `retries={"max_attempts": 5, "initial_interval_ms": 1000}` - Specify retry parameters
- `backoff={"type": "exponential", "multiplier": 2.0}` - Specify backoff parameters

**Backoff Types:**
- `"constant"`: Fixed delay between retries
- `"linear"`: Linearly increasing delay
- `"exponential"`: Exponentially increasing delay (recommended)

### Using Pydantic Models (Automatic Validation)

Functions support Pydantic models for rich type validation and schema generation:

```python
from pydantic import BaseModel
from agnt5 import function, Context

class UserInput(BaseModel):
    name: str
    age: int
    email: str

class UserOutput(BaseModel):
    greeting: str
    is_adult: bool
    user_id: str

@function
async def process_user(ctx: Context, user: UserInput) -> UserOutput:
    """Process user data with automatic validation."""
    ctx.logger.info(f"Processing user: {user.name}")

    return UserOutput(
        greeting=f"Hello, {user.name}!",
        is_adult=user.age >= 18,
        user_id=f"user_{user.email.split('@')[0]}"
    )
```

**Benefits of Pydantic:**
- ✅ Automatic input validation
- ✅ Rich schema generation (nested objects, constraints, etc.)
- ✅ Better IDE autocomplete
- ✅ Clear type contracts
- ✅ Supports both Pydantic v1 and v2

### Using the Context

The `Context` object provides powerful capabilities for durable execution:

#### Checkpointing Steps (Improved API)

For long-running functions, you can checkpoint intermediate results. Pass coroutines directly - no lambda wrapping needed:

```python
@function
async def process_data_pipeline(ctx: Context, dataset_id: str) -> dict:
    """Process a multi-step data pipeline with checkpoints."""

    # Step 1: Load data (checkpointed) - pass coroutine directly
    data = await ctx.step("load_data", load_from_storage(dataset_id))

    # Step 2: Transform data (checkpointed)
    transformed = await ctx.step("transform", apply_transformations(data))

    # Step 3: Validate (checkpointed)
    validation = await ctx.step("validate", validate_results(transformed))

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

**Improved API:**
```python
# ✅ NEW: Pass coroutines directly (recommended)
result = await ctx.step("load", load_data())

# ✅ Also supported: Pass async function reference (called automatically)
result = await ctx.step("load", load_data)

# ❌ OLD (still works but not recommended): Lambda wrapping
result = await ctx.step("load", lambda: load_data())
```

#### State Management (Async API)

Store and retrieve data within function execution:

```python
@function
async def stateful_processing(ctx: Context, data: dict) -> dict:
    """Function with state management."""

    # Store state
    await ctx.set("input_data", data)
    await ctx.set("processed_count", 0)

    # Retrieve state
    stored_data = await ctx.get("input_data")
    count = await ctx.get("processed_count", default=0)

    # Update state
    await ctx.set("processed_count", count + 1)

    # Delete state
    await ctx.delete("processed_count")

    return {"processed": stored_data}
```

**State API:**
- `await ctx.get(key, default=None)` - Get value from state
- `await ctx.set(key, value)` - Set value in state
- `await ctx.delete(key)` - Delete key from state

#### Logging

Use structured logging through the context:

```python
@function
async def monitored_function(ctx: Context, data: str) -> dict:
    """Function with detailed logging."""
    ctx.logger.info(f"Processing started for: {data}")
    ctx.logger.debug(f"Debug details: {len(data)} chars")

    result = await ctx.step("process", process_data(data))

    ctx.logger.info(f"Processing completed successfully")
    return result
```

### Calling Other Functions (Type-Safe)

Use type-safe function references instead of strings:

```python
from agnt5 import function, Context

# Define reusable functions
@function
async def fetch_user_data(ctx: Context, user_id: str) -> dict:
    """Fetch user data from database."""
    return {"user_id": user_id, "name": "Alice", "role": "admin"}

@function
async def validate_permissions(ctx: Context, user: dict, action: str) -> bool:
    """Validate user has permission for action."""
    return user.get("role") == "admin"

@function
async def process_request(ctx: Context, user_id: str, action: str) -> dict:
    """Process request with type-safe function calls."""

    # ✅ Type-safe: Pass function reference directly
    user = await ctx.task(fetch_user_data, input=user_id)

    # ✅ Type-safe: IDE autocomplete works
    has_permission = await ctx.task(validate_permissions, input={
        "user": user,
        "action": action
    })

    return {
        "user_id": user_id,
        "action": action,
        "allowed": has_permission
    }
```

**Benefits of Type-Safe Calls:**
- ✅ IDE autocomplete shows available functions
- ✅ Refactoring tools work correctly
- ✅ Compile-time error detection
- ✅ Clear dependencies between functions

**String-based calls still work (backward compatible):**
```python
# Still supported but not recommended
user = await ctx.task("fetch_user_data", input=user_id)
```

### Error Handling

Functions should handle errors appropriately:

```python
@function
async def safe_processing(ctx: Context, data: dict) -> dict:
    """Function with proper error handling."""

    try:
        result = await ctx.step("process", process_data(data))
        return {"success": True, "data": result}

    except ValueError as e:
        # Log the error
        ctx.logger.error(f"Validation error: {e}")

        # Return error response
        return {
            "success": False,
            "error": "validation_error",
            "message": str(e)
        }

    except Exception as e:
        # Log unexpected error
        ctx.logger.error(f"Unexpected error: {e}", exc_info=True)

        # Re-raise to trigger retry
        raise
```

## Registration and Discovery

Functions are automatically registered when your worker starts:

```python
# worker.py
from agnt5 import Worker, function, Context

@function
async def my_function(ctx: Context, data: str) -> str:
    return f"Processed: {data}"

# Create and start worker
async def main():
    worker = Worker(
        service_name="my-service",
        service_version="1.0.0",
        coordinator_endpoint="http://localhost:34186"
    )
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

The worker discovers all decorated functions and registers them with the AGNT5 platform, making them available for invocation.

### Worker-Scoped Registries (Advanced)

For advanced use cases (testing, multi-worker processes), you can use worker-scoped registries instead of global registration. See [Worker-Scoped Registry Plan](../worker-scoped-registry.md) for details.

## Best Practices

### 1. Keep Functions Focused

Each function should do one thing well. Break complex operations into multiple functions composed through workflows.

```python
# ✅ Good: Focused functions
@function
async def validate_input(data: dict) -> bool:
    return is_valid(data)

@function
async def process_data(data: dict) -> dict:
    return transform(data)

# ❌ Less ideal: Doing too much in one function
@function
async def validate_and_process(data: dict) -> dict:
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
    content = await ctx.step("download", download_file(file_url))

    # Checkpoint expensive processing
    result = await ctx.step("process", process_content(content))

    return result
```

### 3. Design for Idempotency

Functions may be retried, so ensure they're idempotent:

```python
@function
async def update_record(ctx: Context, record_id: str, data: dict) -> dict:
    """Idempotent update using upsert."""
    # Use upsert instead of insert to handle retries
    result = await ctx.step(
        "upsert_record",
        database.upsert(record_id, data)
    )
    return result
```

### 4. Leverage Async for I/O Operations

Use async functions for I/O-bound operations:

```python
@function
async def fetch_multiple_sources(ctx: Context, sources: list[str]) -> dict:
    """Efficiently fetch from multiple sources in parallel."""
    import asyncio

    results = await asyncio.gather(*[
        fetch_source(source) for source in sources
    ])

    return {"sources": len(sources), "results": results}
```

### 5. Use Pydantic for Complex Types

Prefer Pydantic models over plain dicts for better validation:

```python
from pydantic import BaseModel, EmailStr, Field

class CreateUserInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)

class CreateUserOutput(BaseModel):
    user_id: str
    created_at: str

@function
async def create_user(ctx: Context, user: CreateUserInput) -> CreateUserOutput:
    """Create user with automatic validation."""
    # Input is already validated by Pydantic
    user_id = await ctx.step("create", save_to_db(user))

    return CreateUserOutput(
        user_id=user_id,
        created_at=datetime.utcnow().isoformat()
    )
```

### 6. Configure Appropriate Retry Policies

Match retry configuration to operation characteristics:

```python
# Short-lived, quick retries for transient network issues
@function(retries=5, backoff="exponential")
async def quick_api_call(url: str) -> dict:
    # ... API call logic
    return {"result": "..."}

# Longer backoff for rate-limited APIs
@function(
    retries={"max_attempts": 3, "initial_interval_ms": 5000},
    backoff={"type": "exponential", "multiplier": 3.0}
)
async def rate_limited_call(query: str) -> dict:
    # ... API call with rate limiting
    return {"result": "..."}
```

### 7. Omit Context When Not Needed

For simple pure functions, omit the context parameter:

```python
# ✅ Clean: No context needed
@function
async def calculate_total(items: list[dict]) -> float:
    return sum(item["price"] * item["quantity"] for item in items)

# ❌ Unnecessary: Context not used
@function
async def calculate_total(ctx: Context, items: list[dict]) -> float:
    return sum(item["price"] * item["quantity"] for item in items)
```

## Function Lifecycle

### 1. Registration
When your worker starts, the `@function` decorator registers the function in the registry (global by default, or worker-scoped if configured).

### 2. Invocation
Functions can be invoked through:
- **Gateway HTTP/gRPC API**: External invocations
- **Workflows**: Orchestrated execution via `ctx.task()`
- **Other functions**: Function composition via `ctx.task()`
- **Platform API**: Direct invocation via platform services

### 3. Execution
The function runs on an available worker with:
- Automatic input deserialization (including Pydantic validation)
- Context injection (if needed)
- Checkpoint replay (for resumed executions)
- Output serialization

### 4. Completion or Retry
On success, results are returned to the caller. On failure, the retry policy determines whether to retry or fail permanently.

## Advanced Patterns

### Function Composition

```python
@function
async def extract_data(ctx: Context, source: str) -> dict:
    return await ctx.step("extract", fetch_from_source(source))

@function
async def transform_data(ctx: Context, data: dict) -> dict:
    return await ctx.step("transform", apply_transformations(data))

@function
async def load_data(ctx: Context, data: dict) -> bool:
    return await ctx.step("load", save_to_destination(data))

@function
async def etl_pipeline(ctx: Context, source: str) -> dict:
    """Compose functions into an ETL pipeline."""
    # Type-safe function calls
    extracted = await ctx.task(extract_data, input=source)
    transformed = await ctx.task(transform_data, input=extracted)
    loaded = await ctx.task(load_data, input=transformed)

    return {"success": loaded, "records": len(transformed)}
```

### Conditional Execution

```python
@function
async def conditional_processing(ctx: Context, data: dict, mode: str) -> dict:
    """Execute different logic based on mode."""

    if mode == "fast":
        result = await ctx.step("fast_process", quick_process(data))
    elif mode == "thorough":
        result = await ctx.step("thorough_process", detailed_process(data))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result
```

### Parallel Execution

```python
@function
async def parallel_tasks(ctx: Context, items: list[str]) -> dict:
    """Execute multiple operations in parallel."""
    import asyncio

    # Process all items in parallel
    results = await asyncio.gather(*[
        process_item(item) for item in items
    ])

    return {
        "total": len(items),
        "results": results
    }
```

## Debugging and Monitoring

### Structured Logging

```python
@function
async def monitored_function(ctx: Context, data: str) -> dict:
    """Function with detailed logging."""
    ctx.logger.info(
        "Processing started",
        extra={"data_length": len(data), "data_type": type(data).__name__}
    )

    result = await ctx.step("process", process_data(data))

    ctx.logger.info(
        "Processing completed",
        extra={"result_keys": list(result.keys())}
    )

    return result
```

### Accessing Execution Metadata

```python
@function
async def introspective_function(ctx: Context, data: str) -> dict:
    """Access execution metadata."""
    ctx.logger.info(f"Execution attempt: {ctx.attempt}")

    return {
        "run_id": ctx.run_id,
        "component_type": ctx.component_type,
        "attempt": ctx.attempt,
        "result": process(data)
    }
```

## Migration Notes

### From Old Checkpointing API

The checkpointing API has been simplified:

```python
# ❌ Old: Lambda wrapping required
result = await ctx.step("load", lambda: load_data())

# ✅ New: Pass coroutines directly (recommended)
result = await ctx.step("load", load_data())

# ✅ New: Pass function reference (called automatically)
result = await ctx.step("load", load_data)
```

### From String-Based Function Calls

Function calls are now type-safe:

```python
# ❌ Old: String-based (no autocomplete, no type safety)
result = await ctx.task("process_data", input=data)

# ✅ New: Type-safe function reference (recommended)
result = await ctx.task(process_data, input=data)
```

### From Old Retry Configuration

Retry configuration has been simplified:

```python
# ❌ Old: Verbose object creation
from agnt5.durable import RetryPolicy, BackoffPolicy

@function(
    retries=RetryPolicy(max_attempts=5),
    backoff=BackoffPolicy(type="exponential")
)
async def my_func(ctx: Context, data: str) -> str:
    pass

# ✅ New: Simple forms (recommended)
@function(retries=5, backoff="exponential")
async def my_func(ctx: Context, data: str) -> str:
    pass

# ✅ New: Dict form (when you need more control)
@function(
    retries={"max_attempts": 5, "initial_interval_ms": 1000},
    backoff={"type": "exponential", "multiplier": 2.0}
)
async def my_func(ctx: Context, data: str) -> str:
    pass
```

## See Also

- [Entity Component](entity.md) - Stateful components with persistent state
- [Workflow Component](workflow.md) - Multi-step orchestration
- [Context API](context.md) - Detailed context capabilities
- [Worker-Scoped Registries](../worker-scoped-registry.md) - Advanced registration patterns
- [Pydantic Documentation](https://docs.pydantic.dev/) - Learn more about Pydantic models
