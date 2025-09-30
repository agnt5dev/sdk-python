# Context API

## What is Context?

The **Context** (`ctx`) is the execution environment provided to all AGNT5 components (functions, entities, workflows). It provides APIs for orchestration, state management, LLM interactions, signals, timers, and observability.

**Key Capabilities:**
- **Orchestration**: Execute tasks, spawn functions, parallel execution
- **State Management**: Get/set/delete state for entities
- **Coordination**: Signals, timers, human approvals
- **AI Integration**: LLM calls, tool registration
- **Observability**: Logging, metrics, tracing

## Core Orchestration APIs

### `ctx.task()` - Execute a Function

Call a function and wait for the result (workflows only):

```python
result = await ctx.task(
    service_name="analytics",
    handler_name="process_data",
    input={"dataset": "users"}
)
```

### `ctx.parallel()` - Concurrent Execution

Run multiple tasks in parallel (workflows only):

```python
# Returns list of results in order
results = await ctx.parallel(
    ctx.task("service1", "handler1"),
    ctx.task("service2", "handler2"),
    ctx.task("service3", "handler3")
)
```

### `ctx.gather()` - Named Parallel Results

Like `parallel()` but with dict keys (workflows only):

```python
results = await ctx.gather(
    db=ctx.task("analytics", "analyze_db"),
    api=ctx.task("analytics", "analyze_api")
)
# Access: results["db"], results["api"]
```

### `ctx.spawn()` - Async Child Invocation

Spawn a child function without waiting:

```python
handle = ctx.spawn(my_function, arg1, arg2, key="unique-id")
# Continue doing other work...
result = await handle.result()
```

### `ctx.step()` - Checkpointing

Checkpoint expensive operations (functions only):

```python
# If function crashes, won't re-execute this step
data = await ctx.step(
    "load_data",
    lambda: expensive_database_query()
)
```

## State Management (Entities)

### `ctx.get()` - Get State

Read from entity state:

```python
history = await ctx.get("history", default=[])
```

### `ctx.set()` - Set State

Write to entity state:

```python
ctx.set("history", updated_history)
```

### `ctx.delete()` - Delete State

Remove key from state:

```python
ctx.delete("temporary_data")
```

### `ctx.entity()` - Call Entity Method

Invoke an entity method:

```python
result = await ctx.entity(
    "ChatAgent",
    "conversation-123"
).send_message("Hello!")
```

## Coordination APIs

### `ctx.signal()` - Wait for External Event

Pause execution until signal received:

```python
approval = await ctx.signal(
    "manager_approved",
    timeout_ms=86400000,  # 24 hours
    default={"approved": False}
)
```

### `ctx.signal.emit()` - Send Signal

Send a signal to waiting workflow:

```python
await ctx.signal.emit(
    "deployment_ready",
    payload={"version": "1.0.0"}
)
```

### `ctx.timer()` - Wait for Delay

Pause for a duration:

```python
# Wait 5 seconds
await ctx.timer(delay_ms=5000)

# Wait until specific time
await ctx.timer(cron="0 0 * * *")  # Daily at midnight
```

### `ctx.sleep()` - Durable Sleep

Sleep with checkpoint (alternative to `timer()`):

```python
await ctx.sleep(30)  # Sleep for 30 seconds
```

### `ctx.human.approval()` - Human-in-the-Loop

Request human approval:

```python
result = await ctx.human.approval(
    "deploy_production",
    payload={"version": "2.0.0"},
    timeout=timedelta(minutes=30),
    required_roles=["admin"]
)

if result.decision == "approved":
    # Proceed with deployment
    ...
```

## AI Integration

### `ctx.llm.generate()` - Generate Text or Structured Data

Generate text or structured responses from language models:

```python
# Simple text generation
response = await ctx.llm.generate(
    prompt="Explain quantum computing in simple terms",
    model="gpt-4o-mini"
)
print(response.text)

# Chat-style conversation
response = await ctx.llm.generate(
    prompt=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    model="gpt-4"
)

# Structured JSON output
response = await ctx.llm.generate(
    prompt="Extract key information from this text: ...",
    response_format="json",
    model="gpt-4o-mini"
)
print(response.object)  # Parsed JSON object

# JSON Schema-constrained responses
response = await ctx.llm.generate(
    prompt="Create a user profile",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    },
    model="gpt-4o-mini"
)
```

### `ctx.llm.stream()` - Stream Generated Text

Stream responses for real-time output:

```python
# Stream text generation
async for chunk in await ctx.llm.stream(
    prompt="Write a long story about AI",
    model="gpt-4o"
):
    print(chunk.text, end="", flush=True)

# Stream with chat messages
async for chunk in await ctx.llm.stream(
    prompt=[
        {"role": "system", "content": "You are a storyteller"},
        {"role": "user", "content": "Tell me a story"}
    ],
    model="gpt-4o"
):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

### `ctx.tools.register()` - Register Tool

Register a tool for LLM use:

```python
search_tool = ctx.tools.register(
    "web_search",
    handler=perform_search,
    description="Search the web for information",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    }
)
```

## Memory & State

### `ctx.memory.get()` / `set()` / `delete()`

Durable memory operations (alternative to ctx.get/set):

```python
# Get value
data = await ctx.memory.get("conversation_history", default=[])

# Set value
await ctx.memory.set("conversation_history", updated_data)

# Delete value
await ctx.memory.delete("temporary_cache")
```

### `ctx.memory.append()` - Append to List

Append with automatic size limiting:

```python
await ctx.memory.append(
    "messages",
    new_message,
    limit=100  # Keep last 100 items
)
```

## Observability

### `ctx.log()` - Structured Logging

Access logger with context:

```python
logger = ctx.log()
logger.info("Processing started", extra={"user_id": "123"})
logger.error("Operation failed", exc_info=True)
```

### `ctx.metrics()` - Record Metrics

Track custom metrics:

```python
metrics = ctx.metrics()
metrics.increment("requests.count", service="api")
metrics.observe("latency.ms", 42.5, endpoint="/users")
```

### `ctx.trace_span()` - Distributed Tracing

Create spans for tracing:

```python
with ctx.trace_span().start("external_api_call", service="payments"):
    result = await call_payment_api()
```

## Configuration & Secrets

### `ctx.secrets()` - Access Secrets

Retrieve secrets securely:

```python
api_key = ctx.secrets().get("openai_api_key")
db_password = ctx.secrets().get("database_password")
```

### `ctx.config()` - Feature Flags

Access configuration:

```python
config = ctx.config()
enabled = config.get("new_feature_enabled", default=False)
variant = config.variant("experiment_group", default="control")
```

### `ctx.headers()` - Request Headers

Access incoming request headers:

```python
headers = ctx.headers()
user_agent = headers.get("user-agent", "unknown")
```

## Messaging

### `ctx.send_to()` - Send Message

Send message to another participant:

```python
message_id = await ctx.send_to(
    target="agent-coordinator",
    message={"status": "completed", "result": data},
    metadata={"priority": "high"}
)
```

### `ctx.subscribe()` - Subscribe to Messages

Receive messages asynchronously:

```python
async for message in ctx.subscribe("my-agent"):
    process_message(message.payload)
    await message.ack()  # Acknowledge receipt
```

## Context Properties

### Execution Metadata

Access execution context:

```python
ctx.run_id           # Workflow/run identifier
ctx.step_id          # Current step identifier
ctx.attempt          # Retry attempt number
ctx.component_type   # "function", "entity", "workflow"
ctx.object_id        # Entity key (for entities)
ctx.method_name      # Entity method name (for entities)
```

## Common Patterns

### Parallel with Error Handling

```python
results = await ctx.gather(
    task1=ctx.task("svc", "task1"),
    task2=ctx.task("svc", "task2")
)

if results["task1"] and results["task2"]:
    # Both succeeded
    ...
```

### Conditional Signal Waiting

```python
if needs_approval:
    approval = await ctx.signal("approval_signal", timeout_ms=60000)
    if not approval.get("approved"):
        return {"status": "rejected"}
```

### LLM with Tool Execution

```python
# Register tools
search_tool = ctx.tools.register("search", handler=search_handler, ...)
calc_tool = ctx.tools.register("calculator", handler=calc_handler, ...)

# Generate with tools
response = await ctx.llm.generate(
    prompt="What is 25 * 4 and what are the latest news on AI?",
    tools=[search_tool, calc_tool],
    model="gpt-4o"
)

# Execute tool calls if needed
if response.tool_calls:
    for tool_call in response.tool_calls:
        handler = ctx.tools.handler(tool_call.name)
        result = await handler(**tool_call.arguments)
```

### Checkpointed Multi-Step Process

```python
@function
async def process_pipeline(ctx, data_id: str):
    # Each step is checkpointed
    raw = await ctx.step("extract", lambda: extract_data(data_id))
    cleaned = await ctx.step("clean", lambda: clean_data(raw))
    result = await ctx.step("analyze", lambda: analyze(cleaned))
    return result
```

## See Also

- [Function Component](function.md) - Using context in functions
- [Entity Component](entity.md) - Using context in entities
- [Workflow Component](workflow.md) - Using context in workflows