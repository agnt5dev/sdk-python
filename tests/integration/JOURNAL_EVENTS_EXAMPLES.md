# Journal Events Testing Examples

This guide shows how to test different journal event scenarios using the flexible `verify_journal_events` helper.

## Key Principle

**Tests control what events to verify** - The helper doesn't assume anything. Each test explicitly specifies the expected event sequence.

## Basic Function Execution

```python
@pytest.mark.integration
def test_function_success(client, worker_process, platform):
    """Test successful function execution."""
    result = client.run("my_function", {"arg": "value"})

    # Test specifies expected events in order
    verify_journal_events(platform, [
        "run.enqueued",
        "run.assigned",
        "run.started",
        "function.started",
        "function.completed",
        "run.completed",
    ])
```

## Testing Different Scenarios

### 1. Fast Function (No Queue Wait)

```python
def test_fast_function_events(client, worker_process, platform):
    """Test function that executes immediately."""
    result = client.run("instant_function")

    # Might skip enqueued/assigned if executed immediately
    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.completed",
        "run.completed",
    ])
```

### 2. Function with Retries

```python
def test_function_with_retry(client, worker_process, platform):
    """Test function that retries."""
    result = client.run("retry_function")

    # Verify retry events appear
    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.failed",      # First attempt fails
        "function.retry",       # Retry triggered
        "function.started",     # Second attempt
        "function.completed",   # Success
        "run.completed",
    ])
```

### 3. Function That Fails

```python
def test_function_failure(client, worker_process, platform):
    """Test function that fails."""
    with pytest.raises(RunError):
        client.run("failing_function")

    # Verify failure events
    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.failed",
        "run.failed",
    ])
```

### 4. Workflow Execution

```python
def test_workflow_events(client, worker_process, platform):
    """Test workflow with multiple steps."""
    result = client.run("multi_step_workflow")

    # Test specifies exact workflow event sequence
    verify_journal_events(platform, [
        "run.enqueued",
        "run.assigned",
        "run.started",
        "workflow.started",
        "workflow.step.started",   # Step 1
        "workflow.step.completed",
        "workflow.step.started",   # Step 2
        "workflow.step.completed",
        "workflow.step.started",   # Step 3
        "workflow.step.completed",
        "workflow.completed",
        "run.completed",
    ])
```

### 5. Agent Execution

```python
def test_agent_events(client, worker_process, platform):
    """Test agent with LLM calls."""
    result = client.run("my_agent", {"prompt": "Hello"})

    verify_journal_events(platform, [
        "run.started",
        "agent.started",
        "agent.iteration.started",
        "lm.started",
        "lm.content_block.started",
        "lm.content_block.delta",
        "lm.content_block.completed",
        "lm.completed",
        "agent.iteration.completed",
        "agent.completed",
        "run.completed",
    ])
```

### 6. Agent with Tool Calls

```python
def test_agent_with_tools(client, worker_process, platform):
    """Test agent that calls tools."""
    result = client.run("agent_with_tools")

    verify_journal_events(platform, [
        "run.started",
        "agent.started",
        "agent.iteration.started",
        "lm.started",
        "lm.completed",           # LLM decides to use tool
        "tool.started",           # Tool execution
        "tool.completed",
        "lm.started",             # LLM processes tool result
        "lm.completed",
        "agent.iteration.completed",
        "agent.completed",
        "run.completed",
    ])
```

### 7. Testing Partial Events (Flexibility)

```python
def test_only_critical_events(client, worker_process, platform):
    """Test focusing only on critical events."""
    result = client.run("my_function")

    # Test can choose to verify ONLY these events
    # (ignoring queue/assignment events if not relevant)
    verify_journal_events(platform, [
        "function.started",
        "function.completed",
    ])
```

### 8. Testing Event Count

```python
def test_event_count(client, worker_process, platform):
    """Test exact number of events."""
    result = client.run("simple_function")

    run_id, event_types = verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.completed",
        "run.completed",
    ])

    # Can assert exact count
    assert len(event_types) == 4
```

### 9. Testing Timeout

```python
def test_timeout_events(client, worker_process, platform):
    """Test function that times out."""
    with pytest.raises(TimeoutError):
        client.run("slow_function", timeout=1)

    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.timeout",
        "run.timeout",
    ])
```

### 10. Testing Concurrent Execution

```python
def test_concurrent_functions(client, worker_process, platform):
    """Test multiple concurrent function executions."""
    # Execute first function
    result1 = client.run("func1")

    # Verify its events
    run_id1, events1 = verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.completed",
        "run.completed",
    ])

    # Execute second function
    result2 = client.run("func2")

    # Verify its events (gets latest run)
    run_id2, events2 = verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.completed",
        "run.completed",
    ])

    # Verify they're different runs
    assert run_id1 != run_id2
```

## Advanced Patterns

### Custom Event Validation

```python
from test_helpers import get_db_connection, get_latest_run_id, get_journal_events

def test_custom_event_logic(client, worker_process, platform):
    """Test with custom event validation logic."""
    result = client.run("my_function")

    # Get events manually for custom assertions
    conn = get_db_connection(platform)
    run_id = get_latest_run_id(conn)
    events = get_journal_events(conn, run_id)
    conn.close()

    # Custom validations
    event_types = [e[0] for e in events]

    # Check that started comes before completed
    start_idx = event_types.index("function.started")
    complete_idx = event_types.index("function.completed")
    assert start_idx < complete_idx

    # Check no errors occurred
    assert "function.failed" not in event_types
    assert "run.failed" not in event_types
```

### Testing Event Metadata

```python
def test_event_metadata(client, worker_process, platform):
    """Test event metadata fields."""
    result = client.run("my_function", {"value": 42})

    conn = get_db_connection(platform)
    run_id = get_latest_run_id(conn)

    # Query with metadata
    cursor = conn.cursor()
    cursor.execute("""
        SELECT event_type, metadata
        FROM journal_events
        WHERE run_id = ? AND event_type = 'function.started'
    """, (run_id,))

    row = cursor.fetchone()
    conn.close()

    # Verify metadata
    assert row is not None
    metadata = json.loads(row[1]) if row[1] else {}
    # Add your metadata assertions here
```

## Best Practices

1. **Be Explicit** - Each test specifies exactly what events to expect
2. **Test Order** - Event order matters for correctness (default behavior)
3. **Test Variations** - Test success, failure, retry, timeout scenarios
4. **Keep it Simple** - Most tests need just `verify_journal_events()`
5. **Debug with Prints** - Use `print_journal_events()` during development

## Common Event Sequences Reference

### Function Events
```python
# Success
["run.started", "function.started", "function.completed", "run.completed"]

# Failure
["run.started", "function.started", "function.failed", "run.failed"]

# With queueing
["run.enqueued", "run.assigned", "run.started", "function.started", "function.completed", "run.completed"]
```

The journal sequence is the same in pull mode. When the runtime negotiates
`pull_completion_lifecycle_v1`, a non-streaming pull worker no longer appends
`run.started`, `function.started`, and `function.completed` one RPC at a time:
sdk-core holds them and delivers them inside `CompleteJob`, and the runtime
appends them atomically just below the fenced completion request. Tests that
read the journal see identical event types and source timestamps; tests that
count `Append`/`AppendBatch` RPCs per run should expect zero for plain
functions under that capability.

### Error Scenarios

#### Function Failures

```python
def test_function_failure(client, worker_process, platform):
    """Test function that raises an exception."""
    with pytest.raises(RunError):
        client.run("failing_function", {"message": "test"})

    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.failed",
        "run.failed",
    ])
```

#### Missing Parameters

```python
def test_missing_parameters(client, worker_process, platform):
    """Test function called without required parameters."""
    with pytest.raises(RunError):
        client.run("add", {})  # Missing required a and b

    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.failed",
        "run.failed",
    ])
```

#### Invalid Parameter Types

```python
def test_invalid_type(client, worker_process, platform):
    """Test function with wrong parameter type."""
    with pytest.raises(RunError):
        client.run("add", {"a": "not_an_int", "b": 5})

    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.failed",
        "run.failed",
    ])
```

#### Runtime Errors (Division by Zero)

```python
def test_division_by_zero(client, worker_process, platform):
    """Test function that raises ZeroDivisionError."""
    with pytest.raises(RunError):
        client.run("divide", {"a": 10, "b": 0})

    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "function.failed",
        "run.failed",
    ])
```

#### Timeout Errors

```python
def test_timeout(client, worker_process, platform):
    """Test function that exceeds timeout."""
    with pytest.raises(RunError):
        client.run("slow_function", {"delay_seconds": 70})

    verify_journal_events(platform, [
        "run.started",
        "function.started",
        "run.failed",  # Timeout triggers failure
    ])
```

### Workflow Events
```python
# 2-step workflow
["run.started", "workflow.started",
 "workflow.step.started", "workflow.step.completed",  # Step 1
 "workflow.step.started", "workflow.step.completed",  # Step 2
 "workflow.completed", "run.completed"]
```

### Agent Events
```python
# Simple agent
["run.started", "agent.started", "agent.iteration.started",
 "lm.started", "lm.completed",
 "agent.iteration.completed", "agent.completed", "run.completed"]

# Agent with tool
["run.started", "agent.started", "agent.iteration.started",
 "lm.started", "lm.completed",  # Decides to use tool
 "tool.started", "tool.completed",
 "lm.started", "lm.completed",  # Processes tool result
 "agent.iteration.completed", "agent.completed", "run.completed"]
```

## Example Test File

See `test_functions.py` for working examples:
- `test_add_journal_events` - Complete function event verification
- `test_add_minimal_journal_events` - Flexible event specification
