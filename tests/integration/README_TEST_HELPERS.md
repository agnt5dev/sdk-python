# Integration Test Helpers

Common utilities for testing journal events and database queries.

## Overview

The `test_helpers.py` module provides reusable functions for verifying journal events in integration tests. This eliminates code duplication and makes tests cleaner and easier to maintain.

## Usage

### Basic Function Event Verification

```python
import pytest
from test_helpers import verify_function_events, print_journal_events

@pytest.mark.integration
def test_my_function(client, worker_process, platform):
    """Test that journal events are recorded."""
    # Execute your function
    result = client.run("my_function", {"arg": "value"})
    assert result == "expected"

    # Verify journal events (one line!)
    run_id, event_types = verify_function_events(platform)

    # Optional: print for debugging
    print_journal_events(run_id, event_types)
```

**Before (50+ lines):**
```python
# Manual database queries, cursor management, assertions...
db_path = platform["host_db_path"]
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""SELECT DISTINCT run_id...""")
# ... 40+ more lines
```

**After (1 line):**
```python
run_id, event_types = verify_function_events(platform)
```

## Available Helper Functions

### Database Connection

```python
get_db_connection(platform: Dict) -> sqlite3.Connection
```
Get a connection to the agnt5.db database.

### Event Queries

```python
get_latest_run_id(conn: sqlite3.Connection) -> Optional[str]
```
Get the most recent run_id from journal_events.

```python
get_journal_events(conn: sqlite3.Connection, run_id: str) -> List[Tuple]
```
Get all journal events for a specific run.

```python
get_event_types(events: List[Tuple]) -> List[str]
```
Extract event types from journal events.

### Event Verification

```python
verify_function_events(
    platform: Dict,
    expected_events: Optional[List[str]] = None
) -> Tuple[str, List[str]]
```
Verify that journal events were recorded for a function execution.

**Default expected events:**
- `run.enqueued`
- `run.assigned`
- `run.started`
- `function.started`
- `function.completed`
- `run.completed`

**Custom events:**
```python
# Verify custom event sequence
run_id, event_types = verify_function_events(
    platform,
    expected_events=["run.started", "function.started", "function.completed"]
)
```

```python
verify_workflow_events(
    platform: Dict,
    expected_steps: Optional[List[str]] = None
) -> Tuple[str, List[str]]
```
Verify that journal events were recorded for a workflow execution.

### Debugging

```python
print_journal_events(run_id: str, event_types: List[str])
```
Pretty-print journal events for debugging.

Output:
```
✅ Journal events verified for run 3fb0a54d-5050-4e7e-8ddd-0f036736e572:
   • run.enqueued
   • run.assigned
   • run.started
   • function.started
   • function.completed
   • run.completed
```

```python
count_events_by_type(conn: sqlite3.Connection) -> Dict[str, int]
```
Count all journal events grouped by type.

## Advanced Usage

### Custom Event Verification

```python
from test_helpers import get_db_connection, get_latest_run_id, get_journal_events

def test_custom_events(client, worker_process, platform):
    result = client.run("my_function")

    # Manual query for custom logic
    conn = get_db_connection(platform)
    run_id = get_latest_run_id(conn)
    events = get_journal_events(conn, run_id)
    conn.close()

    # Custom assertions
    assert len(events) == 10
    assert events[0][0] == "custom.event.type"
```

### Workflow Testing

```python
from test_helpers import verify_workflow_events

def test_workflow_events(client, worker_process, platform):
    result = client.run("my_workflow")

    # Verify workflow-specific events
    run_id, event_types = verify_workflow_events(
        platform,
        expected_steps=["step_1", "step_2", "step_3"]
    )
```

### Event Counting

```python
from test_helpers import get_db_connection, count_events_by_type

def test_event_statistics(platform):
    conn = get_db_connection(platform)
    counts = count_events_by_type(conn)
    conn.close()

    print(f"Total function.started events: {counts.get('function.started', 0)}")
```

## Database Schema

The helpers query the `journal_events` table in `agnt5.db`:

```sql
CREATE TABLE IF NOT EXISTS "journal_events" (
  "id" TEXT NOT NULL,
  "run_id" TEXT NOT NULL,
  "event_type" TEXT NOT NULL,
  "step_key" TEXT NULL,
  "input_data" BLOB NULL,
  "input_type" TEXT NULL,
  "input_hash" TEXT NULL,
  "input_ref" TEXT NULL,
  "output_data" BLOB NULL,
  "output_type" TEXT NULL,
  "output_ref" TEXT NULL,
  "metadata" TEXT NULL,
  "timestamp_ns" INTEGER NOT NULL,
  "created_at" INTEGER NOT NULL,
  PRIMARY KEY ("id"),
  FOREIGN KEY ("run_id") REFERENCES "runs" ("id") ON DELETE CASCADE
);
```

## Common Event Types

### Function Execution
- `run.enqueued` - Function queued for execution
- `run.assigned` - Function assigned to worker
- `run.started` - Function execution started
- `function.started` - Function code execution began
- `function.completed` - Function code execution finished
- `run.completed` - Function execution completed

### Workflow Execution
- `workflow.started` - Workflow execution started
- `workflow.step.started` - Workflow step started
- `workflow.step.completed` - Workflow step completed
- `workflow.completed` - Workflow execution completed

### Agent Execution
- `agent.started` - Agent execution started
- `agent.iteration.started` - Agent iteration started
- `agent.iteration.completed` - Agent iteration completed
- `agent.completed` - Agent execution completed
- `lm.started` - LLM call started
- `lm.completed` - LLM call completed

### Tool Execution
- `tool.started` - Tool execution started
- `tool.completed` - Tool execution completed

## Examples

See `test_functions.py::test_add_journal_events` for a complete example.

## Best Practices

1. **Use helpers for common cases** - `verify_function_events()` covers 90% of use cases
2. **Print events for debugging** - Use `print_journal_events()` during development
3. **Custom queries for edge cases** - Use low-level helpers when needed
4. **Keep tests simple** - One-line verification is the goal

## File Location

- Helper module: `tests/integration/test_helpers.py`
- Example usage: `tests/integration/test_functions.py`
- Database: Platform creates `agnt5.db` in temp directory (accessible via `platform["host_db_path"]`)
