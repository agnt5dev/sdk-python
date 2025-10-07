# Crash Recovery Testing

This guide explains how to test AGNT5's Phase 2 durability features by simulating server crashes during execution.

## Overview

AGNT5 Phase 2 implements event sourcing with SQLite persistence. This means:

✅ **All execution state survives crashes**
- Run records persisted to database
- Events logged to journal_events table
- Consumer checkpoints track processing progress

❌ **Known Limitation in Dev Mode**
- In-flight worker responses are lost if dev-server crashes
- Worker is separate process but coordinator is in dev-server
- Production mode with Redpanda/Kafka will handle this properly

## Test Components

### 1. Long-Running Function

The `long_running_task` function in `01_basic_function.py` simulates work that takes 30 seconds:

```python
@function
async def long_running_task(ctx: Context, duration_seconds: int = 30) -> dict:
    """Long-running task with checkpoints at 25%, 50%, 75%, 100%"""
    # ... executes for specified duration
```

### 2. Crash Test Script

`test_crash_recovery.py` automates the crash test:

1. Submits long-running task
2. Kills dev-server after 5 seconds
3. Verifies database state
4. Provides instructions for manual restart

### 3. Verification Script

`verify_recovery.py` checks recovery status:

```bash
python verify_recovery.py <run_id>
```

## Running the Test

### Automated Test (Recommended)

**Terminal 1: Start dev-server**
```bash
cd /path/to/agnt5/platform
just dev-server /path/to/your/project default-service
```

**Terminal 2: Start Python worker**
```bash
cd /path/to/your/project
python examples/01_basic_function.py
```

**Terminal 3: Run crash test**
```bash
cd /path/to/agnt5/sdk/sdk-python
python examples/test_crash_recovery.py
```

The script will:
1. ✅ Submit long-running task (30s)
2. ⏱️  Wait 5 seconds
3. 💥 Kill dev-server
4. 📊 Show database state
5. 📝 Provide restart instructions

### Manual Restart

After the crash test kills the server:

**Terminal 1: Restart dev-server**
```bash
just dev-server /path/to/your/project default-service
```

**Terminal 2: Restart worker** (if it died)
```bash
python examples/01_basic_function.py
```

**Terminal 3: Verify recovery**
```bash
# Option 1: Use verification script
python examples/verify_recovery.py <run_id>

# Option 2: Watch database
watch -n 1 "sqlite3 /tmp/agnt5-dev-data/agnt5-dev-orchestration.db \
  \"SELECT id, status, completed_at FROM runs ORDER BY submitted_at DESC LIMIT 1;\""
```

## Manual Test

For more control, run the test manually:

### Step 1: Submit Long-Running Task

```python
from agnt5 import Client

with Client("http://localhost:34181") as client:
    result = client.run("long_running_task", {"duration_seconds": 30})
    print(result)
```

### Step 2: Kill Server Mid-Execution

```bash
# Find and kill dev-server
ps aux | grep dev-server
kill -9 <PID>
```

### Step 3: Check Database State

```bash
# Check run status
sqlite3 /tmp/agnt5-dev-data/agnt5-dev-orchestration.db \
  "SELECT id, status, started_at, completed_at FROM runs ORDER BY submitted_at DESC LIMIT 1;"

# Check events
sqlite3 /tmp/agnt5-dev-data/agnt5-dev-orchestration.db \
  "SELECT event_type, created_at FROM journal_events ORDER BY sequence_num DESC LIMIT 10;"
```

### Step 4: Restart and Verify

```bash
# Restart dev-server
just dev-server /path/to/project default-service

# Restart worker
python examples/01_basic_function.py

# Check if run completes
python examples/verify_recovery.py <run_id>
```

## What to Expect

### ✅ Expected Results

**After Crash:**
- Run record exists with status "running"
- Events persisted: run.started, run.assigned
- No data corruption

**After Restart:**
- ⚠️ Run stays "running" (worker response was lost)
- New requests work normally
- All state is intact

### ❌ Current Limitations

**Worker Response Lost:**
- If server crashes while worker is processing, the response is lost
- Run will stay in "running" status indefinitely
- This is a known limitation of the in-memory event channels in dev mode

**No Automatic Retry:**
- Stuck runs are not automatically retried
- Manual intervention needed to clean up or retry

**Client Timeout:**
- Client gets connection error instead of graceful timeout
- Client should implement retry logic

### 🚀 Production Improvements (Phase 7+)

These limitations will be fixed in production mode:

- **Persistent Message Queue (Redpanda/Kafka):**
  - Worker responses persisted even if coordinator crashes
  - Guaranteed delivery with at-least-once semantics

- **Automatic Retry:**
  - Detect stuck runs and retry automatically
  - Configurable timeout and retry policies

- **Graceful Client Handling:**
  - Long-running tasks use async endpoints
  - Client polls for status instead of blocking

## Verification Queries

### Check Run Status
```sql
SELECT id, component_name, status, error_message,
       submitted_at, started_at, completed_at
FROM runs
WHERE id = '<run_id>';
```

### Check Events
```sql
SELECT event_type, aggregate_id, created_at
FROM journal_events
WHERE aggregate_id = '<run_id>'
ORDER BY sequence_num;
```

### Check Output
```sql
SELECT output_data, output_type, created_at
FROM run_outputs
WHERE run_id = '<run_id>';
```

### Check Consumer Progress
```sql
SELECT consumer_name, topic, partition, offset, updated_at
FROM consumer_checkpoints
ORDER BY updated_at DESC;
```

## Debugging

### Server Won't Start After Crash

**Symptom:** Database locked or corrupted

**Solution:**
```bash
# Backup database
cp /tmp/agnt5-dev-data/agnt5-dev-orchestration.db /tmp/backup.db

# Delete and recreate
rm /tmp/agnt5-dev-data/agnt5-dev-orchestration.db*
just dev-server /path/to/project default-service
```

### Run Stuck in "running" Status

**Symptom:** Run never completes after restart

**Cause:** Worker response lost during crash

**Solution:**
```bash
# Manually mark as failed
sqlite3 /tmp/agnt5-dev-data/agnt5-dev-orchestration.db \
  "UPDATE runs SET status='failed', error_message='Lost during crash', completed_at=strftime('%s','now')*1000 WHERE id='<run_id>';"
```

### Events Not Processing

**Symptom:** Consumer checkpoints not updating

**Cause:** Consumer goroutine may have crashed

**Solution:** Restart dev-server (consumers start fresh on boot)

## Success Criteria

The crash recovery test is successful if:

✅ Run record persisted with correct status
✅ Events logged to journal_events table
✅ Consumer checkpoints track progress
✅ No database corruption after crash
✅ Server restarts and processes new requests
✅ Event sourcing flow continues normally

⚠️ Known limitation: In-flight worker responses are lost in dev mode
