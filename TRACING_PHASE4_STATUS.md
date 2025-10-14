# Phase 4 Tracing Implementation - Status Report

## Overview
Phase 4 implementation adds worker-level span creation for functions and workflows, completing the distributed tracing hierarchy from gateway to leaf components.

## ✅ Completed Work

### 1. Worker-Level Span Creation (src/agnt5/worker.py)

#### Function Spans (lines 345-364)
```python
with create_span(
    config.name,
    "function",
    request.runtime_context,
    {
        "function.name": config.name,
        "service.name": self.service_name,
    },
) as span:
    # Execute function handler
    result = config.handler(ctx, **input_dict or {})
```

**What it does:**
- Wraps function handler execution in a span
- Links to parent span via runtime_context (from gateway)
- Adds function.name and service.name attributes
- Span automatically captures exceptions and success status

#### Workflow Spans (lines 495-509)
```python
with create_span(
    config.name,
    "workflow",
    request.runtime_context,
    {
        "workflow.name": config.name,
        "service.name": self.service_name,
    },
) as span:
    # Execute workflow handler
    result = await config.handler(ctx, **input_dict or {})
```

**What it does:**
- Same as function spans but for workflow execution
- Enables workflow → tool → agent span hierarchy
- Workflow steps and state changes happen within this span

### 2. Test Infrastructure (tests/integration/conftest.py)

#### Container Configuration (line 98)
```python
dev_server.with_exposed_ports(34181, 34182, 34186, 4317, 34180)
# HTTP, gRPC, Coordinator, OTLP, MCP
```

**Added ports:**
- **4317**: OTLP gRPC endpoint for span export
- **34180**: MCP HTTP endpoint for observability queries

#### Worker Environment (lines 413, 571)
```python
env = {
    ...
    "OTEL_EXPORTER_OTLP_ENDPOINT": platform['otlp_endpoint'],
}
```

**What it does:**
- Configures worker to export telemetry to dev-server OTLP endpoint
- Worker logs confirm configuration: `Endpoint: http://localhost:PORT`

### 3. Span Verification Tests (tests/integration/test_tracing.py)

#### MCP Query Helpers (lines 14-95)
```python
def query_mcp_observability(mcp_endpoint: str, service: str, name: str = None):
    """Query spans via MCP HTTP endpoint."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "obs.list_traces",
            "arguments": {"service": service, "name": name, "limit": 10}
        }
    }
    response = requests.post(f"{mcp_endpoint}/mcp/rpc", json=payload, timeout=5)
    return response.json().get("result", {}).get("items", [])

def get_trace_spans(mcp_endpoint: str, trace_id: str):
    """Get all spans for a trace via MCP."""
    # Similar implementation for obs.get_trace
```

**What it does:**
- Uses MCP JSON-RPC protocol to query observability database
- Leverages built-in `obs.list_traces` and `obs.get_trace` tools
- Returns trace summaries and full span hierarchies

#### Test Cases
1. **test_function_span_creation**: Verifies function spans are created and exported
2. **test_workflow_with_tools_span_hierarchy**: Verifies complete workflow → tool span tree

## Complete Span Hierarchy

With Phase 4 complete, the full tracing hierarchy is:

```
Gateway Span (created in gateway)
  ├─ Worker Function/Workflow Span (Phase 4 - NEW)
      ├─ Tool Span (Phase 2)
      │   └─ Nested operations
      └─ Agent Span (Phase 3)
          ├─ LLM calls
          └─ Tool invocations
```

**Trace linking via RuntimeContext:**
- Gateway creates RuntimeContext with trace_id/span_id
- Passed through gRPC to worker
- Worker creates child span with parent context
- All nested operations (tools/agents) inherit trace context

## ⚠️ Outstanding Issue: Span Export Not Working

### Problem
Spans are being created but not appearing in the observability database.

### Evidence
**Working:**
- ✅ Worker telemetry initialized: `🔭 AGNT5 OpenTelemetry Configuration: Endpoint: http://localhost:PORT`
- ✅ Function executed successfully: `[INFO] 🔥 WORKER: Executing function greet`
- ✅ Test infrastructure correct: OTLP port exposed, MCP queries work
- ✅ Integration tests execute successfully

**Not Working:**
- ❌ No spans found in observability.db after 3-second wait
- ❌ MCP query returns empty result: `assert len(traces) > 0` fails

### Possible Root Causes

#### 1. Span Batching/Flushing
**Hypothesis**: OpenTelemetry batches spans and doesn't flush before worker shutdown.

**Evidence**:
- Default batch span processor has 5s timeout
- Worker terminates immediately after function completes
- No explicit flush before shutdown

**Solution**:
```python
# In worker shutdown or after execution
if hasattr(self, '_telemetry'):
    self._telemetry.flush()
```

#### 2. OTLP Receiver Not Storing Spans
**Hypothesis**: Dev-server OTLP receiver receives spans but fails to write to SQLite.

**Check**:
- Examine dev-server logs for OTLP receiver activity
- Look for SQLite write errors
- Verify observability.db file is created

**Debug**:
```bash
# Check dev-server logs
docker logs <container-id> | grep -i "otlp\|span\|observability"

# Check if database exists
docker exec <container-id> ls -la /data/observability.db
```

#### 3. Service Name Mismatch
**Hypothesis**: Spans created with wrong service name, query filters them out.

**Check**:
- Worker logs show: `Service: test-service`
- Test queries for: `service="test-service"`
- Should match, but verify in database

**Debug**:
```python
# Query all traces without filter
traces = query_mcp_observability(mcp_endpoint, service="")
```

#### 4. Rust SDK Not Creating Spans
**Hypothesis**: `create_span()` in Rust FFI not actually creating OpenTelemetry spans.

**Check**:
- Review `sdk-core/src/telemetry.rs:create_component_span()`
- Verify span is added to tracer
- Check if OTLP exporter is initialized

## Next Steps for Debugging

### Step 1: Verify Span Creation
Add debug logging in `sdk-core/src/telemetry.rs`:
```rust
pub fn create_component_span(...) -> Span {
    eprintln!("🔍 Creating span: name={}, type={}", name, component_type);
    let span = tracer.span_builder(name).start(&tracer);
    eprintln!("🔍 Span created: trace_id={:?}", span.span_context().trace_id());
    span
}
```

### Step 2: Add Explicit Flush
Modify worker to flush telemetry before shutdown:
```python
# In worker.py after function/workflow execution
try:
    from ._core import flush_telemetry
    flush_telemetry()
    logger.debug("Telemetry flushed")
except Exception as e:
    logger.warning(f"Failed to flush telemetry: {e}")
```

### Step 3: Increase Wait Time
Rule out batching by waiting longer:
```python
# In test_tracing.py
time.sleep(10)  # Increased from 3 to 10 seconds
```

### Step 4: Check Dev-Server Logs
During test execution:
```bash
# Get container ID
container_id=$(docker ps | grep dev-server | awk '{print $1}')

# Stream logs
docker logs -f $container_id | grep -i "span\|otlp\|observability"
```

### Step 5: Direct Database Query
Verify database contents directly:
```python
# In test, query all spans without filters
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "obs.list_traces",
        "arguments": {"limit": 100}  # No service filter
    }
}
```

## Code Files Modified

### Production Code
- `src/agnt5/worker.py`: Added function and workflow span creation
- `src/agnt5/tool.py`: Already had tool span creation (Phase 2)
- `src/agnt5/agent.py`: Already had agent span creation (Phase 3)

### Test Infrastructure
- `tests/integration/conftest.py`: Exposed OTLP/MCP ports, configured worker env
- `tests/integration/test_tracing.py`: New file with span verification tests

## How to Continue

### For Immediate Debugging
```bash
cd [private-monorepo]/sdk/sdk-python

# Run test with verbose output
uv run pytest tests/integration/test_tracing.py::test_function_span_creation -v -s

# While test is running (in another terminal):
# 1. Find container ID
docker ps | grep dev-server

# 2. Stream logs
docker logs -f <container-id>

# 3. Check database
docker exec <container-id> ls -la /data/
docker exec <container-id> sqlite3 /data/observability.db "SELECT COUNT(*) FROM spans;"
```

### For Fixing the Issue
1. Start with Step 2 (Add Explicit Flush) - most likely cause
2. Add Rust debug logging (Step 1) to verify spans are created
3. Check dev-server logs (Step 4) to see if OTLP receiver is working
4. If still failing, examine OTLP exporter configuration in sdk-core

## Testing Strategy

Once spans appear in database:

1. **Basic verification**: Function span created with correct attributes
2. **Hierarchy verification**: Workflow → tool → agent span tree
3. **Trace linking**: All spans share same trace_id
4. **Parent-child**: Correct parent_span_id relationships
5. **Attributes**: service.name, component names, etc.

## Success Criteria

Phase 4 is complete when:
- [x] Function spans wrap handler execution
- [x] Workflow spans wrap handler execution
- [x] Spans linked via runtime_context
- [x] Test infrastructure configured
- [x] MCP query helpers implemented
- [ ] **Spans appear in observability database** ⚠️
- [ ] Integration tests pass

**Current Status**: 5/6 complete - Only span export pipeline needs debugging.
