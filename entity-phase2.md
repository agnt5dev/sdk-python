# Entity Phase 2: Platform-Backed State Persistence

## Executive Summary

**Current State (Phase 1)**: Entities use global in-memory state that doesn't persist across Worker restarts.

**Gap**: SDK doesn't load pre-loaded state from request metadata, even though the platform infrastructure is 90% complete.

**Solution**: Wire up SDK to consume platform state from metadata and send version information back.

## Current Architecture Analysis

### What Works ✅

1. **Platform Infrastructure (90% Complete)**:
   - ✅ Entity table with JSONB state storage (`platform/pkg/models/orchestration/entity.go`)
   - ✅ Gateway pre-loads state from DB before execution (`platform/internal/gateway/http_entity_handlers.go`)
   - ✅ Worker Coordinator publishes state updates to journal (`platform/internal/worker-coordinator/run_consumer.go`)
   - ✅ Entity Projector consumes events and persists to DB (`platform/internal/execution-engine/entity_projector.go`)
   - ✅ Optimistic locking with version conflict detection
   - ✅ State passed through request/response metadata

2. **SDK Features**:
   - ✅ Entity class with `self.state` API
   - ✅ State capture after execution (`worker.py:609-622`)
   - ✅ Single-writer consistency with async locks
   - ✅ Global `_entity_states` dict for in-memory storage

### What Doesn't Work ❌

1. **SDK Limitations**:
   - ❌ Worker ignores incoming state in `request.metadata["entity_state"]`
   - ❌ No version tracking in SDK (`_entity_versions` dict doesn't exist)
   - ❌ Hardcoded versions in platform (`expected_version: "0"`, `new_version: "1"`)
   - ❌ State lost on Worker restart
   - ❌ No conflict detection from SDK side

### The Gap

**Current Flow**:
```
1. Gateway loads state from DB → Passes in metadata ✅
2. Worker receives request → IGNORES metadata state ❌
3. Worker executes entity → Uses empty dict {} ❌
4. Worker sends state back → No version info ❌
5. Worker Coordinator publishes → Hardcoded versions ❌
6. Entity Projector saves → Works but no conflict detection ✅
```

**Expected Flow**:
```
1. Gateway loads state from DB → Passes in metadata ✅
2. Worker receives request → Loads state from metadata ✅
3. Worker executes entity → Uses platform state ✅
4. Worker sends state back → Includes version info ✅
5. Worker Coordinator publishes → Real versions ✅
6. Entity Projector saves → Full conflict detection ✅
```

## Phase 2 Implementation Plan

### Part 1: SDK Changes (worker.py)

**File**: `sdk/sdk-python/src/agnt5/worker.py`

#### Change 1.1: Add Version Tracking

**Location**: After line 24 (after `_entity_locks` declaration)

```python
# Add new global dict for version tracking
_entity_versions: Dict[Tuple[str, str], int] = {}  # (type, key) -> version
```

#### Change 1.2: Load State from Metadata

**Location**: Lines 609-622 (before entity execution)

**Current Code**:
```python
# Phase 5B: Capture entity state after execution for persistence
state_key = (entity_type.name, entity_key)
metadata = {}
if state_key in _entity_states:
    entity_state = _entity_states[state_key]
    state_json = json.dumps(entity_state)
    metadata = {
        "entity_state": state_json,
        "entity_type": entity_type.name,
        "entity_key": entity_key,
    }
```

**New Code**:
```python
# Phase 5A: Load pre-existing state from platform (BEFORE execution)
state_key = (entity_type.name, entity_key)

# Check if Gateway pre-loaded state in request metadata
if "entity_state" in request.metadata:
    try:
        # Platform sent existing state - load it
        platform_state_json = request.metadata["entity_state"]
        platform_state = json.loads(platform_state_json)

        # Initialize global state with platform state
        _entity_states[state_key] = platform_state

        # Track version for optimistic locking
        if "entity_version" in request.metadata:
            version = int(request.metadata["entity_version"])
            _entity_versions[state_key] = version
        else:
            _entity_versions[state_key] = 0

        logger.info(
            f"Loaded entity state from platform: {entity_type.name}/{entity_key} "
            f"(version {_entity_versions[state_key]})"
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse entity state from metadata: {e}")
        # Fall back to empty state
        if state_key not in _entity_states:
            _entity_states[state_key] = {}
        _entity_versions[state_key] = 0
else:
    # No platform state - this is a new entity
    if state_key not in _entity_states:
        _entity_states[state_key] = {}
    _entity_versions[state_key] = 0

# ... execute entity method ...

# Phase 5B: Capture entity state after execution for persistence
metadata = {}
if state_key in _entity_states:
    entity_state = _entity_states[state_key]
    state_json = json.dumps(entity_state)

    # Get expected version (version BEFORE this execution)
    expected_version = _entity_versions.get(state_key, 0)

    # Calculate new version (increment)
    new_version = expected_version + 1
    _entity_versions[state_key] = new_version

    metadata = {
        "entity_state": state_json,
        "entity_type": entity_type.name,
        "entity_key": entity_key,
        "expected_version": str(expected_version),  # Version we expect in DB
        "new_version": str(new_version),            # Version after this update
    }

    logger.info(
        f"Captured entity state: {entity_type.name}/{entity_key} "
        f"(version {expected_version} → {new_version})"
    )
```

### Part 2: Platform Changes (run_consumer.go)

**File**: `platform/internal/worker-coordinator/run_consumer.go`

#### Change 2.1: Extract Versions from Metadata

**Location**: Lines 448-469 (`publishEntityStateUpdate` function)

**Current Code**:
```go
metadata := map[string]string{
    "tenant_id":        tenantID,
    "entity_type":      entityType,
    "entity_key":       entityKey,
    "operation":        "replace",
    "event_id":         eventID,
    "expected_version": "0",  // TODO: Track version in SDK
    "new_version":      "1",  // TODO: Track version in SDK
}
```

**New Code**:
```go
// Extract versions from SDK response metadata (if available)
expectedVersion := responseMetadata["expected_version"]
newVersion := responseMetadata["new_version"]

// Fall back to defaults if SDK didn't provide versions
if expectedVersion == "" {
    expectedVersion = "0"
}
if newVersion == "" {
    newVersion = "1"
}

metadata := map[string]string{
    "tenant_id":        tenantID,
    "entity_type":      entityType,
    "entity_key":       entityKey,
    "operation":        "replace",
    "event_id":         eventID,
    "expected_version": expectedVersion,  // From SDK
    "new_version":      newVersion,       // From SDK
}
```

#### Change 2.2: Pass State Metadata to Publish Function

**Location**: Find where `publishEntityStateUpdate` is called and ensure metadata is passed

```go
// Extract entity state from response metadata
if entityState, ok := responseMetadata["entity_state"]; ok {
    entityType := responseMetadata["entity_type"]
    entityKey := responseMetadata["entity_key"]

    err := s.publishEntityStateUpdate(
        ctx,
        tenantID,
        entityType,
        entityKey,
        []byte(entityState),
        responseMetadata,  // Pass full metadata for version extraction
    )
    if err != nil {
        slog.Error("Failed to publish entity state update", "error", err)
    }
}
```

### Part 3: Gateway Changes (Optional - Already Works)

**File**: `platform/internal/gateway/http_entity_handlers.go`

The Gateway already loads state and passes it in metadata (lines 97-112). No changes needed unless we want to optimize:

**Optional Enhancement**: Cache loaded state to avoid duplicate DB queries

```go
// Before executing, cache the loaded state
stateCache := map[string]interface{}{
    "state":   currentState,
    "version": stateVersion,
}

// Pass in request metadata
metadata["entity_state"] = currentState
metadata["entity_version"] = fmt.Sprintf("%d", stateVersion)
```

## Testing Strategy

### Test 1: State Persistence Across Invocations

```python
# Test: Create entity, set state, verify it persists

# Invocation 1: Set initial state
client = Client("http://localhost:34181")
counter = client.entity("Counter", "test-counter-1")
result1 = counter.increment(amount=5)
assert result1 == 5

# Wait for state to persist
time.sleep(1)

# Invocation 2: State should persist
result2 = counter.increment(amount=3)
assert result2 == 8  # Should be 5 + 3, not 0 + 3

# Invocation 3: Verify get_value
value = counter.get_value()
assert value == 8
```

### Test 2: State Persistence Across Worker Restarts

```bash
# 1. Start worker
just dev-blueprint blueprints/sdk-python-benchmark

# 2. Create entity and set state
python -c "
from agnt5 import Client
client = Client('http://localhost:34181')
counter = client.entity('Counter', 'restart-test')
print(counter.increment(amount=10))
"

# 3. Restart worker (kill and restart)
# Just kill the worker process

# 4. Start new worker
just dev-blueprint blueprints/sdk-python-benchmark

# 5. Verify state persisted
python -c "
from agnt5 import Client
client = Client('http://localhost:34181')
counter = client.entity('Counter', 'restart-test')
print(counter.get_value())  # Should print 10, not 0
print(counter.increment(amount=5))  # Should print 15, not 5
"
```

### Test 3: Version Conflict Detection

```python
# Test: Simulate concurrent updates with version conflicts

# This test requires manual intervention in the platform
# to create a version conflict scenario

# 1. Load entity state (version 1)
# 2. Manually update DB to version 2
# 3. Try to apply update expecting version 1
# 4. Should fail with version conflict

# Platform should log:
# "Version conflict detected - skipping update"
# (from entity_projector.go:209)
```

### Test 4: New Entity Creation

```python
# Test: First invocation of a new entity

client = Client("http://localhost:34181")
new_entity = client.entity("Counter", "brand-new-entity")

# Should work without errors
result = new_entity.increment(amount=1)
assert result == 1

# Second call should build on first
result = new_entity.increment(amount=1)
assert result == 2
```

## Rollout Plan

### Phase 2.1: SDK Changes (Week 1)

1. **Day 1-2**: Implement version tracking (`_entity_versions` dict)
2. **Day 3-4**: Implement state loading from metadata
3. **Day 5**: Write unit tests for state loading logic
4. **Day 6-7**: Integration testing with dev server

**Success Criteria**:
- ✅ SDK loads state from `request.metadata["entity_state"]`
- ✅ SDK sends `expected_version` and `new_version` in response
- ✅ All existing examples continue to work
- ✅ Tests pass

### Phase 2.2: Platform Changes (Week 2)

1. **Day 1-2**: Update Worker Coordinator to extract versions
2. **Day 3-4**: Test version conflict detection
3. **Day 5**: Test state persistence across Worker restarts
4. **Day 6-7**: End-to-end testing with all components

**Success Criteria**:
- ✅ Platform uses real versions from SDK
- ✅ Version conflicts are detected and logged
- ✅ State persists across Worker restarts
- ✅ No data loss in failure scenarios

### Phase 2.3: Documentation and Examples (Week 3)

1. Update Entity documentation with durability guarantees
2. Add example showing state persistence
3. Add example showing Worker restart scenario
4. Document version conflict behavior

## Backward Compatibility

### SDK Backward Compatibility

**Approach**: Graceful fallback to in-memory state

```python
# If no metadata present, fall back to current behavior
if "entity_state" not in request.metadata:
    # Use in-memory state (Phase 1 behavior)
    if state_key not in _entity_states:
        _entity_states[state_key] = {}
    _entity_versions[state_key] = 0
```

**Impact**: Zero breaking changes for existing code

### Platform Backward Compatibility

**Approach**: Accept missing versions from SDK

```go
// Fall back to defaults if SDK didn't provide versions
if expectedVersion == "" {
    expectedVersion = "0"  // Treat as first version
}
if newVersion == "" {
    newVersion = "1"
}
```

**Impact**: Old SDK versions continue to work (without conflict detection)

## Success Criteria

### Functional Requirements

- ✅ Entity state persists across invocations
- ✅ Entity state persists across Worker restarts
- ✅ Version conflicts are detected and handled
- ✅ No state corruption in concurrent scenarios
- ✅ Backward compatible with Phase 1 code

### Performance Requirements

- ✅ State loading adds <10ms latency
- ✅ Version tracking adds <1ms overhead
- ✅ No memory leaks from version tracking
- ✅ State persistence doesn't block execution

### Observability Requirements

- ✅ Log when state is loaded from platform
- ✅ Log when state is captured and sent
- ✅ Log version conflicts with details
- ✅ Metrics for state load/save operations

## Risk Mitigation

### Risk 1: State Size Growth

**Risk**: JSONB state grows unbounded

**Mitigation**:
- Document state size best practices
- Add warning logs for state >100KB
- Consider state compression for large entities
- Document cleanup patterns (e.g., session expiration)

### Risk 2: Version Conflict Storms

**Risk**: High-frequency updates cause version conflicts

**Mitigation**:
- Entity Projector logs conflicts (already implemented)
- Monitor conflict rate via metrics
- Document single-writer pattern for high-frequency entities
- Consider retry logic for transient conflicts

### Risk 3: Migration Issues

**Risk**: Existing in-memory state lost during upgrade

**Mitigation**:
- Document that in-memory state is ephemeral (expected)
- Provide migration guide for critical entities
- Recommend testing with --client mode first
- Phase rollout: SDK first, then platform

### Risk 4: Metadata Size Limits

**Risk**: Large entity state exceeds gRPC metadata limits (8KB default)

**Mitigation**:
- Document state size limits
- Consider state compression if needed
- Warn on state >5KB
- Future: Use separate state channel if needed

## Code Review Checklist

Before merging Phase 2:

- [ ] SDK loads state from `request.metadata["entity_state"]`
- [ ] SDK tracks versions in `_entity_versions` dict
- [ ] SDK sends `expected_version` and `new_version` in response
- [ ] Worker Coordinator extracts versions from response metadata
- [ ] Platform uses real versions (not hardcoded "0" and "1")
- [ ] All tests pass (unit + integration)
- [ ] Examples updated with persistence examples
- [ ] Documentation updated with durability guarantees
- [ ] Backward compatibility tested
- [ ] Performance benchmarks run
- [ ] Error handling tested (malformed state, version conflicts)
- [ ] Logging and metrics verified

## Future Enhancements (Phase 3+)

1. **State Compression**: Compress large state before sending
2. **State Snapshots**: Periodic snapshots for fast recovery
3. **State Sharding**: Partition large entities across multiple keys
4. **State Expiration**: TTL-based cleanup for session entities
5. **State Queries**: Query entity state without execution
6. **State Export**: Bulk export for analytics
7. **State Replay**: Replay events to rebuild state

## Appendix: File References

### SDK Files
- `sdk/sdk-python/src/agnt5/entity.py` - Entity class definition
- `sdk/sdk-python/src/agnt5/worker.py` - Entity execution and state capture
- `sdk/sdk-python/examples/18_entity_e2e_test.py` - E2E test example

### Platform Files
- `platform/pkg/models/orchestration/entity.go` - Entity table schema
- `platform/internal/execution-engine/entity_projector.go` - State projection
- `platform/internal/gateway/http_entity_handlers.go` - State pre-loading
- `platform/internal/worker-coordinator/run_consumer.go` - State publishing

### Documentation
- `sdk/sdk-python/CLAUDE.md` - SDK documentation
- `platform/CLAUDE.md` - Platform documentation
- `docs/architecture/orchestration-plane.md` - Architecture overview
