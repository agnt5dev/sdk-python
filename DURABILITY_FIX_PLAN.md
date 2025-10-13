# AGNT5 SDK Durability Fix - Incremental Task Plan

**Status**: 🔴 Planning
**Duration**: 6-8 weeks (30-40 tasks)
**Priority**: P0 - Critical

---

## Overview

This plan breaks down the durability fixes into **small, incremental tasks** that can be completed one at a time. Each task is designed to be completable in 2-4 hours and includes clear acceptance criteria.

### Strategy: Bottom-Up Approach
1. Fix foundation (Context, State Manager) first
2. Enable workflows (highest value, existing infrastructure)
3. Fix entities (already partially working)
4. Enable functions (dependent on Context)
5. Enable agents (dependent on Context + workflows)

---

## Phase 1: Foundation - State Management (Week 1-2)

### 🎯 Goal: Make Context state backend-backed instead of in-memory

#### Task 1.1: Define State Manager Protocol in Python ⏱️ 2 hours
**File**: `src/agnt5/state.py` (new file)

**Description**: Create Python interface for state management that mirrors Rust `StateManager` trait.

**Acceptance Criteria**:
- [ ] Create `StateManager` protocol class with `get()`, `set()`, `delete()` methods
- [ ] All methods are async
- [ ] Type hints for state keys and values
- [ ] Docstrings with examples

**Implementation**:
```python
# src/agnt5/state.py
from typing import Any, Optional, Protocol

class StateManager(Protocol):
    """Protocol for state persistence backends."""

    async def get(self, key: str) -> Optional[bytes]:
        """Get state value by key."""
        ...

    async def set(self, key: str, value: bytes) -> None:
        """Set state value for key."""
        ...

    async def delete(self, key: str) -> None:
        """Delete state key."""
        ...
```

**Test**:
```python
# Test that protocol is properly defined
from agnt5.state import StateManager
assert hasattr(StateManager, 'get')
assert hasattr(StateManager, 'set')
assert hasattr(StateManager, 'delete')
```

---

#### Task 1.2: Create In-Memory State Manager (for testing) ⏱️ 2 hours
**File**: `src/agnt5/state.py`

**Description**: Implement in-memory state manager for local testing (temporary, will be replaced).

**Acceptance Criteria**:
- [ ] Implements `StateManager` protocol
- [ ] Uses dict for storage (same as current behavior)
- [ ] Handles serialization (JSON or pickle)
- [ ] Thread-safe with locks
- [ ] Unit tests pass

**Implementation**:
```python
class InMemoryStateManager:
    """In-memory state manager for testing."""

    def __init__(self):
        self._store: Dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[bytes]:
        async with self._lock:
            return self._store.get(key)

    async def set(self, key: str, value: bytes) -> None:
        async with self._lock:
            self._store[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)
```

**Test**:
```python
async def test_in_memory_state_manager():
    mgr = InMemoryStateManager()
    await mgr.set("key1", b"value1")
    assert await mgr.get("key1") == b"value1"
    await mgr.delete("key1")
    assert await mgr.get("key1") is None
```

---

#### Task 1.3: Add State Manager to RuntimeContext ⏱️ 1 hour
**File**: `rust-src/types.rs`

**Description**: Expose Rust `StateManager` to Python via PyRuntimeContext.

**Acceptance Criteria**:
- [ ] `PyRuntimeContext` has `state_manager()` method
- [ ] Returns Python-compatible state manager interface
- [ ] Can be called from Python Context

**Implementation**:
```rust
// rust-src/types.rs
#[pyclass]
pub struct PyRuntimeContext {
    pub inner: Arc<RuntimeContext>,
}

#[pymethods]
impl PyRuntimeContext {
    // Add state_manager accessor
    fn state_manager(&self) -> PyStateManager {
        PyStateManager {
            inner: self.inner.state_manager.clone(),
        }
    }
}
```

**Test**:
```python
# In Python
runtime_context = get_runtime_context()  # From Rust
state_mgr = runtime_context.state_manager()
assert state_mgr is not None
```

---

#### Task 1.4: Wrap Rust StateManager in Python ⏱️ 3 hours
**File**: `rust-src/types.rs`, `src/agnt5/state.py`

**Description**: Create Python wrapper for Rust StateManager that implements our protocol.

**Acceptance Criteria**:
- [ ] `PyStateManager` class in Rust exposes async methods
- [ ] Python `RustStateManager` wrapper implements `StateManager` protocol
- [ ] Handles serialization/deserialization
- [ ] Error handling for Rust errors

**Implementation**:
```python
# src/agnt5/state.py
class RustStateManager:
    """Wrapper for Rust-backed state manager."""

    def __init__(self, py_state_manager):
        self._inner = py_state_manager

    async def get(self, key: str) -> Optional[bytes]:
        try:
            return await self._inner.get_async(key)
        except Exception as e:
            logger.error(f"State get failed: {e}")
            return None

    async def set(self, key: str, value: bytes) -> None:
        await self._inner.set_async(key, value)

    async def delete(self, key: str) -> None:
        await self._inner.delete_async(key)
```

**Test**:
```python
async def test_rust_state_manager_wrapper():
    rust_mgr = get_rust_state_manager()
    wrapper = RustStateManager(rust_mgr)
    await wrapper.set("test", b"data")
    assert await wrapper.get("test") == b"data"
```

---

#### Task 1.5: Integrate State Manager into Context.__init__() ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Initialize state manager in Context from RuntimeContext.

**Acceptance Criteria**:
- [ ] Context accepts optional `runtime_context` parameter
- [ ] Extracts state_manager from runtime_context
- [ ] Falls back to InMemoryStateManager if no runtime_context
- [ ] `_state_manager` attribute set correctly

**Implementation**:
```python
# src/agnt5/context.py
from .state import StateManager, InMemoryStateManager, RustStateManager

class Context:
    def __init__(
        self,
        run_id: str,
        runtime_context: Optional[Any] = None,
        ...
    ):
        # Initialize state manager
        if runtime_context and hasattr(runtime_context, 'state_manager'):
            rust_mgr = runtime_context.state_manager()
            self._state_manager: StateManager = RustStateManager(rust_mgr)
        else:
            self._state_manager: StateManager = InMemoryStateManager()

        # Keep in-memory cache for fast access
        self._state: Dict[str, Any] = {}
```

**Test**:
```python
def test_context_with_state_manager():
    ctx = Context(run_id="test")
    assert isinstance(ctx._state_manager, InMemoryStateManager)

    ctx_with_runtime = Context(run_id="test", runtime_context=mock_runtime)
    assert isinstance(ctx_with_runtime._state_manager, RustStateManager)
```

---

#### Task 1.6: Implement Context.set() with Backend Persistence ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Update `ctx.set()` to persist to backend via state manager.

**Acceptance Criteria**:
- [ ] `set()` writes to both local cache and backend
- [ ] Handles serialization (JSON for simple types)
- [ ] Error handling if backend fails (log but don't crash)
- [ ] Unit tests for persistence

**Implementation**:
```python
# src/agnt5/context.py
import json

class Context:
    async def set(self, key: str, value: Any) -> None:
        """Set value in state (local cache + backend)."""
        # Update local cache
        self._state[key] = value

        # Persist to backend
        try:
            serialized = json.dumps(value).encode('utf-8')
            await self._state_manager.set(
                f"{self.run_id}:{key}",  # Namespace by run_id
                serialized
            )
            self.logger.debug(f"State persisted: {key}")
        except Exception as e:
            self.logger.error(f"Failed to persist state {key}: {e}")
            # Don't fail the operation if persistence fails
```

**Test**:
```python
async def test_context_set_persists():
    ctx = Context(run_id="test-123")
    await ctx.set("counter", 42)

    # Verify in backend
    backend_value = await ctx._state_manager.get("test-123:counter")
    assert json.loads(backend_value) == 42
```

---

#### Task 1.7: Implement Context.get() with Backend Hydration ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Update `ctx.get()` to load from backend if not in local cache.

**Acceptance Criteria**:
- [ ] `get()` checks local cache first (fast path)
- [ ] Falls back to backend if not in cache
- [ ] Handles deserialization
- [ ] Updates local cache on backend hit
- [ ] Returns default if not found anywhere

**Implementation**:
```python
# src/agnt5/context.py
class Context:
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from state (cache -> backend -> default)."""
        # Try local cache first
        if key in self._state:
            return self._state[key]

        # Try backend
        try:
            backend_key = f"{self.run_id}:{key}"
            value_bytes = await self._state_manager.get(backend_key)

            if value_bytes:
                value = json.loads(value_bytes.decode('utf-8'))
                # Update local cache
                self._state[key] = value
                self.logger.debug(f"State hydrated from backend: {key}")
                return value
        except Exception as e:
            self.logger.error(f"Failed to load state {key}: {e}")

        # Return default
        return default
```

**Test**:
```python
async def test_context_get_hydrates():
    ctx1 = Context(run_id="test-123")
    await ctx1.set("data", {"value": 100})

    # Create new context (empty cache)
    ctx2 = Context(run_id="test-123")
    value = await ctx2.get("data")
    assert value == {"value": 100}
```

---

#### Task 1.8: Implement Context.delete() with Backend Cleanup ⏱️ 1 hour
**File**: `src/agnt5/context.py`

**Description**: Update `ctx.delete()` to remove from both cache and backend.

**Acceptance Criteria**:
- [ ] Deletes from local cache
- [ ] Deletes from backend
- [ ] Handles missing keys gracefully
- [ ] Error handling

**Implementation**:
```python
# src/agnt5/context.py
class Context:
    async def delete(self, key: str) -> None:
        """Delete key from state (cache + backend)."""
        # Remove from cache
        self._state.pop(key, None)

        # Remove from backend
        try:
            await self._state_manager.delete(f"{self.run_id}:{key}")
            self.logger.debug(f"State deleted: {key}")
        except Exception as e:
            self.logger.error(f"Failed to delete state {key}: {e}")
```

**Test**:
```python
async def test_context_delete_backend():
    ctx = Context(run_id="test-123")
    await ctx.set("temp", "data")
    await ctx.delete("temp")

    # Verify deleted from backend
    value = await ctx._state_manager.get("test-123:temp")
    assert value is None
```

---

#### Task 1.9: Add Integration Tests for Context State Persistence ⏱️ 3 hours
**File**: `tests/test_context_state_persistence.py` (new file)

**Description**: Comprehensive tests for Context state management with backend.

**Acceptance Criteria**:
- [ ] Test state persistence across context instances
- [ ] Test state hydration from backend
- [ ] Test cache behavior (fast path)
- [ ] Test error scenarios (backend unavailable)
- [ ] Test different data types (str, int, dict, list)

**Implementation**:
```python
# tests/test_context_state_persistence.py
import pytest
from agnt5 import Context

class TestContextStatePersistence:
    @pytest.mark.asyncio
    async def test_state_survives_context_recreation(self):
        """State persists when creating new context with same run_id."""
        ctx1 = Context(run_id="persist-test")
        await ctx1.set("counter", 1)
        await ctx1.set("data", {"key": "value"})

        # Create new context (simulates restart)
        ctx2 = Context(run_id="persist-test")
        assert await ctx2.get("counter") == 1
        assert await ctx2.get("data") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_state_isolation_by_run_id(self):
        """Different run_ids have isolated state."""
        ctx1 = Context(run_id="run-1")
        ctx2 = Context(run_id="run-2")

        await ctx1.set("value", 100)
        await ctx2.set("value", 200)

        assert await ctx1.get("value") == 100
        assert await ctx2.get("value") == 200

    @pytest.mark.asyncio
    async def test_state_types(self):
        """Various data types persist correctly."""
        ctx = Context(run_id="types-test")

        await ctx.set("str", "hello")
        await ctx.set("int", 42)
        await ctx.set("float", 3.14)
        await ctx.set("list", [1, 2, 3])
        await ctx.set("dict", {"a": 1, "b": 2})

        ctx2 = Context(run_id="types-test")
        assert await ctx2.get("str") == "hello"
        assert await ctx2.get("int") == 42
        assert await ctx2.get("float") == 3.14
        assert await ctx2.get("list") == [1, 2, 3]
        assert await ctx2.get("dict") == {"a": 1, "b": 2}
```

---

#### Task 1.10: Update Worker to Pass RuntimeContext to Context ⏱️ 1 hour
**File**: `src/agnt5/worker.py`

**Description**: Ensure worker passes runtime_context when creating Context.

**Acceptance Criteria**:
- [ ] Function execution creates Context with runtime_context
- [ ] Workflow execution creates Context with runtime_context
- [ ] Entity execution creates Context with runtime_context
- [ ] Agent execution creates Context with runtime_context

**Implementation**:
```python
# src/agnt5/worker.py
async def _execute_function(self, config, input_data: bytes, request):
    # Create context with runtime_context for state persistence
    ctx = Context(
        run_id=f"{self.service_name}:{config.name}",
        component_type="function",
        runtime_context=request.runtime_context,  # ✅ Pass it through
    )

    # Execute function
    result = await config.handler(ctx, **input_dict)
```

**Test**:
```python
async def test_worker_provides_runtime_context():
    # Mock execution
    response = await worker._execute_function(config, input_data, request)
    # Context should have state_manager from runtime_context
    # (Validated by integration test)
```

---

## Phase 2: Workflow Durability (Week 2-3)

### 🎯 Goal: Enable workflow replay with proper step recording

#### Task 2.1: Add Step Event Recording to Context.step() ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Record step events for workflow replay.

**Acceptance Criteria**:
- [ ] `ctx.step()` appends to `_step_events` list
- [ ] Event includes: step_name, result, timestamp
- [ ] Events stored in Context for worker to collect
- [ ] Unit tests pass

**Implementation**:
```python
# src/agnt5/context.py
import time

class Context:
    def __init__(self, ...):
        self._step_events: List[Dict[str, Any]] = []
        ...

    async def step(self, name: str, func_or_awaitable) -> T:
        # Check replay cache
        if name in self._checkpoints:
            self.logger.info(f"⏭️ Replaying step: {name}")
            return self._checkpoints[name]

        # Execute step
        start_time = time.time()
        if inspect.iscoroutine(func_or_awaitable) or inspect.isawaitable(func_or_awaitable):
            result = await func_or_awaitable
        else:
            result = await func_or_awaitable()
        duration = time.time() - start_time

        # Cache result
        self._checkpoints[name] = result

        # ✅ Record step event for persistence
        self._step_events.append({
            "step_name": name,
            "step_type": "checkpoint",
            "result": result,
            "timestamp": time.time(),
            "duration_ms": int(duration * 1000),
        })

        self.logger.info(f"✅ Completed step: {name} ({duration*1000:.0f}ms)")
        return result
```

**Test**:
```python
async def test_step_records_events():
    ctx = Context(run_id="test")

    await ctx.step("step1", async_func())
    await ctx.step("step2", async_func())

    assert len(ctx._step_events) == 2
    assert ctx._step_events[0]["step_name"] == "step1"
    assert ctx._step_events[1]["step_name"] == "step2"
    assert "result" in ctx._step_events[0]
```

---

#### Task 2.2: Add Task Event Recording to Context.task() ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Record task invocations for workflow replay.

**Acceptance Criteria**:
- [ ] `ctx.task()` records task calls as step events
- [ ] Event includes: handler_name, input, result
- [ ] Task results checkpointed for replay
- [ ] Unit tests pass

**Implementation**:
```python
# src/agnt5/context.py
class Context:
    async def task(
        self,
        handler: Union[str, Callable],
        input: Any = None,
        *,
        service_name: Optional[str] = None,
    ) -> Any:
        # Extract handler name
        if callable(handler):
            handler_name = handler.__name__
        else:
            handler_name = handler

        # Create checkpoint key
        checkpoint_key = f"task:{handler_name}"

        # Check if already executed (replay)
        if checkpoint_key in self._checkpoints:
            self.logger.info(f"⏭️ Replaying task: {handler_name}")
            return self._checkpoints[checkpoint_key]

        # Execute task
        result = await self._execute_task(handler_name, input, service_name)

        # Checkpoint result
        self._checkpoints[checkpoint_key] = result

        # ✅ Record task event
        self._step_events.append({
            "step_name": checkpoint_key,
            "step_type": "task",
            "handler": handler_name,
            "input": input,
            "result": result,
            "timestamp": time.time(),
        })

        return result
```

**Test**:
```python
async def test_task_records_events():
    ctx = Context(run_id="test", component_type="workflow")

    result = await ctx.task(my_function, input={"data": "test"})

    assert len(ctx._step_events) == 1
    event = ctx._step_events[0]
    assert event["step_type"] == "task"
    assert event["handler"] == "my_function"
```

---

#### Task 2.3: Load Completed Steps on Workflow Startup ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Initialize Context with completed steps from previous execution.

**Acceptance Criteria**:
- [ ] Context accepts `completed_steps` parameter
- [ ] Populates `_checkpoints` dict from completed steps
- [ ] Maps step names to results correctly
- [ ] Unit tests verify replay

**Implementation**:
```python
# src/agnt5/context.py
class Context:
    def __init__(
        self,
        run_id: str,
        completed_steps: Optional[Dict[str, Any]] = None,
        ...
    ):
        # Initialize checkpoints from replay data
        self._checkpoints: Dict[str, Any] = {}
        if completed_steps:
            for step_name, result in completed_steps.items():
                self._checkpoints[step_name] = result
                self.logger.info(f"🔄 Loaded checkpoint: {step_name}")
```

**Test**:
```python
async def test_context_loads_completed_steps():
    completed = {
        "step1": "result1",
        "step2": "result2",
    }

    ctx = Context(run_id="test", completed_steps=completed)

    # Steps should replay from cache
    result1 = await ctx.step("step1", async_func())
    assert result1 == "result1"  # From replay, not execution
```

---

#### Task 2.4: Update Worker to Send Step Events to Backend ⏱️ 2 hours
**File**: `src/agnt5/worker.py`

**Description**: Ensure worker collects and sends step events in metadata.

**Acceptance Criteria**:
- [ ] Worker checks `ctx._step_events` after workflow execution
- [ ] Serializes step events to JSON
- [ ] Includes in response metadata
- [ ] Logs step count

**Implementation**:
```python
# src/agnt5/worker.py (already exists, verify it works)
async def _execute_workflow(self, config, input_data: bytes, request):
    # ... execute workflow ...

    metadata = {}

    # Collect step events
    if ctx._step_events:
        metadata["step_events"] = json.dumps(ctx._step_events)
        logger.info(f"📊 Workflow recorded {len(ctx._step_events)} steps")

    # Collect state snapshot
    if hasattr(ctx, '_state_manager'):
        # Get all state keys for this run_id
        state_snapshot = await ctx.get_state_snapshot()
        if state_snapshot:
            metadata["workflow_state"] = json.dumps(state_snapshot)

    return PyExecuteComponentResponse(
        ...
        metadata=metadata,  # ✅ Send to backend
    )
```

**Test**:
```python
async def test_worker_sends_step_events():
    response = await worker._execute_workflow(config, input_data, request)

    assert response.metadata is not None
    assert "step_events" in response.metadata
    step_events = json.loads(response.metadata["step_events"])
    assert len(step_events) > 0
```

---

#### Task 2.5: Add Context.get_state_snapshot() Helper ⏱️ 2 hours
**File**: `src/agnt5/context.py`

**Description**: Helper to get all state for current run_id.

**Acceptance Criteria**:
- [ ] Returns dict of all state keys/values
- [ ] Only includes keys for current run_id
- [ ] Handles serialization
- [ ] Unit tests pass

**Implementation**:
```python
# src/agnt5/context.py
class Context:
    async def get_state_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all state for this run_id."""
        # Return local cache (already has all state)
        return self._state.copy()

    def has_state_changes(self) -> bool:
        """Check if state was modified."""
        return len(self._state) > 0
```

**Test**:
```python
async def test_get_state_snapshot():
    ctx = Context(run_id="test")
    await ctx.set("key1", "value1")
    await ctx.set("key2", "value2")

    snapshot = await ctx.get_state_snapshot()
    assert snapshot == {"key1": "value1", "key2": "value2"}
```

---

#### Task 2.6: Write Workflow Replay Integration Test ⏱️ 3 hours
**File**: `tests/test_workflow_replay.py` (new file)

**Description**: End-to-end test of workflow replay mechanism.

**Acceptance Criteria**:
- [ ] Workflow executes steps 1, 2, 3
- [ ] Simulate failure after step 2
- [ ] Restart with completed_steps from first run
- [ ] Verify steps 1, 2 replayed from cache
- [ ] Verify only step 3 re-executed

**Implementation**:
```python
# tests/test_workflow_replay.py
import pytest
from agnt5 import workflow, Context

@pytest.mark.asyncio
async def test_workflow_replay_on_failure():
    call_log = []

    @workflow
    async def multi_step_workflow(ctx: Context, input_data: str):
        # Step 1
        step1_result = await ctx.step("fetch_data", async_fetch(input_data))
        call_log.append("step1")

        # Step 2
        step2_result = await ctx.step("process_data", async_process(step1_result))
        call_log.append("step2")

        # Step 3 (will fail first time)
        step3_result = await ctx.step("save_data", async_save(step2_result))
        call_log.append("step3")

        return {"status": "completed"}

    # First execution (fails at step 3)
    ctx1 = Context(run_id="replay-test-1", component_type="workflow")
    try:
        await multi_step_workflow(ctx1, "test-data")
    except Exception:
        pass

    # Collect completed steps from first run
    completed_steps = {
        event["step_name"]: event["result"]
        for event in ctx1._step_events
        if event.get("result") is not None
    }

    # Second execution (replay)
    call_log.clear()
    ctx2 = Context(
        run_id="replay-test-2",
        component_type="workflow",
        completed_steps=completed_steps
    )
    result = await multi_step_workflow(ctx2, "test-data")

    # Verify replay behavior
    assert call_log == ["step3"]  # Only step3 executed
    assert result["status"] == "completed"
```

---

## Phase 3: Entity Durability (Week 3-4)

### 🎯 Goal: Fix entity state hydration and distributed locking

#### Task 3.1: Add State Hydration to Entity Initialization ⏱️ 3 hours
**File**: `src/agnt5/entity.py`

**Description**: Load entity state from backend on first access.

**Acceptance Criteria**:
- [ ] Entity checks backend for existing state
- [ ] Loads state into `_entity_states` on first access
- [ ] Handles missing state (empty state)
- [ ] Unit tests pass

**Implementation**:
```python
# src/agnt5/entity.py
async def _load_entity_state(
    entity_type: str,
    key: str,
    state_manager: Optional[StateManager]
) -> Dict[str, Any]:
    """Load entity state from backend."""
    if not state_manager:
        return {}

    try:
        state_key = f"entity:{entity_type}:{key}"
        state_bytes = await state_manager.get(state_key)

        if state_bytes:
            state = json.loads(state_bytes.decode('utf-8'))
            logger.info(f"💾 Loaded entity state: {entity_type}:{key}")
            return state
    except Exception as e:
        logger.error(f"Failed to load entity state: {e}")

    return {}

# In Entity.__getattribute__ wrapper:
async with lock:
    if state_key not in _entity_states:
        # ✅ Load from backend
        loaded_state = await _load_entity_state(
            entity_type,
            key,
            getattr(ctx, '_state_manager', None)
        )
        _entity_states[state_key] = loaded_state
```

**Test**:
```python
async def test_entity_loads_state_from_backend():
    # Pre-populate backend
    state_manager = get_state_manager()
    await state_manager.set(
        "entity:Counter:user-123",
        json.dumps({"count": 42}).encode()
    )

    # Create entity (should load state)
    counter = Counter(key="user-123")
    count = await counter.get_count()
    assert count == 42  # ✅ Loaded from backend
```

---

#### Task 3.2: Save Entity State to Backend on Every Execution ⏱️ 2 hours
**File**: `src/agnt5/entity.py`, `src/agnt5/worker.py`

**Description**: Ensure entity state persists after method execution.

**Acceptance Criteria**:
- [ ] After entity method, save state to state_manager
- [ ] Worker sends state in metadata (already done)
- [ ] State keyed by entity type and key
- [ ] Unit tests verify persistence

**Implementation**:
```python
# src/agnt5/entity.py
async def _save_entity_state(
    entity_type: str,
    key: str,
    state: Dict[str, Any],
    state_manager: Optional[StateManager]
) -> None:
    """Save entity state to backend."""
    if not state_manager:
        return

    try:
        state_key = f"entity:{entity_type}:{key}"
        state_bytes = json.dumps(state).encode('utf-8')
        await state_manager.set(state_key, state_bytes)
        logger.info(f"💾 Saved entity state: {entity_type}:{key}")
    except Exception as e:
        logger.error(f"Failed to save entity state: {e}")

# In Entity.__getattribute__ wrapper (after execution):
finally:
    # Save state to backend
    await _save_entity_state(
        entity_type,
        key,
        state_dict,
        getattr(ctx, '_state_manager', None)
    )
```

**Test**:
```python
async def test_entity_saves_state_to_backend():
    counter = Counter(key="user-456")
    await counter.increment()

    # Verify state in backend
    state_manager = get_state_manager()
    state_bytes = await state_manager.get("entity:Counter:user-456")
    state = json.loads(state_bytes.decode())
    assert state["count"] == 1
```

---

#### Task 3.3: Add Distributed Lock Service Interface ⏱️ 2 hours
**File**: `src/agnt5/distributed_lock.py` (new file)

**Description**: Define interface for distributed locking (to replace asyncio.Lock).

**Acceptance Criteria**:
- [ ] Protocol for distributed lock
- [ ] Async `acquire()` and `release()` methods
- [ ] Context manager support
- [ ] In-memory implementation for testing

**Implementation**:
```python
# src/agnt5/distributed_lock.py
from typing import Protocol
import asyncio

class DistributedLock(Protocol):
    """Protocol for distributed locks."""

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire lock, returns True if acquired."""
        ...

    async def release(self) -> None:
        """Release lock."""
        ...

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()

class AsyncioDistributedLock:
    """Local asyncio lock (fallback for single-worker)."""

    def __init__(self, key: str):
        self._lock = asyncio.Lock()
        self._key = key

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        if timeout:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout)
                return True
            except asyncio.TimeoutError:
                return False
        else:
            await self._lock.acquire()
            return True

    async def release(self) -> None:
        self._lock.release()
```

---

#### Task 3.4: Implement Redis-based Distributed Lock ⏱️ 4 hours
**File**: `src/agnt5/distributed_lock.py`

**Description**: Real distributed lock using Redis (optional, for multi-worker).

**Acceptance Criteria**:
- [ ] Uses Redis SET with NX and PX options
- [ ] Implements lock acquisition with timeout
- [ ] Implements lock release with Lua script
- [ ] Handles connection failures gracefully
- [ ] Unit tests with Redis mock

**Implementation**:
```python
# src/agnt5/distributed_lock.py
import uuid
import asyncio
from typing import Optional

class RedisDistributedLock:
    """Redis-based distributed lock."""

    def __init__(self, redis_client, key: str, ttl_ms: int = 30000):
        self._redis = redis_client
        self._key = f"lock:{key}"
        self._token = str(uuid.uuid4())
        self._ttl_ms = ttl_ms

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire lock using Redis SET NX PX."""
        deadline = time.time() + (timeout or 30)

        while time.time() < deadline:
            # Try to acquire lock
            acquired = await self._redis.set(
                self._key,
                self._token,
                px=self._ttl_ms,  # TTL in milliseconds
                nx=True  # Only set if not exists
            )

            if acquired:
                return True

            # Wait before retry
            await asyncio.sleep(0.01)

        return False

    async def release(self) -> None:
        """Release lock using Lua script (check token)."""
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        await self._redis.eval(lua_script, 1, self._key, self._token)
```

**Test**:
```python
async def test_redis_distributed_lock():
    redis = await aioredis.create_redis_pool("redis://localhost")

    lock = RedisDistributedLock(redis, "test-lock")
    acquired = await lock.acquire(timeout=1.0)
    assert acquired

    # Try to acquire same lock (should fail)
    lock2 = RedisDistributedLock(redis, "test-lock")
    acquired2 = await lock2.acquire(timeout=0.1)
    assert not acquired2

    await lock.release()
```

---

#### Task 3.5: Replace asyncio.Lock with Distributed Lock in Entities ⏱️ 2 hours
**File**: `src/agnt5/entity.py`

**Description**: Use distributed lock instead of asyncio.Lock for entity methods.

**Acceptance Criteria**:
- [ ] Entity uses DistributedLock from context
- [ ] Falls back to asyncio.Lock if no distributed lock available
- [ ] Lock key includes entity type and key
- [ ] Unit tests pass

**Implementation**:
```python
# src/agnt5/entity.py
from .distributed_lock import DistributedLock, AsyncioDistributedLock

# In Entity.__getattribute__ wrapper:
async def entity_method_wrapper(*args, **kwargs):
    state_key = object.__getattribute__(self, '_state_key')

    # Get distributed lock from context or use local lock
    if hasattr(ctx, '_distributed_lock_manager'):
        lock = ctx._distributed_lock_manager.get_lock(
            f"entity:{state_key[0]}:{state_key[1]}"
        )
    else:
        # Fallback to asyncio.Lock
        if state_key not in _entity_locks:
            _entity_locks[state_key] = AsyncioDistributedLock(str(state_key))
        lock = _entity_locks[state_key]

    async with lock:
        # Execute method with distributed lock
        ...
```

---

## Phase 4: Function & Agent Durability (Week 5-6)

### 🎯 Goal: Enable checkpointing for functions and conversation persistence for agents

#### Task 4.1: Add @checkpoint Decorator for Functions ⏱️ 3 hours
**File**: `src/agnt5/checkpoint.py` (new file)

**Description**: Decorator to auto-checkpoint function steps.

**Acceptance Criteria**:
- [ ] `@checkpoint` decorator wraps function sections
- [ ] Uses `ctx.step()` internally
- [ ] Generates checkpoint names automatically
- [ ] Works with both sync and async functions

**Implementation**:
```python
# src/agnt5/checkpoint.py
def checkpoint(name: Optional[str] = None):
    """Decorator to checkpoint a function section."""
    def decorator(func):
        checkpoint_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract context from args
            ctx = None
            if args and isinstance(args[0], Context):
                ctx = args[0]

            if ctx and hasattr(ctx, 'step'):
                # Use context.step() for checkpointing
                return await ctx.step(checkpoint_name, func(*args, **kwargs))
            else:
                # No context, execute normally
                return await func(*args, **kwargs)

        return wrapper
    return decorator

# Usage:
@function
async def process_order(ctx: Context, order_id: str):
    @checkpoint("validate")
    async def validate():
        return await validate_order(order_id)

    @checkpoint("charge")
    async def charge():
        return await charge_payment(order_id)

    order = await validate()
    payment = await charge()
    return {"order": order, "payment": payment}
```

---

#### Task 4.2: Add Conversation Persistence for Agents ⏱️ 3 hours
**File**: `src/agnt5/agent.py`

**Description**: Save and restore agent conversation history.

**Acceptance Criteria**:
- [ ] Agent loads conversation from `ctx.get("conversation")`
- [ ] Agent saves conversation to `ctx.set("conversation")`
- [ ] Conversation includes all messages and tool calls
- [ ] Resume works across agent restarts

**Implementation**:
```python
# src/agnt5/agent.py
class Agent:
    async def run(self, user_message: str, context: Optional[Context] = None) -> AgentResult:
        # Load previous conversation
        previous_messages = []
        if context:
            previous_messages = await context.get("conversation_history", [])
            if previous_messages:
                logger.info(f"📜 Loaded {len(previous_messages)} previous messages")

        # Initialize conversation
        messages = [Message.from_dict(m) for m in previous_messages]
        messages.append(Message.user(user_message))

        # ... agent loop ...

        # Save conversation
        if context:
            conversation_data = [m.to_dict() for m in messages]
            await context.set("conversation_history", conversation_data)
            logger.info(f"💾 Saved {len(messages)} messages")

        return result
```

---

## Phase 5: Testing & Validation (Week 7-8)

### 🎯 Goal: Comprehensive durability testing

#### Task 5.1: Multi-Worker Entity Consistency Test ⏱️ 3 hours
#### Task 5.2: Workflow Failure Recovery Test ⏱️ 3 hours
#### Task 5.3: Agent Conversation Resume Test ⏱️ 2 hours
#### Task 5.4: Load Testing with State Persistence ⏱️ 4 hours
#### Task 5.5: Chaos Testing (Random Failures) ⏱️ 4 hours

---

## Task Tracking

### Week 1: Foundation
- [ ] Task 1.1: State Manager Protocol (2h)
- [ ] Task 1.2: In-Memory State Manager (2h)
- [ ] Task 1.3: RuntimeContext Integration (1h)
- [ ] Task 1.4: Rust Wrapper (3h)
- [ ] Task 1.5: Context.__init__ (2h)
- [ ] Task 1.6: Context.set() (2h)
- [ ] Task 1.7: Context.get() (2h)
- [ ] Task 1.8: Context.delete() (1h)
- [ ] Task 1.9: Integration Tests (3h)
- [ ] Task 1.10: Worker Integration (1h)

**Total: 19 hours (~2.5 days)**

### Week 2: Workflows
- [ ] Task 2.1: Step Event Recording (2h)
- [ ] Task 2.2: Task Event Recording (2h)
- [ ] Task 2.3: Load Completed Steps (2h)
- [ ] Task 2.4: Worker Metadata (2h)
- [ ] Task 2.5: State Snapshot (2h)
- [ ] Task 2.6: Replay Test (3h)

**Total: 13 hours (~2 days)**

### Week 3: Entities
- [ ] Task 3.1: State Hydration (3h)
- [ ] Task 3.2: State Persistence (2h)
- [ ] Task 3.3: Lock Interface (2h)
- [ ] Task 3.4: Redis Lock (4h)
- [ ] Task 3.5: Entity Lock Integration (2h)

**Total: 13 hours (~2 days)**

### Week 4-6: Functions & Agents
- [ ] Task 4.1: Checkpoint Decorator (3h)
- [ ] Task 4.2: Agent Persistence (3h)

**Total: 6 hours (~1 day)**

### Week 7-8: Testing
- [ ] Task 5.1-5.5: Comprehensive Tests (16h)

**Total: 16 hours (~2 days)**

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `ctx.set()` persists to backend
- [ ] `ctx.get()` loads from backend
- [ ] State survives context recreation
- [ ] All state persistence tests pass

### Phase 2 Complete When:
- [ ] Workflow steps are recorded
- [ ] Workflow replay works end-to-end
- [ ] Step execution is skipped on replay
- [ ] All workflow tests pass

### Phase 3 Complete When:
- [ ] Entity state loads from backend
- [ ] Entity state persists after execution
- [ ] Distributed locks prevent race conditions
- [ ] Multi-worker entity tests pass

### Phase 4 Complete When:
- [ ] Functions can checkpoint steps
- [ ] Agents save/restore conversations
- [ ] Handoffs preserve state
- [ ] All agent tests pass

### Phase 5 Complete When:
- [ ] Load tests pass with state persistence
- [ ] Chaos tests show resilience
- [ ] No state corruption under load
- [ ] Production-ready durability

---

## Next Steps

1. **Review this plan** with team
2. **Start with Task 1.1** (State Manager Protocol)
3. **Complete tasks sequentially** - each task builds on previous
4. **Run tests after each task** - ensure no regressions
5. **Update this document** as tasks complete

**Let's start with Task 1.1!** 🚀
