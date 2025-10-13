# AGNT5 SDK Durability Integration Analysis

**Date**: October 9, 2025
**Author**: Critical Architecture Review
**Scope**: Python SDK Components - Backend Integration & Durability

---

## Executive Summary

This report provides a critical analysis of all AGNT5 Python SDK components for their integration with the backend Worker Coordinator and durability infrastructure. The analysis reveals **significant gaps** in durability implementation across most components, with only **partial durability** achieved for workflows and entities.

### Overall Durability Score: **4/10** ⚠️

| Component | Durability Status | Integration Quality | Critical Issues |
|-----------|------------------|---------------------|-----------------|
| Functions | ❌ **None** | ✅ Full | No state persistence, no replay |
| Workflows | 🟡 **Partial** | ✅ Full | Step replay works, state persistence incomplete |
| Entities | 🟡 **Partial** | ✅ Full | State persistence works, no distributed locking |
| Agents | ❌ **None** | 🟡 Basic | No state persistence, no handoff durability |
| Tools | ❌ **None** | ✅ Full | No state tracking |
| Context | 🟡 **Partial** | 🟡 Mixed | In-memory only, no backend integration |

---

## 1. Functions (`function.py`) - Durability: ❌ **NONE**

### Integration Status: ✅ **COMPLETE**
- ✅ Full registration with Worker via `FunctionRegistry`
- ✅ Schema extraction (input/output) working correctly
- ✅ Retry logic implemented with exponential backoff
- ✅ Trace propagation via Rust bridge
- ✅ Pydantic model support for validation

### Durability Status: ❌ **MISSING**

**Critical Issues:**

1. **No State Persistence**
   ```python
   @function
   async def process_order(ctx: Context, order_id: str) -> dict:
       # Problem: If this function fails after partial processing,
       # there's NO way to resume from where it left off
       await validate_order(order_id)  # ❌ Not checkpointed
       await charge_payment(order_id)   # ❌ Not checkpointed
       await ship_order(order_id)       # ❌ Not checkpointed
       return {"status": "completed"}
   ```

2. **No Checkpoint/Replay Mechanism**
   - Functions execute from scratch on every retry
   - No idempotency guarantees beyond retry count
   - Lost work on transient failures

3. **Context State Not Persisted**
   - `ctx.set()` / `ctx.get()` are **in-memory only** (src/agnt5/context.py:49-65)
   - State dies with the worker process
   - No backend state manager integration

**Code Evidence:**
```python
# context.py:49-65 - IN-MEMORY ONLY
async def get(self, key: str, default: Any = None) -> Any:
    """Get value from state."""
    return self._state.get(key, default)  # ❌ self._state is dict

async def set(self, key: str, value: Any) -> None:
    """Set value in state."""
    self._state[key] = value  # ❌ Never persisted to backend
```

**Missing Backend Integration:**
- No calls to `runtime_context.state_manager` (available in Rust bridge)
- No state snapshots sent to Worker Coordinator
- No metadata for state persistence in `PyExecuteComponentResponse`

**Impact:** 🔴 **CRITICAL**
- Functions are **NOT durable** despite the SDK claiming durability
- Any failure loses all intermediate state
- Cannot implement reliable multi-step operations

---

## 2. Workflows (`workflow.py`) - Durability: 🟡 **PARTIAL** (40%)

### Integration Status: ✅ **COMPLETE**
- ✅ Full registration with Worker
- ✅ Schema extraction working
- ✅ Context injection with `component_type="workflow"`

### Durability Status: 🟡 **PARTIAL IMPLEMENTATION**

**What Works:** ✅

1. **Step Replay** (worker.py:436-468)
   ```python
   # Workflows receive completed_steps for replay
   if "completed_steps" in request.metadata:
       completed_steps = json.loads(request.metadata["completed_steps"])
       logger.info(f"🔄 Replaying workflow with {len(completed_steps)} cached steps")
   ```

2. **Step Event Recording** (worker.py:483-485)
   ```python
   # Step events are captured and sent to backend
   if ctx._step_events:
       metadata["step_events"] = json.dumps(ctx._step_events)
   ```

3. **State Snapshot** (worker.py:488-491)
   ```python
   if ctx.state.has_changes():
       state_snapshot = ctx.state.get_state_snapshot()
       metadata["workflow_state"] = json.dumps(state_snapshot)
   ```

**What's Broken:** ❌

1. **Context Step Implementation Incomplete**
   - `ctx.step()` in context.py:209-228 does **in-memory caching only**
   - No event recording to `ctx._step_events`
   - Step checkpointing not functional

   ```python
   # context.py:209-228 - INCOMPLETE
   async def step(self, name: str, func_or_awaitable: ...) -> T:
       if name in self._checkpoints:
           return self._checkpoints[name]  # ✅ Replay works

       result = await func_or_awaitable
       self._checkpoints[name] = result  # ❌ Only stored in memory
       # ❌ MISSING: Record to self._step_events for persistence
       return result
   ```

2. **No Distributed Task Coordination**
   - `ctx.task()` calls are **not checkpointed** (context.py:376-442)
   - Workflow cannot resume if worker dies mid-execution
   - Sub-task results not persisted

3. **State Manager Not Used**
   - Context has `_state_client` but never initialized (context.py:54-56)
   - No integration with `runtime_context.state_manager`
   - State is dictionary-based, not backend-backed

**Code Evidence:**
```python
# context.py:54-56 - State manager exists but NEVER SET
self._state_client: Optional[Any] = None  # ❌ Always None
self._state: Dict[str, Any] = {}         # ❌ In-memory dict
```

**Missing Integration:**
- Step events never recorded to `_step_events` list
- No platform API calls for step persistence
- Replay depends on metadata but step recording is broken

**Impact:** 🟡 **MEDIUM**
- Workflows have **skeleton durability** but incomplete
- Step replay theoretically works but step recording is broken
- Cannot reliably resume long-running workflows

---

## 3. Entities (`entity.py`) - Durability: 🟡 **PARTIAL** (60%)

### Integration Status: ✅ **COMPLETE**
- ✅ Auto-registration via `__init_subclass__`
- ✅ Method schema extraction
- ✅ Worker execution with key-based routing

### Durability Status: 🟡 **WORKING BUT LIMITED**

**What Works:** ✅

1. **State Persistence** (worker.py:609-622)
   ```python
   # Entity state IS persisted to backend
   if state_key in _entity_states:
       entity_state = _entity_states[state_key]
       state_json = json.dumps(entity_state)
       metadata = {
           "entity_state": state_json,
           "entity_type": entity_type.name,
           "entity_key": entity_key,
       }
   ```

2. **Single-Writer Semantics** (entity.py:296-298)
   ```python
   # Lock per entity key ensures consistency
   if state_key not in _entity_locks:
       _entity_locks[state_key] = asyncio.Lock()
   async with lock:
       # Execute with exclusive access
   ```

3. **State Lifecycle** (entity.py:301-315)
   ```python
   # State properly loaded and saved
   if state_key not in _entity_states:
       _entity_states[state_key] = {}
   state_dict = _entity_states[state_key]
   ctx._state = state_dict  # ✅ Shared state reference
   ```

**Critical Limitations:** ❌

1. **In-Memory State Store**
   ```python
   # entity.py:23-24 - GLOBAL IN-MEMORY STORAGE
   _entity_states: Dict[Tuple[str, str], Dict[str, Any]] = {}
   _entity_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
   ```
   - State persisted to backend **per execution** but loaded from memory
   - If worker restarts, all entity state is **lost**
   - No state hydration from backend on startup

2. **No Distributed Locking**
   - `asyncio.Lock()` only works **within single worker process**
   - **Race conditions** possible across multiple workers
   - Entity methods can execute concurrently across workers

3. **Missing State Hydration**
   - No mechanism to load state from backend when entity is first accessed
   - State only exists if previously executed in same worker instance
   - Cold start = empty state

**Code Evidence:**
```python
# entity.py:301-304 - State initialized empty, never loaded from backend
if state_key not in _entity_states:
    _entity_states[state_key] = {}  # ❌ Should load from backend
state_dict = _entity_states[state_key]
```

**Missing Backend Integration:**
- No `state_manager.get()` call to hydrate state
- No distributed lock acquisition from platform
- State persistence is **write-only**, never read back

**Impact:** 🟡 **MEDIUM-HIGH**
- Entities **appear durable** but are not truly distributed
- Multi-worker deployments will have **inconsistent state**
- State loss on worker restart

---

## 4. Agents (`agent.py`) - Durability: ❌ **NONE**

### Integration Status: 🟡 **BASIC**
- ✅ Registration with Worker via `AgentRegistry`
- ✅ Execution through worker handler
- 🟡 Basic input/output schemas
- ❌ No durability features

### Durability Status: ❌ **COMPLETELY MISSING**

**Critical Issues:**

1. **No Conversation State Persistence**
   ```python
   # agent.py:467-468 - IN-MEMORY ONLY
   messages: List[Message] = [Message.user(user_message)]
   # ❌ Conversation lost on failure
   # ❌ Cannot resume multi-turn reasoning
   ```

2. **No Tool Call Durability**
   ```python
   # agent.py:539-577 - Tool calls not checkpointed
   result = await tool.invoke(context, **tool_args)
   # ❌ If agent crashes after tool execution, result is lost
   # ❌ Tool may be re-executed on retry (not idempotent)
   ```

3. **No Handoff State Tracking**
   ```python
   # agent.py:553-565 - Handoff result returned directly
   if result.get("_handoff"):
       return AgentResult(output=result["output"], ...)
   # ❌ Handoff chain not persisted
   # ❌ Cannot resume after handoff failure
   ```

4. **Iteration Count Lost**
   - Agent loops up to `max_iterations` (agent.py:472)
   - If agent crashes at iteration 5/10, restarts from iteration 1
   - Wastes LLM calls and time

**Missing Integration:**
- No state snapshots sent to Worker Coordinator
- No context state usage (could use `ctx.step()` but doesn't)
- No metadata for conversation persistence

**Impact:** 🔴 **CRITICAL**
- Agents cannot be used for reliable multi-turn workflows
- Tool execution may be duplicated on retry
- Handoff chains are fragile

---

## 5. Tools (`tool.py`) - Durability: ❌ **NONE**

### Integration Status: ✅ **COMPLETE**
- ✅ Full registration via `ToolRegistry`
- ✅ Schema extraction from docstrings and type hints
- ✅ Execution through worker with context injection

### Durability Status: ❌ **NO DURABILITY NEEDED** (by design)

**Analysis:**

Tools are **stateless by design** and **don't need durability**:

```python
# tool.py:211-250 - Clean execution pattern
async def invoke(self, ctx: Context, **kwargs) -> Any:
    with create_span(self.name, "tool", {...}):
        result = await self.handler(ctx, **kwargs)
        return result
```

**However:**
- Tools **depend on Context** for state
- If Context doesn't persist state, tools can't be durable
- Tool results should be captured in parent component's checkpoints

**Issues:**
1. **No Idempotency Tracking**
   - Tools may be called multiple times on retry
   - No mechanism to detect duplicate calls
   - External API calls not protected

2. **Context State Not Reliable**
   - Tools use `ctx.set()` for state (tool.py:244)
   - But Context state is in-memory only
   - Tool state lost on failure

**Impact:** 🟡 **MEDIUM**
- Tools themselves are fine, but context limitations affect them
- Duplicate tool executions on retry can cause issues
- Need calling component (agent/workflow) to checkpoint results

---

## 6. Context (`context.py`) - Durability: 🟡 **FOUNDATION MISSING** (20%)

### Integration Status: 🟡 **INCOMPLETE**
- ✅ RuntimeContext passed from Rust bridge
- 🟡 State manager available but unused
- ❌ No backend state integration

### Durability Status: 🟡 **BROKEN FOUNDATION**

**Critical Architecture Flaws:**

1. **State Manager Ignored** (context.py:54-65)
   ```python
   class Context:
       def __init__(self, ...):
           self._state_client: Optional[Any] = None  # ❌ NEVER SET
           self._state: Dict[str, Any] = {}          # ❌ In-memory dict

       async def get(self, key: str, default: Any = None) -> Any:
           return self._state.get(key, default)  # ❌ Reading from dict

       async def set(self, key: str, value: Any) -> None:
           self._state[key] = value  # ❌ Writing to dict, not backend
   ```

2. **RuntimeContext Available But Unused**
   ```python
   # worker.py:286-295 - RuntimeContext has state_manager
   runtime_context = RuntimeContext.with_trace_context(
       ...
       state_manager=Arc::new(DummyStateManager),  # ⚠️ Dummy only!
   )

   # context.py:54 - But Context never uses it
   self._runtime_context = runtime_context  # ✅ Stored
   # ❌ self._state_client never initialized from runtime_context.state_manager
   ```

3. **Step Checkpointing Incomplete**
   ```python
   # context.py:209-228 - No event recording
   async def step(self, name: str, func_or_awaitable) -> T:
       if name in self._checkpoints:
           return self._checkpoints[name]  # ✅ Replay from memory

       result = await func_or_awaitable
       self._checkpoints[name] = result  # ✅ Cache in memory
       # ❌ MISSING: self._step_events.append({"step": name, "result": result})
       return result
   ```

4. **Task Orchestration Not Checkpointed**
   ```python
   # context.py:376-442 - ctx.task() doesn't record steps
   async def task(self, handler, input, *, service_name=None) -> Any:
       # ... execute task via platform ...
       result = await self._platform_client.call_function(...)
       # ❌ NOT checkpointed - will re-execute on replay
       return result
   ```

**Missing Implementations:**

1. **State Client Integration**
   ```python
   # NEEDED in context.py __init__:
   if runtime_context and runtime_context.state_manager:
       self._state_client = StateClient(runtime_context.state_manager)
   ```

2. **Step Event Recording**
   ```python
   # NEEDED in context.step():
   self._step_events.append({
       "step_name": name,
       "step_type": "checkpoint",
       "result": result,
       "timestamp": ...
   })
   ```

3. **State Persistence Hooks**
   ```python
   # NEEDED in context.set():
   async def set(self, key: str, value: Any) -> None:
       self._state[key] = value
       if self._state_client:
           await self._state_client.set(key, value)  # Persist to backend
   ```

**Impact:** 🔴 **CRITICAL - ROOT CAUSE**
- Context is the **foundation** for all component durability
- Current implementation makes ALL components non-durable
- Fixing Context would enable durability for functions, workflows, agents

---

## 7. Worker (`worker.py`) - Backend Integration: ✅ **EXCELLENT**

### Integration Status: ✅ **PRODUCTION-READY**
- ✅ Full gRPC integration via Rust bridge
- ✅ Component discovery and registration
- ✅ Message handling with trace propagation
- ✅ Streaming support for functions
- ✅ Error handling and response formatting

### Durability Support: 🟡 **INFRASTRUCTURE READY**

**What Works Perfectly:** ✅

1. **Trace Propagation** (worker.py:280-302, rust-src/worker.rs:281-361)
   ```rust
   // Extract trace context from request metadata
   let parent_context = extract_context_from_runtime_message(&invoke_request.metadata);

   // Create RuntimeContext with OTel context
   let runtime_context = RuntimeContext::with_trace_context(
       invoke_request.invocation_id.clone(),
       ...
       parent_context.clone(),
       Arc::new(DummyStateManager),  // ⚠️ Dummy for now
   );
   ```

2. **Workflow Replay Support** (worker.py:436-468)
   ```python
   # Parse completed steps for replay
   if "completed_steps" in request.metadata:
       completed_steps = json.loads(request.metadata["completed_steps"])
       ctx = Context(..., completed_steps=completed_steps)
   ```

3. **Entity State Persistence** (worker.py:609-622)
   ```python
   # Entity state sent to backend
   metadata = {
       "entity_state": json.dumps(entity_state),
       "entity_type": entity_type.name,
       "entity_key": entity_key,
   }
   ```

4. **Streaming Function Support** (worker.py:352-389)
   ```python
   # Streaming responses properly handled
   if inspect.isasyncgen(result):
       for chunk in result:
           responses.append(PyExecuteComponentResponse(..., is_chunk=True))
   ```

**What's Missing:** ⚠️

1. **Real State Manager** (runtime_adapter.rs:145-167)
   ```rust
   // DummyStateManager returns errors
   pub struct DummyStateManager;

   async fn get(&self, _key: String) -> Result<Vec<u8>> {
       Err(SdkError::Other(anyhow!("State management not implemented")))
   }
   ```

2. **Function Metadata for State** (worker.py:398-424)
   ```python
   # Functions don't send state metadata
   return PyExecuteComponentResponse(
       ...
       state_update=None,  # ❌ Always None
       metadata=None,      # ❌ No state metadata
   )
   ```

3. **No Distributed Coordination**
   - No distributed locks for entity methods
   - No leader election for singleton components
   - No work stealing for parallel execution

**Impact:** 🟡 **INFRASTRUCTURE EXISTS**
- Worker has all hooks for durability
- Just needs components to use them correctly
- State manager needs real implementation

---

## 8. Critical Gaps & Root Causes

### Root Cause Analysis

**Primary Issue:** Context State Management is **100% In-Memory**

```python
# context.py:49-65 - THE CORE PROBLEM
class Context:
    def __init__(self, ...):
        self._state: Dict[str, Any] = {}  # ❌ Python dict, not backend-backed

    async def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)  # ❌ No backend call

    async def set(self, key: str, value: Any) -> None:
        self._state[key] = value  # ❌ No backend persistence
```

**This Cascades To:**
1. Functions → No state persistence
2. Workflows → Step recording broken
3. Agents → Conversation state lost
4. Entities → State not hydrated
5. Tools → Dependent on broken Context

### Architectural Debt

1. **State Manager Interface Exists But Unused**
   - `RuntimeContext.state_manager` available (runtime_adapter.rs:43)
   - Context receives `runtime_context` (worker.py:336-340)
   - But Context never initializes `_state_client` from it

2. **No Backend State Hydration**
   - Components never load state from backend on initialization
   - State only exists if set in current execution
   - Cold start = empty state

3. **No Distributed Coordination**
   - Entity locks are `asyncio.Lock()` (single process)
   - No distributed mutex service
   - Race conditions in multi-worker deployments

4. **Incomplete Workflow Durability**
   - Replay infrastructure exists (worker.py:436-468)
   - But step recording never happens (context.py:209-228)
   - Cannot actually resume workflows

---

## 9. Recommendations - Priority Order

### 🔥 **P0 - CRITICAL (Fix Immediately)**

1. **Implement Real State Manager in Context**
   ```python
   # context.py - REQUIRED CHANGES
   class Context:
       def __init__(self, runtime_context: Optional[RuntimeContext] = None, ...):
           if runtime_context and runtime_context.state_manager:
               self._state_client = StateClient(runtime_context.state_manager)
           else:
               self._state_client = None
           self._state: Dict[str, Any] = {}  # Local cache

       async def get(self, key: str, default: Any = None) -> Any:
           # Try backend first, fall back to local cache
           if self._state_client:
               try:
                   value = await self._state_client.get(key)
                   return value if value is not None else default
               except:
                   pass
           return self._state.get(key, default)

       async def set(self, key: str, value: Any) -> None:
           self._state[key] = value  # Local cache
           if self._state_client:
               await self._state_client.set(key, value)  # Persist
   ```

2. **Implement Real StateManager in Rust**
   ```rust
   // Replace DummyStateManager with:
   pub struct PlatformStateManager {
       client: StateServiceClient,
       run_id: String,
   }

   impl StateManager for PlatformStateManager {
       async fn get(&self, key: String) -> Result<Vec<u8>> {
           // Call state service via gRPC
           let response = self.client.get_state(GetStateRequest {
               run_id: self.run_id.clone(),
               key,
           }).await?;
           Ok(response.value)
       }
   }
   ```

3. **Fix Workflow Step Recording**
   ```python
   # context.py - Complete step() implementation
   async def step(self, name: str, func_or_awaitable) -> T:
       # Check replay cache
       if name in self._checkpoints:
           self.logger.info(f"Replaying step: {name}")
           return self._checkpoints[name]

       # Execute step
       result = await func_or_awaitable

       # Cache locally
       self._checkpoints[name] = result

       # CRITICAL: Record event for persistence
       self._step_events.append({
           "step_name": name,
           "step_type": "checkpoint",
           "result": result,
           "timestamp": time.time(),
       })

       return result
   ```

### ⚠️ **P1 - HIGH (Next Sprint)**

4. **Add State Hydration for Entities**
   ```python
   # entity.py - Load state from backend on first access
   async with lock:
       if state_key not in _entity_states:
           # Load from backend
           if self._state_client:
               backend_state = await self._state_client.get_entity_state(
                   entity_type, key
               )
               _entity_states[state_key] = backend_state or {}
           else:
               _entity_states[state_key] = {}
   ```

5. **Implement Distributed Locks for Entities**
   - Add distributed lock service to platform
   - Replace `asyncio.Lock()` with distributed mutex
   - Ensure single-writer across all workers

6. **Add Agent Conversation Persistence**
   ```python
   # agent.py - Save conversation state
   async def run(self, user_message: str, context: Context) -> AgentResult:
       # Load previous conversation
       messages = await context.get("conversation_history", [])

       # ... agent loop ...

       # Save updated conversation
       await context.set("conversation_history", messages)
   ```

### 📋 **P2 - MEDIUM (Future)**

7. **Function State Checkpointing**
   - Add `@checkpoint` decorator for function steps
   - Automatically record intermediate results
   - Enable resume from last checkpoint

8. **Tool Idempotency Tracking**
   - Generate idempotency keys for tool calls
   - Store tool results by idempotency key
   - Return cached result for duplicate calls

9. **Distributed Task Coordination**
   - Implement work stealing for parallel tasks
   - Add leader election for singleton functions
   - Support distributed workflows across workers

---

## 10. Testing Requirements

### Critical Test Scenarios

1. **State Persistence Across Restarts**
   ```python
   # Test: Worker restart preserves state
   async def test_state_persistence():
       ctx = Context(...)
       await ctx.set("counter", 42)

       # Simulate worker restart
       new_ctx = Context(same_run_id)
       value = await new_ctx.get("counter")
       assert value == 42  # ❌ Currently fails
   ```

2. **Workflow Replay**
   ```python
   # Test: Workflow resumes from failure point
   async def test_workflow_replay():
       @workflow
       async def multi_step(ctx: Context):
           step1 = await ctx.step("step1", expensive_operation())
           step2 = await ctx.step("step2", may_fail_operation())
           return step1 + step2

       # First execution fails at step2
       # Second execution should replay step1 from cache
   ```

3. **Entity Consistency**
   ```python
   # Test: Multiple workers don't corrupt entity state
   async def test_entity_distributed_consistency():
       # Start 2 workers
       entity1 = Counter(key="shared")  # Worker 1
       entity2 = Counter(key="shared")  # Worker 2

       # Concurrent increments
       await asyncio.gather(
           entity1.increment(),
           entity2.increment(),
       )

       # Final count should be 2, not 1 (race condition)
   ```

---

## 11. Migration Path

### Phase 1: Foundation (Week 1-2)
- ✅ Implement `PlatformStateManager` in Rust
- ✅ Integrate state manager in Context
- ✅ Add state persistence tests
- ✅ Deploy to staging

### Phase 2: Workflows (Week 3)
- ✅ Fix `ctx.step()` event recording
- ✅ Complete workflow replay mechanism
- ✅ Add workflow durability tests
- ✅ Document workflow patterns

### Phase 3: Entities (Week 4)
- ✅ Add state hydration from backend
- ✅ Implement distributed locks
- ✅ Add multi-worker consistency tests
- ✅ Migration guide for entities

### Phase 4: Agents & Tools (Week 5-6)
- ✅ Agent conversation persistence
- ✅ Tool idempotency tracking
- ✅ Handoff state management
- ✅ Integration tests

### Phase 5: Functions (Week 7-8)
- ✅ Function checkpointing API
- ✅ Automatic state capture
- ✅ Resume from failure
- ✅ Production rollout

---

## 12. Conclusion

### Current State: **NOT PRODUCTION-READY FOR DURABLE WORKLOADS**

The AGNT5 Python SDK has **excellent integration** with the backend Worker Coordinator but **severely limited durability**. The root cause is the **in-memory-only Context state management**, which makes all components non-durable by default.

### Key Findings:

1. ✅ **Worker infrastructure is solid** - gRPC, tracing, streaming all work
2. ❌ **State persistence is completely broken** - Context uses Python dicts
3. 🟡 **Workflows have durability skeleton** - But step recording doesn't work
4. 🟡 **Entities have partial durability** - State saved but not loaded
5. ❌ **Functions, Agents, Tools have no durability** - Dependent on Context

### Immediate Actions Required:

1. 🔥 **Stop claiming "durable" in SDK docs** until state persistence works
2. 🔥 **Implement real StateManager** in Rust and integrate with Context
3. 🔥 **Fix workflow step recording** to enable actual replay
4. ⚠️ **Add state hydration** for entities to work across workers
5. ⚠️ **Implement distributed locks** for entity consistency

### Risk Assessment:

- **Current SDK is suitable ONLY for stateless workloads**
- **Multi-step workflows WILL FAIL on retries** (lost intermediate state)
- **Entities WILL HAVE RACE CONDITIONS** in multi-worker deployments
- **Agents CANNOT BE USED** for reliable multi-turn conversations

**Recommendation:** Treat this as a **P0 bug** and dedicate 1-2 sprints to fix the foundation before adding more features.

---

*This analysis is based on code review as of October 9, 2025. Review covers Python SDK only; Rust core and platform services not fully analyzed.*
