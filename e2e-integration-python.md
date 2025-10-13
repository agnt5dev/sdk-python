# E2E Integration Testing Plan for Python SDK

## Architecture: Client → Platform → Worker

This document describes the end-to-end integration testing strategy for the AGNT5 Python SDK using the **Client API** as the primary test interface.

## Overview

The key insight is that **the Client (`agnt5.Client`) is the user-facing API** - testing through it validates the entire system:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Integration Test                          │
│                                                                   │
│  from agnt5 import Client                                        │
│                                                                   │
│  client = Client("http://localhost:34181")                       │
│  result = client.entity("ShoppingCart", "user-123").add_item(...) │
│                                     ↓                             │
└─────────────────────────────────────┼───────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │   HTTP POST to Gateway            │
                    │   /entity/ShoppingCart/user-123/  │
                    │         add_item                  │
                    └─────────────────┼─────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Platform (Testcontainers)                     │
│                                                                   │
│  ┌──────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│  │ Gateway  │───▶│ Execution Engine│───▶│Worker Coordinator│   │
│  │ :34181   │    │     :34185      │    │     :34186      │   │
│  └──────────┘    └─────────────────┘    └────────┼────────┘   │
│                                                    │             │
│  ┌──────────────┐       ┌──────────────┐         │             │
│  │ CockroachDB  │       │  Redpanda    │         │             │
│  │   (State)    │       │  (Events)    │         │             │
│  └──────────────┘       └──────────────┘         │             │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│              Python Worker Process (SDK Code)                    │
│                                                                   │
│  from agnt5 import Entity                                        │
│                                                                   │
│  class ShoppingCart(Entity):                                     │
│      async def add_item(self, item_id, quantity, price):        │
│          items = self.state.get("items", {})                     │
│          items[item_id] = {"quantity": quantity, "price": price} │
│          self.state.set("items", items)  # ← SAVES TO PLATFORM  │
│          return {"total_items": len(items)}                      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓
                            Result flows back up
                            Client receives response
```

## What This Tests

✅ **Client → Gateway HTTP communication**
✅ **Gateway → Execution Engine orchestration**
✅ **Execution Engine → Worker Coordinator routing**
✅ **Worker Coordinator → Worker gRPC invocation**
✅ **Worker → Platform state persistence** (via gRPC back to coordinator)
✅ **Result propagation** back to client
✅ **Platform persistence** (CockroachDB, Redpanda)
✅ **Event sourcing** and projections

## Why This Approach is Correct

### 1. Tests the Actual User Experience

```python
# This is EXACTLY what users write:
from agnt5 import Client

client = Client("http://localhost:34181")
result = client.entity("ShoppingCart", "user-123").add_item(
    item_id="widget-1",
    quantity=2,
    price=29.99
)

# If THIS works in tests, it works for users!
```

### 2. Validates End-to-End Integration

Unlike unit tests that mock platform components, these tests use:
- **Real Gateway** (built from local code)
- **Real Execution Engine** (built from local code)
- **Real Worker Coordinator** (built from local code)
- **Real CockroachDB** (Testcontainer)
- **Real Redpanda** (Testcontainer)
- **Real Worker** (Python subprocess)

### 3. Tests Platform Persistence (The Critical Gap!)

```python
@pytest.mark.integration
def test_entity_persistence_via_client(platform, worker):
    """Test entity state persists using Client API."""

    client = Client(platform["gateway_url"])

    # 1. Add item via client
    result = client.entity("ShoppingCart", "user-123").add_item(
        item_id="widget-1",
        quantity=2,
        price=29.99
    )
    assert result["total_items"] == 1

    # 2. Restart worker (simulate crash)
    worker.restart()

    # 3. Get total via client (state should be restored)
    total = client.entity("ShoppingCart", "user-123").get_total()

    # ✅ This ONLY passes if platform persistence works!
    assert total == 59.98
```

### 4. SDK-Agnostic Platform Testing

Each SDK just needs:
1. A `Client` implementation (already exists in Python)
2. Worker implementation (already exists)
3. Integration tests using that Client

```python
# Python SDK tests
sdk/sdk-python/tests/integration/test_client_e2e.py

# TypeScript SDK tests (future)
sdk/sdk-typescript/tests/integration/test_client_e2e.ts

# Go SDK tests (future)
sdk/sdk-go/tests/integration/client_e2e_test.go

# All test the SAME platform via their SDK's Client!
```

### 5. Can Test Cross-SDK Scenarios

```python
@pytest.mark.integration
def test_python_client_calls_typescript_worker(platform):
    """Python client invoking TypeScript worker."""

    # Start TypeScript worker
    ts_worker = start_typescript_worker(platform)

    # Use Python client to invoke TypeScript function
    from agnt5 import Client
    client = Client(platform["gateway_url"])

    result = client.run("typescript_function", {"data": "test"})

    # ✅ Cross-language integration works!
    assert result["processed"] == True
```

## Test-Driven Development with Integration Tests

### Why Integration TDD is Essential for AGNT5

From the critical analysis (see `analysis.md`), we discovered:

```python
# Current situation:
# Unit tests: ✅ 90%+ passing
# Platform integration: ❌ 0% working

# Example from entity.py:355-363
# TODO: Load state from platform if not in memory
# TODO: Save state to platform after successful execution

# Unit tests PASS (use in-memory state)
# But real platform integration DOESN'T WORK!
```

**Integration TDD solves this by testing end-to-end from day one.**

### The Integration TDD Cycle

```
┌─────────────────────────────────────────────────────────────┐
│  1. Write Integration Test (RED)                            │
│     Define the goal - what should work end-to-end          │
│                                                              │
│     @pytest.mark.integration                                │
│     def test_entity_state_persists(client, worker):         │
│         # This will FAIL - exposes TODO!                    │
│         client.entity("Cart", "user").add_item(...)         │
│         worker.restart()                                     │
│         assert client.entity("Cart", "user").get_total()    │
└─────────────────────┬───────────────────────────────────────┘
                      │ ❌ FAILS - Feature doesn't work
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Write Unit Tests (RED)                                  │
│     Break down the problem into components                  │
│                                                              │
│     def test_entity_manager_calls_rust_load():              │
│         # Mock test - will pass with stub                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Implement Feature (GREEN)                               │
│     Build the actual implementation                         │
│                                                              │
│     - Wire Rust EntityStateManager to Python                │
│     - Implement load_state() gRPC call                      │
│     - Implement save_state() gRPC call                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ ✅ Unit tests pass
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Run Integration Test (GREEN?)                           │
│     Does it work end-to-end now?                            │
│                                                              │
│     $ pytest tests/integration/test_entity_persistence.py   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
         ┌────────────┴────────────┐
         │                          │
    ✅ GREEN                   ❌ Still RED
    │                          │
    │                          ↓
    │                 Fix integration issues
    │                 (wire components together)
    │                          │
    └──────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Refactor (keep GREEN)                                   │
│     Improve code while tests remain passing                 │
└─────────────────────────────────────────────────────────────┘
```

### Example: Entity Persistence Feature (Integration TDD)

#### Step 1: Write Integration Test FIRST (RED)

```python
# tests/integration/test_entity_persistence.py

import pytest
from agnt5 import Client

@pytest.mark.integration
def test_entity_state_survives_worker_restart(platform, worker, client):
    """
    GOAL: Entity state should persist to platform and survive restarts.

    This test will FAIL initially because:
    - EntityStateManager._rust_manager is None
    - load_state() is stubbed (TODO comment)
    - save_state() is stubbed (TODO comment)
    """

    # 1. Modify entity state
    result = client.entity("BankAccount", "alice").deposit(amount=100.0)
    assert result["balance"] == 100.0

    # 2. Restart worker (simulate crash)
    worker.restart()

    # 3. State should be restored from platform
    balance = client.entity("BankAccount", "alice").get_balance()

    # ❌ FAILS HERE - balance is None because state was lost
    assert balance == 100.0
```

**Run the test:**
```bash
$ pytest tests/integration/test_entity_persistence.py -v

test_entity_state_survives_worker_restart ... FAILED  ❌

FAILURES:
  AssertionError: assert None == 100.0
  # State was not persisted to platform
```

#### Step 2: Write Unit Tests to Break Down Problem (RED)

```python
# tests/unit/test_entity_state_manager.py

def test_entity_state_manager_loads_from_rust():
    """Unit test for load_state integration."""

    # Mock Rust manager
    rust_manager = Mock()
    rust_manager.load_state.return_value = LoadResult(
        found=True,
        state_json=b'{"balance": 100.0}',
        version=1
    )

    state_manager = EntityStateManager(rust_manager)
    state = state_manager.load_state("BankAccount", "alice")

    # ✅ Unit test passes with mock
    assert state["balance"] == 100.0
    rust_manager.load_state.assert_called_once()


def test_entity_state_manager_saves_to_rust():
    """Unit test for save_state integration."""

    rust_manager = Mock()
    rust_manager.save_state.return_value = SaveResult(new_version=2)

    state_manager = EntityStateManager(rust_manager)
    state_manager.save_state(
        "BankAccount", "alice",
        {"balance": 150.0},
        version=1
    )

    # ✅ Unit test passes with mock
    rust_manager.save_state.assert_called_once()
```

#### Step 3: Implement Feature (GREEN)

```python
# src/agnt5/entity.py

class EntityStateManager:
    def __init__(self, rust_entity_state_manager=None):
        self._rust_manager = rust_entity_state_manager  # ✅ Wire it up!

    async def _execute_method_with_lock(self, ...):
        """Execute entity method with state management."""

        # ✅ IMPLEMENT: Load state from platform if not in memory
        if state_key not in self._states:
            if self._rust_manager:
                result = await self._rust_manager.load_state(
                    entity_type=entity_type,
                    entity_key=entity_key
                )
                if result.found:
                    state_dict = json.loads(result.state_json)
                    self._states[state_key] = state_dict
                    self._versions[state_key] = result.version

        # Execute method...
        result = await method(entity_instance, **method_kwargs)

        # ✅ IMPLEMENT: Save state to platform after execution
        if self._rust_manager:
            state_dict = self._states.get(state_key, {})
            expected_version = self._versions.get(state_key, 0)

            save_result = await self._rust_manager.save_state(
                entity_type=entity_type,
                entity_key=entity_key,
                state_json=json.dumps(state_dict).encode(),
                expected_version=expected_version
            )
            self._versions[state_key] = save_result.new_version

        return result
```

**Run unit tests:**
```bash
$ pytest tests/unit/test_entity_state_manager.py -v

test_entity_state_manager_loads_from_rust ... PASSED ✅
test_entity_state_manager_saves_to_rust ... PASSED ✅
```

#### Step 4: Run Integration Test Again (GREEN!)

```bash
$ pytest tests/integration/test_entity_persistence.py -v

test_entity_state_survives_worker_restart ... PASSED ✅

# Success! Feature works end-to-end!
```

### Practical Integration TDD Workflow

#### Week 1: Write All Integration Tests (RED)

Write tests for all features, expect most to fail:

```python
# Day 1: Entity tests
@pytest.mark.integration
def test_entity_persistence():
    # ❌ FAILS - platform integration not wired
    ...

@pytest.mark.integration
def test_entity_concurrency():
    # ❌ FAILS - distributed locking not implemented
    ...

# Day 2: Workflow tests
@pytest.mark.integration
def test_workflow_recovery():
    # ❌ FAILS - checkpoint persistence not implemented
    ...

@pytest.mark.integration
def test_workflow_idempotency():
    # ❌ FAILS - step replay broken (non-deterministic names)
    ...

# Day 3: Advanced features
@pytest.mark.integration
def test_durable_timers():
    # ❌ FAILS - not implemented (commented out)
    ...

@pytest.mark.integration
def test_signals():
    # ❌ FAILS - not implemented (commented out)
    ...
```

**Result after Week 1:**
```bash
$ pytest tests/integration/ -v

============================== test session starts ==============================
collected 25 items

test_entity_persistence.py::test_state_survives_restart FAILED         [  4%]
test_entity_persistence.py::test_concurrent_updates FAILED             [  8%]
test_entity_persistence.py::test_multi_worker_sync FAILED              [ 12%]
test_workflow_recovery.py::test_checkpoint_replay FAILED               [ 16%]
test_workflow_recovery.py::test_idempotency FAILED                     [ 20%]
test_durable_timers.py::test_timer_persistence FAILED                  [ 24%]
test_signals.py::test_signal_coordination FAILED                       [ 28%]
...

========================== 5 passed, 20 failed in 45.23s ==========================

Current Production Readiness: 20% ❌
```

**This gives you a CLEAR TODO list!**

#### Week 2-4: Implement Features (GREEN)

Work through failures by priority:

```bash
# P0: Critical for any production use
✅ Week 2: Entity persistence to platform
✅ Week 2: Workflow checkpoint persistence
✅ Week 2: Error handling improvements

$ pytest tests/integration/ -v
# 12 passed, 13 failed (48% ready)

# P1: Important for reliability
✅ Week 3: Retry error filtering
✅ Week 3: Graceful shutdown
✅ Week 3: Performance limits

$ pytest tests/integration/ -v
# 18 passed, 7 failed (72% ready)

# P2: Advanced features
✅ Week 4: Durable timers
✅ Week 4: Signal coordination
✅ Week 4: Chaos engineering

$ pytest tests/integration/ -v
# 25 passed, 0 failed (100% ready) ✅
```

### Benefits of Integration TDD for AGNT5

#### 1. **Exposes Real Gaps Immediately**

```python
# Without integration tests:
Unit tests: 90%+ passing ✅
Platform integration: Unknown ❓

# With integration tests:
$ pytest tests/integration/ -v

❌ test_entity_persistence ... FAILED (TODO: platform integration)
❌ test_workflow_recovery ... FAILED (TODO: checkpoint persistence)
❌ test_durable_timers ... FAILED (TODO: not implemented)
❌ test_signals ... FAILED (TODO: not implemented)

# Clear visibility into what needs work!
```

#### 2. **Objective Production Readiness Metrics**

```bash
# Before v1.0 release:
$ pytest tests/integration/ -v

================================ test session starts =================================
collected 25 items

test_entity_persistence.py::test_state_survives_restart PASSED         [  4%]
test_entity_persistence.py::test_concurrent_updates PASSED             [  8%]
test_workflow_recovery.py::test_checkpoint_replay PASSED               [ 12%]
test_workflow_recovery.py::test_idempotency PASSED                     [ 16%]
...
test_performance.py::test_latency_p99_under_200ms PASSED              [100%]

========================= 25 passed in 45.23s ==========================

Production Readiness: 100% ✅
Safe to release v1.0!
```

#### 3. **Prevents False Confidence**

```python
# Traditional approach (unit tests only):
def test_entity_state_manager():
    manager = EntityStateManager()
    manager.set_state("key", "value")
    assert manager.get_state("key") == "value"
    # ✅ PASSES but only tests in-memory!

# Integration TDD approach:
@pytest.mark.integration
def test_entity_state_platform_roundtrip(client, worker):
    client.entity("Cart", "user").add_item(...)
    worker.restart()  # Crash test
    assert client.entity("Cart", "user").get_total() == 10.0
    # ❌ FAILS if platform integration broken
    # ✅ PASSES only when end-to-end works
```

#### 4. **Documents Expected Behavior**

Integration tests serve as executable specifications:

```python
@pytest.mark.integration
def test_entity_optimistic_locking(client):
    """
    SPECIFICATION: When two concurrent updates conflict,
    the second one should retry with fresh state.

    This documents AND validates the expected behavior.
    """
    # Test serves as both specification and validation
```

### Recommended Hybrid Strategy

Use **both** integration and unit tests:

```python
┌─────────────────────────────────────────────────────────────┐
│         Integration Tests (Slow, Comprehensive)             │
│                                                              │
│  Write FIRST to define goal                                 │
│  Run LAST to validate                                       │
│  Example: test_entity_state_persists_across_restart()       │
│                                                              │
│  Speed: Seconds per test                                    │
│  Coverage: Entire stack (Client → Platform → Worker)       │
│  Value: Proves it works end-to-end                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Drives development
                         │
┌────────────────────────▼────────────────────────────────────┐
│           Unit Tests (Fast, Focused)                        │
│                                                              │
│  Write during implementation                                │
│  Run constantly (on every save)                             │
│  Example: test_entity_state_manager_calls_rust_load()       │
│                                                              │
│  Speed: Milliseconds per test                               │
│  Coverage: Individual components                            │
│  Value: Fast feedback during development                    │
└─────────────────────────────────────────────────────────────┘

Workflow:
1. Write integration test (RED) → defines "done"
2. Write unit tests (RED) → breaks down problem
3. Implement code (GREEN) → unit tests pass
4. Run integration test (GREEN?) → validates end-to-end
5. If still RED, debug integration issues
6. Refactor when GREEN
```

### Action Plan: Integration TDD Sprint

#### Week 1: Write Integration Tests (Accept RED)

```bash
# Goal: Create comprehensive integration test suite
# Expected: 60-70% will FAIL (this is good!)

Day 1: Entity tests (persistence, concurrency, isolation)
Day 2: Workflow tests (recovery, idempotency, orchestration)
Day 3: Function tests (execution, retry, error handling)
Day 4: Advanced tests (timers, signals, streaming)
Day 5: Performance tests (latency, throughput, memory)
```

#### Week 2-4: Implement Features (Make GREEN)

```bash
# Goal: Make tests pass by priority

Week 2 (P0 - Blocking):
  ✅ Entity persistence to platform
  ✅ Workflow checkpoint persistence
  ✅ Error handling improvements
  Target: 50% tests passing

Week 3 (P1 - Important):
  ✅ Retry error filtering
  ✅ Graceful shutdown
  ✅ Performance limits
  Target: 75% tests passing

Week 4 (P2 - Advanced):
  ✅ Durable timers
  ✅ Signal coordination
  ✅ Chaos engineering
  Target: 100% tests passing
```

#### Release Gate

```bash
# Require 100% integration test pass rate for v1.0

$ pytest tests/integration/ -v --tb=short

# Only release when:
✅ All integration tests passing
✅ No skipped tests (no @pytest.mark.skip)
✅ No TODO comments in test code
✅ Performance benchmarks meet targets
```

### Comparison: Traditional vs Integration TDD

| Aspect | Traditional Unit TDD | Integration TDD (This Plan) |
|--------|---------------------|----------------------------|
| **Test First** | ✅ Yes (unit tests) | ✅ Yes (integration tests) |
| **Fast Feedback** | ✅ Milliseconds | ⚠️ Seconds (acceptable) |
| **Real Confidence** | ❌ May have gaps | ✅ Tests complete system |
| **False Positives** | ⚠️ High (mocks may lie) | ✅ Low (tests reality) |
| **Gap Detection** | ❌ Misses integration | ✅ Catches everything |
| **AGNT5 Value** | ⚠️ Already have good unit tests | ✅ **Critical - exposes TODOs!** |
| **Production Ready** | ❓ Unknown | ✅ Measurable (test pass %) |

### Key Insight

**Integration TDD is perfect for AGNT5 because:**

1. ✅ **Exposes stubbed code immediately** - No more hidden TODOs
2. ✅ **Objective metrics** - Production readiness = test pass rate
3. ✅ **Prevents regressions** - Refactoring safely with test coverage
4. ✅ **Documents behavior** - Tests show how system should work
5. ✅ **Builds confidence** - If tests pass, platform actually works

When integration tests pass at 100%, the SDK is **objectively** production-ready.

## Proposed Test Structure

```
sdk/sdk-python/tests/integration/
├── conftest.py                          # Testcontainers setup
├── test_client_functions.py            # Function invocation via client
├── test_client_entities.py             # Entity persistence via client
├── test_client_workflows.py            # Workflow orchestration via client
├── test_client_streaming.py            # Streaming via client.stream()
├── test_durability.py                   # Crash recovery tests
├── test_concurrency.py                  # Multi-worker tests
├── utils.py                             # Platform verification utilities
└── blueprints/
    └── test-service/                    # Test worker code
        ├── app.py                       # Worker entry point
        ├── components.py                # Test entities/functions/workflows
        └── requirements.txt             # Dependencies
```

## Implementation Details

### 1. Testcontainers Platform Setup

**File: `tests/integration/conftest.py`**

```python
import pytest
import subprocess
import time
import os
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
import requests


@pytest.fixture(scope="session")
def platform_stack():
    """Start entire AGNT5 platform stack using Testcontainers."""

    # Start CockroachDB
    cockroach = DockerContainer("cockroachdb/cockroach:latest")
    cockroach.with_exposed_ports(26257, 8080)
    cockroach.with_command("start-single-node --insecure")
    cockroach.start()
    wait_for_logs(cockroach, "nodeID", timeout=30)

    # Start Redpanda
    redpanda = DockerContainer("docker.redpanda.com/vectorized/redpanda:latest")
    redpanda.with_exposed_ports(9092, 8081)
    redpanda.with_command(
        "redpanda start --smp 1 --overprovisioned "
        "--kafka-addr PLAINTEXT://0.0.0.0:29092"
    )
    redpanda.start()
    wait_for_logs(redpanda, "Successfully started Redpanda", timeout=30)

    # Get connection details
    cockroach_url = f"postgresql://root@{cockroach.get_container_host_ip()}:{cockroach.get_exposed_port(26257)}/orchestration?sslmode=disable"
    redpanda_broker = f"{redpanda.get_container_host_ip()}:{redpanda.get_exposed_port(9092)}"

    # Start AGNT5 Gateway (built from local code)
    # TODO: Use docker build from platform/cmd/gateway or pre-built image
    gateway = DockerContainer("agnt5/gateway:dev")
    gateway.with_exposed_ports(34181, 34182)
    gateway.with_env("COCKROACH_URL", cockroach_url)
    gateway.with_env("REDPANDA_BROKERS", redpanda_broker)
    gateway.start()

    # Wait for gateway health
    gateway_url = f"http://{gateway.get_container_host_ip()}:{gateway.get_exposed_port(34181)}"
    for _ in range(30):
        try:
            if requests.get(f"{gateway_url}/health").status_code == 200:
                break
        except:
            time.sleep(1)

    yield {
        "gateway_url": gateway_url,
        "gateway_grpc_port": gateway.get_exposed_port(34182),
        "cockroach_url": cockroach_url,
        "redpanda_broker": redpanda_broker,
        "containers": {
            "gateway": gateway,
            "cockroach": cockroach,
            "redpanda": redpanda,
        }
    }

    # Cleanup
    gateway.stop()
    cockroach.stop()
    redpanda.stop()


@pytest.fixture
def worker_process(platform_stack):
    """Start Python worker connected to platform."""

    worker = subprocess.Popen(
        ["uv", "run", "python", "app.py"],
        cwd="./tests/integration/blueprints/test-service",
        env={
            **os.environ,
            "AGNT5_COORDINATOR_ENDPOINT": f"{platform_stack['gateway_url']}/coordinator",
            "AGNT5_TENANT_ID": "test-tenant-001",
            "AGNT5_DEPLOYMENT_ID": "test-deployment-001",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for worker registration
    time.sleep(2)

    yield worker

    # Cleanup
    worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()


@pytest.fixture
def client(platform_stack):
    """Create Client instance connected to platform."""
    from agnt5 import Client
    return Client(platform_stack["gateway_url"])
```

### 2. Function Invocation Tests

**File: `tests/integration/test_client_functions.py`**

```python
import pytest


@pytest.mark.integration
def test_sync_function_call(client):
    """Test synchronous function invocation via client."""
    result = client.run("greet", {"name": "Alice"})
    assert result["message"] == "Hello, Alice!"


@pytest.mark.integration
def test_async_function_call(client):
    """Test async function submission and polling."""
    run_id = client.submit("long_task", {"duration": 1})

    # Wait for completion
    result = client.wait_for_result(run_id, timeout=10)

    assert result["status"] == "completed"
    assert result["duration"] >= 1


@pytest.mark.integration
def test_function_retry_on_failure(client):
    """Test function retry logic works end-to-end."""
    # Function fails twice, succeeds on 3rd attempt
    result = client.run("flaky_function", {"fail_count": 2})

    assert result["succeeded"] == True
    assert result["attempts"] == 3


@pytest.mark.integration
def test_function_error_propagation(client):
    """Test errors propagate correctly to client."""
    from agnt5 import RunError

    with pytest.raises(RunError) as exc_info:
        client.run("failing_function", {"error": "Test error"})

    assert "Test error" in str(exc_info.value)
```

### 3. Entity Persistence Tests

**File: `tests/integration/test_client_entities.py`**

```python
import pytest


@pytest.mark.integration
def test_entity_state_persists_across_restart(client, worker_process, platform_stack):
    """Test entity state survives worker restart."""

    # 1. Add item to shopping cart
    result = client.entity("ShoppingCart", "user-123").add_item(
        item_id="widget-1",
        quantity=2,
        price=29.99
    )
    assert result["total_items"] == 1

    # 2. Restart worker (simulate crash)
    worker_process.terminate()
    worker_process.wait()

    # Start new worker
    import subprocess
    import os
    new_worker = subprocess.Popen(
        ["uv", "run", "python", "app.py"],
        cwd="./tests/integration/blueprints/test-service",
        env={
            **os.environ,
            "AGNT5_COORDINATOR_ENDPOINT": f"{platform_stack['gateway_url']}/coordinator",
        }
    )
    time.sleep(2)

    # 3. Verify state was restored from platform
    total = client.entity("ShoppingCart", "user-123").get_total()

    # ✅ This ONLY passes if platform persistence works!
    assert total == 59.98

    # Cleanup
    new_worker.terminate()


@pytest.mark.integration
def test_entity_concurrent_updates_no_lost_writes(client):
    """Test single-writer guarantee with concurrent calls."""
    import asyncio

    # Reset counter
    client.entity("Counter", "shared").reset()

    # 50 concurrent increments
    async def run_increments():
        tasks = []
        for _ in range(50):
            # Each call is synchronous HTTP but we can parallelize
            task = asyncio.to_thread(
                lambda: client.entity("Counter", "shared").increment()
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

    asyncio.run(run_increments())

    # Verify all updates applied (no lost writes)
    count = client.entity("Counter", "shared").get_count()
    assert count == 50  # ✅ No lost updates!


@pytest.mark.integration
def test_entity_isolation_between_keys(client):
    """Test different entity keys have isolated state."""

    # Modify two different shopping carts
    client.entity("ShoppingCart", "alice").add_item("item-1", 1, 10.0)
    client.entity("ShoppingCart", "bob").add_item("item-2", 2, 20.0)

    # Verify isolation
    alice_total = client.entity("ShoppingCart", "alice").get_total()
    bob_total = client.entity("ShoppingCart", "bob").get_total()

    assert alice_total == 10.0
    assert bob_total == 40.0  # 2 * 20.0
```

### 4. Workflow Recovery Tests

**File: `tests/integration/test_client_workflows.py`**

```python
import pytest
import time


@pytest.mark.integration
def test_workflow_execution(client):
    """Test basic workflow execution via client."""
    result = client.run("order_fulfillment", {"order_id": "order-123"})

    assert result["status"] == "completed"
    assert result["order_id"] == "order-123"
    assert "payment_id" in result
    assert "tracking_number" in result


@pytest.mark.integration
def test_workflow_resumes_after_crash(client, worker_process, platform_stack):
    """Test workflow checkpoint replay after worker crash."""

    # 1. Submit workflow
    run_id = client.submit("long_workflow", {"steps": 5})

    # 2. Wait for partial completion (step 2)
    time.sleep(2)

    # 3. Crash worker
    worker_process.kill()
    worker_process.wait()

    # 4. Start new worker
    import subprocess
    import os
    new_worker = subprocess.Popen(
        ["uv", "run", "python", "app.py"],
        cwd="./tests/integration/blueprints/test-service",
        env={
            **os.environ,
            "AGNT5_COORDINATOR_ENDPOINT": f"{platform_stack['gateway_url']}/coordinator",
        }
    )
    time.sleep(2)

    # 5. Workflow should resume and complete
    result = client.wait_for_result(run_id, timeout=30)
    assert result["status"] == "completed"

    # 6. Verify steps were NOT re-executed (idempotency)
    # TODO: Add platform inspection to verify execution count

    # Cleanup
    new_worker.terminate()
```

### 5. Streaming Tests

**File: `tests/integration/test_client_streaming.py`**

```python
import pytest


@pytest.mark.integration
def test_sse_streaming(client):
    """Test Server-Sent Events streaming via client.stream()."""

    chunks = []
    for chunk in client.stream("generate_text", {"prompt": "Count to 5"}):
        chunks.append(chunk)

    full_text = "".join(chunks)

    # Verify we got multiple chunks (streaming worked)
    assert len(chunks) > 1
    assert len(full_text) > 0


@pytest.mark.integration
def test_llm_token_streaming(client):
    """Test LLM token streaming end-to-end."""

    tokens = []
    for token in client.stream("llm_generate", {
        "prompt": "Write a haiku about coding",
        "model": "gpt-4o-mini"
    }):
        tokens.append(token)
        print(token, end="", flush=True)

    # Should receive multiple tokens
    assert len(tokens) > 5
```

### 6. Test Service Blueprint

**File: `tests/integration/blueprints/test-service/app.py`**

```python
"""Test worker service for integration tests."""

from agnt5 import Worker
from components import *

if __name__ == "__main__":
    worker = Worker(service_name="test-service")
    worker.start()
```

**File: `tests/integration/blueprints/test-service/components.py`**

```python
"""Test components for integration testing."""

import asyncio
from agnt5 import function, Entity, workflow, Context


# ==================== Functions ====================

@function
async def greet(ctx: Context, name: str) -> dict:
    """Simple greeting function."""
    return {"message": f"Hello, {name}!"}


@function
async def long_task(ctx: Context, duration: int) -> dict:
    """Simulates a long-running task."""
    await asyncio.sleep(duration)
    return {"status": "completed", "duration": duration}


@function(retries=5)
async def flaky_function(ctx: Context, fail_count: int) -> dict:
    """Fails `fail_count` times, then succeeds."""
    attempt = ctx.attempt

    if attempt < fail_count:
        raise Exception(f"Simulated failure (attempt {attempt + 1})")

    return {"succeeded": True, "attempts": attempt + 1}


@function
async def failing_function(ctx: Context, error: str) -> dict:
    """Always fails with given error message."""
    raise Exception(error)


# ==================== Entities ====================

class ShoppingCart(Entity):
    """Shopping cart entity for testing state persistence."""

    async def add_item(self, item_id: str, quantity: int, price: float) -> dict:
        """Add item to cart."""
        items = self.state.get("items", {})
        items[item_id] = {"quantity": quantity, "price": price}
        self.state.set("items", items)
        return {"total_items": len(items)}

    async def get_total(self) -> float:
        """Calculate cart total."""
        items = self.state.get("items", {})
        total = sum(
            item["quantity"] * item["price"]
            for item in items.values()
        )
        return total

    async def clear(self) -> dict:
        """Clear cart."""
        self.state.set("items", {})
        return {"status": "cleared"}


class Counter(Entity):
    """Counter entity for testing concurrency."""

    async def increment(self) -> int:
        """Increment counter."""
        count = self.state.get("count", 0)
        # Simulate delay to expose race conditions
        await asyncio.sleep(0.01)
        self.state.set("count", count + 1)
        return count + 1

    async def get_count(self) -> int:
        """Get current count."""
        return self.state.get("count", 0)

    async def reset(self) -> dict:
        """Reset counter to zero."""
        self.state.set("count", 0)
        return {"status": "reset"}


# ==================== Workflows ====================

@workflow
async def order_fulfillment(ctx: Context, order_id: str) -> dict:
    """Multi-step order processing workflow."""

    # Step 1: Validate order
    ctx.state.set("status", "validating")
    await asyncio.sleep(0.1)

    # Step 2: Process payment
    ctx.state.set("status", "processing_payment")
    await asyncio.sleep(0.1)
    payment_id = f"pay_{order_id}"

    # Step 3: Ship order
    ctx.state.set("status", "shipping")
    await asyncio.sleep(0.1)
    tracking_number = f"TRACK_{order_id}"

    # Complete
    ctx.state.set("status", "completed")

    return {
        "status": "completed",
        "order_id": order_id,
        "payment_id": payment_id,
        "tracking_number": tracking_number,
    }


@workflow
async def long_workflow(ctx: Context, steps: int) -> dict:
    """Multi-step workflow for testing crash recovery."""
    ctx.state.set("completed_steps", [])

    for i in range(steps):
        ctx.state.set("current_step", i)
        await asyncio.sleep(1)

        completed = ctx.state.get("completed_steps", [])
        completed.append(i)
        ctx.state.set("completed_steps", completed)

    return {
        "status": "completed",
        "steps_completed": steps
    }


@function
async def generate_text(ctx: Context, prompt: str) -> str:
    """Simulates streaming text generation."""
    # For streaming, this would use async generator
    # For now, return simple response
    return f"Generated response for: {prompt}"
```

### 7. Platform Verification Utilities

**File: `tests/integration/utils.py`**

```python
"""Utilities for verifying platform state."""

import psycopg2
import json


def get_entity_state_from_platform(cockroach_url: str, entity_type: str, key: str) -> dict:
    """Query CockroachDB directly for entity state.

    This verifies state was actually persisted to platform,
    not just stored in-memory in the worker.
    """
    conn = psycopg2.connect(cockroach_url)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT state_json
        FROM entity_state
        WHERE entity_type = %s AND entity_key = %s
        """,
        (entity_type, key)
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        return json.loads(row[0])
    return None


def get_workflow_run_status(cockroach_url: str, run_id: str) -> dict:
    """Get workflow run status from platform database."""
    conn = psycopg2.connect(cockroach_url)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT status, total_steps, completed_steps
        FROM runs
        WHERE id = %s
        """,
        (run_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "status": row[0],
            "total_steps": row[1],
            "completed_steps": row[2],
        }
    return None


def get_step_execution_count(cockroach_url: str, run_id: str, step_name: str) -> int:
    """Count how many times a step was executed.

    Used to verify idempotency - steps should only execute once
    even if workflow is replayed after crash.
    """
    conn = psycopg2.connect(cockroach_url)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM invocations
        WHERE run_id = %s AND step_name = %s
        """,
        (run_id, step_name)
    )

    count = cursor.fetchone()[0]
    conn.close()

    return count
```

## Running the Tests

### Prerequisites

```bash
# Install dependencies
cd sdk/sdk-python
pip install -e ".[test]"

# Install testcontainers
pip install testcontainers

# Ensure Docker is running
docker ps
```

### Run Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_client_entities.py -v

# Run specific test
pytest tests/integration/test_client_entities.py::test_entity_state_persists_across_restart -v

# Run with detailed output
pytest tests/integration/ -v -s
```

### CI/CD Integration

**File: `.github/workflows/integration-tests.yml`**

```yaml
name: Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd sdk/sdk-python
          pip install -e ".[test]"
          pip install testcontainers

      - name: Run integration tests
        run: |
          cd sdk/sdk-python
          pytest tests/integration/ -v --tb=short
```

## Success Metrics

These integration tests provide objective validation of:

### ✅ Functionality
- [ ] Functions execute via client API
- [ ] Entities persist state across restarts
- [ ] Workflows resume after crashes
- [ ] Streaming works end-to-end
- [ ] Errors propagate correctly

### ✅ Reliability
- [ ] No lost writes in concurrent entity updates
- [ ] State isolation between entity keys
- [ ] Idempotent workflow replay (steps not re-executed)
- [ ] Worker crash recovery

### ✅ Platform Integration
- [ ] Client → Gateway communication
- [ ] Gateway → Execution Engine → Worker Coordinator flow
- [ ] Worker registration and invocation
- [ ] State persistence to CockroachDB
- [ ] Event publishing to Redpanda

### ✅ Production Readiness Score

```
Production Readiness = (Passing Tests / Total Tests) × 100%

Current Status: TBD (tests not yet implemented)
Target: 100% before v1.0 release
```

## Next Steps

1. **Phase 1: Foundation** (Week 1)
   - [ ] Set up Testcontainers platform (conftest.py)
   - [ ] Create test service blueprint
   - [ ] Implement basic function invocation tests
   - [ ] Verify end-to-end flow works

2. **Phase 2: Core Tests** (Week 2)
   - [ ] Entity persistence tests
   - [ ] Workflow recovery tests
   - [ ] Concurrency tests
   - [ ] Platform verification utilities

3. **Phase 3: Advanced** (Week 3)
   - [ ] Streaming tests
   - [ ] Error handling tests
   - [ ] Performance benchmarks
   - [ ] CI/CD integration

## Advantages Over Unit Tests

| Aspect | Unit Tests | Integration Tests (This Plan) |
|--------|------------|------------------------------|
| **Platform** | Mocked | Real (Testcontainers) |
| **State Persistence** | In-memory | CockroachDB + Redpanda |
| **Worker** | Mocked | Real Python subprocess |
| **Client API** | Direct function calls | HTTP/gRPC via Client |
| **Confidence** | Tests individual units | Tests complete system |
| **Production Readiness** | Cannot validate | Objectively validates |

## Conclusion

This E2E integration testing strategy using the Client API provides:

1. ✅ **User-centric validation** - Tests what users actually use
2. ✅ **Complete stack coverage** - Client → Gateway → Platform → Worker
3. ✅ **Objective metrics** - Pass/fail based on real behavior
4. ✅ **Production confidence** - If tests pass, platform works
5. ✅ **SDK-agnostic** - Same approach for all language SDKs

**When these tests pass at 100%, the SDK is production-ready.**
