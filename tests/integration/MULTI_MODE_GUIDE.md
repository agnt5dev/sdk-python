# Multi-Mode Integration Testing Guide

This guide explains how AGNT5 integration tests run across **three different runtime modes** to ensure the SDK works correctly in all deployment scenarios.

---

## Overview

AGNT5 supports three runtime modes for different use cases:

| Mode | Use Case | Journal Backend | State Backend | Containers |
|------|----------|----------------|---------------|------------|
| **Embedded** | Local development | In-memory | SQLite | 1 (dev-server) |
| **Postgres** | Community self-hosted | Embedded | PostgreSQL | 2 (dev-server + PostgreSQL) |
| **Managed** | Production SaaS | Redpanda | CockroachDB | 3 (dev-server + Redpanda + CockroachDB) |

Integration tests run against **all three modes** by default, validating that the SDK works consistently across different backend implementations.

---

## Architecture

### Mode 1: Embedded (Dev Server)

**Purpose**: Local development, fastest tests

**Architecture**:
```
┌─────────────────────────────────────┐
│         Dev Server Container        │
│                                     │
│  ┌───────────┐    ┌──────────────┐ │
│  │  Journal  │    │ Orchestration│ │
│  │ (Embedded)│    │   (SQLite)   │ │
│  │ In-memory │    │ File-based   │ │
│  └───────────┘    └──────────────┘ │
│                                     │
│  Gateway + Engine + Coordinator    │
└─────────────────────────────────────┘
```

**Configuration**:
```python
{
    "journal_backend": "embedded",
    "orchestration_backend": "sqlite",
    "db_url": "/data/orchestration.db"
}
```

**Characteristics**:
- ✅ Fastest startup time (~2s)
- ✅ No external dependencies
- ✅ Perfect for rapid iteration
- ⚠️ Data not persisted between restarts
- ⚠️ Single-node only

---

### Mode 2: Postgres (Community Edition)

**Purpose**: Community self-hosted deployments

**Architecture**:
```
┌──────────────────┐    ┌─────────────────────────────┐
│   PostgreSQL     │◄───│      Dev Server Container   │
│   Container      │    │                             │
│                  │    │  ┌───────────┐  ┌─────────┐│
│  Database:       │    │  │  Journal  │  │  Orch.  ││
│  - orchestration │    │  │(Embedded) │  │(Postgres│
│  - entity_states │    │  │In-memory  │  │Backend) ││
│  - runs          │    │  └───────────┘  └─────────┘│
│  - workflow_steps│    │                             │
└──────────────────┘    │  Gateway + Engine + Coord.  │
                        └─────────────────────────────┘
```

**Configuration**:
```python
{
    "journal_backend": "embedded",
    "orchestration_backend": "postgres",
    "db_url": "postgresql://agnt5:agnt5@localhost:5432/orchestration"
}
```

**Characteristics**:
- ✅ Durable state storage
- ✅ Production-grade database
- ✅ Can scale vertically
- ⚠️ Slower startup (~5s)
- ⚠️ Requires PostgreSQL management

---

### Mode 3: Managed (Production)

**Purpose**: Production managed deployments

**Architecture**:
```
┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────────┐
│   Redpanda       │    │  CockroachDB     │◄───│      Dev Server Container   │
│   Container      │    │  Container       │    │                             │
│                  │    │                  │    │  ┌───────────┐  ┌─────────┐│
│  Event Log:      │    │  Database:       │    │  │  Journal  │  │  Orch.  ││
│  - invocation    │    │  - orchestration │    │  │(Redpanda) │  │(CockDB) ││
│    requests      │    │  - entity_states │    │  │Kafka API  │  │ Backend ││
│  - invocation    │    │  - runs          │    │  └───────────┘  └─────────┘│
│    results       │    │  - workflow_steps│    │                             │
│  - workflow      │    │                  │    │  Gateway + Engine + Coord.  │
│    events        │    │                  │    │                             │
└──────────────────┘    └──────────────────┘    └─────────────────────────────┘
```

**Configuration**:
```python
{
    "journal_backend": "redpanda",
    "orchestration_backend": "cockroach",
    "db_url": "postgresql://root@localhost:26257/defaultdb",
    "redpanda_broker": "localhost:9092"
}
```

**Characteristics**:
- ✅ Full production architecture
- ✅ Distributed event sourcing
- ✅ Horizontal scalability
- ✅ High availability
- ⚠️ Slowest startup (~10s)
- ⚠️ Complex infrastructure

---

## Running Tests

### Run All Modes (Default)

Tests run against all three modes by default:

```bash
pytest tests/integration/ -v

# Output:
test_entity_basic_operation[embedded] PASSED      [  4%]
test_entity_basic_operation[postgres] PASSED      [  8%]
test_entity_basic_operation[managed] PASSED       [ 12%]
...

==================== 75 tests passed (25 tests × 3 modes) ====================
```

### Run Single Mode

Filter by mode using `-k` flag:

```bash
# Embedded mode only (fastest)
pytest tests/integration/ -v -k embedded

# Postgres mode only
pytest tests/integration/ -v -k postgres

# Managed mode only (production-like)
pytest tests/integration/ -v -k managed
```

### Run Specific Test Across Modes

```bash
# Run entity persistence test in all modes
pytest tests/integration/test_client_entities.py::test_entity_state_persists_across_restart -v

# Output shows 3 results (one per mode):
test_entity_state_persists_across_restart[embedded] FAILED   ❌
test_entity_state_persists_across_restart[postgres] FAILED   ❌
test_entity_state_persists_across_restart[managed] FAILED    ❌
```

### Run Multiple Modes

```bash
# Exclude embedded (run postgres + managed only)
pytest tests/integration/ -v -k "not embedded"

# Run only embedded + postgres (exclude managed)
pytest tests/integration/ -v -k "embedded or postgres"
```

---

## Understanding Test Results

### Expected Baseline (Week 1 - RED Phase)

```bash
$ pytest tests/integration/ -v

collected 75 items (25 tests × 3 modes)

test_client_entities.py::test_entity_basic_operation[embedded] PASSED         [  1%]
test_client_entities.py::test_entity_basic_operation[postgres] PASSED         [  3%]
test_client_entities.py::test_entity_basic_operation[managed] PASSED          [  4%]

test_client_entities.py::test_entity_state_persists[embedded] FAILED          [  5%] ❌
test_client_entities.py::test_entity_state_persists[postgres] FAILED          [  7%] ❌
test_client_entities.py::test_entity_state_persists[managed] FAILED           [  9%] ❌

test_client_entities.py::test_concurrent_updates[embedded] FAILED             [ 11%] ❌
test_client_entities.py::test_concurrent_updates[postgres] FAILED             [ 12%] ❌
test_client_entities.py::test_concurrent_updates[managed] FAILED              [ 13%] ❌

...

==================== 45 passed, 30 failed in 120s ====================

Production Readiness by Mode:
- Embedded: 60% (15/25 tests)
- Postgres: 60% (15/25 tests)
- Managed: 60% (15/25 tests)

Overall: 60% ready across all deployment scenarios ✅
```

### Interpreting Failures

**Same failure in all modes** = SDK issue (e.g., entity persistence not implemented)
```
test_entity_state_persists[embedded] FAILED  ❌
test_entity_state_persists[postgres] FAILED  ❌
test_entity_state_persists[managed] FAILED   ❌
```
→ Implement platform persistence in SDK

**Failure in specific mode** = Mode-specific issue
```
test_entity_state_persists[embedded] PASSED  ✅
test_entity_state_persists[postgres] FAILED  ❌
test_entity_state_persists[managed] PASSED   ✅
```
→ Debug PostgreSQL-specific configuration

---

## Test Fixture Architecture

### How Parametrization Works

```python
# conftest.py

@pytest.fixture(scope="session", params=["embedded", "postgres", "managed"])
def runtime_mode(request):
    """Parametrize tests across all three modes."""
    return request.param

@pytest.fixture(scope="function")
def platform(runtime_mode):
    """Start platform in specified mode."""
    if runtime_mode == "embedded":
        return setup_embedded_mode()
    elif runtime_mode == "postgres":
        return setup_postgres_mode()
    elif runtime_mode == "managed":
        return setup_managed_mode()
```

### Platform Fixture Output

Each mode provides consistent interface:

```python
platform = {
    "mode": "embedded" | "postgres" | "managed",
    "gateway_url": "http://localhost:34181",
    "coordinator_url": "http://localhost:34186",
    "db_url": "/data/orchestration.db" | "postgresql://...",
    "db_type": "sqlite" | "postgres" | "cockroach",
    "journal_backend": "embedded" | "redpanda",
    "orchestration_backend": "sqlite" | "postgres" | "cockroach",
}
```

### Backend-Agnostic Utilities

Utils automatically detect backend from `db_url`:

```python
from tests.integration.utils import get_entity_state_from_platform

# Works with SQLite (embedded)
state = get_entity_state_from_platform(
    db_url="/data/orchestration.db",  # SQLite
    entity_type="ShoppingCart",
    key="user-123"
)

# Works with PostgreSQL (postgres)
state = get_entity_state_from_platform(
    db_url="postgresql://agnt5:agnt5@localhost:5432/orchestration",
    entity_type="ShoppingCart",
    key="user-123"
)

# Works with CockroachDB (managed)
state = get_entity_state_from_platform(
    db_url="postgresql://root@localhost:26257/defaultdb",
    entity_type="ShoppingCart",
    key="user-123"
)
```

---

## Performance Characteristics

### Test Execution Times

| Mode | Setup Time | Per-Test Time | Total (25 tests) |
|------|-----------|---------------|------------------|
| **Embedded** | ~2s | ~0.5s | ~15s |
| **Postgres** | ~5s | ~0.8s | ~25s |
| **Managed** | ~10s | ~1.0s | ~35s |
| **All Modes** | ~17s | ~2.3s | ~75s |

### Recommendations

**Development workflow**:
```bash
# Quick iteration (embedded only)
pytest tests/integration/ -v -k embedded

# Before PR (all modes)
pytest tests/integration/ -v
```

**CI/CD**:
```bash
# Run all modes in parallel
pytest tests/integration/ -v -n auto --dist loadgroup
```

---

## Troubleshooting

### Common Issues

#### 1. Platform Not Healthy

**Symptom**:
```
Exception: Platform failed to become healthy after 30s
```

**Solution**:
- Check dev-server is running: `docker ps | grep dev-server`
- Check logs: `docker logs agnt5-dev-server`
- Restart: `just dev-server-restart`

#### 2. Database Connection Failed

**Symptom** (Postgres mode):
```
psycopg2.OperationalError: could not connect to server
```

**Solution**:
- PostgreSQL container may not be ready
- Increase wait timeout in `setup_postgres_mode()`
- Check container: `docker ps | grep postgres`

#### 3. Container Cleanup Errors

**Symptom**:
```
Failed to stop postgres: ...
```

**Solution**:
- Containers may be orphaned from previous run
- Clean up manually: `docker rm -f $(docker ps -aq)`
- Restart Docker if persists

---

## Best Practices

### 1. Write Mode-Agnostic Tests

✅ **Good** - Uses platform fixture:
```python
def test_entity_persistence(client, worker_process, platform):
    client.entity("Cart", "user").add_item(...)

    # Restart worker
    restart_worker(worker_process, platform)

    # Verify using platform["db_url"]
    state = get_entity_state_from_platform(
        db_url=platform["db_url"],  # Works with all backends
        entity_type="Cart",
        key="user"
    )
```

❌ **Bad** - Hardcodes backend:
```python
def test_entity_persistence(client, worker_process):
    client.entity("Cart", "user").add_item(...)

    # Only works with PostgreSQL!
    conn = psycopg2.connect("postgresql://...")
```

### 2. Use Appropriate Markers

```python
@pytest.mark.integration  # Runs in all modes
def test_basic_operation(client):
    ...

@pytest.mark.embedded  # Only runs in embedded mode
def test_sqlite_specific_feature(client, platform):
    assert platform["db_type"] == "sqlite"
    ...
```

### 3. Clean Up Resources

```python
@pytest.fixture
def my_fixture(platform):
    # Setup
    resource = create_resource()

    yield resource

    # Cleanup - important for mode-specific resources
    resource.cleanup()
```

---

## Production Readiness Validation

### When All Tests Pass

```bash
$ pytest tests/integration/ -v

==================== 75 passed in 75s ====================

Production Readiness:
- Embedded: 100% ✅
- Postgres: 100% ✅
- Managed: 100% ✅

SDK is production-ready across ALL deployment modes! 🎉
```

**This means**:
- ✅ SDK works on developer laptops (embedded)
- ✅ SDK works in community self-hosted (postgres)
- ✅ SDK works in production managed (redpanda + cockroachdb)
- ✅ Backend portability validated
- ✅ No mode-specific bugs
- ✅ Safe to release v1.0

---

## Next Steps

1. **Week 1 (RED)**: Run tests, watch them fail across all modes
2. **Week 2-4 (GREEN)**: Implement features, watch pass rate climb
3. **Week 5+ (REFACTOR)**: Optimize while keeping 100% pass rate

**Key Insight**: Multi-mode testing provides objective, comprehensive validation that the SDK works correctly in all real-world deployment scenarios.
