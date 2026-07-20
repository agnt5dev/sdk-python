# Integration Tests

This directory contains end-to-end integration tests for the AGNT5 Python SDK.

## Overview

These tests use the **Client API** to validate the complete stack:
- Client → Gateway → Execution Engine → Worker Coordinator → Worker
- Real platform services running in Testcontainers
- Real database persistence
- Real event sourcing

**Multi-Mode Testing**: Tests run across **three runtime modes** to ensure the SDK works in all deployment scenarios:

1. **Embedded** - SQLite + in-memory journal (dev-server)
2. **Postgres** - PostgreSQL backend (community edition)
3. **Managed** - Redpanda + CockroachDB (production)

See `MULTI_MODE_GUIDE.md` for complete multi-mode documentation.

## Architecture

See `/sdk/sdk-python/e2e-integration-python.md` for complete E2E testing documentation.

## Quick Start

```bash
# 1. Install integration test dependencies
cd sdk/sdk-python
uv sync --group integration

# 2. Start dev-server
cd [private-monorepo]
just dev-server

# 3. Run integration tests
cd sdk/sdk-python
pytest tests/integration/ -v
```

## Running Tests

### All Modes (Default)

Run tests against all three runtime modes:

```bash
pytest tests/integration/ -v

# Output shows tests × 3 modes:
test_entity_basic_operation[embedded] PASSED      [  4%]
test_entity_basic_operation[postgres] PASSED      [  8%]
test_entity_basic_operation[managed] PASSED       [ 12%]
...

==================== 75 tests passed (25 tests × 3 modes) ====================
```

### Single Mode

Filter by specific runtime mode:

```bash
# Embedded only (fastest, local dev)
pytest tests/integration/ -v -k embedded

# Postgres only (community edition)
pytest tests/integration/ -v -k postgres

# Managed only (production architecture)
pytest tests/integration/ -v -k managed
```

### Specific Test File

```bash
# Run entity tests in all modes
pytest tests/integration/test_client_entities.py -v

# Run entity tests in embedded mode only
pytest tests/integration/test_client_entities.py -v -k embedded
```

### Specific Test

```bash
# Run one test across all modes
pytest tests/integration/test_client_entities.py::test_entity_state_persists_across_restart -v

# Output:
test_entity_state_persists_across_restart[embedded] FAILED  ❌
test_entity_state_persists_across_restart[postgres] FAILED  ❌
test_entity_state_persists_across_restart[managed] FAILED   ❌
```

### With Detailed Output

```bash
# Show platform setup logs
pytest tests/integration/ -v -s

# Show only failures
pytest tests/integration/ -v --tb=short
```

## Test Structure

- `conftest.py` - Multi-mode platform fixtures
- `test_functions.py` - Function invocation tests
- `test_batch_eval.py` - Batch evaluation tests
- `test_worker_app.py` - Worker application tests
- `test_helpers.py` - Integration helper tests
- `utils.py` - Backend-agnostic platform verification utilities

## Understanding Test Failures

### Same Failure Across All Modes

```
test_entity_state_persists[embedded] FAILED  ❌
test_entity_state_persists[postgres] FAILED  ❌
test_entity_state_persists[managed] FAILED   ❌
```

→ **SDK issue** - Feature not implemented (e.g., platform persistence)

### Mode-Specific Failure

```
test_entity_state_persists[embedded] PASSED  ✅
test_entity_state_persists[postgres] FAILED  ❌
test_entity_state_persists[managed] PASSED   ✅
```

→ **Backend-specific issue** - Debug PostgreSQL configuration

## Performance

Test execution times vary by mode:

| Mode | Setup | Per-Test | Total (25 tests) |
|------|-------|----------|------------------|
| **Embedded** | ~2s | ~0.5s | ~15s |
| **Postgres** | ~5s | ~0.8s | ~25s |
| **Managed** | ~10s | ~1.0s | ~35s |
| **All Modes** | ~17s | ~2.3s | ~75s |

**Development workflow**:
```bash
# Quick iteration (embedded only)
pytest tests/integration/ -v -k embedded  # ~15s

# Before PR (all modes)
pytest tests/integration/ -v  # ~75s
```

## Documentation

- **`README.md`** (this file) - Quick start and usage
- **`JOURNAL_EVENTS_EXAMPLES.md`** - Journal event examples
- **`README_TEST_HELPERS.md`** - Test helper reference

## Benefits of Multi-Mode Testing

1. ✅ **Comprehensive Coverage** - Validates all deployment scenarios
2. ✅ **Backend Portability** - Ensures no mode-specific bugs
3. ✅ **Production Confidence** - Production mode explicitly tested
4. ✅ **Flexibility** - Fast tests (embedded) or realistic tests (managed)
5. ✅ **Objective Metrics** - Pass rate = production readiness

When all tests pass at 100%, the SDK is **objectively production-ready** across all deployment modes!
