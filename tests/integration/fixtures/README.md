# Integration Test Fixtures

This directory contains reusable test fixtures for integration testing. These fixtures are separate from pedagogical examples, allowing tests to cover comprehensive scenarios including error cases, edge cases, and failure modes.

## Purpose

- **Separation of Concerns**: Examples stay clean and pedagogical, while fixtures can be complex and error-focused
- **Comprehensive Coverage**: Fixtures cover happy paths, error scenarios, edge cases, and failure modes
- **Test-Driven Development**: Fixtures support TDD by providing deterministic test backends
- **Error Testing**: Systematic testing of failure modes (wrong parameters, LLM failures, worker crashes, timeouts)

## Structure

```
fixtures/
├── __init__.py                  # Exports all fixtures
├── function_fixtures.py         # 40+ function fixtures
├── entity_fixtures.py           # 13 entity fixtures
├── workflow_fixtures.py         # 17 workflow fixtures
├── agent_fixtures.py            # 19+ agent fixtures + tools
└── README.md                    # This file
```

## Fixture Categories

### Function Fixtures (`function_fixtures.py`)

**Happy Path (10 functions):**
- `add_numbers`, `multiply_numbers` - Basic arithmetic
- `slow_function` - Timeout testing
- `large_payload_function` - Size limit testing
- `concurrent_function` - Concurrency testing
- `unicode_function` - Internationalization
- `echo_function`, `process_dict`, `process_list` - Type testing

**Error Scenarios (14 functions):**
- `sometimes_fails`, `always_fails` - Random and deterministic failures
- `parameter_validator` - Strict parameter validation
- `timeout_function` - Timeout testing
- `retry_eventually_succeeds`, `retry_always_fails` - Retry logic
- `wrong_parameter_count_function` - Parameter validation
- `missing_required_param` - Required parameter testing
- `type_strict_function` - Type validation
- `division_function` - Division by zero and edge cases

**Edge Cases (5 functions):**
- `empty_string_function` - Empty string handling
- `very_long_string_function` - Large string testing
- `negative_number_function` - Negative number handling
- `float_precision_function` - Floating point precision
- `special_characters_function` - Unicode and special characters

### Entity Fixtures (`entity_fixtures.py`)

**Happy Path (3 entities):**
- `CounterEntity` - Simple counter
- `ComplexStateEntity` - Nested state structures
- `LargeStateEntity` - Large state (100KB+)

**Error Scenarios (4 entities):**
- `ValidationEntity` - Strict validation rules
- `CorruptibleEntity` - State corruption testing
- `ConcurrentEntity` - Race condition testing
- `FailingEntity` - Deterministic and random failures

**Edge Cases (4 entities):**
- `EmptyStateEntity` - Default state handling
- `NegativeNumberEntity` - Negative value testing
- `UnicodeEntity` - Unicode/internationalization
- `TimeoutEntity` - Slow operations

### Workflow Fixtures (`workflow_fixtures.py`)

**Happy Path (5 workflows):**
- `simple_two_step_workflow` - Basic 2-step workflow
- `three_step_workflow` - 3-step sequencing
- `long_running_workflow` - Many steps (configurable)
- `workflow_with_checkpoints` - Explicit checkpoints
- `data_processing_workflow` - List processing

**Error Scenarios (5 workflows):**
- `failing_workflow` - Fails at specific step
- `sometimes_failing_workflow` - Random failures
- `timeout_workflow` - Slow steps
- `step_failure_workflow` - Named step failures
- `corrupted_checkpoint_workflow` - Checkpoint corruption

**Edge Cases (5 workflows):**
- `empty_workflow` - No steps
- `single_step_workflow` - One step only
- `very_long_workflow` - 100+ steps
- `conditional_workflow` - Branching logic
- `nested_data_workflow` - Complex data structures

### Agent Fixtures (`agent_fixtures.py`)

**Happy Path (4 agent functions):**
- `basic_agent` - Simple agent
- `agent_with_tools` - Agent with tool calling
- `multi_turn_agent` - Conversation history
- `streaming_agent` - Streaming responses

**Tools (3 tools):**
- `calculator_tool` - Math operations
- `search_tool` - Mock search
- `weather_tool` - Mock weather data

**Error Scenarios (5 agent functions):**
- `failing_agent` - Deterministic failures
- `timeout_agent` - Timeout scenarios
- `agent_with_broken_tools` - Tool failure testing
- `llm_error_agent` - LLM API error simulation
- `sometimes_failing_agent` - Random failures

**Edge Cases (6 agent functions):**
- `empty_response_agent` - Empty responses
- `large_response_agent` - Large payload responses
- `slow_agent` - Artificial delay
- `unicode_agent` - Unicode handling
- `max_iterations_agent` - Iteration limits
- `context_aware_agent` - Context usage

**Utilities:**
- `MockLLM` - Mock language model (no API key required)
- `create_basic_agent` - Agent factory function

## Usage

### In Test Worker App

The `test_worker_app.py` imports all fixtures to register them with the platform:

```python
from fixtures import function_fixtures
from fixtures import entity_fixtures
from fixtures import workflow_fixtures
from fixtures import agent_fixtures
```

### In Integration Tests

Tests call fixtures by name via the client:

```python
@pytest.mark.integration
def test_addition(client, worker_process):
    result = client.run("add_numbers", {"a": 5, "b": 3})
    assert result == 8

@pytest.mark.integration
def test_parameter_validation_error(client, worker_process):
    with pytest.raises(RunError):
        client.run("parameter_validator", {"value": -1})  # Invalid: negative
```

### Importing Specific Fixtures

```python
from tests.integration.fixtures import (
    add_numbers,
    CounterEntity,
    simple_two_step_workflow,
    basic_agent,
)
```

## Design Principles

### 1. Deterministic

All fixtures produce deterministic results (except those explicitly testing randomness like `sometimes_fails`). This ensures reliable tests.

### 2. Comprehensive

Fixtures cover:
- **Happy paths** - Basic functionality works
- **Error scenarios** - Failures handled gracefully
- **Edge cases** - Boundary conditions
- **Failure modes** - Real-world issues (timeouts, crashes, corruption)

### 3. Self-Contained

Each fixture is independent and doesn't depend on other fixtures. Tests can use any combination of fixtures.

### 4. Well-Documented

Every fixture has:
- Clear docstring explaining purpose
- Parameter documentation
- Return value documentation
- Example usage

### 5. Test-Focused

Fixtures are designed for testing, not pedagogy. They:
- Use simple, predictable logic
- Avoid unnecessary complexity
- Focus on testable behavior
- Include error injection points

## Fixture Naming Conventions

- **Happy path**: Descriptive names (`add_numbers`, `simple_two_step_workflow`)
- **Error scenarios**: Include "fail" or error type (`always_fails`, `timeout_function`)
- **Edge cases**: Descriptive of edge condition (`empty_string_function`, `unicode_agent`)
- **Utilities**: Clear purpose (`MockLLM`, `create_basic_agent`)

## Adding New Fixtures

When adding new fixtures:

1. **Choose the right file**: Function, entity, workflow, or agent
2. **Follow naming conventions**: Descriptive, purpose-clear names
3. **Add documentation**: Complete docstring with examples
4. **Export in `__init__.py`**: Add to imports and `__all__`
5. **Test the fixture**: Verify it works before using in tests

Example:

```python
# In function_fixtures.py

@function
async def my_new_fixture(ctx: FunctionContext, value: int) -> dict:
    """
    Description of what this fixture does.

    Args:
        ctx: Function context
        value: Description of parameter

    Returns:
        Dict with result

    Example:
        result = client.run("my_new_fixture", {"value": 42})
        assert result["value"] == 42
    """
    return {"value": value, "processed": True}
```

```python
# In __init__.py

from .function_fixtures import (
    ...
    my_new_fixture,  # Add here
)

__all__ = [
    ...
    "my_new_fixture",  # And here
]
```

## Testing Error Scenarios

The fixtures support systematic error scenario testing:

### Parameter Errors
```python
# Wrong type
client.run("type_strict_function", {"count": "not_an_int", "name": "test", "active": True})

# Missing required
client.run("missing_required_param", {"optional": "value"})  # Missing 'required'

# Out of range
client.run("parameter_validator", {"value": 9999})  # Exceeds max (1000)
```

### LLM Failures
```python
# Rate limit
client.run("llm_error_agent", {"message": "test", "error_type": "rate_limit"})

# Timeout
client.run("llm_error_agent", {"message": "test", "error_type": "timeout"})

# Auth error
client.run("llm_error_agent", {"message": "test", "error_type": "auth"})
```

### Worker Failures
```python
# Worker crashes during execution (requires chaos fixtures - see Task 1.3)
# Timeout scenarios
client.run("slow_function", {"duration": 100.0})  # Exceeds timeout
```

### State Corruption
```python
# Entity corruption
entity = client.entity("CorruptibleEntity", "test-1")
entity.corrupt_state()
# Test recovery...
```

## Backward Compatibility

The `test_worker_app.py` also imports examples for backward compatibility:

```python
import ex_01_functions  # Provides: greet, add, process_data
import ex_02_entities   # Provides: Counter, ConversationMemory, etc.
import ex_03_workflows  # Provides: data_pipeline, order_fulfillment
```

This ensures existing tests that reference example function names (like "greet", "add") continue to work while we gradually migrate to fixture-based tests.

## Next Steps

1. **Migrate existing tests** to use fixtures (gradually)
2. **Add error scenario tests** using these fixtures (Task 1.3)
3. **Add load tests** using fixture-based workers (Task 2.1)
4. **Add chaos tests** using failure-injection fixtures (Task 2.2)

## Related Documentation

- Integration Test README: `tests/integration/README.md`
- Examples README: `examples/README.md`
- Testing Plan: `/dev/active/sdk-python-refactor-tests.md`
