# Integration Test Fixtures

This directory is reserved for test-specific fixtures that should NOT be pedagogical examples.

## Current Status

Integration tests currently use examples from `../../examples/` directory:
- `ex_01_functions.py` - greet, add, process_data, failing_function, etc.
- `ex_02_entities.py` - Counter, ConversationMemory, KeyValueStore
- `ex_03_workflows.py` - data_pipeline, order_fulfillment, etc.
- Agent examples from ex_06_agents.py and ex_07_agents_with_tools.py

## When to Add Fixtures Here

Add fixtures to this directory when you need:
- **Error scenario testing** - Functions that intentionally fail in specific ways
- **Chaos testing** - Components that simulate failures, timeouts, crashes
- **Load testing** - Specialized high-throughput test backends
- **Test-only behavior** - Anything that doesn't make sense as a pedagogical example

## When NOT to Add Fixtures Here

Don't add fixtures here if:
- The component is a good pedagogical example → Add to `examples/` instead
- It demonstrates a real use case → Add to `examples/` instead
- Users would benefit from seeing it → Add to `examples/` instead

## Design Principles

**Keep it minimal:**
- Only add fixtures when examples can't serve the purpose
- Prefer using examples with error injection over creating duplicate fixtures
- Build fixtures incrementally as testing needs arise

**Examples are not tests:**
- Examples should show best practices and happy paths
- Tests should cover edge cases and error scenarios
- Use this directory to bridge the gap when needed

Useful error fixtures include:
- Parameter validation errors
- LLM API failures (rate limits, timeouts, auth errors)
- Worker crash scenarios
- Network partition simulations
- State corruption testing
- Resource exhaustion scenarios
