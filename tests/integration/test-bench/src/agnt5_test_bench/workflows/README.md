# Test Bench Workflows

Comprehensive test workflows for integration testing of the AGNT5 Python SDK.

## Overview

This directory contains 35 workflows (wf_01 through wf_35) organized into 7 categories, testing various SDK features including functions, workflows, agents, tools, error handling, and retry logic.

## Workflow Categories

### WF_01-05: Simple Workflows (wf_01_simple_workflows.py)
Basic workflow execution patterns testing fundamental workflow features:
- `wf_01_basic_execution` - Basic workflow with no dependencies
- `wf_02_state_management` - State get/set/delete operations
- `wf_03_function_invocation` - Invoke functions using ctx.task()
- `wf_04_agent_integration` - Execute agent in workflow context
- `wf_05_multi_step_workflow` - Multi-step workflow combining state, functions, and logic

### WF_06-10: Function Workflows (wf_02_function_workflows.py)
Function invocation and composition patterns:
- `wf_06_different_param_counts` - Functions with 0-3 parameters
- `wf_07_sequential_function_calls` - Sequential calls with data dependencies
- `wf_08_function_result_chaining` - Chain function results directly
- `wf_09_function_with_state` - Functions combined with state management
- `wf_10_conditional_function_calls` - Conditional function execution

### WF_11-15: Error Workflows (wf_03_error_workflows.py)
Error handling and recovery scenarios:
- `wf_11_function_error_propagation` - Errors bubble up to workflow
- `wf_12_error_handling_with_try_catch` - Catch and handle errors
- `wf_13_error_recovery_continue` - Continue execution after handling errors
- `wf_14_mixed_error_types` - Multiple error types in one workflow
- `wf_15_partial_success_with_errors` - Some operations succeed, some fail

### WF_16-20: Retry Workflows (wf_04_retry_workflows.py)
Retry behavior and exhaustion testing:
- `wf_16_retry_eventual_success` - Function succeeds after retries
- `wf_17_retry_exhaustion` - All retries exhausted (should fail)
- `wf_18_retry_with_error_handling` - Handle retry exhaustion gracefully
- `wf_19_multiple_retry_operations` - Multiple operations with different retry configs
- `wf_20_retry_state_persistence` - State persists across retry attempts

### WF_21-25: Agent Workflows (wf_05_agent_workflows.py)
Basic agent execution patterns:
- `wf_21_basic_agent_execution` - Agent without tools (pure reasoning)
- `wf_22_agent_conversation` - Multi-turn conversation with memory
- `wf_23_agent_with_workflow_state` - Agent + workflow state interaction
- `wf_24_multiple_agents_in_workflow` - Multiple independent agents
- `wf_25_agent_with_structured_output` - Structured agent responses

### WF_26-30: Agent Tool Workflows (wf_06_agent_tool_workflows.py)
Agents using tools for enhanced capabilities:
- `wf_26_agent_with_single_tool` - Single tool usage
- `wf_27_agent_with_multiple_tools` - Multiple tool options, agent selects correct one
- `wf_28_agent_multi_tool_coordination` - Chain multiple tools in sequence
- `wf_29_agent_tool_with_parameters` - Various parameter types (lists, dicts, primitives)
- `wf_30_agent_tool_iteration` - Multiple tool iterations

### WF_31-35: Agent Handoff Workflows (wf_07_agent_handoff_workflows.py)
Agent handoff patterns (control transfer):
- `wf_31_simple_triage_handoff` - Triage agent routes to specialist
- `wf_32_handoff_with_context` - Context state sharing across handoff
- `wf_33_handoff_with_tools` - Specialist uses tools after handoff
- `wf_34_multiple_handoff_options` - Multiple specialists available for routing
- `wf_35_handoff_with_state` - Workflow state persistence through handoff

## Usage

### Importing Workflows

```python
from agnt5_test_bench.workflows import wf_01_basic_execution, wf_31_simple_triage_handoff

# Run via client
result = await client.workflow("wf_01_basic_execution").run()
```

### Running the Test Bench

```bash
# Start dev server with test-bench
just platform start-dev-server python

# Restart test-bench after code changes
pm2 restart agnt5-python-test-bench
```

## Design Principles

### Sequential Numbering
All workflows are numbered sequentially (wf_01 through wf_35) making it easy to:
- Reference workflows in tests and documentation
- Track which scenarios are covered
- Add new workflows incrementally

### Categorization
Workflows are organized into logical categories (simple, functions, errors, retry, agents, tools, handoffs) for:
- Easy navigation
- Clear test coverage mapping
- Organized test execution

### Comprehensive Documentation
Each workflow includes:
- **Test Scenario**: What the workflow tests
- **Validates**: Specific behaviors being verified
- **Args/Returns**: Clear parameter and return documentation
- **MCP Verification Points**: What to check in logs, traces, events, and entity state

### Reusable Components
Workflows import from shared modules:
- **Functions** (`../functions/`): fn_01 through fn_11
- **Tools** (`../tools/`): calculate_total, search_database, etc.
- **Agents** (`../agents/`): research_specialist, researcher_agent, analyst_agent

### MCP Verification Points
Each workflow documents what to look for in:
- **Logs**: Key operations and results
- **Traces**: Span structure and timing
- **Events**: Lifecycle events
- **Entity**: State persistence and consistency

## Testing Strategy

### Unit Testing
Individual workflows can be invoked directly to test specific features:

```python
# Test basic workflow execution
await client.workflow("wf_01_basic_execution").run()

# Test error handling
await client.workflow("wf_12_error_handling_with_try_catch").run()

# Test agent with tools
await client.workflow("wf_26_agent_with_single_tool").run()
```

### Integration Testing
Workflows designed to work together for comprehensive testing:

1. Run simple workflows first to verify basic functionality
2. Progress to function and error workflows for SDK features
3. Test retry and agent workflows for advanced features
4. Validate handoff patterns for multi-agent scenarios

### Debugging
Each workflow includes detailed logging for debugging:
- Input parameters logged at start
- Key operations logged during execution
- Results logged at completion
- Error messages with context

### MCP Inspection
Use MCP tools to inspect workflow execution:

```bash
# Query workflow runs
sqlite3 /path/to/orchestration.db "SELECT * FROM runs WHERE run_type = 'workflow';"

# Check traces
sqlite3 /path/to/observability.db "SELECT * FROM spans WHERE span_name LIKE 'wf_%';"

# View entity state
sqlite3 /path/to/orchestration.db "SELECT * FROM entities WHERE entity_key LIKE 'workflow:%';"
```

## Adding New Workflows

To add new workflows:

1. **Choose a category** or create a new file following the naming pattern
2. **Number sequentially** starting from wf_36
3. **Follow the template**:
   - Module docstring with test coverage
   - Clear workflow documentation
   - MCP verification points
   - Comprehensive logging
4. **Import reusable components** from shared modules
5. **Update `__init__.py`** to export the new workflows
6. **Test imports** to verify everything works

### Template

```python
@workflow
async def wf_XX_descriptive_name(ctx: WorkflowContext, param: Type) -> dict:
    """
    Test Scenario: Clear description of what this tests.

    Validates:
    - Specific behavior 1
    - Specific behavior 2
    - Specific behavior 3

    Args:
        param: Parameter description

    Returns:
        Result description

    MCP Verification Points:
        - Logs: What to check in logs
        - Traces: Expected span structure
        - Events: Lifecycle events to verify
        - Entity: State consistency checks
    """
    ctx.logger.info("=== WF_XX: Descriptive Name ===")
    ctx.logger.info(f"Param: {param}")

    # Implementation
    result = {"status": "success"}

    ctx.logger.info(f"Result: {result}")
    return result
```

## Related Documentation

- **Functions**: `../functions/README.md` (if exists)
- **Tools**: `../tools.py` - Reusable tool definitions
- **Agents**: `../agents.py` - Canonical agent definitions
- **Test Bench**: `../app.py` - Main test-bench application

## Notes

- Workflows use relative imports (`from ..functions import ...`)
- Module-level agents are exported for reuse in other workflows
- All workflows are async and use `WorkflowContext`
- Error workflows (wf_11, wf_17) are expected to fail - they test error propagation
