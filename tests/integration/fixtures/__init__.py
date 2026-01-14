"""
Test fixtures for integration tests.

This module provides reusable test backends for functions, entities, workflows,
and agents. These fixtures are separate from pedagogical examples, allowing
tests to cover complex scenarios including error cases.

Usage:
    from tests.integration.fixtures import add_numbers, CounterEntity

    def test_function(client):
        result = client.run(add_numbers, a=5, b=3)
        assert result == 8
"""

# Re-export all fixtures for convenient imports
from .function_fixtures import (
    # Happy path functions
    add_numbers,
    multiply_numbers,
    slow_function,
    large_payload_function,
    concurrent_function,
    unicode_function,
    nested_function_call,
    echo_function,
    process_dict,
    process_list,

    # Error scenario functions
    sometimes_fails,
    always_fails,
    parameter_validator,
    invalid_return_type,
    timeout_function,
    memory_intensive_function,
    retry_eventually_succeeds,
    retry_always_fails,
    wrong_parameter_count_function,
    missing_required_param,
    type_strict_function,
    null_handling_function,
    division_function,
    context_access_function,

    # Edge case functions
    empty_string_function,
    very_long_string_function,
    negative_number_function,
    float_precision_function,
    special_characters_function,
)

from .entity_fixtures import (
    # Happy path entities
    CounterEntity,
    ComplexStateEntity,
    LargeStateEntity,

    # Error scenario entities
    ValidationEntity,
    CorruptibleEntity,
    ConcurrentEntity,
    FailingEntity,

    # Edge case entities
    EmptyStateEntity,
    NegativeNumberEntity,
    UnicodeEntity,
    TimeoutEntity,
)

from .workflow_fixtures import (
    # Happy path workflows
    simple_two_step_workflow,
    three_step_workflow,
    long_running_workflow,
    workflow_with_checkpoints,
    data_processing_workflow,

    # Error scenario workflows
    failing_workflow,
    sometimes_failing_workflow,
    timeout_workflow,
    step_failure_workflow,
    corrupted_checkpoint_workflow,

    # Edge case workflows
    empty_workflow,
    single_step_workflow,
    very_long_workflow,
    conditional_workflow,
    nested_data_workflow,
)

from .agent_fixtures import (
    # Happy path agents/functions
    basic_agent,
    agent_with_tools,
    multi_turn_agent,
    streaming_agent,

    # Tools
    calculator_tool,
    search_tool,
    weather_tool,

    # Error scenario agents
    failing_agent,
    timeout_agent,
    agent_with_broken_tools,
    llm_error_agent,
    sometimes_failing_agent,

    # Edge case agents
    empty_response_agent,
    large_response_agent,
    slow_agent,
    unicode_agent,
    max_iterations_agent,
    context_aware_agent,

    # Mock LLM
    MockLLM,
    create_basic_agent,
)

__all__ = [
    # Functions - Happy Path
    "add_numbers",
    "multiply_numbers",
    "slow_function",
    "large_payload_function",
    "concurrent_function",
    "unicode_function",
    "nested_function_call",
    "echo_function",
    "process_dict",
    "process_list",

    # Functions - Error Scenarios
    "sometimes_fails",
    "always_fails",
    "parameter_validator",
    "invalid_return_type",
    "timeout_function",
    "memory_intensive_function",
    "retry_eventually_succeeds",
    "retry_always_fails",
    "wrong_parameter_count_function",
    "missing_required_param",
    "type_strict_function",
    "null_handling_function",
    "division_function",
    "context_access_function",

    # Functions - Edge Cases
    "empty_string_function",
    "very_long_string_function",
    "negative_number_function",
    "float_precision_function",
    "special_characters_function",

    # Entities - Happy Path
    "CounterEntity",
    "ComplexStateEntity",
    "LargeStateEntity",

    # Entities - Error Scenarios
    "ValidationEntity",
    "CorruptibleEntity",
    "ConcurrentEntity",
    "FailingEntity",

    # Entities - Edge Cases
    "EmptyStateEntity",
    "NegativeNumberEntity",
    "UnicodeEntity",
    "TimeoutEntity",

    # Workflows - Happy Path
    "simple_two_step_workflow",
    "three_step_workflow",
    "long_running_workflow",
    "workflow_with_checkpoints",
    "data_processing_workflow",

    # Workflows - Error Scenarios
    "failing_workflow",
    "sometimes_failing_workflow",
    "timeout_workflow",
    "step_failure_workflow",
    "corrupted_checkpoint_workflow",

    # Workflows - Edge Cases
    "empty_workflow",
    "single_step_workflow",
    "very_long_workflow",
    "conditional_workflow",
    "nested_data_workflow",

    # Agents - Happy Path
    "basic_agent",
    "agent_with_tools",
    "multi_turn_agent",
    "streaming_agent",

    # Agents - Tools
    "calculator_tool",
    "search_tool",
    "weather_tool",

    # Agents - Error Scenarios
    "failing_agent",
    "timeout_agent",
    "agent_with_broken_tools",
    "llm_error_agent",
    "sometimes_failing_agent",

    # Agents - Edge Cases
    "empty_response_agent",
    "large_response_agent",
    "slow_agent",
    "unicode_agent",
    "max_iterations_agent",
    "context_aware_agent",

    # Mock utilities
    "MockLLM",
    "create_basic_agent",
]
