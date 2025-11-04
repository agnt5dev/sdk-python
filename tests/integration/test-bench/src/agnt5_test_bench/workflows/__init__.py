"""
Test Workflows

Comprehensive test workflows for testing SDK functionality.
Workflows are organized by category and numbered sequentially.
"""

# WF_01: Simple Workflows (wf_01 - wf_05)
from .wf_01_simple_workflows import (
    wf_01_basic_execution,
    wf_02_state_management,
    wf_03_function_invocation,
    wf_04_agent_integration,
    wf_05_multi_step_workflow,
)

# WF_02: Function Invocation Workflows (wf_06 - wf_10)
from .wf_02_function_workflows import (
    wf_06_different_param_counts,
    wf_07_sequential_function_calls,
    wf_08_function_result_chaining,
    wf_09_function_with_state,
    wf_10_conditional_function_calls,
)

# WF_03: Error Handling Workflows (wf_11 - wf_15)
from .wf_03_error_workflows import (
    wf_11_function_error_propagation,
    wf_12_error_handling_with_try_catch,
    wf_13_error_recovery_continue,
    wf_14_mixed_error_types,
    wf_15_partial_success_with_errors,
)

# WF_04: Retry Workflows (wf_16 - wf_20)
from .wf_04_retry_workflows import (
    wf_16_retry_eventual_success,
    wf_17_retry_exhaustion,
    wf_18_retry_with_error_handling,
    wf_19_multiple_retry_operations,
    wf_20_retry_state_persistence,
)

# WF_05: Agent Workflows (wf_21 - wf_25)
from .wf_05_agent_workflows import (
    test_agent,  # Module-level agent
    wf_21_basic_agent_execution,
    wf_22_agent_conversation,
    wf_23_agent_with_workflow_state,
    wf_24_multiple_agents_in_workflow,
    wf_25_agent_with_structured_output,
)

# WF_06: Agent Tool Workflows (wf_26 - wf_30)
from .wf_06_agent_tool_workflows import (
    calculator_agent,  # Module-level agent
    wf_26_agent_with_single_tool,
    wf_27_agent_with_multiple_tools,
    wf_28_agent_multi_tool_coordination,
    wf_29_agent_tool_with_parameters,
    wf_30_agent_tool_iteration,
)

# WF_07: Agent Handoff Workflows (wf_31 - wf_35)
from .wf_07_agent_handoff_workflows import (
    save_preference,  # Shared tool
    technical_specialist,  # Module-level agents
    business_specialist,
    research_specialist,
    wf_31_simple_triage_handoff,
    wf_32_handoff_with_context,
    wf_33_handoff_with_tools,
    wf_34_multiple_handoff_options,
    wf_35_handoff_with_state,
)

__all__ = [
    # WF_01: Simple workflows
    "wf_01_basic_execution",
    "wf_02_state_management",
    "wf_03_function_invocation",
    "wf_04_agent_integration",
    "wf_05_multi_step_workflow",
    # WF_02: Function invocation
    "wf_06_different_param_counts",
    "wf_07_sequential_function_calls",
    "wf_08_function_result_chaining",
    "wf_09_function_with_state",
    "wf_10_conditional_function_calls",
    # WF_03: Error handling
    "wf_11_function_error_propagation",
    "wf_12_error_handling_with_try_catch",
    "wf_13_error_recovery_continue",
    "wf_14_mixed_error_types",
    "wf_15_partial_success_with_errors",
    # WF_04: Retry workflows
    "wf_16_retry_eventual_success",
    "wf_17_retry_exhaustion",
    "wf_18_retry_with_error_handling",
    "wf_19_multiple_retry_operations",
    "wf_20_retry_state_persistence",
    # WF_05: Agent workflows
    "test_agent",
    "wf_21_basic_agent_execution",
    "wf_22_agent_conversation",
    "wf_23_agent_with_workflow_state",
    "wf_24_multiple_agents_in_workflow",
    "wf_25_agent_with_structured_output",
    # WF_06: Agent tool workflows
    "calculator_agent",
    "wf_26_agent_with_single_tool",
    "wf_27_agent_with_multiple_tools",
    "wf_28_agent_multi_tool_coordination",
    "wf_29_agent_tool_with_parameters",
    "wf_30_agent_tool_iteration",
    # WF_07: Agent handoff workflows
    "save_preference",
    "technical_specialist",
    "business_specialist",
    "research_specialist",
    "wf_31_simple_triage_handoff",
    "wf_32_handoff_with_context",
    "wf_33_handoff_with_tools",
    "wf_34_multiple_handoff_options",
    "wf_35_handoff_with_state",
]
