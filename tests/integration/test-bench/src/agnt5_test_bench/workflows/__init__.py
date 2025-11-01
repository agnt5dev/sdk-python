"""
Test Bench Workflows

This package contains workflow definitions for testing AGNT5 functionality.

Workflow Categories:
- simple_workflows.py: Basic workflows for setup verification
- replay_test_workflows.py: Workflows for testing Phase 1 replay functionality
- memory_test_workflows.py: Workflows for testing entity memory architecture

Agent Test Workflows (Organized by Feature):
- agent_basic_workflows.py: Simple agents without tools
- agent_tools_workflows.py: Agents with tools (single and multiple)
- agent_collaboration_workflows.py: Agents using other agents as tools
- agent_handoff_workflows.py: Explicit handoff patterns between agents
- agent_hitl_workflows.py: Human-in-the-loop interactions
- comprehensive_integration_test.py: Ultimate test combining all features
"""

# Import workflows from simple_workflows
from .simple_workflows import (
    test_workflow,
    test_workflow_with_state,
)

# Import replay test workflows
from .replay_test_workflows import (
    # Primary crash/recovery test
    workflow_that_crashes_at_step,

    # Cost/time savings demonstration
    workflow_with_expensive_steps,

    # Mixed step types (functions, state, etc.)
    workflow_with_mixed_steps,

    # State consistency verification
    workflow_with_state_changes,

    # Performance/scale testing
    workflow_replay_stress_test,
)

# Import memory test workflows
from .memory_test_workflows import (
    test_session_memory_workflow,
    test_user_memory_workflow,
    test_multi_agent_session_workflow,
)

# Import agent test workflows (organized by feature)
from .agent_basic_workflows import (
    test_agent_basic,
)

from .agent_tools_workflows import (
    test_agent_with_simple_tool,
    test_agent_with_multiple_tools,
    # Shared tools (can be imported for reuse)
    simple_calculator,
    search_web,
    get_weather,
)

from .agent_collaboration_workflows import (
    test_agent_with_agent_as_tool,
    # Shared tools and helpers
    domain_specific_analysis,
    create_domain_specialist,
)

from .agent_handoff_workflows import (
    test_handoff_simple,
    test_handoff_with_context,
    test_handoff_complex,
    # Shared tools and helpers
    save_user_preference,
    create_technical_specialist,
    create_business_specialist,
    create_research_specialist,
)

from .agent_hitl_workflows import (
    test_agent_with_hitl,
)

from .comprehensive_integration_test import (
    test_comprehensive_scenario,
)

__all__ = [
    # Simple workflows
    "test_workflow",
    "test_workflow_with_state",

    # Replay test workflows
    "workflow_that_crashes_at_step",
    "workflow_with_expensive_steps",
    "workflow_with_mixed_steps",
    "workflow_with_state_changes",
    "workflow_replay_stress_test",

    # Memory test workflows
    "test_session_memory_workflow",
    "test_user_memory_workflow",
    "test_multi_agent_session_workflow",

    # Agent test workflows
    "test_agent_basic",
    "test_agent_with_simple_tool",
    "test_agent_with_multiple_tools",
    "test_agent_with_agent_as_tool",
    "test_handoff_simple",
    "test_handoff_with_context",
    "test_handoff_complex",
    "test_agent_with_hitl",
    "test_comprehensive_scenario",

    # Shared tools (exported for reuse in custom workflows)
    "simple_calculator",
    "search_web",
    "get_weather",
    "domain_specific_analysis",
    "save_user_preference",

    # Agent factory helpers (exported for reuse)
    "create_domain_specialist",
    "create_technical_specialist",
    "create_business_specialist",
    "create_research_specialist",
]
