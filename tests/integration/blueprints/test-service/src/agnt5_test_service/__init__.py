"""
AGNT5 Test Service - Integration Testing Components

This package provides test functions, entities, and workflows for AGNT5 integration testing.
"""

from agnt5_test_service.functions import (
    # Simple functions
    greet,
    long_task,
    flaky_function,
    failing_function,
    generate_text,
    # OpenAI LM functions
    generate_greeting,
    analyze_sentiment,
    generate_story,
    chat_with_context,
    generate_with_invalid_model,
    generate_joke,
    SentimentAnalysis,
    # Anthropic LM functions
    generate_text_anthropic,
    summarize_with_anthropic,
    # OpenRouter LM functions
    generate_text_openrouter,
    compare_models_openrouter,
)

from agnt5_test_service.entities import (
    ShoppingCart,
    Counter,
    BankAccount,
)

from agnt5_test_service.workflows import (
    order_fulfillment,
    long_workflow,
    data_pipeline,
    tool_orchestrated_workflow,
    agent_research_workflow,
    agent_multi_step_workflow,
)

from agnt5_test_service.tools import (
    calculate_total,
    search_database,
    format_report,
    validate_data,
)

from agnt5_test_service.agents import (
    chat_agent,
)

__all__ = [
    # Simple functions
    "greet",
    "long_task",
    "flaky_function",
    "failing_function",
    "generate_text",
    # OpenAI LM functions
    "generate_greeting",
    "analyze_sentiment",
    "generate_story",
    "chat_with_context",
    "generate_with_invalid_model",
    "generate_joke",
    "SentimentAnalysis",
    # Anthropic LM functions
    "generate_text_anthropic",
    "summarize_with_anthropic",
    # OpenRouter LM functions
    "generate_text_openrouter",
    "compare_models_openrouter",
    # Entities
    "ShoppingCart",
    "Counter",
    "BankAccount",
    # Workflows
    "order_fulfillment",
    "long_workflow",
    "data_pipeline",
    "tool_orchestrated_workflow",
    "agent_research_workflow",
    "agent_multi_step_workflow",
    # Tools
    "calculate_total",
    "search_database",
    "format_report",
    "validate_data",
    # Agents
    "chat_agent",
]

__version__ = "1.0.0"
