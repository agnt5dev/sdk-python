"""
Test Functions for AGNT5 Integration Testing

This package provides test functions organized by functionality:
- simple_functions: Basic platform functionality without LM dependencies
- openai_lm_functions: OpenAI LM integration tests
- anthropic_lm_functions: Anthropic Claude integration tests
- openrouter_lm_functions: OpenRouter multi-provider integration tests
"""

# Import all functions from submodules
from .simple_functions import (
    greet,
    long_task,
    flaky_function,
    failing_function,
    generate_text,
)

from .openai_lm_functions import (
    generate_greeting,
    analyze_sentiment,
    generate_story,
    chat_with_context,
    generate_with_invalid_model,
    generate_joke,
    SentimentAnalysis,
)

from .anthropic_lm_functions import (
    generate_text_anthropic,
    summarize_with_anthropic,
)

from .openrouter_lm_functions import (
    generate_text_openrouter,
    compare_models_openrouter,
)

# Export all functions
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
]
