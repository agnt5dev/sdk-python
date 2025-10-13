# Test Service Functions

This document describes the organization of test functions in the test-service blueprint.

## Directory Structure

```
src/agnt5_test_service/functions/
├── __init__.py                    # Imports and exports all functions
├── simple_functions.py            # Basic platform tests (no LM)
├── openai_lm_functions.py         # OpenAI integration tests
├── anthropic_lm_functions.py      # Anthropic Claude integration tests
└── openrouter_lm_functions.py     # OpenRouter multi-provider tests
```

## Function Categories

### Simple Functions (`simple_functions.py`)

Basic platform functionality tests that don't require LM API keys:

- `greet(name)` - Simple greeting for basic execution tests
- `long_task(duration)` - Simulates long-running tasks
- `flaky_function(fail_count)` - Tests retry logic
- `failing_function(error)` - Always fails with error message
- `generate_text(prompt)` - Mock text generation

**API Keys Required**: None

### OpenAI Functions (`openai_lm_functions.py`)

Tests OpenAI GPT models integration:

- `generate_greeting(name, style)` - Basic text generation
- `analyze_sentiment(text)` - Structured output with dataclasses
- `generate_story(topic)` - Streaming responses
- `chat_with_context(user_message, context)` - Multi-turn conversations
- `generate_with_invalid_model(prompt)` - Error handling test
- `generate_joke(topic)` - Live API integration test

**API Keys Required**: `OPENAI_API_KEY`

**Models Used**:
- `openai/gpt-4o-mini` - Fast, cost-effective model
- `openai/gpt-4o` - For structured output tests

### Anthropic Functions (`anthropic_lm_functions.py`)

Tests Anthropic Claude integration:

- `generate_text_anthropic(prompt)` - Basic text generation
- `summarize_with_anthropic(text)` - Text summarization

**API Keys Required**: `ANTHROPIC_API_KEY`

**Models Used**:
- `anthropic/claude-3-5-sonnet-20241022` - Latest Claude model

### OpenRouter Functions (`openrouter_lm_functions.py`)

Tests OpenRouter multi-provider access:

- `generate_text_openrouter(prompt, model_name)` - Generate with specified model
- `compare_models_openrouter(prompt)` - Test with free models

**API Keys Required**: `OPENROUTER_API_KEY`

**Models Used**:
- `openrouter/anthropic/claude-3-5-sonnet` (default)
- `openrouter/meta-llama/llama-3.1-8b-instruct:free` - Free model for testing

## Usage

### Import Functions

All functions are automatically imported through `agnt5_test_service.functions`:

```python
from agnt5_test_service.functions import (
    greet,
    generate_greeting,
    generate_text_anthropic,
    generate_text_openrouter,
)
```

### Running Tests

1. **Simple Functions** - No setup required:
   ```bash
   pytest tests/integration/test_client.py -v
   ```

2. **OpenAI Functions** - Requires API key:
   ```bash
   export OPENAI_API_KEY=sk-...
   pytest tests/integration/test_client_lm.py -v
   ```

3. **Anthropic Functions** - Requires API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   # Add tests for Anthropic functions
   ```

4. **OpenRouter Functions** - Requires API key:
   ```bash
   export OPENROUTER_API_KEY=sk-or-...
   # Add tests for OpenRouter functions
   ```

### Using .env File

For easier development, create a `.env` file in the test-service directory:

```bash
cd tests/integration/blueprints/test-service
cp .env.example .env
# Edit .env and add your API keys
```

The worker will automatically load API keys from the .env file at startup.

## Adding New Functions

### 1. Choose the Right Module

- **No LM required** → `simple_functions.py`
- **OpenAI models** → `openai_lm_functions.py`
- **Anthropic models** → `anthropic_lm_functions.py`
- **OpenRouter models** → `openrouter_lm_functions.py`
- **New provider** → Create new file: `{provider}_lm_functions.py`

### 2. Add Function

```python
@function
async def my_new_function(ctx: Context, param: str) -> dict:
    """Description.

    Requires {PROVIDER}_API_KEY - will fail if not available.
    """
    response = await lm.generate(
        model="provider/model-name",
        prompt=param,
        max_tokens=100
    )
    return {"result": response.text}
```

### 3. Export Function

Add to `__all__` in the module file and in `functions/__init__.py`.

### 4. Add Test

Create corresponding test in `tests/integration/test_client_{provider}.py`.

## Important Notes

⚠️ **No Fallbacks**: All LM functions will FAIL if API keys are not available. This is intentional to ensure real integration testing.

⚠️ **Real API Calls**: Integration tests make actual API calls and cost money. Use cost-effective models when possible.

⚠️ **API Key Security**: Never commit .env files or expose API keys in code or logs.

## Cost Considerations

Approximate costs per test run:
- OpenAI tests: ~$0.01-0.05 (using gpt-4o-mini)
- Anthropic tests: ~$0.01-0.03 (using claude-3-5-sonnet)
- OpenRouter tests: Variable (free models available)

For frequent testing, consider:
1. Running unit tests instead (mocked, free)
2. Running specific tests only
3. Using free tier models when available
