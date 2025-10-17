# Language Model Integration Testing

This document explains how to run LM integration tests for the AGNT5 Python SDK.

## Overview

The LM integration tests validate that language models work correctly within AGNT5 platform functions. These are **true integration tests** that make real API calls to LM providers.

## Important Notes

⚠️ **These tests REQUIRE API keys and will FAIL without them**

- No mocking or fallbacks
- Tests make real LM API calls
- This is intentional - we need to catch real integration issues
- Fallbacks would give false positives

## Setup

### 1. Get an OpenAI API Key

Get your API key from: https://platform.openai.com/api-keys

### 2. Configure Environment Variables

**Option A: Export environment variable**
```bash
export OPENAI_API_KEY=sk-...
```

**Option B: Use .env file (recommended)**
```bash
cd tests/integration/blueprints/test-service
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

The worker will automatically load the .env file at startup.

## Running Tests

### Run All LM Integration Tests

```bash
# From SDK root directory
uv run pytest tests/integration/test_client_lm.py -v
```

### Run Specific Test

```bash
uv run pytest tests/integration/test_client_lm.py::test_function_with_lm_generate -v
```

### Run Unit Tests (Mocked, No API Key Required)

```bash
uv run pytest tests/test_lm.py -v
```

## Test Coverage

### Integration Tests (Require API Key)

Located in: `tests/integration/test_client_lm.py`

- `test_function_with_lm_generate` - Basic LM generation within functions
- `test_function_with_lm_structured_output` - Structured output with dataclasses
- `test_function_with_lm_streaming` - Streaming responses
- `test_function_with_lm_multi_turn` - Multi-turn conversations
- `test_function_with_lm_error_handling` - Error propagation
- `test_function_with_live_lm` - Live API test (skipped by default)

### Unit Tests (No API Key Required)

Located in: `tests/test_lm.py`

- 18 unit tests with mocked LM responses
- Test API surface without making real API calls
- Always safe to run in CI/CD

## CI/CD Integration

For CI/CD pipelines, you have two options:

### Option 1: Skip LM Integration Tests
```bash
# Run all tests except LM integration
pytest tests/ -v -k "not test_client_lm"
```

### Option 2: Provide API Key in CI
```yaml
# GitHub Actions example
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

steps:
  - name: Run LM integration tests
    run: uv run pytest tests/integration/test_client_lm.py -v
```

## Troubleshooting

### Tests Fail with "API key not found"

This is expected if OPENAI_API_KEY is not set. Add your API key using one of the methods above.

### Tests Fail with "Rate limit exceeded"

OpenAI has rate limits. Wait a few seconds and try again, or use a higher-tier API key.

### Tests Pass but Output Looks Wrong

This likely means you're running unit tests (which are mocked) instead of integration tests. Make sure you're running:
```bash
pytest tests/integration/test_client_lm.py -v
```

## Cost Considerations

These integration tests make real API calls which cost money:

- Each test run costs approximately $0.01-0.05 (OpenAI pricing)
- Tests use `gpt-4o-mini` for cost efficiency
- Total test suite: ~5 API calls per run

For frequent testing during development, consider:
1. Running unit tests instead: `pytest tests/test_lm.py -v`
2. Running specific tests: `pytest tests/integration/test_client_lm.py::test_function_with_lm_generate -v`
3. Using .env file to easily enable/disable API access

## Adding New LM Tests

When adding new LM integration tests:

1. **No Fallbacks**: Test functions should not have try/except with fallbacks
2. **Real API Calls**: Always use real LM provider APIs
3. **Clear Docstrings**: Document which API keys are required
4. **Example Functions**: Add corresponding test functions to `tests/integration/blueprints/test-service/src/agnt5_test_service/functions.py`

Example:
```python
@function
async def my_lm_function(ctx: Context, prompt: str) -> dict:
    """My LM test function.

    Requires OPENAI_API_KEY - will fail if not available.
    """
    response = await lm.generate(
        model="openai/gpt-4o-mini",
        prompt=prompt,
        max_tokens=50
    )
    return {"result": response.text}
```

## Questions?

See the main SDK documentation or open an issue on GitHub.
