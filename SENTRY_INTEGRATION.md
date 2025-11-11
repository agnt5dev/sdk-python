# Sentry Integration for AGNT5 Python SDK

The AGNT5 Python SDK includes built-in Sentry integration for error tracking and performance monitoring. This integration is **opt-in** and requires minimal configuration.

## Prerequisites

1. Create a Sentry project at [sentry.io](https://sentry.io)
2. Obtain your project's DSN (Data Source Name)

## Quick Start

### 1. Set Environment Variables

The easiest way to enable Sentry is by setting environment variables:

```bash
# Required: Your Sentry project DSN
export AGNT5_SENTRY_DSN="https://your-key@o123456.ingest.sentry.io/789012"

# Optional: Environment tag (default: "development")
export AGNT5_SENTRY_ENVIRONMENT="production"

# Optional: Performance trace sampling rate 0.0-1.0 (default: 0.1)
export AGNT5_SENTRY_TRACES_SAMPLE_RATE="0.2"

# Optional: Explicitly enable/disable (default: auto from DSN)
export AGNT5_SENTRY_ENABLED="true"
```

### 2. Run Your Worker

No code changes required! The Worker automatically initializes Sentry if `AGNT5_SENTRY_DSN` is set:

```python
from agnt5 import Worker, function

@function
async def process_data(ctx, data: str) -> dict:
    # Errors here will be automatically captured by Sentry
    result = risky_operation(data)
    return {"result": result}

async def main():
    worker = Worker(
        service_name="data-processor",
        service_version="1.0.0",
    )
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## What Gets Captured

The integration automatically captures:

### Exceptions
- All unhandled exceptions in functions, workflows, agents, entities, and tools
- Full stack traces with source code context
- Rich metadata including:
  - `run_id`: Unique execution identifier
  - `component_name`: Name of the component that failed
  - `component_type`: Type (function, workflow, agent, entity, tool)
  - `service_name`: Your service name
  - `error_type`: Exception class name
  - Additional context specific to each component type

### Performance (APM)
- Transaction traces for workflow and agent executions
- Sampling controlled by `AGNT5_SENTRY_TRACES_SAMPLE_RATE`

### Breadcrumbs
- Execution flow leading up to errors
- State transitions
- API calls

## Advanced Usage

### Manual Error Capture

For custom error tracking in your code:

```python
from agnt5 import sentry, function

@function
async def complex_operation(ctx, data: dict):
    try:
        result = await process(data)
        return result
    except ValidationError as e:
        # Capture non-critical errors with custom context
        sentry.capture_exception(
            e,
            context={
                "data_size": len(data),
                "validation_step": "schema_check"
            },
            tags={
                "severity": "warning",
                "component": "validator"
            },
            level="warning"
        )
        # Handle gracefully
        return {"status": "validation_failed"}
```

### Adding Breadcrumbs

Track execution flow for debugging:

```python
from agnt5 import sentry, workflow

@workflow
async def data_pipeline(ctx, input_data: dict):
    sentry.add_breadcrumb(
        message="Starting data validation",
        category="workflow",
        data={"record_count": len(input_data)}
    )

    validated = await validate_data(input_data)

    sentry.add_breadcrumb(
        message="Data validated successfully",
        category="workflow",
        data={"valid_records": len(validated)}
    )

    return await process_validated_data(validated)
```

### Setting User Context

Track errors by user:

```python
from agnt5 import sentry, function

@function
async def user_action(ctx, user_id: str, action: str):
    # Set user context for this execution
    sentry.set_user(user_id=user_id)

    # Errors will now be tagged with user_id
    result = await perform_action(action)
    return result
```

### Custom Tags and Context

Enrich error reports with custom metadata:

```python
from agnt5 import sentry, function

@function
async def api_handler(ctx, endpoint: str, method: str):
    # Set custom tags for filtering in Sentry
    sentry.set_tag("api_endpoint", endpoint)
    sentry.set_tag("http_method", method)

    # Set structured context
    sentry.set_context("api", {
        "endpoint": endpoint,
        "method": method,
        "version": "v2"
    })

    return await handle_request(endpoint, method)
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `AGNT5_SENTRY_DSN` | Sentry project DSN (required) | None | `https://key@o123.ingest.sentry.io/456` |
| `AGNT5_SENTRY_ENVIRONMENT` | Environment tag | `development` | `production`, `staging` |
| `AGNT5_SENTRY_TRACES_SAMPLE_RATE` | APM sampling rate | `0.1` | `0.0` to `1.0` |
| `AGNT5_SENTRY_ENABLED` | Explicit enable/disable | Auto from DSN | `true`, `false` |

### Sampling Rates

Choose sampling based on your traffic and Sentry plan:

- **Development**: `1.0` (100% - capture everything)
- **Staging**: `0.5` (50% - good coverage)
- **Low-traffic Production**: `0.5` (50%)
- **High-traffic Production**: `0.1` (10% - reduce costs)
- **Very High-traffic**: `0.01` (1% - minimal overhead)

## Testing Sentry Integration

Test that Sentry is working correctly:

```python
from agnt5 import Worker, function, sentry

@function
async def test_sentry(ctx):
    # Check if Sentry is initialized
    if sentry.is_sentry_enabled():
        print("✅ Sentry is enabled")

        # Send a test message
        sentry.capture_message(
            "Test message from AGNT5",
            level="info",
            tags={"test": "true"}
        )

        # Trigger a test error (don't do this in production!)
        # raise Exception("Test exception for Sentry")
    else:
        print("❌ Sentry is not enabled (AGNT5_SENTRY_DSN not set)")

    return {"sentry_enabled": sentry.is_sentry_enabled()}

async def main():
    worker = Worker(
        service_name="sentry-test",
        service_version="1.0.0",
        functions=[test_sentry]
    )
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Disabling Sentry

To temporarily disable Sentry without removing the DSN:

```bash
export AGNT5_SENTRY_ENABLED="false"
```

Or unset the DSN:

```bash
unset AGNT5_SENTRY_DSN
```

## Best Practices

### 1. Use Different Projects for Environments

Create separate Sentry projects for dev/staging/prod:

```bash
# Development
export AGNT5_SENTRY_DSN="https://dev-key@o123.ingest.sentry.io/111"
export AGNT5_SENTRY_ENVIRONMENT="development"

# Production
export AGNT5_SENTRY_DSN="https://prod-key@o123.ingest.sentry.io/222"
export AGNT5_SENTRY_ENVIRONMENT="production"
```

### 2. Add Release Tags

Use semantic versioning for better tracking:

```python
worker = Worker(
    service_name="my-service",
    service_version="1.2.3",  # Sent as release tag to Sentry
)
```

### 3. Filter Sensitive Data

Never log sensitive information:

```python
from agnt5 import sentry

# ❌ Bad - logs sensitive data
sentry.capture_exception(e, context={"password": user_password})

# ✅ Good - redact sensitive fields
sentry.capture_exception(e, context={"user_id": user_id})
```

### 4. Set Alert Rules

Configure Sentry alerts for:
- High error rates (> 10 errors/minute)
- New error types (first occurrence)
- Regression issues (previously resolved errors)

## Troubleshooting

### Sentry Not Initializing

Check the worker logs for:

```
INFO: Sentry error tracking initialized
```

If you see:

```
DEBUG: Sentry not initialized (AGNT5_SENTRY_DSN not set or disabled)
```

Verify:
1. `AGNT5_SENTRY_DSN` is set correctly
2. DSN format is valid
3. `AGNT5_SENTRY_ENABLED` is not set to `false`

### No Events in Sentry

1. **Check connectivity**: Ensure worker can reach `*.ingest.sentry.io`
2. **Verify DSN**: Test DSN in Sentry project settings
3. **Check sampling**: Increase `AGNT5_SENTRY_TRACES_SAMPLE_RATE` to `1.0`
4. **Review filters**: Check Sentry inbound filters aren't blocking events

### Too Many Events

1. **Reduce sampling**: Lower `AGNT5_SENTRY_TRACES_SAMPLE_RATE`
2. **Add filters**: Use Sentry's inbound filters to ignore known issues
3. **Fix errors**: The best way to reduce events is to fix bugs!

## Performance Impact

The Sentry integration is designed for minimal overhead:

- **Initialization**: One-time setup cost (~10ms)
- **Error capture**: Only triggered on exceptions (~5-10ms per error)
- **APM tracing**: Controlled by sampling rate (default 10%)
- **Async sending**: Events sent asynchronously, non-blocking
- **Batching**: Events batched for efficient network usage

Typical overhead in production: **< 1% CPU and memory**

## Support

For issues with the AGNT5 Sentry integration:
- GitHub Issues: [agnt5/issues](https://github.com/arunreddy/agnt5/issues)
- Sentry-specific questions: [Sentry Support](https://sentry.io/support/)

## References

- [Sentry Python SDK Documentation](https://docs.sentry.io/platforms/python/)
- [Sentry Best Practices](https://docs.sentry.io/product/best-practices/)
- [AGNT5 SDK Documentation](https://agnt5.com/sdk/python)
