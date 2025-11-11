# Sentry Integration for AGNT5 Python SDK

The AGNT5 Python SDK includes built-in Sentry integration for **SDK-level error tracking**. This helps us identify and fix SDK bugs, initialization failures, and Python-specific issues.

## What Gets Captured

This integration captures **SDK errors**, not user code execution errors:

✅ **Captured (SDK Issues):**
- SDK initialization failures
- Rust FFI import errors
- Component auto-registration failures
- Worker startup/lifecycle errors
- SDK internal bugs and crashes
- Configuration errors

❌ **Not Captured (User Code Errors):**
- Exceptions in your @function/@workflow/@agent code
- User application logic errors
- Business logic failures

> **Note:** Users should handle their own application errors. The SDK integration only captures SDK-level issues to help us improve the SDK itself.

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
    # SDK errors (initialization, FFI issues) will be captured
    # Your application errors should be handled by your code
    result = risky_operation(data)
    return {"result": result}

async def main():
    worker = Worker(
        service_name="data-processor",
        service_version="1.0.0",
    )
    # SDK startup failures will be captured here
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## SDK Errors Captured

The integration automatically captures SDK-level issues:

### SDK Initialization Errors
- Rust core import failures (PyO3/maturin issues)
- Worker initialization failures
- Entity state manager setup errors
- Configuration errors

### Component Registration Errors
- Auto-discovery import failures
- Component registration bugs
- Schema validation errors

### Worker Lifecycle Errors
- Worker startup failures
- Event loop configuration issues
- Coordinator communication failures
- Critical runtime errors

### Rich Context
All captured errors include:
- `service_name` and `service_version`
- `error_location`: Where in the SDK the error occurred
- `error_phase`: What operation was being performed
- Stack traces with source code context
- SDK version and Python version

## For User Application Errors

**Important:** The SDK integration only captures SDK bugs. For your application errors, you should:

### Option 1: Use Your Own Sentry Project

Set up a separate Sentry project for your application:

```python
import sentry_sdk
from agnt5 import Worker, function

# Initialize your app's Sentry separately
sentry_sdk.init(
    dsn="your-app-sentry-dsn",  # Different from AGNT5_SENTRY_DSN
    environment="production",
)

@function
async def process_order(ctx, order_id: str):
    try:
        result = await dangerous_operation(order_id)
        return result
    except ValidationError as e:
        # Capture in YOUR Sentry project
        sentry_sdk.capture_exception(e)
        return {"status": "failed"}
```

### Option 2: Use the SDK's Sentry Utilities

If you want to use the same Sentry project, you can access the utilities:

```python
from agnt5 import sentry, function

@function
async def risky_operation(ctx, data: dict):
    try:
        result = await process(data)
        return result
    except Exception as e:
        # Manually capture your application error
        sentry.capture_exception(
            e,
            context={"operation": "process_data"},
            tags={"app_error": "true"},
        )
        raise  # Re-raise for retry handling
```

### Option 3: Let Errors Propagate

The platform handles retries and error tracking:

```python
@function
async def my_function(ctx, data: str):
    # Just let exceptions propagate
    # Platform will retry based on your retry policy
    result = await risky_call(data)
    return result
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

## Testing SDK Error Tracking

Test that SDK error tracking is working:

```python
from agnt5 import Worker, function, sentry

@function
async def test_sdk_tracking(ctx):
    # Check if SDK error tracking is enabled
    if sentry.is_sentry_enabled():
        print("✅ SDK error tracking is enabled")
        return {"sentry_enabled": True}
    else:
        print("❌ SDK error tracking not enabled (AGNT5_SENTRY_DSN not set)")
        return {"sentry_enabled": False}

async def main():
    # To test SDK error capture, introduce SDK-level issues:
    # - Try with invalid coordinator endpoint
    # - Use malformed component registration
    # - Trigger auto-discovery import errors

    worker = Worker(
        service_name="sentry-test",
        service_version="1.0.0",
        functions=[test_sdk_tracking],
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

### 1. Separate SDK and Application Errors

Use different Sentry projects for SDK vs application errors:

```bash
# SDK error tracking (AGNT5 team)
export AGNT5_SENTRY_DSN="https://sdk-key@o123.ingest.sentry.io/111"

# Your application errors (separate project)
# Initialize separately in your code with sentry_sdk.init()
```

### 2. Use Semantic Versioning

Helps track which SDK versions have issues:

```python
worker = Worker(
    service_name="my-service",
    service_version="1.2.3",  # SDK sees this as release tag
)
```

### 3. Report SDK Bugs

If you see SDK errors in Sentry:
1. Check if it's a known issue in [agnt5/issues](https://github.com/arunreddy/agnt5/issues)
2. Report with full context from Sentry
3. Include SDK version and Python version

### 4. Don't Over-Report

The SDK integration is meant for SDK bugs, not:
- User code errors (handle in your app)
- Expected business logic failures
- Validation errors
- Rate limiting errors

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

If you're seeing too many SDK errors:
1. **Check your setup**: SDK errors should be rare
2. **Report the issue**: Frequent SDK errors indicate a bug we should fix
3. **Disable temporarily**: Set `AGNT5_SENTRY_ENABLED=false` while we investigate

## Performance Impact

The SDK error tracking has negligible overhead:

- **Initialization**: One-time setup cost (~10ms)
- **Error capture**: Only triggered on SDK exceptions (rare)
- **No execution overhead**: User code execution not affected
- **Async sending**: Events sent asynchronously, non-blocking

Since SDK errors are rare, the overhead is effectively **0% in normal operation**.

## Support

For issues with the AGNT5 Sentry integration:
- GitHub Issues: [agnt5/issues](https://github.com/arunreddy/agnt5/issues)
- Sentry-specific questions: [Sentry Support](https://sentry.io/support/)

## References

- [Sentry Python SDK Documentation](https://docs.sentry.io/platforms/python/)
- [Sentry Best Practices](https://docs.sentry.io/product/best-practices/)
- [AGNT5 SDK Documentation](https://agnt5.com/sdk/python)
