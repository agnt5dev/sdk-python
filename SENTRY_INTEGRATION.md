# Sentry Integration for AGNT5 Python SDK

The AGNT5 Python SDK includes **automatic SDK error tracking** to help us identify and fix SDK bugs, initialization failures, and Python-specific issues.

## How It Works

### For Alpha/Beta Releases (e.g., 0.2.8a12, 1.0.0b3)

**✅ Telemetry Automatically Enabled**
- SDK errors sent to AGNT5 team
- Helps identify bugs before stable release
- All data anonymized (no secrets, IP addresses, or personal data)
- **Opt-out**: `export AGNT5_DISABLE_SDK_TELEMETRY=true`

```bash
# Disable telemetry in alpha/beta if needed
export AGNT5_DISABLE_SDK_TELEMETRY=true
```

### For Stable Releases (e.g., 1.0.0, 2.1.3)

**✅ Telemetry Disabled by Default (Privacy First)**
- No data sent unless you explicitly opt-in
- **Opt-in**: `export AGNT5_ENABLE_SDK_TELEMETRY=true`

```bash
# Enable telemetry in stable releases (to help AGNT5 team)
export AGNT5_ENABLE_SDK_TELEMETRY=true
```

## What Gets Captured

This integration captures **SDK errors only**, not user code execution errors:

### ✅ Captured (SDK Issues)
- SDK initialization failures
- Rust FFI import errors
- Component auto-registration failures
- Worker startup/lifecycle errors
- SDK internal bugs and crashes
- Configuration errors

### ❌ NOT Captured (User Code)
- Exceptions in your @function/@workflow/@agent code
- User application logic errors
- Business logic failures
- User data or secrets

> **Note:** Users should handle their own application errors. This integration only captures SDK-level issues to help us improve the SDK itself.

## Quick Start

**No configuration required!** The SDK automatically manages telemetry based on your version.

```python
from agnt5 import Worker, function

@function
async def process_data(ctx, data: str) -> dict:
    # SDK errors (initialization, FFI issues) are automatically captured
    # Your application errors should be handled by your code
    result = risky_operation(data)
    return {"result": result}

async def main():
    worker = Worker(
        service_name="data-processor",
        service_version="1.0.0",
    )
    # SDK automatically initializes telemetry based on version
    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Alpha/Beta users will see:**
```
INFO: SDK telemetry enabled for alpha/beta version 0.2.8a12
      (helps AGNT5 team find bugs). To disable: export AGNT5_DISABLE_SDK_TELEMETRY=true
```

**Stable users will see:**
```
DEBUG: SDK telemetry disabled by default for stable version 1.0.0
       (set AGNT5_ENABLE_SDK_TELEMETRY=true to help AGNT5 team)
```

## Privacy & Security

### What's Anonymized

All events are scrubbed of sensitive data before sending:

- ✅ **IP addresses** - Completely removed
- ✅ **Environment variables** - Stripped (prevents secret leakage)
- ✅ **Stack trace local variables** - Removed (may contain API keys, passwords)
- ✅ **Request data and headers** - Removed (may contain auth tokens)
- ✅ **Breadcrumb data** - Only safe metadata kept

**Example of what's protected:**
```python
def risky_function():
    api_key = "sk-abc123..."  # ← This will NOT be sent to Sentry
    password = "prod_pass"     # ← This will NOT be sent to Sentry
    result = sdk_operation()   # SDK error here
    # Only the error message and SDK code context are sent
```

### What Gets Sent

**Rich SDK Context (Safe Metadata):**
- SDK version and Python version
- Error type and message
- Stack trace (SDK code only, no local variables)
- Error location (file and function in SDK)
- Service name (you control this)
- Timestamp and environment

**All context is safe and contains no user secrets or data.**

## SDK Errors Captured

### Initialization Errors
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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGNT5_DISABLE_SDK_TELEMETRY` | Disable telemetry (alpha/beta) | `false` |
| `AGNT5_ENABLE_SDK_TELEMETRY` | Enable telemetry (stable) | `false` |
| `AGNT5_SENTRY_ENVIRONMENT` | Environment tag | `production` |
| `AGNT5_SENTRY_TRACES_SAMPLE_RATE` | APM sampling rate | `0.1` |
| `AGNT5_SDK_SENTRY_DSN` | Override DSN (testing only) | (hardcoded) |

## For User Application Errors

**Important:** The SDK integration only captures SDK bugs. For your application errors:

### Option 1: Use Your Own Sentry Project (Recommended)

```python
import sentry_sdk
from agnt5 import Worker, function

# Initialize your app's Sentry separately
sentry_sdk.init(
    dsn="your-app-sentry-dsn",  # Your own Sentry project
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

### Option 2: Use SDK's Sentry Utilities

```python
from agnt5 import sentry, function

@function
async def risky_operation(ctx, data: dict):
    try:
        result = await process(data)
        return result
    except Exception as e:
        # Manually capture your application error
        # This goes to AGNT5's Sentry project
        sentry.capture_exception(
            e,
            context={"operation": "process_data"},
            tags={"app_error": "true"},
        )
        raise  # Re-raise for retry handling
```

### Option 3: Let Errors Propagate (Recommended)

```python
@function
async def my_function(ctx, data: str):
    # Just let exceptions propagate
    # Platform will retry based on your retry policy
    # OpenTelemetry already captures execution errors
    result = await risky_call(data)
    return result
```

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
        print("❌ SDK error tracking not enabled")
        return {"sentry_enabled": False}

async def main():
    worker = Worker(
        service_name="sentry-test",
        service_version="1.0.0",
        functions=[test_sdk_tracking],
    )
    await worker.run()
```

## Best Practices

### 1. Let SDK Handle Telemetry Automatically

Don't override unless testing:
```bash
# ❌ Don't set unless you have a specific reason
# export AGNT5_SDK_SENTRY_DSN="..."

# ✅ Let SDK use version-based defaults
```

### 2. Use Semantic Versioning

Helps track which SDK versions have issues:
```python
worker = Worker(
    service_name="my-service",
    service_version="1.2.3",
)
```

### 3. Report SDK Bugs

If you see SDK errors:
1. Check [agnt5/issues](https://github.com/arunreddy/agnt5/issues)
2. Report with full context
3. Include SDK version and Python version

### 4. Separate SDK and App Errors

- SDK errors → AGNT5's Sentry (automatic)
- App errors → Your Sentry project (separate `sentry_sdk.init()`)

## Troubleshooting

### Telemetry Not Working (Alpha/Beta)

Check worker logs for initialization message:
```
INFO: SDK telemetry enabled for alpha/beta version 0.2.8a12...
```

If missing:
1. Verify you're using alpha/beta version (check `__version__`)
2. Check `AGNT5_DISABLE_SDK_TELEMETRY` is not set
3. Ensure `sentry-sdk` is installed

### Want to Enable in Stable

```bash
export AGNT5_ENABLE_SDK_TELEMETRY=true
# Restart worker to see:
# INFO: SDK telemetry enabled for version 1.0.0 (thank you for helping improve AGNT5!)
```

### Too Many SDK Errors

If you're seeing frequent SDK errors:
1. **Report immediately** - This indicates a critical SDK bug
2. Check if it's a known issue
3. Temporarily disable: `export AGNT5_DISABLE_SDK_TELEMETRY=true`

## Performance Impact

The SDK error tracking has **negligible overhead**:

- **Initialization**: One-time ~10ms cost
- **Runtime**: Only triggered on SDK exceptions (rare)
- **No execution overhead**: User code unaffected
- **Async sending**: Non-blocking

Since SDK errors are rare, overhead is **effectively 0% in normal operation**.

## Security Notes

### DSN Security

The Sentry DSN is hardcoded in the SDK and visible in source code. This is **acceptable** because:
- It's an ingestion-only key (not sensitive)
- Rate limiting is configured in Sentry project
- Events are anonymized before sending

### Abuse Prevention

The AGNT5 team has configured:
- Rate limiting (max events per minute)
- Spike protection
- Inbound filters for suspicious events
- IP allowlists if needed

## Advanced Configuration

### Override DSN (Testing Only)

```bash
# For AGNT5 team testing only
export AGNT5_SDK_SENTRY_DSN="https://test-key@..."
```

### Change Environment Tag

```bash
export AGNT5_SENTRY_ENVIRONMENT="staging"
```

### Adjust Sampling Rate

```bash
# Capture more traces (0.0 to 1.0)
export AGNT5_SENTRY_TRACES_SAMPLE_RATE="0.5"
```

## Support

**For SDK errors:**
- GitHub Issues: [agnt5/issues](https://github.com/arunreddy/agnt5/issues)

**For Sentry integration questions:**
- Check this documentation
- Open an issue with `[telemetry]` tag

**For application errors:**
- Use your own Sentry project
- See Sentry documentation: [docs.sentry.io](https://docs.sentry.io)

## Summary

✅ **Alpha/Beta**: Telemetry ON (opt-out available)
✅ **Stable**: Telemetry OFF (opt-in available)
✅ **Privacy**: All data anonymized
✅ **Zero Config**: Works automatically
✅ **SDK Errors Only**: User code not tracked

This helps us fix SDK bugs faster while respecting user privacy! 🎉
