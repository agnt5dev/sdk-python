# Sentry Project Setup for AGNT5 SDK

This document is for the AGNT5 team to configure the Sentry project for SDK error tracking.

## Current Configuration

**Project**: AGNT5 Python SDK Error Tracking
**DSN**: `https://a25fea6eeec2e8b393a77f1e2cc7fe2c@o4509047159521280.ingest.us.sentry.io/4509047294656512`
**Hardcoded in**: `sdk/sdk-python/src/agnt5/_sentry.py:46`

## Version Detection Logic

The SDK automatically detects alpha/beta releases using regex pattern matching:

```python
def _is_prerelease_version(version: str) -> bool:
    # Pattern: <major>.<minor>.<patch>(a|b)<number>
    # Examples:
    #   "0.2.8a12" → True (alpha 12)
    #   "1.0.0b3"  → True (beta 3)
    #   "1.2.3"    → False (stable)
    #   "1.2.3rc1" → False (release candidate not considered pre-release)
    return bool(re.search(r'\d+\.\d+\.\d+(a|b)\d+', version))
```

**Version Examples:**
- `0.2.8a1` through `0.2.8a99` → Alpha (telemetry ON)
- `1.0.0b1` through `1.0.0b99` → Beta (telemetry ON)
- `1.0.0`, `2.1.3`, etc. → Stable (telemetry OFF)

**Not Covered (intentionally):**
- Release candidates: `1.0.0rc1` → Treated as stable
- Dev versions: `1.0.0.dev0` → Treated as stable

## Required Sentry Configuration

### 1. Rate Limiting (Critical)

**Why**: The DSN is public in source code, vulnerable to abuse.

**Settings** → **General** → **Client Keys (DSN)** → **Configure**:

```
Rate Limits:
- Events per minute: 100
- Events per hour: 1000
- Events per day: 10000

Spike Protection:
- Enable spike protection: YES
- Spike threshold: 50 events in 1 minute
```

### 2. Inbound Filters

**Settings** → **Inbound Filters**:

**Enable these filters:**
- ✅ Filter out known legacy browsers
- ✅ Filter out localhost and private IP addresses (already anonymized, but extra layer)
- ✅ Filter out web crawlers

**Add custom filters:**
```
Error Message Patterns (if needed):
- Block: "test" (case-insensitive)
- Block: "fake" (case-insensitive)
```

### 3. Data Scrubbing

**Settings** → **Security & Privacy** → **Data Scrubbing**:

- ✅ **Enable**: "Use default scrubbers"
- ✅ **Enable**: "Prevent Storing of IP Addresses"
- ✅ **Additional field names to scrub**: `password,secret,api_key,token,auth,authorization,apikey,api-key`

**Note**: Our `_anonymize_event()` already handles this, but this is defense-in-depth.

### 4. Alert Rules

**Alerts** → **Create Alert Rule**:

**Alert 1: High SDK Error Rate**
```
When: Number of events
Condition: is greater than 50
In: 5 minutes
Filter: sdk_error = true

Action: Send email to team@agnt5.com
```

**Alert 2: Critical SDK Failures**
```
When: An event is tagged
Condition: severity = critical
Filter: sdk_error = true

Action: Send Slack notification + email
```

**Alert 3: New SDK Error Type**
```
When: A new issue is created
Filter: sdk_error = true

Action: Send Slack notification
```

### 5. Issue Grouping

**Settings** → **Issue Grouping**:

```
Grouping Strategy: Recommended
Stack Trace Rules: Default

Custom Fingerprinting (optional):
- Group by error_location + error_phase
```

### 6. Retention & Sampling

**Settings** → **Data Management**:

```
Event Retention: 90 days
Delete & Discard old events: YES

Dynamic Sampling:
- Keep all error events (don't sample)
- Sample transaction events at configured rate (default 0.1)
```

## Monitoring & Maintenance

### Regular Checks

**Weekly:**
- Review error volume trends
- Check for new error types
- Verify no abuse (check spike protection logs)

**Monthly:**
- Review and close resolved issues
- Update inbound filters if needed
- Check rate limit logs

### Dashboards to Create

**Dashboard 1: SDK Health**
```
Widgets:
- Total events (last 7 days)
- Events by SDK version
- Top 10 error types
- Events by is_prerelease tag
```

**Dashboard 2: Release Tracking**
```
Widgets:
- Errors by release (agnt5-python-sdk@version)
- First seen / Last seen per release
- Adoption rate (events per release)
```

## Team Access

**Roles:**
- **Admin**: Arun (full access)
- **Member**: Core team (can view, create issues, manage alerts)
- **Billing**: Finance team (billing only)

## Cost Estimation

**Current Plan**: [Fill in your Sentry plan]

**Expected Volume:**
- Alpha/Beta releases: ~100-500 events/day (SDK errors are rare)
- Stable releases: ~10-50 events/day (opt-in only)
- Est. monthly cost: [Fill in based on plan]

**Cost Control:**
- Rate limiting prevents runaway costs
- Sampling configured for traces
- Event retention: 90 days

## Testing the Integration

### Test Alpha/Beta Behavior

```bash
# In SDK repo
cd sdk/sdk-python

# Check current version
python -c "from agnt5 import __version__; print(__version__)"
# Should see something like: 0.2.8a12

# Start worker (telemetry should be ON)
just platform dev-server python

# Look for log:
# INFO: SDK telemetry enabled for alpha/beta version 0.2.8a12
```

### Trigger Test Error

```python
# Create test_sentry.py
from agnt5 import Worker

async def main():
    # This will fail (missing required params)
    worker = Worker()  # Intentional error for testing

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Expected in Sentry:**
- Event with `sdk_error=true`
- Release tag: `agnt5-python-sdk@0.2.8a12`
- Error type: `import_error` or similar
- Anonymized context (no secrets)

### Verify Anonymization

**Check that these are NOT present:**
- Local variables (should be stripped)
- Environment variables
- IP addresses
- Request headers

**Should see:**
- Error message and type
- SDK stack trace (no vars)
- Service name (user-controlled)
- SDK version
- Python version

## Troubleshooting

### No Events Appearing

1. Check DSN is correct in code
2. Verify network access to `*.ingest.sentry.io`
3. Check version detection: `_is_prerelease_version("0.2.8a12")` should return `True`
4. Check logs for initialization message

### Too Many Events

1. Check for runaway error loop
2. Verify rate limiting is configured
3. Check if someone is abusing public DSN
4. Temporarily increase rate limits if legitimate

### Events Missing Context

1. Check `_anonymize_event()` isn't too aggressive
2. Verify Sentry data scrubbing settings
3. Review event JSON in Sentry UI

## Future Enhancements

- [ ] Add Sentry for Rust sdk-core (separate project)
- [ ] Add usage telemetry (not errors) - component counts, etc.
- [ ] Add monthly "help improve AGNT5?" prompt for stable users
- [ ] Integrate with GitHub issues (auto-create from Sentry)
- [ ] Add Sentry session tracking (SDK usage patterns)

## Security Incident Response

**If DSN is abused:**
1. Regenerate DSN in Sentry project
2. Update `AGNT5_SDK_SENTRY_DSN` in code
3. Release hotfix version
4. Review and tighten rate limits
5. Consider IP allowlisting

**If sensitive data is accidentally sent:**
1. Delete affected events immediately (Sentry UI)
2. Review `_anonymize_event()` for gaps
3. Add additional scrubbing rules
4. Notify affected users if needed

## Contacts

**Sentry Account Owner**: team@agnt5.com
**Escalation**: team@agnt5.com
**Billing Questions**: finance@agnt5.com

---

**Last Updated**: 2025-11-11
**Next Review**: 2025-12-11
