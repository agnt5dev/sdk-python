# OpenTelemetry Integration Guide

## Overview

The AGNT5 SDK has comprehensive OpenTelemetry integration for distributed tracing, structured logging, and metrics.

## Architecture

### Rust SDK Core (`sdk-core`)

The Rust core provides the OpenTelemetry foundation:

- **Traces**: Spans for all operations (functions, workflows, LLM calls)
- **Logs**: Structured logs forwarded to OTLP exporter
- **Metrics**: Counters and histograms for LLM operations
- **Console Output**: Formatted logs to stdout/stderr

### Python SDK (`sdk-python`)

Python integrates with Rust telemetry via:

- **Custom Logging Handler**: Forwards `ctx.logger` logs to Rust
- **Automatic Span Context**: Logs inherit invocation.id, trace_id
- **Dual Output**: Logs go to both console AND OpenTelemetry

## Configuration

### Environment Variables

```bash
# OpenTelemetry Collector Endpoint (gRPC)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Log Level (optional, defaults to info)
export RUST_LOG=agnt5=debug,info

# Trace Content (includes LLM prompt/response in spans)
export AGNT5_TRACE_CONTENT_ENABLED=true
```

### Default Endpoint

If `OTEL_EXPORTER_OTLP_ENDPOINT` is not set:
- Defaults to `http://localhost:4317` (local OpenTelemetry Collector)

## What Gets Traced

### 1. Function Invocations

Every `@function` execution creates a span with:

```
Span: function.{function_name}
Attributes:
  - function.name: "my_function"
  - service.name: "my-service"
  - worker.id: "{worker_id}"
  - invocation.id: "{invocation_id}"
  - tenant.id: "{tenant_id}"
  - run.id: "{run_id}" (if workflow)
  - function.status: "success" | "error"
  - function.output_size: {bytes}
```

**Code**: `sdk-core/src/telemetry.rs:213-293`

### 2. Workflow Execution

Workflows create spans for:
- Main workflow execution
- Each step/task invocation
- State transitions

**Planned Enhancement**: Add explicit workflow step spans

### 3. LLM Calls

Comprehensive LLM telemetry with:

```
Span: llm.chat_completion | llm.completion | llm.embeddings
Attributes:
  - gen_ai.system: "openai" | "anthropic" | "groq" | ...
  - gen_ai.request.model: "{model}"
  - gen_ai.request.temperature: {temp}
  - gen_ai.request.max_tokens: {tokens}
  - gen_ai.response.id: "{completion_id}"
  - gen_ai.usage.input_tokens: {count}
  - gen_ai.usage.output_tokens: {count}
  - gen_ai.prompt.{i}.role: "{role}"
  - gen_ai.prompt.{i}.content: "{content}" (if AGNT5_TRACE_CONTENT_ENABLED)
  - gen_ai.completion.{i}.content: "{content}"

Metrics:
  - llm.request.count (counter)
  - llm.latency.ms (histogram)
  - llm.tokens.total (counter)
```

**Code**: `sdk-core/src/llm/telemetry.rs`

### 4. Structured Logging

All logs from `ctx.logger` are structured with:

```
Log Entry:
  - timestamp: {unix_timestamp}
  - level: DEBUG | INFO | WARN | ERROR
  - message: "{message}"
  - invocation.id: "{invocation_id}" (inherited from span)
  - trace_id: "{trace_id}"
  - span_id: "{span_id}"
  - python.module: "{module}"
  - python.filename: "{file}"
  - python.line: {line_number}
```

**Console Output Format**:
```
invocation.id={invocation_id} {file}:{line} {message}
```

## Python Usage

### Function with Logging

```python
from agnt5 import Context, function

@function
async def process_data(ctx: Context, data: str) -> dict:
    # All logs automatically go to OpenTelemetry + console
    ctx.logger.debug(f"Processing: {data}")
    ctx.logger.info("Starting validation")
    ctx.logger.warning("Cache miss")
    ctx.logger.error("Failed to connect")

    return {"result": data.upper()}
```

### Workflow with Logging

```python
from agnt5 import Context, workflow

@workflow
async def order_flow(ctx: Context, order_id: str) -> dict:
    ctx.logger.info(f"Starting order: {order_id}")

    # Each task gets its own span
    result = await ctx.task("service", "validate", input=order_id)
    ctx.logger.info(f"Validated: {result}")

    return result
```

### LLM with Automatic Telemetry

```python
from agnt5 import lm

# Automatically creates span + metrics
response = await lm.generate(
    model="openai/gpt-4o-mini",
    prompt="Explain AGNT5",
    temperature=0.7
)
# Span includes: model, tokens, latency, prompt, response
```

## Verifying Telemetry

### 1. Console Output

You should see logs like:

```
invocation.id=abc-123 worker.py:355 🔥 WORKER: Received request for function: my_function
invocation.id=abc-123 context.py:21 INFO: Processing message
```

### 2. OpenTelemetry Collector

Check collector logs:

```bash
# If using Docker
docker logs agnt5-otel-collector
```

Expected output:
```
2024-01-15T12:00:00.000Z info TracesExporter {"#spans": 5}
2024-01-15T12:00:00.000Z info LogsExporter {"#logs": 12}
2024-01-15T12:00:00.000Z info MetricsExporter {"#metrics": 3}
```

### 3. Observability Backend

Query your backend (Jaeger, Tempo, Loki, etc.):

```
# Jaeger (traces)
http://localhost:16686

# Grafana (logs)
{service_name="my-service"} |= "invocation.id"
```

## Troubleshooting

### Logs Not Appearing in OpenTelemetry

**Symptom**: Logs visible in console but not in backend

**Possible Causes**:
1. **OpenTelemetry Collector not running**
   ```bash
   # Check if collector is accessible
   curl http://localhost:4317
   ```

2. **Wrong endpoint configured**
   ```bash
   # Verify endpoint
   echo $OTEL_EXPORTER_OTLP_ENDPOINT
   # Should be: http://localhost:4317
   ```

3. **Collector not configured for logs**
   ```yaml
   # otel-collector-config.yaml
   receivers:
     otlp:
       protocols:
         grpc:
           endpoint: 0.0.0.0:4317

   exporters:
     logging:
       loglevel: debug

   service:
     pipelines:
       traces:
         receivers: [otlp]
         exporters: [logging]
       logs:  # ← Must be configured!
         receivers: [otlp]
         exporters: [logging]
   ```

4. **Network issues**
   ```bash
   # Test connectivity from worker container
   nc -zv localhost 4317
   ```

### Spans Not Created

**Symptom**: No traces in Jaeger/Tempo

**Possible Causes**:
1. **Telemetry not initialized**
   - Check worker startup logs for "Initializing OpenTelemetry"

2. **No parent span context**
   - For distributed tracing, ensure traceparent header is passed

3. **Batch not flushed**
   - Spans are batched, wait ~10 seconds before checking backend

### Console Logs Missing invocation.id

**Symptom**: Logs don't show invocation.id

**Check**:
1. Is `ctx.logger` being used? (not `print()` or `logging.getLogger()`)
2. Is the logger inside a function/workflow execution?
3. Check if OpenTelemetryHandler is attached:
   ```python
   print(ctx.logger.handlers)
   # Should include OpenTelemetryHandler
   ```

## Performance Considerations

### Batching

Spans and logs are batched for efficiency:
- **Default batch size**: 512 records
- **Default timeout**: 5 seconds
- **Flush on shutdown**: Automatic

### Sampling (Future)

Currently all spans are recorded. For high-volume production:
- Configure trace sampling in collector
- Or implement head-based sampling in SDK

### Overhead

Telemetry overhead is minimal:
- Span creation: ~1-5 μs
- Log forwarding: ~2-10 μs
- LLM telemetry: ~10-50 μs (includes serialization)

## Future Enhancements

### Phase 2 (Planned)

1. **Workflow Step Spans**: Explicit spans for each workflow step
2. **Entity Operation Spans**: Track entity method invocations
3. **Tool Execution Spans**: Trace tool usage in agents
4. **Custom Attributes**: User-defined span attributes via `ctx.trace`
5. **Metrics API**: Expose metrics in Context (`ctx.metrics`)

### Phase 3 (Planned)

1. **Distributed Context**: Baggage propagation for user_id, tenant_id
2. **Trace Linking**: Link workflows to parent runs
3. **Custom Samplers**: SDK-level sampling configuration
4. **Performance Profiling**: CPU/memory profiling integration
