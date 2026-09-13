# AGNT5 Python SDK

[![CI](https://github.com/agnt5dev/agnt5/actions/workflows/sdk-python-tests.yml/badge.svg)](https://github.com/agnt5dev/agnt5/actions/workflows/sdk-python-tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Build reliable AI agents and durable workflows in Python. The AGNT5 SDK
provides typed components, workflow checkpoints, retries, streaming, tools,
human-in-the-loop coordination, evaluation, and runtime observability.

## Requirements

- Python 3.11 or newer
- An AGNT5 runtime for deployed execution

## Installation

```bash
pip install agnt5
```

## Quick start

Define a function. Type hints are used to derive its input and output schema.

```python
from agnt5 import FunctionContext, function


@function(retries=3, backoff="exponential")
async def greet(ctx: FunctionContext, name: str) -> dict[str, str]:
    ctx.logger.info("greeting user", extra={"name": name})
    return {"message": f"Hello, {name}!"}
```

Register application components with a worker for runtime-backed execution:

```python
import asyncio

from agnt5 import Worker


async def main() -> None:
    worker = Worker(service_name="hello-python")
    await worker.run()


asyncio.run(main())
```

Decorated functions, workflows, agents, tools, and scorers are registered when
their modules are imported. See [`examples/app.py`](examples/app.py) for a
complete application entrypoint.

## Invoke a deployed component

```python
from agnt5 import Client

client = Client(
    gateway_url="https://gw.agnt5.com",
    api_key="agnt5_sk_...",
    deployment_id="deployment-id",
)

result = client.run("greet", {"name": "Ada"})
print(result)
```

`Client` also supports asynchronous submission, status and result polling,
streaming events, batches, cancellation, workflow resume, chat, and evaluation.
Configuration can be supplied explicitly or through `AGNT5_GATEWAY_URL`,
`AGNT5_API_KEY`, and `AGNT5_DEPLOYMENT_ID`.

## Core APIs

| API | Purpose |
| --- | --- |
| `@function` | Typed, retryable units of work |
| `@workflow` | Durable multi-step orchestration and checkpointing |
| `Agent` and `@agent` | Model and tool orchestration |
| `@tool` | Typed tools with generated schemas |
| `ctx.state` | Durable workflow and component state |
| `ctx.memory` | Conversation and application memory |
| `Client` / `AsyncClient` | Invoke and observe deployed components |
| `Worker` | Register components and serve runtime dispatch |

In async workflows, use `await ctx.state.set_async(key, value)` and
`await ctx.state.delete_async(key)`. These await the runtime's durable
acknowledgment while allowing other workflows to run. Reads remain local via
`ctx.state.get(key)`. The existing synchronous `set` and `delete` methods remain
supported, but block their calling thread while waiting for persistence.

The shared Rust runtime foundation lives in
[`agnt5dev/sdk-core`](https://github.com/agnt5dev/sdk-core). Vendor sandbox
adapters live in
[`agnt5dev/sdk-integrations`](https://github.com/agnt5dev/sdk-integrations).

## Examples and documentation

- [`examples/`](examples/) contains functions, workflows, agents, tools,
  streaming, HITL, MCP, chat, and evaluation examples.
- [`docs/`](docs/) contains SDK-specific guides.
- [AGNT5 documentation](https://agnt5.com/docs) covers platform concepts and
  deployment.

## Development

```bash
uv sync --all-groups
uv run ruff check src tests
uv run pytest tests/unit -q
uv build --sdist
```

Development builds use a sibling checkout of
[`sdk-core`](https://github.com/agnt5dev/sdk-core).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues according to
[SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).

## Response wait

Run and stream calls wait up to **5 minutes** by default. Set the per-call wait
to any value from zero to 24 hours. Zero returns a pending receipt immediately
after acceptance. This controls response waiting, not the workflow execution
deadline: accepted work continues when the wait expires or the client disconnects.

```python
result = client.run("process_order", order, component_type="workflow", wait_timeout=60)
for event in client.stream_events("process_order", order, component_type="workflow", wait_timeout=60):
    print(event.event_type, event.run_id)
```

`wait_timeout` uses seconds and also applies to async `run` and `stream_events`.
Run calls return `202` pending receipts directly, without additional polling.
Event streams emit `stream.wait_expired` when the wait expires, or
`stream.detached` for a `202` receipt. Use the run ID to read status/results.
Chunk-only `stream` raises `RunError` with the run ID when waiting ends.

The default HTTP timeout allows at least the wait plus 10 seconds, or the client
timeout if longer. Pass `timeout=75` to set it explicitly. Python's HTTP timeout
limits individual network operations; `wait_timeout` bounds the gateway wait.
