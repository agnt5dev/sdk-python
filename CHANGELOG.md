# Changelog

All notable changes to the AGNT5 Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-09-03

### Changed

- Preserve provider-specific tool-call data across native prompt, response,
  and streaming conversions, and update the Google API compatibility matrix.
- Build the native extension against `agnt5-sdk-core` 0.2.5.

### Fixed

- Isolate durable step hierarchy state between concurrent asyncio tasks while
  retaining inherited nesting within each task.
- Complete the router iteration and source agent lifecycles after a delegated
  handoff settles.
- Serialize durable sleep continuations with the SDK serializer so typed step
  outputs such as Pydantic models remain resumable.

## [0.11.0b5] - 2026-09-02

### Fixed

- Preserve the runtime-authored assignment commit offset on lifecycle records
  so append-time lease fencing can bridge projection lag immediately after a
  pull claim.

## [0.11.0b4] - 2026-09-02

### Changed

- Durable activations are now the step boundary records. For every call that
  goes through the activation RPCs (workflow steps and timers, model calls,
  durable tools, delegated child agents) the SDK no longer emits its own
  `workflow.step.*`, `lm.*`, `tool_call.*`, or `agent.*` lifecycle events and
  instead supplies `display_name` and a bounded JSON `input_data` (64 KiB,
  truncation marker beyond that) on `BeginActivation`, `cached_tokens` on the
  completion usage, and a measured `latency_ms` on `FailActivation`. The
  runtime journals one kind-named record per side keyed by the activation id;
  a REPLAY appends nothing.
- While an activation executes it is the current activation and the ambient
  parent correlation id, so nested `function.*` events, stream deltas, and
  logs parent to the journaled record. Stream consumers still receive the
  in-memory `lm.*` and `tool_call.*` events under the activation id.
- Legacy (non-durable) paths, HITL resume, and executor-managed top-level
  `agent.*` lifecycle are unchanged.

## [0.11.0b3] - 2026-08-26

### Fixed

- Update the native extension to `agnt5-sdk-core` 0.2.3 so token-auth
  customer-hosted workers configure verified TLS for discovered HTTPS runtime
  endpoints, including coordinator reconnects and engine connections.

## [0.11.0b2] - 2026-08-26

### Fixed

- Update the native extension to `agnt5-sdk-core` 0.2.2 so customer-hosted
  workers preserve discovered project authority across reconnects and honor a
  configured `SSL_CERT_FILE` CA bundle without weakening TLS verification.

## [0.11.0b1] - 2026-08-24

### Changed

- Build Linux wheels for both `manylinux_2_28` and `manylinux_2_39`, verify
  installation on a `manylinux_2_34` baseline, and publish a source
  distribution alongside the wheels.
- Update the native extension to `agnt5-sdk-core` 0.2.1.

### Fixed

- Preserve the durable activation client when `WorkflowContext.step()` invokes
  a decorated function so nested model calls remain durable.

## [0.11.0b0] - 2026-08-12

### Added

- Capture installed OpenAI, OpenAI Agents SDK, and Google ADK calls made
  inside AGNT5 components without requiring application-level instrumentation.
- Emit correlated `agent.*`, `lm.*`, and `tool_call.*` journal events with
  provider, model, token, `source`, and `capture_mode=observed` metadata.
- Add optional dependency groups for the supported OpenAI, OpenAI Agents SDK,
  and Google ADK version bands, including Google ADK Python 1.7 and newer.

### Changed

- Auto-enable available capture integrations at worker and serverless startup
  while keeping missing or disabled third-party libraries as no-ops.
- Preserve provider behavior when capture fails and suppress duplicate raw
  OpenAI events inside OpenAI Agents SDK model spans.

## [0.10.0] - 2026-08-08

### Added

- Add the durable activation V1 contract for fenced checkpoint, function,
  tool, model, and delegated-agent execution.
- Add durable workflow sleeps, invocation idempotency keys, replay-safe model
  finals, and required-child recovery.

### Changed

- Build the native extension against `agnt5-sdk-core` 0.2.0 and enable durable
  activation V1 in default package builds.
- Keep worker lifecycle emission asynchronous and run synchronous workflow
  handlers outside the Python event loop.

### Fixed

- Fail closed when durable checkpoints cannot be acknowledged, preserve
  execution and wait authority across activation boundaries, and wait for
  durably detached runs to be accepted.

## [0.9.8] - 2026-08-04

### Fixed

- Let the runtime own gateway-managed agent session history so workers do not
  reload or persist the same conversation through legacy entity storage.
- Exclude the unresolved assistant tool call that triggered an agent handoff
  before passing conversation history to the target agent.
- Make the worker entity-state adapter available while constructing execution
  contexts.

### Changed

- Align API compatibility and integration tests with the current LM streaming
  event shape, runtime endpoints, journal event names, and package imports.

## [0.9.7] - 2026-07-31

### Added

- Expose explicitly selected functions, workflows, tools, and agents through
  Python serverless endpoints.
- Add raw WSGI, Starlette, Flask, and Django adapters alongside the existing
  ASGI and FastAPI integrations.
- Checkpoint agent session history and return function, tool, and agent events
  through workerless responses.

### Fixed

- Preserve registry fallback when component lists are omitted while honoring
  explicit empty lists as exposing no components.

## [0.9.6] - 2026-07-30

### Fixed

- Update the native extension to `agnt5-sdk-core` 0.1.6 so Python receives
  Gemini tool-call parsing and expanded Amazon Bedrock provider support.

## [0.9.5] - 2026-07-29

### Changed

- Update the native extension to `agnt5-sdk-core` 0.1.5 so Python remains
  aligned with the shared agent terminal-output contract.

## [0.9.4] - 2026-07-29

### Fixed

- Stream OpenAI agent responses incrementally when tools are registered while
  preserving completed tool calls across agent iterations.

## [0.9.3] - 2026-07-26

### Added

- Add schema-aware scoped state adapters and agent streaming tool coverage.

### Fixed

- Preserve deterministic event ordering and timeout behavior across concurrent
  agent, workflow, and worker execution.

## [0.9.2] - 2026-07-24

### Added

- Pull workers now complete jobs through the runtime's typed lease, session,
  and attempt fence.
- Pull workflow pause responses retain checkpoint and resume metadata without
  emitting an unfenced terminal event.

### Fixed

- Hosted Agent and nested Workflow executors now forward LM content and tool events to streaming clients.
- Failed LM streams emit exactly one durable `lm.failed` lifecycle event.
- Current `lm.content_block.*` events use the transient streaming path instead of durable checkpoints.
- Push and pull dispatch metadata can no longer override the runtime-provided
  dispatch mode or lease authority.
- Pull completion waits for queued lifecycle events to flush before
  acknowledging the job.

## [0.9.1] - 2026-07-20

### Added

- Standalone GitHub-hosted release builds for Linux x64, Linux ARM64, and macOS ARM64.
- PyPI publishing for the standalone `agnt5dev/sdk-python` repository.
- Published `agnt5-sdk-core` crate dependency for the native Python extension.

[Unreleased]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0...HEAD
[0.11.0]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0b5...v0.11.0
[0.11.0b5]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0b4...v0.11.0b5
[0.11.0b4]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0b3...v0.11.0b4
[0.11.0b3]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0b2...v0.11.0b3
[0.11.0b2]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0b1...v0.11.0b2
[0.11.0b1]: https://github.com/agnt5dev/sdk-python/compare/v0.11.0b0...v0.11.0b1
[0.11.0b0]: https://github.com/agnt5dev/sdk-python/compare/v0.10.0...v0.11.0b0
[0.10.0]: https://github.com/agnt5dev/sdk-python/compare/v0.9.8...v0.10.0
[0.9.8]: https://github.com/agnt5dev/sdk-python/compare/v0.9.7...v0.9.8
[0.9.7]: https://github.com/agnt5dev/sdk-python/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/agnt5dev/sdk-python/compare/v0.9.5...v0.9.6
[0.9.5]: https://github.com/agnt5dev/sdk-python/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/agnt5dev/sdk-python/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/agnt5dev/sdk-python/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/agnt5dev/sdk-python/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/agnt5dev/sdk-python/releases/tag/v0.9.1
