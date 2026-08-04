# Changelog

All notable changes to the AGNT5 Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/agnt5dev/sdk-python/compare/v0.9.8...HEAD
[0.9.8]: https://github.com/agnt5dev/sdk-python/compare/v0.9.7...v0.9.8
[0.9.7]: https://github.com/agnt5dev/sdk-python/compare/v0.9.6...v0.9.7
[0.9.6]: https://github.com/agnt5dev/sdk-python/compare/v0.9.5...v0.9.6
[0.9.5]: https://github.com/agnt5dev/sdk-python/compare/v0.9.4...v0.9.5
[0.9.4]: https://github.com/agnt5dev/sdk-python/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/agnt5dev/sdk-python/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/agnt5dev/sdk-python/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/agnt5dev/sdk-python/releases/tag/v0.9.1
