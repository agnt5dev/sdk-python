# Changelog

All notable changes to the AGNT5 Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2] - 2026-07-24

### Fixed

- Hosted Agent and nested Workflow executors now forward LM content and tool events to streaming clients.
- Failed LM streams emit exactly one durable `lm.failed` lifecycle event.
- Current `lm.content_block.*` events use the transient streaming path instead of durable checkpoints.

## [0.9.1] - 2026-07-20

### Added

- Standalone GitHub-hosted release builds for Linux x64, Linux ARM64, and macOS ARM64.
- PyPI publishing for the standalone `agnt5dev/sdk-python` repository.
- Published `agnt5-sdk-core` crate dependency for the native Python extension.

[Unreleased]: https://github.com/agnt5dev/sdk-python/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/agnt5dev/sdk-python/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/agnt5dev/sdk-python/releases/tag/v0.9.1
