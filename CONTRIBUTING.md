# Contributing

Thank you for contributing to the AGNT5 Python SDK.

Open an issue before making a large or compatibility-affecting change. Keep
pull requests focused, add tests for observable behavior, and update public
documentation when an API changes.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0 included in this repository.

Run the narrow Python checks before opening a pull request:

```bash
uv run pytest tests/unit -q
uv run ruff check src tests
```
