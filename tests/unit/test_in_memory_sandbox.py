"""Deterministic in-memory sandbox contract tests."""

import pytest

from agnt5 import InMemorySandbox


@pytest.mark.asyncio
async def test_file_roundtrip_list_and_delete() -> None:
    sandbox = InMemorySandbox()

    written = await sandbox.write_file("/tmp/example.txt", "hello")
    read = await sandbox.read_file("/tmp/example.txt")
    listed = await sandbox.list_files("/tmp")

    assert written.success is True
    assert written.size == 5
    assert read.content == b"hello"
    assert [file.path for file in listed.files] == ["/tmp/example.txt"]
    assert await sandbox.delete_file("/tmp/example.txt") is True
    assert (await sandbox.list_files("/tmp")).total == 0


@pytest.mark.asyncio
async def test_missing_file_raises() -> None:
    sandbox = InMemorySandbox()

    with pytest.raises(FileNotFoundError, match="sandbox file not found"):
        await sandbox.read_file("/tmp/missing.txt")


@pytest.mark.asyncio
async def test_execution_is_explicitly_deterministic() -> None:
    sandbox = InMemorySandbox()

    result = await sandbox.execute_code("print('hello')", language="python")

    assert result.exit_code == 0
    assert result.stdout == "[python] print('hello')"
