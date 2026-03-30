"""
Pre-built sandbox tools for agent use.

Provides a set of tools that give agents the ability to execute code
and manage files in an isolated sandbox workspace. The sandbox persists
across tool calls within an agent run, so agents can build up a workspace
incrementally (write files, run code, read results).

Usage:
    ```python
    from agnt5 import Agent, sandbox_tools

    # With default sandbox (auto-selects backend)
    agent = Agent(
        name="coder",
        model="openai/gpt-4o-mini",
        instructions="You can write and run code to solve problems.",
        tools=sandbox_tools(),
    )

    result = await agent.run("Calculate the 20th fibonacci number")

    # With explicit sandbox configuration
    from agnt5 import Sandbox
    sandbox = Sandbox(sandbox_id="my-sandbox", http_endpoint="http://localhost:4001")
    agent = Agent(
        name="coder",
        model="openai/gpt-4o-mini",
        tools=sandbox_tools(sandbox=sandbox),
    )
    ```
"""

import secrets
from typing import List, Optional

from .context import Context
from .sandbox import Sandbox
from .sandbox_events import (
    SandboxExecuteCompleted,
    SandboxExecuteFailed,
    SandboxExecuteStarted,
    SandboxFileRead,
    SandboxFileWritten,
)
from .tool import Tool


def sandbox_tools(sandbox: Optional[Sandbox] = None) -> List[Tool]:
    """Create sandbox tools for agent use.

    Returns four tools that agents can invoke:
    - ``sandbox_execute_code``: Execute code (JS, Python, Bash)
    - ``sandbox_write_file``: Write a file to the workspace
    - ``sandbox_read_file``: Read a file from the workspace
    - ``sandbox_list_files``: List files in the workspace

    All tools share the same sandbox instance, so the workspace persists
    across calls within an agent run.

    Args:
        sandbox: Sandbox instance to use. If None, must be set later via
            the returned tools' ``set_sandbox`` class method, or the
            agent context must provide one at ``ctx.sandbox``.

    Returns:
        List of Tool instances to pass to an Agent's ``tools`` parameter.
    """
    # Shared state: the sandbox is captured in the closure so all tools
    # operate on the same workspace.
    state = _SandboxToolState(sandbox)

    execute_code = Tool(
        name="sandbox_execute_code",
        description="Execute code in a sandboxed environment. Returns stdout, stderr, and exit code.",
        handler=_make_execute_code_handler(state),
        auto_schema=True,
    )

    write_file = Tool(
        name="sandbox_write_file",
        description="Write content to a file in the sandbox workspace. Creates parent directories automatically.",
        handler=_make_write_file_handler(state),
        auto_schema=True,
    )

    read_file = Tool(
        name="sandbox_read_file",
        description="Read the contents of a file from the sandbox workspace.",
        handler=_make_read_file_handler(state),
        auto_schema=True,
    )

    list_files = Tool(
        name="sandbox_list_files",
        description="List files and directories in the sandbox workspace.",
        handler=_make_list_files_handler(state),
        auto_schema=True,
    )

    return [execute_code, write_file, read_file, list_files]


class _SandboxToolState:
    """Shared state for sandbox tools within a single agent run."""

    def __init__(self, sandbox: Optional[Sandbox] = None):
        self.sandbox = sandbox

    def get_sandbox(self, ctx: Context) -> Sandbox:
        """Get the sandbox instance, falling back to ctx.sandbox if available."""
        if self.sandbox is not None:
            return self.sandbox
        # Try to get sandbox from the agent context
        if hasattr(ctx, "sandbox") and ctx.sandbox is not None:
            return ctx.sandbox
        raise RuntimeError(
            "No sandbox available. Either pass a Sandbox instance to sandbox_tools(), "
            "or ensure ctx.sandbox is set in the agent context."
        )


def _make_execute_code_handler(state: _SandboxToolState):
    async def sandbox_execute_code(ctx: Context, code: str, language: str = "javascript") -> dict:
        """Execute code in a sandboxed environment.

        Args:
            code: Source code to execute.
            language: Programming language — "javascript", "python", or "bash".
        """
        sandbox = state.get_sandbox(ctx)
        cid = f"sandbox-{secrets.token_hex(5)}"

        ctx.emit(SandboxExecuteStarted(
            name="sandbox_execute_code",
            correlation_id=cid,
            parent_correlation_id=ctx.correlation_id,
            language=language,
            code_length=len(code),
            input_data={"language": language, "code_length": len(code)},
        ))

        try:
            result = await sandbox.execute_code(code, language=language)
            ctx.emit(SandboxExecuteCompleted(
                name="sandbox_execute_code",
                correlation_id=cid,
                parent_correlation_id=ctx.correlation_id,
                language=language,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,
                output_data={
                    "exit_code": result.exit_code,
                    "stdout_length": len(result.stdout),
                    "execution_time_ms": result.execution_time_ms,
                },
            ))
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "execution_time_ms": result.execution_time_ms,
            }
        except Exception as e:
            ctx.emit(SandboxExecuteFailed(
                name="sandbox_execute_code",
                correlation_id=cid,
                parent_correlation_id=ctx.correlation_id,
                language=language,
                error_code="SANDBOX_EXECUTION_FAILED",
                error_message=str(e),
            ))
            raise

    return sandbox_execute_code


def _make_write_file_handler(state: _SandboxToolState):
    async def sandbox_write_file(ctx: Context, path: str, content: str) -> dict:
        """Write content to a file in the sandbox workspace.

        Args:
            path: File path relative to workspace root.
            content: Text content to write.
        """
        sandbox = state.get_sandbox(ctx)
        result = await sandbox.write_file(path, content.encode("utf-8"))
        ctx.emit(SandboxFileWritten(
            name="sandbox_write_file",
            correlation_id=f"sandbox-{secrets.token_hex(5)}",
            parent_correlation_id=ctx.correlation_id,
            file_path=path,
            file_size=result.size,
            output_data={"path": result.path, "size": result.size},
        ))
        return {
            "success": result.success,
            "path": result.path,
            "size": result.size,
        }

    return sandbox_write_file


def _make_read_file_handler(state: _SandboxToolState):
    async def sandbox_read_file(ctx: Context, path: str) -> dict:
        """Read the contents of a file from the sandbox workspace.

        Args:
            path: File path relative to workspace root.
        """
        sandbox = state.get_sandbox(ctx)
        result = await sandbox.read_file(path)
        ctx.emit(SandboxFileRead(
            name="sandbox_read_file",
            correlation_id=f"sandbox-{secrets.token_hex(5)}",
            parent_correlation_id=ctx.correlation_id,
            file_path=path,
            file_size=result.size,
            output_data={"path": result.path, "size": result.size},
        ))
        return {
            "path": result.path,
            "content": result.content.decode("utf-8", errors="replace"),
            "size": result.size,
            "is_dir": result.is_dir,
        }

    return sandbox_read_file


def _make_list_files_handler(state: _SandboxToolState):
    async def sandbox_list_files(ctx: Context, path: str = ".", recursive: bool = False) -> dict:
        """List files and directories in the sandbox workspace.

        Args:
            path: Directory path to list. Defaults to workspace root.
            recursive: If true, list files recursively including subdirectories.
        """
        sandbox = state.get_sandbox(ctx)
        result = await sandbox.list_files(path, recursive=recursive)
        return {
            "path": result.path,
            "total": result.total,
            "files": [
                {
                    "name": f.name,
                    "path": f.path,
                    "size": f.size,
                    "is_dir": f.is_dir,
                }
                for f in result.files
            ],
        }

    return sandbox_list_files
