"""AGNT5 Sandbox SDK for ephemeral code execution environments.

Sandboxes provide isolated execution environments for running code, commands,
and file operations. They are pooled (pre-warmed) and can be claimed for
exclusive use during a workflow execution.

Example usage:
    ```python
    from agnt5 import Sandbox, SandboxPool

    # Claim a sandbox from a pool
    async with SandboxPool(project_id="...", pool_id="...") as pool:
        sandbox = await pool.claim(run_id="...")

        # Execute code
        result = await sandbox.execute_code("print('Hello!')", language="python")
        print(result.stdout)

        # Run a command
        result = await sandbox.run_command("ls", args=["-la"])
        print(result.stdout)

        # Write and read files
        await sandbox.write_file("/workspace/test.txt", b"Hello, World!")
        content = await sandbox.read_file("/workspace/test.txt")
    ```
"""

import base64
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx


# ============= Response Types =============


@dataclass
class ExecuteCodeResult:
    """Result of code execution."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    error: Optional[str] = None


@dataclass
class RunCommandResult:
    """Result of running a command."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    error: Optional[str] = None


@dataclass
class WriteFileResult:
    """Result of writing a file."""

    success: bool
    path: str
    size: int = 0
    error: Optional[str] = None


@dataclass
class ReadFileResult:
    """Result of reading a file."""

    path: str
    content: bytes
    size: int
    mode: int
    is_dir: bool
    error: Optional[str] = None


@dataclass
class FileInfo:
    """Information about a file."""

    name: str
    path: str
    size: int
    mode: int
    is_dir: bool
    mod_time: int  # Unix milliseconds


@dataclass
class ListFilesResult:
    """Result of listing files."""

    path: str
    files: List[FileInfo]
    total: int
    error: Optional[str] = None


@dataclass
class GitCloneResult:
    """Result of git clone operation."""

    success: bool
    path: str = ""
    branch: str = ""
    commit_sha: str = ""
    error: Optional[str] = None


@dataclass
class GitStatusFile:
    """Status of a file in git."""

    path: str
    status: str  # modified, added, deleted, untracked, renamed
    staged: bool


@dataclass
class GitStatusResult:
    """Result of git status operation."""

    branch: str
    commit_sha: str
    is_clean: bool
    ahead: int
    behind: int
    files: List[GitStatusFile]
    error: Optional[str] = None


@dataclass
class GitCommitResult:
    """Result of git commit operation."""

    success: bool
    commit_sha: str = ""
    error: Optional[str] = None


@dataclass
class GitPushResult:
    """Result of git push operation."""

    success: bool
    error: Optional[str] = None


@dataclass
class SandboxHealthResult:
    """Result of sandbox health check."""

    status: str
    sandbox_id: str
    uptime_ms: int
    error: Optional[str] = None


@dataclass
class StreamEvent:
    """Event from streaming command execution."""

    type: str  # "stdout", "stderr", "exit", "error"
    data: str
    time: int  # Unix milliseconds


# ============= Main Classes =============


class Sandbox:
    """HTTP client for sandbox operations.

    Provides methods for executing code, running commands, and managing files
    in an isolated sandbox environment.

    Args:
        sandbox_id: Unique identifier for this sandbox.
        http_endpoint: HTTP endpoint for the sandbox (e.g., "10.0.1.5:4001").
        timeout: Default timeout in seconds for operations.
    """

    def __init__(
        self,
        sandbox_id: str,
        http_endpoint: str,
        timeout: float = 300.0,
    ):
        self.sandbox_id = sandbox_id
        self.http_endpoint = http_endpoint
        self.timeout = timeout

        # Ensure endpoint has protocol
        if not http_endpoint.startswith(("http://", "https://")):
            http_endpoint = f"http://{http_endpoint}"

        self.base_url = http_endpoint
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "Sandbox":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    # ============= Tier 1: Core Operations =============

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout_ms: int = 30000,
        env: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
    ) -> ExecuteCodeResult:
        """Execute code in the sandbox.

        Args:
            code: Code to execute.
            language: Programming language (python, javascript, bash).
            timeout_ms: Execution timeout in milliseconds.
            env: Environment variables to set.
            work_dir: Working directory for execution.

        Returns:
            ExecuteCodeResult with stdout, stderr, exit_code.
        """
        payload: Dict[str, Any] = {
            "code": code,
            "language": language,
            "timeout_ms": timeout_ms,
        }
        if env:
            payload["env"] = env
        if work_dir:
            payload["work_dir"] = work_dir

        resp = await self._client.post(f"{self.base_url}/execute", json=payload)
        data = resp.json()

        return ExecuteCodeResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", 0),
            execution_time_ms=data.get("execution_time_ms", 0),
            error=data.get("error"),
        )

    async def run_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
    ) -> RunCommandResult:
        """Run a shell command in the sandbox.

        Args:
            command: Command to run.
            args: Command arguments.
            working_dir: Working directory.
            env: Environment variables.
            timeout_ms: Execution timeout in milliseconds.

        Returns:
            RunCommandResult with stdout, stderr, exit_code.
        """
        payload: Dict[str, Any] = {
            "command": command,
            "timeout_ms": timeout_ms,
        }
        if args:
            payload["args"] = args
        if working_dir:
            payload["working_dir"] = working_dir
        if env:
            payload["env"] = env

        resp = await self._client.post(f"{self.base_url}/command", json=payload)
        data = resp.json()

        return RunCommandResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", 0),
            execution_time_ms=data.get("execution_time_ms", 0),
            error=data.get("error"),
        )

    async def run_command_stream(
        self,
        command: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
    ) -> AsyncIterator[StreamEvent]:
        """Run a command with streaming output via SSE.

        Args:
            command: Command to run.
            args: Command arguments.
            working_dir: Working directory.
            env: Environment variables.
            timeout_ms: Execution timeout in milliseconds.

        Yields:
            StreamEvent objects for stdout, stderr, and completion.
        """
        import json

        payload: Dict[str, Any] = {
            "command": command,
            "timeout_ms": timeout_ms,
        }
        if args:
            payload["args"] = args
        if working_dir:
            payload["working_dir"] = working_dir
        if env:
            payload["env"] = env

        async with self._client.stream(
            "POST", f"{self.base_url}/command/stream", json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield StreamEvent(
                        type=data.get("type", ""),
                        data=data.get("data", ""),
                        time=data.get("time", 0),
                    )

    async def write_file(
        self,
        path: str,
        content: Union[bytes, str],
        mode: int = 0o644,
    ) -> WriteFileResult:
        """Write a file to the sandbox.

        Args:
            path: File path (relative to workspace or absolute).
            content: File content (bytes or string).
            mode: File mode/permissions.

        Returns:
            WriteFileResult indicating success.
        """
        if isinstance(content, str):
            content_str = content
            is_b64 = False
        else:
            content_str = base64.b64encode(content).decode()
            is_b64 = True

        payload = {
            "path": path,
            "content": content_str,
            "mode": mode,
            "is_base64": is_b64,
        }

        resp = await self._client.post(f"{self.base_url}/files", json=payload)
        data = resp.json()

        return WriteFileResult(
            success=data.get("success", False),
            path=data.get("path", path),
            size=data.get("size", 0),
            error=data.get("error"),
        )

    async def read_file(self, path: str) -> ReadFileResult:
        """Read a file from the sandbox.

        Args:
            path: File path to read.

        Returns:
            ReadFileResult with file content.
        """
        resp = await self._client.get(f"{self.base_url}/files", params={"path": path})
        data = resp.json()

        content_str = data.get("content", "")
        is_b64 = data.get("is_base64", False)

        if is_b64:
            content = base64.b64decode(content_str)
        else:
            content = content_str.encode()

        return ReadFileResult(
            path=data.get("path", path),
            content=content,
            size=data.get("size", 0),
            mode=data.get("mode", 0),
            is_dir=data.get("is_dir", False),
            error=data.get("error"),
        )

    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        """Delete a file or directory from the sandbox.

        Args:
            path: Path to delete.
            recursive: If True, delete directories recursively.

        Returns:
            True if deletion was successful.
        """
        resp = await self._client.delete(
            f"{self.base_url}/files",
            params={"path": path, "recursive": str(recursive).lower()},
        )
        data = resp.json()
        return data.get("success", False)

    async def list_files(
        self, path: str = ".", recursive: bool = False
    ) -> ListFilesResult:
        """List files in a directory.

        Args:
            path: Directory path to list.
            recursive: If True, list recursively.

        Returns:
            ListFilesResult with file information.
        """
        resp = await self._client.get(
            f"{self.base_url}/files/list",
            params={"path": path, "recursive": str(recursive).lower()},
        )
        data = resp.json()

        files = [
            FileInfo(
                name=f.get("name", ""),
                path=f.get("path", ""),
                size=f.get("size", 0),
                mode=f.get("mode", 0),
                is_dir=f.get("is_dir", False),
                mod_time=f.get("mod_time", 0),
            )
            for f in data.get("files", [])
        ]

        return ListFilesResult(
            path=data.get("path", path),
            files=files,
            total=data.get("total", len(files)),
            error=data.get("error"),
        )

    async def get_preview_url(self, port: int = 3000) -> str:
        """Get the preview URL for a web server running in the sandbox.

        Args:
            port: Port number of the web server.

        Returns:
            Preview URL string.
        """
        resp = await self._client.get(
            f"{self.base_url}/preview-url", params={"port": port}
        )
        data = resp.json()
        return data.get("url", "")

    # ============= Tier 2: Extended Operations =============

    async def set_timeout(self, timeout_ms: int) -> bool:
        """Set the default execution timeout.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            True if successful.
        """
        resp = await self._client.post(
            f"{self.base_url}/timeout", json={"timeout_ms": timeout_ms}
        )
        data = resp.json()
        return data.get("success", False)

    async def git_clone(
        self,
        url: str,
        branch: Optional[str] = None,
        target_dir: Optional[str] = None,
        depth: int = 0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssh_key: Optional[str] = None,
    ) -> GitCloneResult:
        """Clone a git repository.

        Args:
            url: Repository URL.
            branch: Branch to checkout.
            target_dir: Target directory for clone.
            depth: Shallow clone depth (0 for full clone).
            username: Auth username (for HTTPS).
            password: Auth password or token (for HTTPS).
            ssh_key: Base64-encoded SSH private key.

        Returns:
            GitCloneResult with clone details.
        """
        payload: Dict[str, Any] = {"url": url}
        if branch:
            payload["branch"] = branch
        if target_dir:
            payload["target_dir"] = target_dir
        if depth > 0:
            payload["depth"] = depth
        if username:
            payload["username"] = username
        if password:
            payload["password"] = password
        if ssh_key:
            payload["ssh_key"] = ssh_key

        resp = await self._client.post(f"{self.base_url}/git/clone", json=payload)
        data = resp.json()

        return GitCloneResult(
            success=data.get("success", False),
            path=data.get("path", ""),
            branch=data.get("branch", ""),
            commit_sha=data.get("commit_sha", ""),
            error=data.get("error"),
        )

    async def git_status(self, path: Optional[str] = None) -> GitStatusResult:
        """Get git status for a repository.

        Args:
            path: Repository path (optional).

        Returns:
            GitStatusResult with status details.
        """
        params = {"path": path} if path else {}
        resp = await self._client.get(f"{self.base_url}/git/status", params=params)
        data = resp.json()

        files = [
            GitStatusFile(
                path=f.get("path", ""),
                status=f.get("status", ""),
                staged=f.get("staged", False),
            )
            for f in data.get("files", [])
        ]

        return GitStatusResult(
            branch=data.get("branch", ""),
            commit_sha=data.get("commit_sha", ""),
            is_clean=data.get("is_clean", True),
            ahead=data.get("ahead", 0),
            behind=data.get("behind", 0),
            files=files,
            error=data.get("error"),
        )

    async def git_commit(
        self,
        message: str,
        path: Optional[str] = None,
        files: Optional[List[str]] = None,
        all: bool = False,
        author: Optional[str] = None,
    ) -> GitCommitResult:
        """Create a git commit.

        Args:
            message: Commit message.
            path: Repository path.
            files: Specific files to commit.
            all: Stage all changes before commit.
            author: Author string ("Name <email>").

        Returns:
            GitCommitResult with commit SHA.
        """
        payload: Dict[str, Any] = {"message": message}
        if path:
            payload["path"] = path
        if files:
            payload["files"] = files
        if all:
            payload["all"] = all
        if author:
            payload["author"] = author

        resp = await self._client.post(f"{self.base_url}/git/commit", json=payload)
        data = resp.json()

        return GitCommitResult(
            success=data.get("success", False),
            commit_sha=data.get("commit_sha", ""),
            error=data.get("error"),
        )

    async def git_push(
        self,
        path: Optional[str] = None,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ssh_key: Optional[str] = None,
    ) -> GitPushResult:
        """Push commits to remote.

        Args:
            path: Repository path.
            remote: Remote name.
            branch: Branch to push.
            force: Force push.
            username: Auth username.
            password: Auth password or token.
            ssh_key: Base64-encoded SSH private key.

        Returns:
            GitPushResult indicating success.
        """
        payload: Dict[str, Any] = {"remote": remote}
        if path:
            payload["path"] = path
        if branch:
            payload["branch"] = branch
        if force:
            payload["force"] = force
        if username:
            payload["username"] = username
        if password:
            payload["password"] = password
        if ssh_key:
            payload["ssh_key"] = ssh_key

        resp = await self._client.post(f"{self.base_url}/git/push", json=payload)
        data = resp.json()

        return GitPushResult(
            success=data.get("success", False),
            error=data.get("error"),
        )

    async def health(self) -> SandboxHealthResult:
        """Check sandbox health.

        Returns:
            SandboxHealthResult with health status.
        """
        resp = await self._client.get(f"{self.base_url}/health")
        data = resp.json()

        return SandboxHealthResult(
            status=data.get("status", "unknown"),
            sandbox_id=data.get("sandbox_id", self.sandbox_id),
            uptime_ms=data.get("uptime_ms", 0),
            error=data.get("error"),
        )


class SandboxPool:
    """Manager for claiming and releasing sandboxes from a pool.

    Provides async context manager support for automatic cleanup.

    Args:
        coordinator_endpoint: Endpoint for the coordinator service.
        project_id: Project ID for the sandbox pool.
        pool_id: Pool ID to claim sandboxes from.
        timeout: Default timeout for operations.
    """

    def __init__(
        self,
        coordinator_endpoint: str,
        project_id: str,
        pool_id: str,
        timeout: float = 30.0,
    ):
        self.coordinator_endpoint = coordinator_endpoint
        self.project_id = project_id
        self.pool_id = pool_id
        self.timeout = timeout

        if not coordinator_endpoint.startswith(("http://", "https://")):
            coordinator_endpoint = f"http://{coordinator_endpoint}"

        self.base_url = coordinator_endpoint
        self._client = httpx.AsyncClient(timeout=timeout)
        self._claimed_sandbox: Optional[Sandbox] = None

    async def __aenter__(self) -> "SandboxPool":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._claimed_sandbox:
            await self.release(self._claimed_sandbox)
        await self._client.aclose()

    async def claim(self, run_id: str) -> Sandbox:
        """Claim a sandbox from the pool.

        Args:
            run_id: Run ID claiming this sandbox.

        Returns:
            Sandbox instance ready for use.

        Raises:
            RuntimeError: If no sandbox is available.
        """
        # Call coordinator to claim a sandbox
        resp = await self._client.post(
            f"{self.base_url}/v1/sandboxes/claim",
            json={
                "project_id": self.project_id,
                "pool_id": self.pool_id,
                "claimed_by": run_id,
            },
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to claim sandbox: {resp.text}")

        data = resp.json()
        sandbox_id = data.get("sandbox_id")
        http_endpoint = data.get("http_endpoint")

        if not sandbox_id or not http_endpoint:
            raise RuntimeError("No sandbox available in pool")

        sandbox = Sandbox(
            sandbox_id=sandbox_id,
            http_endpoint=http_endpoint,
            timeout=self.timeout,
        )
        self._claimed_sandbox = sandbox
        return sandbox

    async def release(self, sandbox: Sandbox) -> None:
        """Release a claimed sandbox back to the pool.

        Args:
            sandbox: Sandbox to release.
        """
        await self._client.post(
            f"{self.base_url}/v1/sandboxes/release",
            json={
                "sandbox_id": sandbox.sandbox_id,
            },
        )
        await sandbox.close()
        if self._claimed_sandbox == sandbox:
            self._claimed_sandbox = None

    async def list_sandboxes(self) -> List[Dict[str, Any]]:
        """List all sandboxes in the pool.

        Returns:
            List of sandbox information dictionaries.
        """
        resp = await self._client.get(
            f"{self.base_url}/v1/sandboxes",
            params={
                "project_id": self.project_id,
                "pool_id": self.pool_id,
            },
        )
        data = resp.json()
        return data.get("sandboxes", [])
