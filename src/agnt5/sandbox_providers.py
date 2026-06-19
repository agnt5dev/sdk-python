"""Managed sandbox provider integrations for the AGNT5 Python SDK.

Native httpx clients for external sandbox vendors, conforming to the
canonical Rust contract in ``sdk-core/src/sandbox/providers/``:

- :class:`E2BSandboxProvider` — E2B (api.e2b.app + envd/code-interpreter)
- :class:`DaytonaSandboxProvider` — Daytona (app.daytona.io + toolbox proxy)
- :class:`VercelSandboxProvider` — Vercel Sandbox (api.vercel.com /v2/sandboxes)
- :class:`NorthflankSandboxProvider` — Northflank (REST + websocket exec)
- :class:`TogetherSandboxProvider` — Together Code Interpreter (/v1/tci)

Modal is not included here: its API is gRPC-only and is integrated
natively in the Rust core (``sdk-core``); a Python surface for it would
need gRPC bindings rather than httpx.

Each provider exposes the control plane (``create`` / ``connect`` /
``destroy`` / ``list_sandboxes``) and returns sandbox handles with the same
data-plane surface as :class:`agnt5.sandbox.Sandbox`: ``execute_code``,
``run_command``, ``write_file``, ``read_file``, ``delete_file``,
``list_files``, ``health`` — plus provider-specific extras (preview URLs,
Daytona git operations).

Example:
    ```python
    from agnt5 import E2BSandboxProvider, CreateSandboxOptions

    async with E2BSandboxProvider.from_env() as provider:
        sandbox = await provider.create(CreateSandboxOptions(timeout_secs=300))
        result = await sandbox.execute_code("print(6 * 7)")
        print(result.stdout)  # "42"
        await provider.destroy(sandbox.sandbox_id)
    ```
"""

import base64
import gzip
import io
import json
import shlex
import tarfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .sandbox import (
    ExecuteCodeResult,
    FileInfo,
    GitCloneResult,
    GitCommitResult,
    GitPushResult,
    GitStatusFile,
    GitStatusResult,
    ListFilesResult,
    ReadFileResult,
    RunCommandResult,
    SandboxHealthResult,
    WriteFileResult,
)

__all__ = [
    "CreateSandboxOptions",
    "SandboxInfo",
    "SandboxProviderError",
    "E2BSandboxProvider",
    "E2BSandbox",
    "DaytonaSandboxProvider",
    "DaytonaSandbox",
    "VercelSandboxProvider",
    "VercelSandbox",
    "NorthflankSandboxProvider",
    "NorthflankSandbox",
    "TogetherSandboxProvider",
    "TogetherSandbox",
    "load_providers_from_env",
]


# ============= Shared types =============


@dataclass
class CreateSandboxOptions:
    """Provider-agnostic sandbox creation options.

    Providers ignore options they don't support (e.g. E2B sizes sandboxes
    via the template, so cpu_cores/memory_mib are ignored there).
    """

    template: Optional[str] = None
    timeout_secs: Optional[int] = None
    env: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, str]] = None
    cpu_cores: Optional[int] = None
    memory_mib: Optional[int] = None


@dataclass
class SandboxInfo:
    """Summary info about a provider sandbox instance."""

    sandbox_id: str
    status: str


class SandboxProviderError(Exception):
    """Error from a sandbox provider API."""

    def __init__(self, provider: str, operation: str, message: str):
        self.provider = provider
        self.operation = operation
        super().__init__(f"{provider} {operation}: {message}")


# ============= Shared helpers =============

_INTERPRETERS = {
    "python": ("python3", "-c"),
    "javascript": ("node", "-e"),
    "bash": ("bash", "-c"),
}


def _interpreter_argv(language: str, code: str) -> Tuple[str, List[str]]:
    """Interpreter argv for executing code in a POSIX sandbox."""
    try:
        program, flag = _INTERPRETERS[language]
    except KeyError:
        raise SandboxProviderError(
            "sandbox", "execute_code", f"unsupported language: {language}"
        ) from None
    return program, [flag, code]


def _interpreter_command_line(language: str, code: str) -> str:
    """Interpreter invocation as a single shell command line."""
    program, args = _interpreter_argv(language, code)
    return " ".join([program] + [shlex.quote(a) for a in args])


def _parse_listing_output(stdout: str) -> List[FileInfo]:
    """Parse ``type|size|mode|mtime|path`` directory-listing lines.

    This is the output of GNU ``find -printf '%y|%s|%m|%T@|%p\\n'`` and of
    the Python os.walk snippets used by providers without a listing API.
    """
    files: List[FileInfo] = []
    for line in stdout.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        kind, size, mode, mtime, path = parts
        try:
            mod_time = int(float(mtime) * 1000)
        except ValueError:
            mod_time = 0
        files.append(
            FileInfo(
                name=path.rsplit("/", 1)[-1],
                path=path,
                size=int(size) if size.isdigit() else 0,
                mode=int(mode, 8) if mode.isdigit() else 0,
                is_dir=kind == "d",
                mod_time=mod_time,
            )
        )
    return files


async def _check(provider: str, operation: str, resp: httpx.Response) -> httpx.Response:
    """Raise SandboxProviderError on non-2xx responses."""
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise SandboxProviderError(provider, operation, f"HTTP {resp.status_code} — {body}")
    return resp


# ============= E2B =============


class E2BSandboxProvider:
    """Control plane for E2B sandboxes (https://e2b.dev).

    Code/command execution uses the code-interpreter data plane (port 49999),
    so the default template is ``code-interpreter-v1``. File operations use
    envd (port 49983) and work on any template.
    """

    name = "e2b"

    def __init__(
        self,
        api_key: str,
        *,
        domain: str = "e2b.app",
        api_url: Optional[str] = None,
        template: str = "code-interpreter-v1",
        timeout: float = 60.0,
    ):
        self.domain = domain
        self.api_url = api_url or f"https://api.{domain}"
        self.template = template
        self._client = httpx.AsyncClient(timeout=timeout, headers={"X-API-Key": api_key})

    @classmethod
    def from_env(cls) -> "E2BSandboxProvider":
        """Build from E2B_API_KEY (+ optional E2B_DOMAIN, E2B_API_URL, E2B_TEMPLATE)."""
        import os

        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            raise SandboxProviderError("e2b", "from_env", "E2B_API_KEY is required")
        domain = os.environ.get("E2B_DOMAIN", "e2b.app")
        return cls(
            api_key,
            domain=domain,
            api_url=os.environ.get("E2B_API_URL"),
            template=os.environ.get("E2B_TEMPLATE", "code-interpreter-v1"),
        )

    async def __aenter__(self) -> "E2BSandboxProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _handle(self, data: Dict[str, Any]) -> "E2BSandbox":
        return E2BSandbox(
            sandbox_id=data["sandboxID"],
            domain=data.get("domain") or self.domain,
            envd_access_token=data.get("envdAccessToken"),
        )

    async def create(self, opts: Optional[CreateSandboxOptions] = None) -> "E2BSandbox":
        opts = opts or CreateSandboxOptions()
        body: Dict[str, Any] = {
            "templateID": opts.template or self.template,
            "timeout": opts.timeout_secs or 300,
        }
        if opts.env:
            body["envVars"] = opts.env
        if opts.metadata:
            body["metadata"] = opts.metadata
        resp = await _check(
            "e2b",
            "create",
            await self._client.post(f"{self.api_url}/sandboxes", json=body),
        )
        return self._handle(resp.json())

    async def connect(self, sandbox_id: str) -> "E2BSandbox":
        """Connect to an existing sandbox, resuming it if paused."""
        resp = await _check(
            "e2b",
            "connect",
            await self._client.post(
                f"{self.api_url}/sandboxes/{sandbox_id}/connect",
                json={"timeout": 300},
            ),
        )
        return self._handle(resp.json())

    async def destroy(self, sandbox_id: str) -> bool:
        await _check(
            "e2b",
            "destroy",
            await self._client.delete(f"{self.api_url}/sandboxes/{sandbox_id}"),
        )
        return True

    async def list_sandboxes(self) -> List[SandboxInfo]:
        resp = await _check(
            "e2b", "list_sandboxes", await self._client.get(f"{self.api_url}/sandboxes")
        )
        return [
            SandboxInfo(
                sandbox_id=item.get("sandboxID", ""),
                status=item.get("state", "running"),
            )
            for item in resp.json()
        ]

    async def set_timeout(self, sandbox_id: str, timeout_secs: int) -> None:
        """Extend the sandbox lifetime to ``timeout_secs`` from now."""
        await _check(
            "e2b",
            "set_timeout",
            await self._client.post(
                f"{self.api_url}/sandboxes/{sandbox_id}/timeout",
                json={"timeout": timeout_secs},
            ),
        )


class E2BSandbox:
    """A running E2B sandbox."""

    languages = ("python", "javascript", "bash")

    def __init__(
        self,
        sandbox_id: str,
        domain: str = "e2b.app",
        envd_access_token: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.sandbox_id = sandbox_id
        self.domain = domain
        self._envd_url = f"https://49983-{sandbox_id}.{domain}"
        self._interpreter_url = f"https://49999-{sandbox_id}.{domain}"
        # envd selects the OS user via Basic auth with an empty password.
        headers = {"Authorization": "Basic " + base64.b64encode(b"user:").decode()}
        if envd_access_token:
            headers["X-Access-Token"] = envd_access_token
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    def preview_url(self, port: int) -> str:
        """Public URL for a port inside the sandbox (no API call required)."""
        return f"https://{port}-{self.sandbox_id}.{self.domain}"

    async def _execute_raw(
        self,
        code: str,
        language: str,
        env: Optional[Dict[str, str]],
        timeout_ms: int,
    ) -> Tuple[str, str, Optional[str], int]:
        body: Dict[str, Any] = {"code": code, "language": language}
        if env:
            body["env_vars"] = env
        started = time.monotonic()
        resp = await _check(
            "e2b",
            "execute_code",
            await self._client.post(
                f"{self._interpreter_url}/execute",
                json=body,
                timeout=timeout_ms / 1000 + 30,
            ),
        )
        stdout, stderr, error = "", "", None
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "stdout":
                stdout += event.get("text", "")
            elif kind == "stderr":
                stderr += event.get("text", "")
            elif kind == "error":
                error = f"{event.get('name', 'Error')}: {event.get('value', '')}"
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return stdout, stderr, error, elapsed_ms

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout_ms: int = 30000,
        env: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
    ) -> ExecuteCodeResult:
        if work_dir and language == "bash":
            code = f"cd {shlex.quote(work_dir)} && {code}"
        stdout, stderr, error, elapsed = await self._execute_raw(code, language, env, timeout_ms)
        return ExecuteCodeResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=1 if error else 0,
            execution_time_ms=elapsed,
            error=error,
        )

    async def run_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
    ) -> RunCommandResult:
        """Run a shell command via the code interpreter's bash kernel.

        Requires a code-interpreter template. Exit codes are synthesized
        (0 on success, 1 when the interpreter reports an error).
        """
        line = ""
        if working_dir:
            line += f"cd {shlex.quote(working_dir)} && "
        line += command
        for arg in args or []:
            line += " " + shlex.quote(arg)
        stdout, stderr, error, elapsed = await self._execute_raw(line, "bash", env, timeout_ms)
        return RunCommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=1 if error else 0,
            execution_time_ms=elapsed,
            error=error,
        )

    async def health(self) -> SandboxHealthResult:
        await _check("e2b", "health", await self._client.get(f"{self._envd_url}/health"))
        return SandboxHealthResult(status="running", sandbox_id=self.sandbox_id, uptime_ms=0)

    async def write_file(self, path: str, content: bytes, mode: int = 0o644) -> WriteFileResult:
        resp = await self._client.post(
            f"{self._envd_url}/files",
            params={"path": path, "username": "user"},
            content=content,
            headers={"Content-Type": "application/octet-stream"},
        )
        await _check("e2b", "write_file", resp)
        return WriteFileResult(success=True, path=path, size=len(content))

    async def read_file(self, path: str) -> ReadFileResult:
        resp = await _check(
            "e2b",
            "read_file",
            await self._client.get(
                f"{self._envd_url}/files",
                params={"path": path, "username": "user"},
            ),
        )
        return ReadFileResult(
            path=path,
            content=resp.content,
            size=len(resp.content),
            mode=0,
            is_dir=False,
        )

    async def _filesystem_rpc(self, method: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Unary Connect-RPC call to envd's filesystem service (JSON encoding)."""
        resp = await _check(
            "e2b",
            method,
            await self._client.post(f"{self._envd_url}/filesystem.Filesystem/{method}", json=body),
        )
        return resp.json()

    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        # envd's Remove deletes files and directories alike.
        await self._filesystem_rpc("Remove", {"path": path})
        return True

    async def list_files(self, path: str, recursive: bool = False) -> ListFilesResult:
        data = await self._filesystem_rpc(
            "ListDir", {"path": path, "depth": 64 if recursive else 1}
        )
        files = []
        for entry in data.get("entries", []):
            size = entry.get("size", 0)
            mod_time = 0
            if entry.get("modifiedTime"):
                try:
                    from datetime import datetime

                    mod_time = int(
                        datetime.fromisoformat(
                            entry["modifiedTime"].replace("Z", "+00:00")
                        ).timestamp()
                        * 1000
                    )
                except ValueError:
                    pass
            files.append(
                FileInfo(
                    name=entry.get("name", ""),
                    path=entry.get("path", ""),
                    # proto3 JSON serializes 64-bit ints as strings.
                    size=int(size) if str(size).isdigit() else 0,
                    mode=entry.get("mode", 0),
                    is_dir=entry.get("type") == "FILE_TYPE_DIRECTORY",
                    mod_time=mod_time,
                )
            )
        return ListFilesResult(path=path, files=files, total=len(files))


# ============= Daytona =============


class DaytonaSandboxProvider:
    """Control plane for Daytona sandboxes (https://daytona.io)."""

    name = "daytona"
    _FAILED_STATES = {"error", "build_failed", "destroyed", "destroying"}

    def __init__(
        self,
        api_key: str,
        *,
        api_url: str = "https://app.daytona.io/api",
        target: Optional[str] = None,
        timeout: float = 60.0,
        ready_timeout: float = 120.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.target = target
        self.ready_timeout = ready_timeout
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    @classmethod
    def from_env(cls) -> "DaytonaSandboxProvider":
        """Build from DAYTONA_API_KEY (+ optional DAYTONA_API_URL, DAYTONA_TARGET)."""
        import os

        api_key = os.environ.get("DAYTONA_API_KEY")
        if not api_key:
            raise SandboxProviderError("daytona", "from_env", "DAYTONA_API_KEY is required")
        return cls(
            api_key,
            api_url=os.environ.get("DAYTONA_API_URL", "https://app.daytona.io/api"),
            target=os.environ.get("DAYTONA_TARGET"),
        )

    async def __aenter__(self) -> "DaytonaSandboxProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_sandbox(self, sandbox_id: str) -> Dict[str, Any]:
        resp = await _check(
            "daytona",
            "get_sandbox",
            await self._client.get(f"{self.api_url}/sandbox/{sandbox_id}"),
        )
        return resp.json()

    async def _handle(self, data: Dict[str, Any]) -> "DaytonaSandbox":
        proxy_url = data.get("toolboxProxyUrl")
        if not proxy_url:
            resp = await _check(
                "daytona",
                "toolbox_proxy_url",
                await self._client.get(f"{self.api_url}/sandbox/{data['id']}/toolbox-proxy-url"),
            )
            proxy_url = resp.json()["url"]
        return DaytonaSandbox(
            sandbox_id=data["id"],
            toolbox_url=f"{proxy_url.rstrip('/')}/{data['id']}",
            api_url=self.api_url,
            client=self._client,
        )

    async def create(self, opts: Optional[CreateSandboxOptions] = None) -> "DaytonaSandbox":
        import asyncio

        opts = opts or CreateSandboxOptions()
        body: Dict[str, Any] = {}
        if opts.template:
            body["snapshot"] = opts.template
        if opts.env:
            body["env"] = opts.env
        if opts.metadata:
            body["labels"] = opts.metadata
        if opts.cpu_cores:
            body["cpu"] = opts.cpu_cores
        if opts.memory_mib:
            body["memory"] = (opts.memory_mib + 1023) // 1024  # GB
        if opts.timeout_secs:
            body["autoStopInterval"] = (opts.timeout_secs + 59) // 60  # minutes
        if self.target:
            body["target"] = self.target

        resp = await _check(
            "daytona",
            "create",
            await self._client.post(f"{self.api_url}/sandbox", json=body),
        )
        sandbox = resp.json()
        deadline = time.monotonic() + self.ready_timeout
        while sandbox.get("state") != "started":
            state = sandbox.get("state", "")
            if state in self._FAILED_STATES:
                raise SandboxProviderError(
                    "daytona", "create", f"sandbox {sandbox['id']} entered state '{state}'"
                )
            if time.monotonic() >= deadline:
                raise SandboxProviderError(
                    "daytona",
                    "create",
                    f"sandbox {sandbox['id']} not started in {self.ready_timeout}s",
                )
            await asyncio.sleep(2)
            sandbox = await self._get_sandbox(sandbox["id"])
        return await self._handle(sandbox)

    async def connect(self, sandbox_id: str) -> "DaytonaSandbox":
        return await self._handle(await self._get_sandbox(sandbox_id))

    async def destroy(self, sandbox_id: str) -> bool:
        await _check(
            "daytona",
            "destroy",
            await self._client.delete(f"{self.api_url}/sandbox/{sandbox_id}"),
        )
        return True

    async def list_sandboxes(self) -> List[SandboxInfo]:
        resp = await _check(
            "daytona", "list_sandboxes", await self._client.get(f"{self.api_url}/sandbox")
        )
        return [
            SandboxInfo(sandbox_id=s.get("id", ""), status=s.get("state", "unknown"))
            for s in resp.json().get("items", [])
        ]


class DaytonaSandbox:
    """A running Daytona sandbox, driven via its toolbox daemon.

    Note: Daytona's exec API merges stdout and stderr; combined output is
    reported as stdout.
    """

    languages = ("python", "javascript", "bash")

    def __init__(
        self,
        sandbox_id: str,
        toolbox_url: str,
        api_url: str,
        client: httpx.AsyncClient,
    ):
        self.sandbox_id = sandbox_id
        self._toolbox_url = toolbox_url
        self._api_url = api_url
        self._client = client

    async def run_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
    ) -> RunCommandResult:
        for arg in args or []:
            command += " " + shlex.quote(arg)
        body: Dict[str, Any] = {
            "command": command,
            "timeout": max(1, timeout_ms // 1000),
        }
        if working_dir:
            body["cwd"] = working_dir
        if env:
            body["envs"] = env
        started = time.monotonic()
        resp = await _check(
            "daytona",
            "run_command",
            await self._client.post(
                f"{self._toolbox_url}/process/execute",
                json=body,
                timeout=timeout_ms / 1000 + 30,
            ),
        )
        data = resp.json()
        return RunCommandResult(
            stdout=data.get("result", ""),
            stderr="",
            exit_code=data.get("exitCode", -1),
            execution_time_ms=int((time.monotonic() - started) * 1000),
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout_ms: int = 30000,
        env: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
    ) -> ExecuteCodeResult:
        if language == "bash":
            # Bash has no native code-run language; go through the shell.
            result = await self.run_command(
                _interpreter_command_line(language, code),
                working_dir=work_dir,
                env=env,
                timeout_ms=timeout_ms,
            )
            return ExecuteCodeResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time_ms=result.execution_time_ms,
                error=result.error,
            )
        body: Dict[str, Any] = {
            "code": code,
            "language": language,
            "timeout": max(1, timeout_ms // 1000),
        }
        if env:
            body["envs"] = env
        started = time.monotonic()
        resp = await _check(
            "daytona",
            "execute_code",
            await self._client.post(
                f"{self._toolbox_url}/process/code-run",
                json=body,
                timeout=timeout_ms / 1000 + 30,
            ),
        )
        data = resp.json()
        return ExecuteCodeResult(
            stdout=data.get("result", ""),
            stderr="",
            exit_code=data.get("exitCode", -1),
            execution_time_ms=int((time.monotonic() - started) * 1000),
        )

    async def health(self) -> SandboxHealthResult:
        resp = await _check(
            "daytona",
            "health",
            await self._client.get(f"{self._api_url}/sandbox/{self.sandbox_id}"),
        )
        return SandboxHealthResult(
            status=resp.json().get("state", "unknown"),
            sandbox_id=self.sandbox_id,
            uptime_ms=0,
        )

    async def write_file(self, path: str, content: bytes, mode: int = 0o644) -> WriteFileResult:
        resp = await self._client.post(
            f"{self._toolbox_url}/files/upload",
            params={"path": path},
            files={"file": (path.rsplit("/", 1)[-1], content)},
        )
        await _check("daytona", "write_file", resp)
        return WriteFileResult(success=True, path=path, size=len(content))

    async def read_file(self, path: str) -> ReadFileResult:
        resp = await _check(
            "daytona",
            "read_file",
            await self._client.get(f"{self._toolbox_url}/files/download", params={"path": path}),
        )
        return ReadFileResult(
            path=path,
            content=resp.content,
            size=len(resp.content),
            mode=0,
            is_dir=False,
        )

    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        await _check(
            "daytona",
            "delete_file",
            await self._client.delete(
                f"{self._toolbox_url}/files",
                params={"path": path, "recursive": str(recursive).lower()},
            ),
        )
        return True

    async def list_files(self, path: str, recursive: bool = False) -> ListFilesResult:
        if recursive:
            # The toolbox files API lists one directory; recurse via find.
            result = await self.run_command(
                f"find {shlex.quote(path)} -mindepth 1 -printf '%y|%s|%m|%T@|%p\\n'"
            )
            if result.exit_code != 0:
                raise SandboxProviderError("daytona", "list_files", result.stdout)
            files = _parse_listing_output(result.stdout)
            return ListFilesResult(path=path, files=files, total=len(files))

        resp = await _check(
            "daytona",
            "list_files",
            await self._client.get(f"{self._toolbox_url}/files", params={"path": path}),
        )
        files = []
        for entry in resp.json():
            mode = 0
            for candidate in (entry.get("permissions"), entry.get("mode")):
                if candidate and str(candidate).isdigit():
                    mode = int(candidate, 8)
                    break
            mod_time = 0
            if entry.get("modifiedAt"):
                try:
                    from datetime import datetime

                    mod_time = int(
                        datetime.fromisoformat(
                            entry["modifiedAt"].replace("Z", "+00:00")
                        ).timestamp()
                        * 1000
                    )
                except ValueError:
                    pass
            files.append(
                FileInfo(
                    name=entry.get("name", ""),
                    path=f"{path.rstrip('/')}/{entry.get('name', '')}",
                    size=entry.get("size", 0),
                    mode=mode,
                    is_dir=entry.get("isDir", False),
                    mod_time=mod_time,
                )
            )
        return ListFilesResult(path=path, files=files, total=len(files))

    async def preview_url(self, port: int) -> Dict[str, Any]:
        """Public preview URL + access token for a sandbox port."""
        resp = await _check(
            "daytona",
            "preview_url",
            await self._client.get(
                f"{self._api_url}/sandbox/{self.sandbox_id}/ports/{port}/preview-url"
            ),
        )
        return resp.json()

    # ----- Git (native toolbox endpoints) -----

    async def git_clone(
        self,
        url: str,
        target_dir: Optional[str] = None,
        branch: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> GitCloneResult:
        path = target_dir or url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        body: Dict[str, Any] = {"url": url, "path": path}
        if branch:
            body["branch"] = branch
        if username:
            body["username"] = username
        if password:
            body["password"] = password
        await _check(
            "daytona",
            "git_clone",
            await self._client.post(f"{self._toolbox_url}/git/clone", json=body),
        )
        return GitCloneResult(success=True, path=path, branch=branch or "")

    async def git_status(self, path: str = ".") -> GitStatusResult:
        resp = await _check(
            "daytona",
            "git_status",
            await self._client.get(f"{self._toolbox_url}/git/status", params={"path": path}),
        )
        data = resp.json()
        files = []
        for f in data.get("fileStatus", []):
            staging = f.get("staging", "")
            staged = bool(staging) and staging.lower() != "unmodified"
            files.append(
                GitStatusFile(
                    path=f.get("name", ""),
                    status=(staging if staged else f.get("worktree", "")).lower(),
                    staged=staged,
                )
            )
        return GitStatusResult(
            branch=data.get("currentBranch", ""),
            commit_sha="",
            is_clean=not files,
            ahead=data.get("ahead", 0),
            behind=data.get("behind", 0),
            files=files,
        )

    async def git_commit(
        self,
        message: str,
        path: str = ".",
        files: Optional[List[str]] = None,
        author: str = "AGNT5 <agnt5@agnt5.dev>",
    ) -> GitCommitResult:
        await _check(
            "daytona",
            "git_add",
            await self._client.post(
                f"{self._toolbox_url}/git/add",
                json={"path": path, "files": files or ["."]},
            ),
        )
        # The toolbox requires separate author name and email fields.
        if "<" in author:
            name, _, rest = author.partition("<")
            name, email = name.strip(), rest.rstrip(">").strip()
        else:
            name, email = author.strip(), "agnt5@agnt5.dev"
        resp = await _check(
            "daytona",
            "git_commit",
            await self._client.post(
                f"{self._toolbox_url}/git/commit",
                json={"path": path, "message": message, "author": name, "email": email},
            ),
        )
        data = resp.json() if resp.content else {}
        return GitCommitResult(success=True, commit_sha=data.get("hash", ""))

    async def git_push(
        self,
        path: str = ".",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> GitPushResult:
        body: Dict[str, Any] = {"path": path}
        if username:
            body["username"] = username
        if password:
            body["password"] = password
        await _check(
            "daytona",
            "git_push",
            await self._client.post(f"{self._toolbox_url}/git/push", json=body),
        )
        return GitPushResult(success=True)


# ============= Vercel =============


class VercelSandboxProvider:
    """Control plane for Vercel Sandboxes (https://vercel.com/docs/sandbox).

    Auth modes (matching @vercel/sandbox): an OIDC token alone, or an access
    token + team_id + project_id.
    """

    name = "vercel"

    def __init__(
        self,
        token: str,
        *,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://api.vercel.com",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.team_id = team_id
        self.project_id = project_id
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {token}"}
        )

    @classmethod
    def from_env(cls) -> "VercelSandboxProvider":
        """Build from VERCEL_OIDC_TOKEN, or VERCEL_TOKEN + VERCEL_TEAM_ID + VERCEL_PROJECT_ID."""
        import os

        base_url = os.environ.get("VERCEL_SANDBOX_BASE_URL", "https://api.vercel.com")
        oidc = os.environ.get("VERCEL_OIDC_TOKEN")
        if oidc:
            return cls(
                oidc,
                team_id=os.environ.get("VERCEL_TEAM_ID"),
                project_id=os.environ.get("VERCEL_PROJECT_ID"),
                base_url=base_url,
            )
        token = os.environ.get("VERCEL_TOKEN")
        team_id = os.environ.get("VERCEL_TEAM_ID")
        project_id = os.environ.get("VERCEL_PROJECT_ID")
        if not token:
            raise SandboxProviderError(
                "vercel", "from_env", "VERCEL_OIDC_TOKEN or VERCEL_TOKEN is required"
            )
        if not team_id or not project_id:
            raise SandboxProviderError(
                "vercel",
                "from_env",
                "VERCEL_TEAM_ID and VERCEL_PROJECT_ID are required with VERCEL_TOKEN",
            )
        return cls(token, team_id=team_id, project_id=project_id, base_url=base_url)

    async def __aenter__(self) -> "VercelSandboxProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _params(self) -> Dict[str, str]:
        params = {}
        if self.team_id:
            params["teamId"] = self.team_id
        if self.project_id:
            params["projectId"] = self.project_id
        return params

    def _handle(self, data: Dict[str, Any]) -> "VercelSandbox":
        return VercelSandbox(
            name=data.get("sandbox", {}).get("name", ""),
            session_id=data["session"]["id"],
            routes=data.get("routes", []),
            base_url=self.base_url,
            params=self._params(),
            client=self._client,
        )

    async def create(self, opts: Optional[CreateSandboxOptions] = None) -> "VercelSandbox":
        opts = opts or CreateSandboxOptions()
        body: Dict[str, Any] = {
            "name": f"agnt5-{uuid.uuid4().hex}",
            "runtime": opts.template or "node24",
            "timeout": (opts.timeout_secs or 300) * 1000,
        }
        if opts.cpu_cores:
            body["resources"] = {"vcpus": opts.cpu_cores}
        if opts.env:
            body["env"] = opts.env
        if self.project_id:
            body["projectId"] = self.project_id
        resp = await _check(
            "vercel",
            "create",
            await self._client.post(
                f"{self.base_url}/v2/sandboxes", params=self._params(), json=body
            ),
        )
        return self._handle(resp.json())

    async def connect(self, name: str) -> "VercelSandbox":
        """Connect to an existing sandbox by name, resuming it if stopped."""
        params = {**self._params(), "resume": "true"}
        resp = await _check(
            "vercel",
            "connect",
            await self._client.get(f"{self.base_url}/v2/sandboxes/{name}", params=params),
        )
        return self._handle(resp.json())

    async def destroy(self, name: str) -> bool:
        await _check(
            "vercel",
            "destroy",
            await self._client.delete(
                f"{self.base_url}/v2/sandboxes/{name}", params=self._params()
            ),
        )
        return True

    async def list_sandboxes(self) -> List[SandboxInfo]:
        resp = await _check(
            "vercel",
            "list_sandboxes",
            await self._client.get(f"{self.base_url}/v2/sandboxes", params=self._params()),
        )
        return [
            SandboxInfo(sandbox_id=s.get("name", ""), status=s.get("status", "unknown"))
            for s in resp.json().get("sandboxes", [])
        ]


class VercelSandbox:
    """A running Vercel Sandbox session."""

    languages = ("python", "javascript", "bash")

    def __init__(
        self,
        name: str,
        session_id: str,
        routes: List[Dict[str, Any]],
        base_url: str,
        params: Dict[str, str],
        client: httpx.AsyncClient,
    ):
        self.name = name
        self.session_id = session_id
        self.routes = routes
        self._base_url = base_url
        self._params = params
        self._client = client

    @property
    def sandbox_id(self) -> str:
        return self.name

    def preview_url(self, port: int) -> Optional[str]:
        """Public URL for a port declared at sandbox creation."""
        for route in self.routes:
            if route.get("port") == port:
                return route.get("url")
        return None

    def _session_url(self, suffix: str = "") -> str:
        return f"{self._base_url}/v2/sandboxes/sessions/{self.session_id}{suffix}"

    async def run_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
        sudo: bool = False,
    ) -> RunCommandResult:
        body: Dict[str, Any] = {
            "command": command,
            "args": args or [],
            "wait": True,
            "logs": True,
            "timeout": timeout_ms,
        }
        if working_dir:
            body["cwd"] = working_dir
        if env:
            body["env"] = env
        if sudo:
            body["sudo"] = True
        started = time.monotonic()
        resp = await _check(
            "vercel",
            "run_command",
            await self._client.post(
                self._session_url("/cmd"),
                params=self._params,
                json=body,
                timeout=timeout_ms / 1000 + 30,
            ),
        )
        stdout, stderr, exit_code, error = "", "", -1, None
        # ND-JSON stream: log lines have stream/data; command lines carry exitCode.
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            stream = event.get("stream")
            if stream == "stdout" and isinstance(event.get("data"), str):
                stdout += event["data"]
            elif stream == "stderr" and isinstance(event.get("data"), str):
                stderr += event["data"]
            elif stream == "error":
                data = event.get("data")
                error = data.get("message") if isinstance(data, dict) else str(data)
            command_obj = event.get("command")
            if isinstance(command_obj, dict) and command_obj.get("exitCode") is not None:
                exit_code = command_obj["exitCode"]
        return RunCommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=int((time.monotonic() - started) * 1000),
            error=error,
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout_ms: int = 30000,
        env: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
    ) -> ExecuteCodeResult:
        program, args = _interpreter_argv(language, code)
        result = await self.run_command(
            program, args, working_dir=work_dir, env=env, timeout_ms=timeout_ms
        )
        return ExecuteCodeResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
            error=result.error,
        )

    async def health(self) -> SandboxHealthResult:
        resp = await _check(
            "vercel",
            "health",
            await self._client.get(self._session_url(), params=self._params),
        )
        data = resp.json()
        status = data.get("status") or data.get("session", {}).get("status", "unknown")
        return SandboxHealthResult(status=status, sandbox_id=self.name, uptime_ms=0)

    async def write_file(self, path: str, content: bytes, mode: int = 0o644) -> WriteFileResult:
        # fs/write takes a gzipped tar; entry paths resolve against x-cwd.
        if path.startswith("/"):
            cwd, entry = "/", path[1:]
        else:
            cwd, entry = "/vercel/sandbox", path
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=entry)
            info.size = len(content)
            info.mode = mode
            tar.addfile(info, io.BytesIO(content))
        archive = gzip.compress(buf.getvalue())

        resp = await self._client.post(
            self._session_url("/fs/write"),
            params=self._params,
            content=archive,
            headers={"Content-Type": "application/gzip", "x-cwd": cwd},
        )
        await _check("vercel", "write_file", resp)
        return WriteFileResult(success=True, path=path, size=len(content))

    async def read_file(self, path: str) -> ReadFileResult:
        resp = await _check(
            "vercel",
            "read_file",
            await self._client.post(
                self._session_url("/fs/read"), params=self._params, json={"path": path}
            ),
        )
        return ReadFileResult(
            path=path,
            content=resp.content,
            size=len(resp.content),
            mode=0,
            is_dir=False,
        )

    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        args = ["-f"] + (["-r"] if recursive else []) + ["--", path]
        result = await self.run_command("rm", args)
        return result.exit_code == 0

    async def list_files(self, path: str, recursive: bool = False) -> ListFilesResult:
        # Amazon Linux 2023 ships GNU findutils, so -printf is available.
        args = [path, "-mindepth", "1"]
        if not recursive:
            args += ["-maxdepth", "1"]
        args += ["-printf", "%y|%s|%m|%T@|%p\\n"]
        result = await self.run_command("find", args)
        if result.exit_code != 0:
            raise SandboxProviderError("vercel", "list_files", result.stderr)
        files = _parse_listing_output(result.stdout)
        return ListFilesResult(path=path, files=files, total=len(files))


# ============= Northflank =============


class NorthflankSandboxProvider:
    """Control plane for Northflank sandboxes (https://northflank.com).

    Sandboxes are deployment services running ``sleep infinity``; commands
    run over the command-exec websocket. File operations are emulated over
    exec (base64 round-trips). Requires the ``websockets`` package.
    """

    name = "northflank"

    def __init__(
        self,
        api_token: str,
        project_id: str,
        *,
        team_id: Optional[str] = None,
        base_url: str = "https://api.northflank.com",
        deployment_plan: str = "nf-compute-200",
        image: str = "python:3.12-slim-bookworm",
        timeout: float = 60.0,
        ready_timeout: float = 180.0,
    ):
        self.api_token = api_token
        self.project_id = project_id
        self.team_id = team_id
        self.base_url = base_url.rstrip("/")
        self.deployment_plan = deployment_plan
        self.image = image
        self.ready_timeout = ready_timeout
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_token}"}
        )

    @classmethod
    def from_env(cls) -> "NorthflankSandboxProvider":
        """Build from NORTHFLANK_API_TOKEN + NORTHFLANK_PROJECT_ID (+ optional vars)."""
        import os

        api_token = os.environ.get("NORTHFLANK_API_TOKEN")
        project_id = os.environ.get("NORTHFLANK_PROJECT_ID")
        if not api_token:
            raise SandboxProviderError("northflank", "from_env", "NORTHFLANK_API_TOKEN is required")
        if not project_id:
            raise SandboxProviderError(
                "northflank", "from_env", "NORTHFLANK_PROJECT_ID is required"
            )
        return cls(
            api_token,
            project_id,
            team_id=os.environ.get("NORTHFLANK_TEAM_ID"),
            base_url=os.environ.get("NORTHFLANK_API_URL", "https://api.northflank.com"),
            deployment_plan=os.environ.get("NORTHFLANK_DEPLOYMENT_PLAN", "nf-compute-200"),
            image=os.environ.get("NORTHFLANK_SANDBOX_IMAGE", "python:3.12-slim-bookworm"),
        )

    async def __aenter__(self) -> "NorthflankSandboxProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _service_url(self, service_id: str) -> str:
        return f"{self.base_url}/v1/projects/{self.project_id}/services/{service_id}"

    @staticmethod
    def _deployment_status(status: Any) -> str:
        if not isinstance(status, dict):
            return "unknown"
        deployment = status.get("deployment", status)
        if isinstance(deployment, dict):
            return str(deployment.get("status", deployment))
        return str(deployment)

    def _handle(self, service_id: str) -> "NorthflankSandbox":
        return NorthflankSandbox(
            service_id=service_id,
            project_id=self.project_id,
            api_token=self.api_token,
            team_id=self.team_id,
            base_url=self.base_url,
            client=self._client,
        )

    async def create(self, opts: Optional[CreateSandboxOptions] = None) -> "NorthflankSandbox":
        import asyncio

        opts = opts or CreateSandboxOptions()
        body: Dict[str, Any] = {
            "name": f"agnt5-{uuid.uuid4().hex[:12]}",
            "billing": {"deploymentPlan": self.deployment_plan},
            "deployment": {
                "external": {"imagePath": opts.template or self.image},
                "docker": {
                    "configType": "customCommand",
                    "customCommand": "sleep infinity",
                },
            },
        }
        if opts.env:
            body["runtimeEnvironment"] = opts.env
        resp = await _check(
            "northflank",
            "create",
            await self._client.post(
                f"{self.base_url}/v1/projects/{self.project_id}/services/deployment",
                json=body,
            ),
        )
        service_id = resp.json()["data"]["id"]

        deadline = time.monotonic() + self.ready_timeout
        while True:
            resp = await _check(
                "northflank",
                "get_service",
                await self._client.get(self._service_url(service_id)),
            )
            status = self._deployment_status(resp.json()["data"].get("status")).lower()
            if "running" in status or "completed" in status:
                break
            if "failed" in status or "error" in status:
                raise SandboxProviderError(
                    "northflank", "create", f"service {service_id} failed ({status})"
                )
            if time.monotonic() >= deadline:
                raise SandboxProviderError(
                    "northflank",
                    "create",
                    f"service {service_id} not running in {self.ready_timeout}s ({status})",
                )
            await asyncio.sleep(3)
        return self._handle(service_id)

    async def connect(self, service_id: str) -> "NorthflankSandbox":
        await _check(
            "northflank",
            "connect",
            await self._client.get(self._service_url(service_id)),
        )
        return self._handle(service_id)

    async def destroy(self, service_id: str) -> bool:
        await _check(
            "northflank",
            "destroy",
            await self._client.delete(self._service_url(service_id)),
        )
        return True

    async def list_sandboxes(self) -> List[SandboxInfo]:
        resp = await _check(
            "northflank",
            "list_sandboxes",
            await self._client.get(f"{self.base_url}/v1/projects/{self.project_id}/services"),
        )
        services = resp.json().get("data", {}).get("services", [])
        return [
            SandboxInfo(
                sandbox_id=s.get("id", ""),
                status=self._deployment_status(s.get("status")),
            )
            for s in services
        ]


class NorthflankSandbox:
    """A running Northflank sandbox service."""

    languages = ("python", "javascript", "bash")

    def __init__(
        self,
        service_id: str,
        project_id: str,
        api_token: str,
        team_id: Optional[str],
        base_url: str,
        client: httpx.AsyncClient,
    ):
        self.service_id = service_id
        self.sandbox_id = service_id
        self._project_id = project_id
        self._api_token = api_token
        self._team_id = team_id
        self._base_url = base_url
        self._client = client

    def _ws_url(self) -> str:
        ws_base = self._base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        team = f"teams/{self._team_id}/" if self._team_id else ""
        return (
            f"{ws_base}/v1/command-exec/{team}projects/{self._project_id}"
            f"/services/{self.service_id}"
        )

    async def run_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_ms: int = 30000,
    ) -> RunCommandResult:
        """Run a command over the command-exec websocket."""
        import asyncio

        try:
            import websockets
        except ImportError as e:
            raise SandboxProviderError(
                "northflank",
                "run_command",
                "the 'websockets' package is required for Northflank exec (pip install websockets)",
            ) from e

        argv = [command] + list(args or [])
        if working_dir or env:
            # The exec context has no cwd/env parameters; wrap in a shell.
            line = ""
            if working_dir:
                line += f"cd {shlex.quote(working_dir)} && "
            for key, value in (env or {}).items():
                line += f"export {key}={shlex.quote(value)} && "
            line += " ".join(shlex.quote(a) for a in argv)
            argv = ["bash", "-c", line]

        started = time.monotonic()

        async def _run() -> RunCommandResult:
            stdout, stderr, exit_code = "", "", -1
            async with websockets.connect(self._ws_url()) as ws:
                # Auth is in-band: the first message carries the API token.
                await ws.send(
                    json.dumps(
                        {
                            "type": "init",
                            "data": {
                                "auth": {"type": "apiToken", "apiToken": self._api_token},
                                "context": {"command": argv},
                            },
                        }
                    )
                )
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    kind = msg.get("type")
                    data = msg.get("data")
                    if kind == "init":
                        auth = data.get("auth") if isinstance(data, dict) else data
                        if auth != "successful":
                            raise SandboxProviderError(
                                "northflank", "run_command", f"auth failed: {msg}"
                            )
                    elif kind == "stdOut" and isinstance(data, str):
                        stdout += data
                    elif kind == "stdErr" and isinstance(data, str):
                        stderr += data
                    elif kind == "completion":
                        if isinstance(data, dict) and data.get("exitCode") is not None:
                            exit_code = data["exitCode"]
                        break
                    elif kind == "error":
                        message = data.get("message") if isinstance(data, dict) else str(data)
                        raise SandboxProviderError("northflank", "run_command", message)
            return RunCommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time_ms=int((time.monotonic() - started) * 1000),
            )

        try:
            return await asyncio.wait_for(_run(), timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            raise SandboxProviderError(
                "northflank", "run_command", f"timed out after {timeout_ms} ms"
            ) from None

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout_ms: int = 30000,
        env: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
    ) -> ExecuteCodeResult:
        program, args = _interpreter_argv(language, code)
        result = await self.run_command(
            program, args, working_dir=work_dir, env=env, timeout_ms=timeout_ms
        )
        return ExecuteCodeResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
            error=result.error,
        )

    async def health(self) -> SandboxHealthResult:
        resp = await _check(
            "northflank",
            "health",
            await self._client.get(
                f"{self._base_url}/v1/projects/{self._project_id}/services/{self.service_id}"
            ),
        )
        status = NorthflankSandboxProvider._deployment_status(
            resp.json().get("data", {}).get("status")
        )
        return SandboxHealthResult(status=status, sandbox_id=self.service_id, uptime_ms=0)

    async def write_file(self, path: str, content: bytes, mode: int = 0o644) -> WriteFileResult:
        encoded = base64.b64encode(content).decode()
        parent = path.rsplit("/", 1)[0] if "/" in path else "."
        parent = parent or "/"
        script = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)} && "
            f"chmod {mode:o} {shlex.quote(path)}"
        )
        result = await self.run_command("bash", ["-c", script], timeout_ms=60000)
        if result.exit_code != 0:
            raise SandboxProviderError("northflank", "write_file", result.stderr)
        return WriteFileResult(success=True, path=path, size=len(content))

    async def read_file(self, path: str) -> ReadFileResult:
        result = await self.run_command("base64", [path], timeout_ms=60000)
        if result.exit_code != 0:
            raise SandboxProviderError("northflank", "read_file", result.stderr)
        content = base64.b64decode("".join(result.stdout.split()))
        return ReadFileResult(path=path, content=content, size=len(content), mode=0, is_dir=False)

    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        args = ["-f"] + (["-r"] if recursive else []) + ["--", path]
        result = await self.run_command("rm", args)
        return result.exit_code == 0

    async def list_files(self, path: str, recursive: bool = False) -> ListFilesResult:
        args = [path, "-mindepth", "1"]
        if not recursive:
            args += ["-maxdepth", "1"]
        args += ["-printf", "%y|%s|%m|%T@|%p\\n"]
        result = await self.run_command("find", args)
        if result.exit_code != 0:
            raise SandboxProviderError("northflank", "list_files", result.stderr)
        files = _parse_listing_output(result.stdout)
        return ListFilesResult(path=path, files=files, total=len(files))


# ============= Together =============


class TogetherSandboxProvider:
    """Control plane for Together Code Interpreter sessions (https://together.ai).

    Sessions persist packages/variables for 60 minutes and cannot be
    destroyed (they expire on their own). Only Python execution is
    supported; file operations are emulated via session-side snippets and
    TCI's native files upload. Each operation is one execution (sessions
    are billed per session).
    """

    name = "together"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.together.ai",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    @classmethod
    def from_env(cls) -> "TogetherSandboxProvider":
        """Build from TOGETHER_API_KEY (+ optional TOGETHER_BASE_URL)."""
        import os

        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise SandboxProviderError("together", "from_env", "TOGETHER_API_KEY is required")
        return cls(
            api_key,
            base_url=os.environ.get("TOGETHER_BASE_URL", "https://api.together.ai"),
        )

    async def __aenter__(self) -> "TogetherSandboxProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _execute(
        self,
        code: str,
        session_id: Optional[str] = None,
        files: Optional[List[Dict[str, str]]] = None,
        timeout_ms: int = 60000,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"code": code, "language": "python"}
        if session_id:
            body["session_id"] = session_id
        if files:
            body["files"] = files
        resp = await _check(
            "together",
            "execute",
            await self._client.post(
                f"{self.base_url}/v1/tci/execute",
                json=body,
                timeout=timeout_ms / 1000 + 30,
            ),
        )
        parsed = resp.json()
        errors = parsed.get("errors")
        if errors:
            raise SandboxProviderError("together", "execute", str(errors))
        data = parsed.get("data")
        if not data:
            raise SandboxProviderError("together", "execute", "response missing data")
        return data

    async def _sessions(self) -> List[Dict[str, Any]]:
        resp = await _check(
            "together",
            "sessions",
            await self._client.get(f"{self.base_url}/v1/tci/sessions"),
        )
        return (resp.json().get("data") or {}).get("sessions", [])

    async def create(self, opts: Optional[CreateSandboxOptions] = None) -> "TogetherSandbox":
        # Sessions are created implicitly; a no-op execution materializes one.
        data = await self._execute("pass")
        session_id = data.get("session_id", "")
        if not session_id:
            raise SandboxProviderError("together", "create", "no session_id returned")
        return TogetherSandbox(session_id=session_id, provider=self)

    async def connect(self, session_id: str) -> "TogetherSandbox":
        sessions = await self._sessions()
        if not any(s.get("id") == session_id for s in sessions):
            raise SandboxProviderError(
                "together", "connect", f"session '{session_id}' not found or expired"
            )
        return TogetherSandbox(session_id=session_id, provider=self)

    async def destroy(self, session_id: str) -> bool:
        raise SandboxProviderError(
            "together",
            "destroy",
            "TCI sessions cannot be destroyed; they expire after 60 minutes",
        )

    async def list_sandboxes(self) -> List[SandboxInfo]:
        return [
            SandboxInfo(sandbox_id=s.get("id", ""), status="running")
            for s in await self._sessions()
        ]


class TogetherSandbox:
    """A live Together Code Interpreter session."""

    languages = ("python",)

    def __init__(self, session_id: str, provider: TogetherSandboxProvider):
        self.session_id = session_id
        self.sandbox_id = session_id
        self._provider = provider

    @staticmethod
    def _map_outputs(data: Dict[str, Any]) -> ExecuteCodeResult:
        stdout, stderr, error = "", "", None
        for output in data.get("outputs", []):
            kind = output.get("type")
            value = output.get("data")
            if kind == "stdout" and isinstance(value, str):
                stdout += value
            elif kind == "stderr" and isinstance(value, str):
                stderr += value
            elif kind == "error":
                error = value if isinstance(value, str) else str(value)
            elif isinstance(value, dict) and "text/plain" in value:
                stdout += str(value["text/plain"])
        succeeded = error is None and data.get("status") in ("success", "completed")
        return ExecuteCodeResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=0 if succeeded else 1,
            execution_time_ms=0,
            error=error,
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout_ms: int = 30000,
        env: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
    ) -> ExecuteCodeResult:
        if language != "python":
            raise SandboxProviderError(
                "together",
                "execute_code",
                f"Together Code Interpreter only supports Python (requested: {language})",
            )
        started = time.monotonic()
        data = await self._provider._execute(
            code, session_id=self.session_id, timeout_ms=timeout_ms
        )
        result = self._map_outputs(data)
        result.execution_time_ms = int((time.monotonic() - started) * 1000)
        return result

    async def _run_python(self, code: str, operation: str) -> ExecuteCodeResult:
        data = await self._provider._execute(code, session_id=self.session_id)
        result = self._map_outputs(data)
        if result.error:
            raise SandboxProviderError("together", operation, result.error)
        return result

    async def health(self) -> SandboxHealthResult:
        sessions = await self._provider._sessions()
        alive = any(s.get("id") == self.session_id for s in sessions)
        return SandboxHealthResult(
            status="running" if alive else "expired",
            sandbox_id=self.session_id,
            uptime_ms=0,
        )

    async def write_file(self, path: str, content: bytes, mode: int = 0o644) -> WriteFileResult:
        await self._provider._execute(
            "pass",
            session_id=self.session_id,
            files=[
                {
                    "name": path,
                    "encoding": "base64",
                    "content": base64.b64encode(content).decode(),
                }
            ],
        )
        return WriteFileResult(success=True, path=path, size=len(content))

    async def read_file(self, path: str) -> ReadFileResult:
        code = (
            "import base64, sys\n"
            f"sys.stdout.write(base64.b64encode(open({path!r}, 'rb').read()).decode())"
        )
        result = await self._run_python(code, "read_file")
        content = base64.b64decode("".join(result.stdout.split()))
        return ReadFileResult(path=path, content=content, size=len(content), mode=0, is_dir=False)

    async def delete_file(self, path: str, recursive: bool = False) -> bool:
        code = (
            "import os, shutil\n"
            f"p = {path!r}\n"
            "if os.path.isdir(p) and not os.path.islink(p):\n"
            f"    shutil.rmtree(p) if {recursive} else os.rmdir(p)\n"
            "else:\n"
            "    os.remove(p)"
        )
        await self._run_python(code, "delete_file")
        return True

    async def list_files(self, path: str, recursive: bool = False) -> ListFilesResult:
        code = (
            "import os\n"
            f"p = {path!r}\n"
            "entries = []\n"
            f"if {recursive}:\n"
            "    for dirpath, dirnames, filenames in os.walk(p):\n"
            "        entries.extend(os.path.join(dirpath, n) for n in dirnames + filenames)\n"
            "else:\n"
            "    entries = [os.path.join(p, n) for n in os.listdir(p)]\n"
            "for e in entries:\n"
            "    st = os.lstat(e)\n"
            "    kind = 'd' if os.path.isdir(e) and not os.path.islink(e) else 'f'\n"
            '    print(f"{kind}|{st.st_size}|{oct(st.st_mode & 0o7777)[2:]}|{st.st_mtime}|{e}")'
        )
        result = await self._run_python(code, "list_files")
        files = _parse_listing_output(result.stdout)
        return ListFilesResult(path=path, files=files, total=len(files))


# ============= Env auto-detection =============


def load_providers_from_env() -> Dict[str, Any]:
    """Detect and construct providers from environment variables.

    Mirrors the Rust ``SandboxRegistry::load_providers_from_environment``:
    a partially configured provider raises rather than being silently
    skipped. Construction is lazy — no network calls are made.

    Triggers: E2B_API_KEY, DAYTONA_API_KEY, VERCEL_OIDC_TOKEN/VERCEL_TOKEN,
    NORTHFLANK_API_TOKEN, TOGETHER_API_KEY.
    """
    import os

    providers: Dict[str, Any] = {}
    if os.environ.get("E2B_API_KEY"):
        providers["e2b"] = E2BSandboxProvider.from_env()
    if os.environ.get("DAYTONA_API_KEY"):
        providers["daytona"] = DaytonaSandboxProvider.from_env()
    if os.environ.get("VERCEL_OIDC_TOKEN") or os.environ.get("VERCEL_TOKEN"):
        providers["vercel"] = VercelSandboxProvider.from_env()
    if os.environ.get("NORTHFLANK_API_TOKEN"):
        providers["northflank"] = NorthflankSandboxProvider.from_env()
    if os.environ.get("TOGETHER_API_KEY"):
        providers["together"] = TogetherSandboxProvider.from_env()
    return providers
