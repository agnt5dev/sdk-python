"""Unit tests for sandbox provider integrations (wire-format helpers).

These mirror the Rust unit tests in sdk-core/src/sandbox/providers/ — the
Rust types are canonical and the Python clients conform to them.
"""

import gzip
import io
import tarfile

import pytest

from agnt5.sandbox_providers import (
    CreateSandboxOptions,
    DaytonaSandboxProvider,
    E2BSandboxProvider,
    NorthflankSandbox,
    NorthflankSandboxProvider,
    SandboxProviderError,
    TogetherSandbox,
    VercelSandboxProvider,
    _interpreter_argv,
    _interpreter_command_line,
    _parse_listing_output,
    load_providers_from_env,
)


class TestSharedHelpers:
    def test_interpreter_argv(self):
        assert _interpreter_argv("python", "print(1)") == ("python3", ["-c", "print(1)"])
        assert _interpreter_argv("javascript", "x") == ("node", ["-e", "x"])
        assert _interpreter_argv("bash", "echo hi") == ("bash", ["-c", "echo hi"])

    def test_interpreter_argv_unsupported(self):
        with pytest.raises(SandboxProviderError):
            _interpreter_argv("cobol", "x")

    def test_interpreter_command_line_quotes(self):
        line = _interpreter_command_line("python", "print('hi')")
        assert line.startswith("python3 -c ")
        assert "'hi'" in line or '"hi"' in line

    def test_parse_listing_output(self):
        stdout = (
            "f|42|644|1718200000.5|/workspace/test.txt\n"
            "d|4096|755|1718200001.0|/workspace/src\n"
            "bogus line\n"
        )
        files = _parse_listing_output(stdout)
        assert len(files) == 2
        assert files[0].name == "test.txt"
        assert files[0].size == 42
        assert files[0].mode == 0o644
        assert not files[0].is_dir
        assert files[0].mod_time == 1718200000500
        assert files[1].is_dir
        assert files[1].name == "src"


class TestEnvDetection:
    def test_nothing_configured(self, monkeypatch):
        for var in (
            "E2B_API_KEY",
            "DAYTONA_API_KEY",
            "VERCEL_OIDC_TOKEN",
            "VERCEL_TOKEN",
            "NORTHFLANK_API_TOKEN",
            "TOGETHER_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        assert load_providers_from_env() == {}

    def test_e2b_detected(self, monkeypatch):
        monkeypatch.setenv("E2B_API_KEY", "e2b_test")
        monkeypatch.delenv("E2B_DOMAIN", raising=False)
        providers = {
            k: v for k, v in load_providers_from_env().items() if k == "e2b"
        }
        assert "e2b" in providers
        assert providers["e2b"].api_url == "https://api.e2b.app"

    def test_vercel_partial_config_raises(self, monkeypatch):
        monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)
        monkeypatch.setenv("VERCEL_TOKEN", "tok")
        monkeypatch.delenv("VERCEL_TEAM_ID", raising=False)
        monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)
        with pytest.raises(SandboxProviderError):
            VercelSandboxProvider.from_env()

    def test_northflank_requires_project(self, monkeypatch):
        monkeypatch.setenv("NORTHFLANK_API_TOKEN", "tok")
        monkeypatch.delenv("NORTHFLANK_PROJECT_ID", raising=False)
        with pytest.raises(SandboxProviderError):
            NorthflankSandboxProvider.from_env()


class TestE2B:
    def test_handle_urls(self):
        provider = E2BSandboxProvider("e2b_test")
        sandbox = provider._handle({"sandboxID": "abc123"})
        assert sandbox.preview_url(3000) == "https://3000-abc123.e2b.app"
        assert sandbox._envd_url == "https://49983-abc123.e2b.app"
        assert sandbox._interpreter_url == "https://49999-abc123.e2b.app"

    def test_handle_uses_response_domain(self):
        provider = E2BSandboxProvider("e2b_test")
        sandbox = provider._handle({"sandboxID": "abc", "domain": "e2b.dev"})
        assert sandbox.preview_url(80) == "https://80-abc.e2b.dev"


class TestDaytona:
    def test_create_options_mapping(self):
        opts = CreateSandboxOptions(memory_mib=1500, timeout_secs=90)
        assert (opts.memory_mib + 1023) // 1024 == 2  # GB round-up
        assert (opts.timeout_secs + 59) // 60 == 2  # minutes round-up


class TestVercel:
    def test_command_stream_parsing(self):
        provider = VercelSandboxProvider("tok", team_id="t", project_id="p")
        sandbox = provider._handle(
            {
                "sandbox": {"name": "agnt5-x"},
                "session": {"id": "sess_1"},
                "routes": [{"url": "https://x.vercel.run", "port": 3000}],
            }
        )
        assert sandbox.preview_url(3000) == "https://x.vercel.run"
        assert sandbox.preview_url(9999) is None
        assert sandbox.sandbox_id == "agnt5-x"

    def test_tar_gz_roundtrip(self):
        # Reproduce the fs/write archive shape used by write_file.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name="dir/test.txt")
            info.size = 5
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(b"hello"))
        archive = gzip.compress(buf.getvalue())
        assert archive[:2] == b"\x1f\x8b"

        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(archive))) as tar:
            member = tar.getmembers()[0]
            assert member.name == "dir/test.txt"
            assert member.mode == 0o644
            assert tar.extractfile(member).read() == b"hello"


class TestNorthflank:
    def _sandbox(self, team_id=None):
        return NorthflankSandbox(
            service_id="svc",
            project_id="proj",
            api_token="tok",
            team_id=team_id,
            base_url="https://api.northflank.com",
            client=None,
        )

    def test_ws_url(self):
        assert (
            self._sandbox()._ws_url()
            == "wss://api.northflank.com/v1/command-exec/projects/proj/services/svc"
        )

    def test_ws_url_with_team(self):
        assert (
            self._sandbox(team_id="team")._ws_url()
            == "wss://api.northflank.com/v1/command-exec/teams/team/projects/proj/services/svc"
        )

    def test_deployment_status(self):
        ds = NorthflankSandboxProvider._deployment_status
        assert ds({"deployment": {"status": "RUNNING"}}) == "RUNNING"
        assert ds({"status": "PAUSED"}) == "PAUSED"
        assert ds(None) == "unknown"


class TestTogether:
    def test_map_outputs_success(self):
        result = TogetherSandbox._map_outputs(
            {
                "session_id": "ses_1",
                "status": "success",
                "outputs": [
                    {"type": "stdout", "data": "hello\n"},
                    {"type": "stderr", "data": "warn\n"},
                    {"type": "execute_result", "data": {"text/plain": "42"}},
                ],
            }
        )
        assert result.stdout == "hello\n42"
        assert result.stderr == "warn\n"
        assert result.exit_code == 0
        assert result.error is None

    def test_map_outputs_error(self):
        result = TogetherSandbox._map_outputs(
            {
                "session_id": "ses_1",
                "status": "error",
                "outputs": [{"type": "error", "data": "NameError: x"}],
            }
        )
        assert result.exit_code == 1
        assert result.error == "NameError: x"


class TestAgentToolsCompatibility:
    """Provider sandboxes must satisfy the surface agents use via sandbox_tools."""

    def test_provider_sandboxes_are_sandbox_like(self):
        from agnt5.sandbox_tools import SandboxLike

        provider = E2BSandboxProvider("e2b_test")
        sandbox = provider._handle({"sandboxID": "abc"})
        assert isinstance(sandbox, SandboxLike)

        nf = NorthflankSandbox(
            service_id="s",
            project_id="p",
            api_token="t",
            team_id=None,
            base_url="https://api.northflank.com",
            client=None,
        )
        assert isinstance(nf, SandboxLike)
