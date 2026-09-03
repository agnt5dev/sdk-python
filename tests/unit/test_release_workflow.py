from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"


def _triggers(path: Path) -> dict:
    # PyYAML parses the bare `on:` key as boolean True.
    return yaml.safe_load(path.read_text())[True]


def test_release_builds_one_portable_linux_wheel_per_architecture_and_sdist():
    workflow = RELEASE_WORKFLOW.read_text()

    assert "manylinux_2_39" not in workflow
    assert "build-wheels-linux-compatible" not in workflow
    assert workflow.count('manylinux: "2_28"') == 1
    assert 'manylinux: "2_28"' in workflow
    assert "quay.io/pypa/manylinux_2_28_x86_64:latest" in workflow
    assert "quay.io/pypa/manylinux_2_28_aarch64:latest" in workflow
    assert "quay.io/pypa/manylinux_2_34_x86_64:latest" in workflow
    assert "--find-links=/dist agnt5" in workflow
    assert "PyO3/maturin-action@v1" in workflow
    assert "command: sdist" in workflow
    assert "wheels-sdist" in workflow
    assert "dist/*.tar.gz" in workflow


def test_release_runs_only_on_published_github_releases():
    triggers = _triggers(RELEASE_WORKFLOW)

    assert set(triggers) == {"release"}
    assert triggers["release"] == {"types": ["published"]}


def test_ci_runs_on_pull_requests_and_manual_dispatch():
    triggers = _triggers(CI_WORKFLOW)

    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert "validate-pr:" in CI_WORKFLOW.read_text()


def test_workflows_use_persistent_compiler_cache():
    for path in (CI_WORKFLOW, RELEASE_WORKFLOW):
        workflow = path.read_text()

        assert "mozilla-actions/sccache-action@v0.0.11" in workflow
        assert 'SCCACHE_GHA_ENABLED: "true"' in workflow
        assert "RUSTC_WRAPPER: sccache" in workflow
        assert "SCCACHE_DIR" not in workflow
        assert "SCCACHE_VERSION" not in workflow
