from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"


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


def test_release_uses_persistent_cache_and_keeps_full_matrix_off_pull_requests():
    workflow = RELEASE_WORKFLOW.read_text()

    assert "validate-pr:" in workflow
    assert "if: github.event_name == 'pull_request'" in workflow
    assert "if: github.event_name != 'pull_request'" in workflow
    assert "mozilla-actions/sccache-action@v0.0.11" in workflow
    assert 'SCCACHE_GHA_ENABLED: "true"' in workflow
    assert "RUSTC_WRAPPER: sccache" in workflow
    assert "SCCACHE_DIR" not in workflow
    assert "SCCACHE_VERSION" not in workflow
