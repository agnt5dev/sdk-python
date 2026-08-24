from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "release.yml"


def test_release_builds_vercel_compatible_linux_wheels_and_sdist():
    workflow = RELEASE_WORKFLOW.read_text()

    assert "--compatibility manylinux_2_39" in workflow
    assert 'manylinux: "2_28"' in workflow
    assert "quay.io/pypa/manylinux_2_28_x86_64:latest" in workflow
    assert "quay.io/pypa/manylinux_2_28_aarch64:latest" in workflow
    assert "quay.io/pypa/manylinux_2_34_x86_64:latest" in workflow
    assert "--find-links=/dist agnt5" in workflow
    assert "PyO3/maturin-action@v1" in workflow
    assert "command: sdist" in workflow
    assert "wheels-sdist" in workflow
    assert "dist/*.tar.gz" in workflow
