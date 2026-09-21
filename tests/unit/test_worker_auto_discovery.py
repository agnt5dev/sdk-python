"""Auto-registration finds a project's importable code and imports all of it once (AGNT5-1194).

Modules are imported under the names the application uses, so nothing runs
twice, and an import that fails stops the worker instead of leaving it Ready
with half its components.
"""

import importlib
import sys
import textwrap
import types
import uuid
from pathlib import Path

import pytest

from agnt5 import Context, function
from agnt5.agent import AgentRegistry
from agnt5.exceptions import AutoDiscoveryError, ConfigurationError
from agnt5.function import FunctionConfig, FunctionRegistry
from agnt5.scorer import ScorerRegistry
from agnt5.tool import ToolRegistry
from agnt5.worker._core import Worker
from agnt5.worker._discovery import (
    default_module_name,
    discover_source_paths,
    discovery_candidates,
    import_candidates,
)
from agnt5.workflow import WorkflowRegistry


@pytest.fixture(autouse=True)
def isolated_registries_and_imports():
    """Every test gets empty registries, and leaves sys.path and sys.modules as it found them."""

    registries = (FunctionRegistry, WorkflowRegistry, ScorerRegistry, ToolRegistry, AgentRegistry)
    for registry in registries:
        registry.clear()
    path_before = list(sys.path)
    modules_before = set(sys.modules)
    yield
    for registry in registries:
        registry.clear()
    sys.path[:] = path_before
    for name in set(sys.modules) - modules_before:
        del sys.modules[name]


def write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def component(name: str) -> str:
    return textwrap.dedent(f'''
        from agnt5 import Context, function

        @function(name="{name}")
        async def {name}(ctx: Context) -> str:
            return "ok"
    ''')


def unique_package() -> str:
    return f"disc_{uuid.uuid4().hex[:8]}"


# --- discover_source_paths ---------------------------------------------------


def test_uv_build_default_layout_is_src_over_the_normalized_project_name(tmp_path):
    write(
        tmp_path,
        "pyproject.toml",
        """
        [project]
        name = "My-Package.x"
        [build-system]
        requires = ["uv-build"]
        build-backend = "uv_build"
    """,
    )
    (tmp_path / "src" / "my_package_x").mkdir(parents=True)
    assert default_module_name("My-Package.x") == "my_package_x"
    assert discover_source_paths(tmp_path / "pyproject.toml") == ["src/my_package_x"]


def test_uv_build_explicit_module_root_and_names(tmp_path):
    write(
        tmp_path,
        "pyproject.toml",
        """
        [project]
        name = "whatever"
        [tool.uv.build-backend]
        module-root = "lib"
        module-name = ["alpha", "ns.beta"]
    """,
    )
    (tmp_path / "lib" / "alpha").mkdir(parents=True)
    (tmp_path / "lib" / "ns" / "beta").mkdir(parents=True)
    assert discover_source_paths(tmp_path / "pyproject.toml") == ["lib/alpha", "lib/ns/beta"]


def test_uv_build_missing_module_dir_scans_the_module_root(tmp_path):
    write(
        tmp_path,
        "pyproject.toml",
        """
        [project]
        name = "renamed"
        [build-system]
        build-backend = "uv_build"
    """,
    )
    (tmp_path / "src" / "actual_package").mkdir(parents=True)
    assert discover_source_paths(tmp_path / "pyproject.toml") == ["src"]


def test_hatch_and_maturin_layouts_are_unchanged(tmp_path):
    write(
        tmp_path,
        "pyproject.toml",
        """
        [tool.hatch.build.targets.wheel]
        packages = ["src/agnt5_quickstart"]
    """,
    )
    assert discover_source_paths(tmp_path / "pyproject.toml") == ["src/agnt5_quickstart"]
    write(
        tmp_path,
        "pyproject.toml",
        """
        [tool.maturin]
        python-source = "python"
    """,
    )
    assert discover_source_paths(tmp_path / "pyproject.toml") == ["python"]


def test_no_build_configuration_defaults_to_src(tmp_path):
    write(tmp_path, "pyproject.toml", '[project]\nname = "plain"\n')
    assert discover_source_paths(tmp_path / "pyproject.toml") == ["src"]
    assert discover_source_paths(tmp_path / "missing.toml") == ["src"]


# --- discovery_candidates ----------------------------------------------------


def test_src_fallback_names_modules_relative_to_src(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/workflow.py")
    write(tmp_path, f"src/{pkg}/tools/__init__.py")
    write(tmp_path, f"src/{pkg}/tools/jira.py")

    candidates = discovery_candidates([str(tmp_path / "src")])

    assert [c.module_name for c in candidates] == [
        pkg,
        f"{pkg}.tools",
        f"{pkg}.tools.jira",
        f"{pkg}.workflow",
    ]
    assert {c.import_root for c in candidates} == {tmp_path / "src"}
    assert not any(c.module_name.startswith("src.") for c in candidates)


def test_a_package_directory_is_named_from_its_parent(tmp_path):
    # The layout the templates' auto_register_paths=["src/<package>"] workaround scans.
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/workflow.py")

    candidates = discovery_candidates([str(tmp_path / "src" / pkg)])

    assert [c.module_name for c in candidates] == [pkg, f"{pkg}.workflow"]
    assert candidates[0].import_root == tmp_path / "src"


def test_candidates_are_sorted_deduplicated_and_skip_what_cannot_be_imported(tmp_path):
    pkg = unique_package()
    for name in ("zeta", "alpha", "mid"):
        write(tmp_path, f"src/{pkg}/{name}.py")
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/test_alpha.py")
    write(tmp_path, f"src/{pkg}/__pycache__/alpha.cpython-312.py")
    write(tmp_path, "src/not-a-package/mod.py")

    names = [
        c.module_name
        for c in discovery_candidates(
            [str(tmp_path / "src"), str(tmp_path / "src"), str(tmp_path / "src" / pkg)]
        )
    ]

    assert names == [pkg, f"{pkg}.alpha", f"{pkg}.mid", f"{pkg}.zeta"]


# --- import_candidates -------------------------------------------------------


def test_a_module_the_application_already_imported_is_not_executed_again(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/_state.py", "RUNS = 0\n")
    write(
        tmp_path,
        f"src/{pkg}/a.py",
        f"from {pkg} import _state\n_state.RUNS += 1\n" + component("first"),
    )
    sys.path.insert(0, str(tmp_path / "src"))
    importlib.import_module(f"{pkg}.a")  # what the application's own imports do

    failures = import_candidates(discovery_candidates([str(tmp_path / "src")]))

    assert failures == []
    assert sys.modules[f"{pkg}._state"].RUNS == 1
    assert f"src.{pkg}.a" not in sys.modules
    assert FunctionRegistry.get("first") is not None


def test_components_in_later_modules_are_not_dropped(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/a.py", component("from_a"))
    write(tmp_path, f"src/{pkg}/b.py", component("from_b"))
    sys.path.insert(0, str(tmp_path / "src"))
    importlib.import_module(f"{pkg}.a")

    failures = import_candidates(discovery_candidates([str(tmp_path / "src")]))

    assert failures == []
    assert sorted(FunctionRegistry.all()) == ["from_a", "from_b"]


def test_a_package_init_is_the_package(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py", component("in_init"))

    failures = import_candidates(discovery_candidates([str(tmp_path / "src")]))

    assert failures == []
    assert FunctionRegistry.get("in_init") is not None
    assert f"{pkg}.__init__" not in sys.modules and f"src.{pkg}.__init__" not in sys.modules


def test_an_import_failure_is_reported_with_context_and_rolled_back(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(
        tmp_path,
        f"src/{pkg}/broken.py",
        component("registered_before_failing") + "\nimport no_such_module_for_agnt5_1194\n",
    )
    write(tmp_path, f"src/{pkg}/fine.py", component("still_imported"))

    failures = import_candidates(discovery_candidates([str(tmp_path / "src")]))

    assert [f.module_name for f in failures] == [f"{pkg}.broken"]
    failure = failures[0]
    assert failure.path == tmp_path / "src" / pkg / "broken.py"
    assert failure.import_root == tmp_path / "src"
    assert isinstance(failure.error, ModuleNotFoundError)
    assert "no_such_module_for_agnt5_1194" in failure.describe()
    # Whatever the broken module registered before failing is gone, and the rest imported.
    assert FunctionRegistry.get("registered_before_failing") is None
    assert FunctionRegistry.get("still_imported") is not None


def test_worker_refuses_to_start_on_a_discovery_failure(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/workflow.py", "import no_such_module_for_agnt5_1194\n")
    worker = types.SimpleNamespace(service_name="test-worker")

    with pytest.raises(AutoDiscoveryError) as caught:
        Worker._auto_discover_components(worker, [str(tmp_path / "src")])

    message = str(caught.value)
    assert isinstance(caught.value, ConfigurationError)
    assert f"{pkg}.workflow" in message
    assert str(tmp_path / "src" / pkg / "workflow.py") in message
    assert f"import root {tmp_path / 'src'}" in message
    assert "No module named 'no_such_module_for_agnt5_1194'" in message
    assert not hasattr(worker, "_explicit_components")


def test_legitimate_duplicate_component_names_still_collide(tmp_path):
    pkg = unique_package()
    write(tmp_path, f"src/{pkg}/__init__.py")
    write(tmp_path, f"src/{pkg}/a.py", component("dup"))
    write(tmp_path, f"src/{pkg}/b.py", component("dup"))

    failures = import_candidates(discovery_candidates([str(tmp_path / "src")]))

    assert [f.module_name for f in failures] == [f"{pkg}.b"]
    assert "Function name collision: 'dup'" in str(failures[0].error)
    assert FunctionRegistry.get("dup").handler.__module__ == f"{pkg}.a"


def test_registering_the_same_function_object_twice_is_not_a_collision():
    @function(name="once")
    async def once(ctx: Context) -> str:
        return "ok"

    config = FunctionRegistry.get("once")
    FunctionRegistry.register(config)  # the same object again
    assert FunctionRegistry.get("once") is config

    async def other(ctx: Context) -> str:
        return "other"

    with pytest.raises(ValueError, match="Function name collision: 'once'"):
        FunctionRegistry.register(FunctionConfig(name="once", handler=other))
