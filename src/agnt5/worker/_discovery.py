"""Auto-registration: where a project's importable code lives, and importing all of it.

`Worker(auto_register=True)` imports every module under the project's source
directories so the components decorated in them register themselves. Each
module is imported under the name the application itself uses for it
(`code_reviewer.workflow`, never `src.code_reviewer.workflow`), so a module the
application already imported is not executed a second time, and a module that
fails to import is reported rather than skipped: a worker must not report
Ready with half of its components (AGNT5-1194).
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .._telemetry import setup_module_logger

logger = setup_module_logger(__name__)

DEFAULT_SOURCE_PATHS = ["src"]


@dataclass(frozen=True)
class DiscoveryCandidate:
    """A module to import: its canonical name, its file, and the directory that makes the name importable."""

    module_name: str
    path: Path
    import_root: Path


@dataclass(frozen=True)
class DiscoveryFailure:
    """A module that could not be imported."""

    module_name: str
    path: Path
    import_root: Path
    error: BaseException

    def describe(self) -> str:
        return (
            f"{self.module_name} ({self.path}, import root {self.import_root}): "
            f"{type(self.error).__name__}: {self.error}"
        )


def default_module_name(project_name: str) -> str:
    """The module name uv_build derives from a project name: normalized, with '-' and '.' as '_'."""

    return re.sub(r"[-_.]+", "_", project_name.strip()).lower()


def discover_source_paths(pyproject_path: str | Path | None = None) -> list[str]:
    """Source directories to scan, read from pyproject.toml.

    Understands, in this order:
    - Hatch: ``[tool.hatch.build.targets.wheel].packages``
    - Maturin: ``[tool.maturin].python-source``
    - uv_build: ``[tool.uv.build-backend]`` ``module-root`` (default ``src``) and
      ``module-name`` (default: the normalized project name), selected by the
      table or by ``build-backend = "uv_build"``
    - otherwise ``["src"]``

    Paths are returned as written in pyproject.toml, relative to the working directory.
    """

    try:
        import tomllib
    except ImportError:
        logger.error("tomllib not available (Python 3.11+ required for auto-registration)")
        return list(DEFAULT_SOURCE_PATHS)

    pyproject_file = Path(pyproject_path) if pyproject_path else Path.cwd() / "pyproject.toml"
    if not pyproject_file.exists():
        logger.warning(
            f"pyproject.toml not found at {pyproject_file}, defaulting to 'src/' directory"
        )
        return list(DEFAULT_SOURCE_PATHS)

    try:
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        logger.error(f"Failed to parse pyproject.toml: {e}")
        return list(DEFAULT_SOURCE_PATHS)

    tool = config.get("tool", {}) if isinstance(config.get("tool"), dict) else {}
    source_paths: list[str] = []

    hatch_config = tool.get("hatch", {})
    if (
        isinstance(hatch_config, dict)
        and "build" in hatch_config
        and "targets" in hatch_config["build"]
    ):
        wheel_config = hatch_config["build"]["targets"].get("wheel", {})
        source_paths.extend(wheel_config.get("packages", []))

    if not source_paths and isinstance(tool.get("maturin"), dict):
        python_source = tool["maturin"].get("python-source")
        if python_source:
            source_paths.append(python_source)

    build_backend = str(config.get("build-system", {}).get("build-backend", ""))
    uv_config = (
        tool.get("uv", {}).get("build-backend") if isinstance(tool.get("uv"), dict) else None
    )
    if not source_paths and (build_backend == "uv_build" or isinstance(uv_config, dict)):
        uv_config = uv_config if isinstance(uv_config, dict) else {}
        module_root = str(uv_config.get("module-root", "src"))
        module_names = uv_config.get("module-name") or default_module_name(
            str(config.get("project", {}).get("name", ""))
        )
        if isinstance(module_names, str):
            module_names = [module_names]
        for module_name in module_names:
            if not module_name:
                continue
            module_dir = Path(module_root, *str(module_name).split("."))
            if (pyproject_file.parent / module_dir).is_dir():
                source_paths.append(module_dir.as_posix())
            else:
                # The package is not where uv_build would build it from; the
                # import root is still right, and it names modules correctly.
                logger.warning(
                    f"uv_build module directory {module_dir} not found; scanning {module_root} instead"
                )
                source_paths.append(module_root)

    if not source_paths:
        source_paths = list(DEFAULT_SOURCE_PATHS)
    return list(dict.fromkeys(source_paths))


def discovery_candidates(source_paths: list[str]) -> list[DiscoveryCandidate]:
    """Every module under the source paths, named as the application imports it.

    A directory that is a package (has ``__init__.py``) is imported by its own
    name from its parent, so ``src/pkg`` yields ``pkg.*``. Any other directory
    is itself the import root, so ``src`` also yields ``pkg.*``: the name the
    installed package has, never ``src.pkg.*``. A package's ``__init__.py`` is
    the package. Candidates are deduplicated by name and sorted, so the import
    order does not depend on the filesystem.
    """

    by_name: dict[str, DiscoveryCandidate] = {}
    for source_path in dict.fromkeys(source_paths):
        path = Path(source_path)
        if not path.is_dir():
            logger.warning(f"Source path does not exist: {source_path}")
            continue
        path = path.resolve()
        import_root = path.parent if (path / "__init__.py").is_file() else path
        for py_file in sorted(path.rglob("*.py")):
            if "__pycache__" in py_file.parts or py_file.name.startswith("test_"):
                continue
            relative = py_file.relative_to(import_root)
            parts = list(relative.parts[:-1])
            if relative.stem != "__init__":
                parts.append(relative.stem)
            if not parts:
                continue
            if not all(part.isidentifier() for part in parts):
                logger.warning(
                    f"Skipping {py_file}: {'.'.join(parts)} is not an importable module name"
                )
                continue
            module_name = ".".join(parts)
            by_name.setdefault(module_name, DiscoveryCandidate(module_name, py_file, import_root))
    return sorted(by_name.values(), key=lambda candidate: candidate.module_name)


def import_candidates(candidates: list[DiscoveryCandidate]) -> list[DiscoveryFailure]:
    """Import each candidate under its canonical name; return the ones that failed.

    A module the application already imported is left alone, so its components
    register once. A module that fails part-way has whatever it registered
    before failing discarded, so a later attempt does not collide with it and a
    caller that swallows the failure does not serve a half-registered module.
    """

    failures: list[DiscoveryFailure] = []
    for candidate in candidates:
        import_root = str(candidate.import_root)
        if import_root not in sys.path:
            # First, so the project's own modules win over anything else of the same name.
            sys.path.insert(0, import_root)
        if candidate.module_name in sys.modules:
            logger.debug(f"Module already imported: {candidate.module_name}")
            continue
        registered_before = _registered_names()
        try:
            importlib.import_module(candidate.module_name)
            logger.debug(f"Auto-imported: {candidate.module_name}")
        except Exception as error:  # reported together, and fatal, in the worker
            _discard_registered_since(registered_before)
            failures.append(
                DiscoveryFailure(
                    candidate.module_name, candidate.path, candidate.import_root, error
                )
            )
    return failures


def _registries():
    from ..agent import AgentRegistry
    from ..function import FunctionRegistry
    from ..scorer import ScorerRegistry
    from ..tool import ToolRegistry
    from ..workflow import WorkflowRegistry

    return (FunctionRegistry, WorkflowRegistry, ScorerRegistry, ToolRegistry, AgentRegistry)


def _registered_names() -> list[set[str]]:
    return [set(registry.all().keys()) for registry in _registries()]


def _discard_registered_since(snapshot: list[set[str]]) -> None:
    for registry, before in zip(_registries(), snapshot):
        for name in set(registry.all().keys()) - before:
            registry.discard(name)
