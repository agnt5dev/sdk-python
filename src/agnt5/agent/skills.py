"""Skills - on-demand capability loading for agents.

A *skill* is a folder containing a ``SKILL.md`` file (Claude's open format):
YAML-style frontmatter with ``name`` and ``description``, followed by a markdown
body of instructions. Optional sibling files (scripts, reference docs) are the
skill's bundled resources.

Skills follow progressive disclosure: only ``name`` + ``description`` sit in the
agent's context (the *catalog*); the full body is loaded on demand when the
agent decides the skill is relevant. This keeps prompts small while giving the
agent access to many capabilities.

This module is pure parsing/resolution/catalog rendering. Wiring the catalog
into the system prompt and the ``load_skill`` tool lives in :mod:`agent.core`.

Usage:
    ```python
    from agnt5 import Agent

    # Curated subset by name, resolved against a shared pool directory
    agent_a = Agent(..., skills_dir="./skills", skills=["pdf-extraction", "sql-reporting"])

    # Load every skill in the pool (single-agent convenience)
    agent_c = Agent(..., skills_dir="./skills")

    # Self-contained Skill objects, no pool directory needed
    from agnt5 import Skill
    agent_d = Agent(..., skills=[Skill.from_path("./skills/pdf-extraction")])
    ```
"""

import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

from .skill_events import SkillLoaded

if TYPE_CHECKING:
    from ..tool import Tool

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"

__all__ = [
    "Skill",
    "discover_skills",
    "resolve_skills",
    "render_catalog",
    "make_load_skill_tool",
]


@dataclass
class Skill:
    """A loadable agent capability parsed from a ``SKILL.md`` folder.

    Attributes:
        name: Catalog identifier the agent uses to load the skill (frontmatter
            ``name``). This is what ``load_skill(name)`` expects.
        description: One-line trigger hint. Lives in the agent's context at all
            times, so the model can decide when the skill is relevant.
        instructions: The markdown body loaded into context on demand.
        resources_dir: Folder containing ``SKILL.md`` and any bundled scripts or
            reference files. ``None`` for skills constructed without a folder.
    """

    name: str
    description: str
    instructions: str
    resources_dir: Optional[Path] = None

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "Skill":
        """Parse a skill from a folder containing ``SKILL.md`` (or the file itself).

        Args:
            path: Path to the skill folder or directly to its ``SKILL.md``.

        Raises:
            FileNotFoundError: If no ``SKILL.md`` exists at the given path.
            ValueError: If the frontmatter is missing a ``name`` or ``description``.
        """
        p = Path(path)
        skill_file = p / SKILL_FILE if p.is_dir() else p
        if not skill_file.is_file():
            raise FileNotFoundError(f"No {SKILL_FILE} found at {p}")

        text = skill_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)

        name = meta.get("name")
        description = meta.get("description")
        if not name:
            raise ValueError(f"{skill_file}: frontmatter is missing required 'name'")
        if not description:
            raise ValueError(f"{skill_file}: frontmatter is missing required 'description'")

        return cls(
            name=name,
            description=description,
            instructions=body.strip(),
            resources_dir=skill_file.parent,
        )


def _parse_frontmatter(text: str) -> tuple[Dict[str, str], str]:
    """Split ``SKILL.md`` text into a frontmatter dict and the markdown body.

    Supports the minimal subset skills need: a leading ``---`` fenced block of
    single-line ``key: value`` pairs (values may be single- or double-quoted).
    No YAML dependency. Text without a frontmatter block parses to an empty
    dict and the full text as body.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta: Dict[str, str] = {}
    body_start = len(lines)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            body_start = i + 1
            break
        line = lines[i].strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("'\"")

    body = "\n".join(lines[body_start:])
    return meta, body


def discover_skills(skills_dir: Union[str, Path]) -> Dict[str, Skill]:
    """Parse every skill in a pool directory, keyed by skill name.

    A skill is any immediate subfolder containing a ``SKILL.md``. Malformed
    skills are skipped with a warning rather than failing discovery, so one bad
    folder doesn't break an agent that never selects it.
    """
    root = Path(skills_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"skills_dir does not exist: {root}")

    found: Dict[str, Skill] = {}
    for child in sorted(root.iterdir()):
        if not (child / SKILL_FILE).is_file():
            continue
        try:
            skill = Skill.from_path(child)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping malformed skill at %s: %s", child, exc)
            continue
        found[skill.name] = skill
    return found


def resolve_skills(
    skills: Optional[Sequence[Union[str, Skill]]],
    skills_dir: Optional[Union[str, Path]],
) -> Dict[str, Skill]:
    """Resolve an agent's skill selection into a ``{name: Skill}`` map.

    - ``skills`` items may be names (resolved against ``skills_dir``) or already
      constructed :class:`Skill` objects (passed through as-is).
    - ``skills`` omitted with a ``skills_dir`` loads every skill in the pool.
    - Neither provided yields an empty map (skills disabled — no behavior change).

    Raises:
        ValueError: If a named skill cannot be found in ``skills_dir`` (the error
            lists the available names), or if a name is given without a pool.
    """
    if skills is None:
        return discover_skills(skills_dir) if skills_dir is not None else {}

    available: Optional[Dict[str, Skill]] = None
    resolved: Dict[str, Skill] = {}
    for item in skills:
        if isinstance(item, Skill):
            resolved[item.name] = item
            continue
        if skills_dir is None:
            raise ValueError(
                f"Skill {item!r} given by name but no skills_dir was provided to resolve it against"
            )
        if available is None:
            available = discover_skills(skills_dir)
        skill = available.get(item)
        if skill is None:
            # Fall back to a direct folder match (folder name != frontmatter name)
            folder = Path(skills_dir) / item
            if (folder / SKILL_FILE).is_file():
                skill = Skill.from_path(folder)
            else:
                raise ValueError(
                    f"Skill {item!r} not found in {skills_dir}. "
                    f"Available: {', '.join(sorted(available)) or '(none)'}"
                )
        resolved[skill.name] = skill
    return resolved


def render_catalog(skills: Dict[str, Skill]) -> str:
    """Render the always-present ``<skills>`` catalog block for the system prompt.

    Only ``name`` + ``description`` are included — the progressive-disclosure
    contract. Returns an empty string when there are no skills, so callers can
    append unconditionally without changing prompts for skill-less agents.
    """
    if not skills:
        return ""

    lines = [
        "<skills>",
        "You have access to the following skills. When a task matches a skill's "
        "purpose, call load_skill(skill_name) to load its full instructions before "
        "proceeding.",
        "",
    ]
    for skill in skills.values():
        lines.append(f"- {skill.name}: {skill.description}")
    lines.append("</skills>")
    return "\n".join(lines)


async def _materialize_resources(ctx, skill: Skill, sandbox) -> Optional[tuple[str, list[str]]]:
    """Copy a skill's bundled files into the sandbox so its scripts can run.

    Files are written under ``skills/<name>/`` preserving their relative layout,
    excluding ``SKILL.md`` itself (already returned as the body). Writes are
    idempotent. Returns ``(base_dir, written_paths)``, or ``None`` when there is
    no sandbox or no resources to copy.
    """
    sandbox = sandbox or getattr(ctx, "sandbox", None)
    if sandbox is None or skill.resources_dir is None:
        return None

    base = f"skills/{skill.name}"
    written: list[str] = []
    for f in sorted(skill.resources_dir.rglob("*")):
        if not f.is_file() or f.name == SKILL_FILE:
            continue
        dest = f"{base}/{f.relative_to(skill.resources_dir).as_posix()}"
        await sandbox.write_file(dest, f.read_bytes())
        written.append(dest)

    if not written:
        return None
    return base, written


def _emit_skill_loaded(ctx, skill: Skill, resource_count: int) -> None:
    """Emit a ``skill.loaded`` event if the context supports emission.

    Guarded so the tool handler stays callable with a minimal context (e.g. in
    unit tests) that lacks ``emit``.
    """
    emit = getattr(ctx, "emit", None)
    if emit is None:
        return
    emit(
        SkillLoaded(
            name="load_skill",
            correlation_id=f"skill-{secrets.token_hex(5)}",
            parent_correlation_id=getattr(ctx, "correlation_id", None),
            skill_name=skill.name,
            instructions_length=len(skill.instructions),
            resources_materialized=resource_count,
            output_data={"skill": skill.name, "resources_materialized": resource_count},
        )
    )


def make_load_skill_tool(skills: Dict[str, Skill], sandbox=None) -> "Tool":
    """Build the ``load_skill`` tool that loads a skill's body on demand.

    The returned tool closes over the agent's resolved ``skills`` map and its
    sandbox. Invoking it returns the skill's instructions (progressive-disclosure
    level 2) and, when a sandbox is present, materializes any bundled resources.
    """
    from ..tool import Tool  # local import: keeps the parsing path Tool-free

    async def load_skill(ctx, skill_name: str) -> str:
        """Load the full instructions for a skill by name.

        Call this when the current task matches a skill listed in the <skills>
        catalog. Any files bundled with the skill are made available in the
        sandbox workspace.

        Args:
            skill_name: Name of the skill to load (as shown in the catalog).
        """
        skill = skills.get(skill_name)
        if skill is None:
            available = ", ".join(sorted(skills)) or "(none)"
            return f"Unknown skill {skill_name!r}. Available skills: {available}"

        materialized = await _materialize_resources(ctx, skill, sandbox)
        resource_count = len(materialized[1]) if materialized else 0
        _emit_skill_loaded(ctx, skill, resource_count)

        if not materialized:
            return skill.instructions

        base, paths = materialized
        files = "\n".join(f"- {p}" for p in paths)
        return (
            f"{skill.instructions}\n\n---\n"
            f"Bundled resources are available in the sandbox under '{base}':\n{files}"
        )

    return Tool(
        name="load_skill",
        description=(
            "Load the full instructions for a skill by name. Call this when a task "
            "matches a skill listed in the <skills> catalog before proceeding."
        ),
        handler=load_skill,
        auto_schema=True,
    )
