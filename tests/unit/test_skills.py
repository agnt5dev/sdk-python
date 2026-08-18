"""Unit tests for agent skills parsing, resolution, and catalog rendering."""

from types import SimpleNamespace

import pytest

from agnt5 import Agent
from agnt5 import Skill as ExportedSkill
from agnt5 import discover_skills as exported_discover_skills
from agnt5 import resolve_skills as exported_resolve_skills
from agnt5.agent import discover_skills as agent_discover_skills
from agnt5.agent import resolve_skills as agent_resolve_skills
from agnt5.agent.skill_events import SkillLoaded
from agnt5.agent.skills import (
    Skill,
    discover_skills,
    make_load_skill_tool,
    render_catalog,
    resolve_skills,
)


class _FakeSandbox:
    """Records write_file calls for materialization tests."""

    def __init__(self):
        self.writes = {}

    async def write_file(self, path, content):
        self.writes[path] = content


class _RecordingCtx:
    """Minimal context that records emitted events, like a real run context."""

    def __init__(self, sandbox=None):
        self.sandbox = sandbox
        self.correlation_id = "tool-call-1"
        self.events = []

    def emit(self, event):
        self.events.append(event)


SKILL_BODY = """---
name: pdf-extraction
description: Extract tables and text from PDFs
---

# PDF Extraction

Run `scripts/extract.py` against the uploaded PDF.
"""


def test_skill_helpers_are_exported_from_public_namespaces():
    assert exported_discover_skills is discover_skills
    assert exported_resolve_skills is resolve_skills
    assert agent_discover_skills is discover_skills
    assert agent_resolve_skills is resolve_skills


def _write_skill(root, folder, name, description, body="Do the thing."):
    skill_dir = root / folder
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_from_path_parses_frontmatter_and_body(tmp_path):
    skill_dir = tmp_path / "pdf-extraction"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(SKILL_BODY, encoding="utf-8")

    skill = Skill.from_path(skill_dir)

    assert skill.name == "pdf-extraction"
    assert skill.description == "Extract tables and text from PDFs"
    assert skill.instructions.startswith("# PDF Extraction")
    assert "scripts/extract.py" in skill.instructions
    assert skill.resources_dir == skill_dir


def test_from_path_accepts_file_directly(tmp_path):
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    f = skill_dir / "SKILL.md"
    f.write_text(SKILL_BODY, encoding="utf-8")

    skill = Skill.from_path(f)
    assert skill.name == "pdf-extraction"


def test_from_path_handles_quoted_values(tmp_path):
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: \"quoted-name\"\ndescription: 'has: a colon'\n---\nbody\n",
        encoding="utf-8",
    )
    skill = Skill.from_path(skill_dir)
    assert skill.name == "quoted-name"
    assert skill.description == "has: a colon"


def test_from_path_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Skill.from_path(tmp_path / "nope")


def test_from_path_missing_required_frontmatter(tmp_path):
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: only-name\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="description"):
        Skill.from_path(skill_dir)


def test_discover_skips_malformed(tmp_path):
    _write_skill(tmp_path, "good", "good", "A good skill")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: incomplete\n---\n", encoding="utf-8")
    # A folder with no SKILL.md is ignored entirely
    (tmp_path / "not-a-skill").mkdir()

    found = discover_skills(tmp_path)
    assert set(found) == {"good"}


def test_resolve_none_without_dir_is_empty():
    assert resolve_skills(None, None) == {}


def test_resolve_none_with_dir_loads_all(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A")
    _write_skill(tmp_path, "b", "b", "Skill B")
    resolved = resolve_skills(None, tmp_path)
    assert set(resolved) == {"a", "b"}


def test_resolve_subset_by_name(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A")
    _write_skill(tmp_path, "b", "b", "Skill B")
    _write_skill(tmp_path, "c", "c", "Skill C")

    resolved = resolve_skills(["a", "c"], tmp_path)
    assert set(resolved) == {"a", "c"}


def test_resolve_unknown_name_lists_available(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A")
    with pytest.raises(ValueError, match="Available: a"):
        resolve_skills(["missing"], tmp_path)


def test_resolve_name_without_dir_raises():
    with pytest.raises(ValueError, match="no skills_dir"):
        resolve_skills(["a"], None)


def test_resolve_passes_through_skill_objects(tmp_path):
    skill_dir = _write_skill(tmp_path, "a", "a", "Skill A")
    obj = Skill.from_path(skill_dir)
    resolved = resolve_skills([obj], None)
    assert resolved == {"a": obj}


def test_resolve_folder_name_differs_from_skill_name(tmp_path):
    # Folder "folder-x" holds a skill named "real-name"
    _write_skill(tmp_path, "folder-x", "real-name", "Real")
    resolved = resolve_skills(["folder-x"], tmp_path)
    assert set(resolved) == {"real-name"}


def test_render_catalog_empty_is_blank():
    assert render_catalog({}) == ""


def test_render_catalog_lists_name_and_description(tmp_path):
    _write_skill(tmp_path, "a", "pdf-extraction", "Extract from PDFs")
    _write_skill(tmp_path, "b", "sql-reporting", "Run SQL reports")
    catalog = render_catalog(resolve_skills(["pdf-extraction", "sql-reporting"], tmp_path))

    assert "<skills>" in catalog and "</skills>" in catalog
    assert "- pdf-extraction: Extract from PDFs" in catalog
    assert "- sql-reporting: Run SQL reports" in catalog
    assert "load_skill(skill_name)" in catalog


# --- load_skill tool ---------------------------------------------------------


async def test_load_skill_returns_body(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A", body="Step 1. Do the thing.")
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path), sandbox=None)
    out = await tool.handler(SimpleNamespace(sandbox=None), skill_name="a")
    assert "Step 1. Do the thing." in out


async def test_load_skill_unknown_lists_available(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A")
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path))
    out = await tool.handler(SimpleNamespace(sandbox=None), skill_name="missing")
    assert "Unknown skill" in out
    assert "a" in out


async def test_load_skill_materializes_resources_into_sandbox(tmp_path):
    skill_dir = _write_skill(tmp_path, "a", "a", "Skill A", body="run scripts/x.py")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "x.py").write_text("print('hi')", encoding="utf-8")

    sandbox = _FakeSandbox()
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path), sandbox=sandbox)
    out = await tool.handler(SimpleNamespace(sandbox=sandbox), skill_name="a")

    assert "skills/a/scripts/x.py" in sandbox.writes
    assert "skills/a/SKILL.md" not in sandbox.writes  # body excluded
    assert "Bundled resources" in out
    assert "skills/a/scripts/x.py" in out


async def test_load_skill_no_sandbox_returns_plain_body(tmp_path):
    skill_dir = _write_skill(tmp_path, "a", "a", "Skill A", body="body text")
    (skill_dir / "extra.txt").write_text("data", encoding="utf-8")
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path), sandbox=None)
    out = await tool.handler(SimpleNamespace(sandbox=None), skill_name="a")
    assert out.strip() == "body text"


async def test_load_skill_emits_skill_loaded_event(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A", body="instructions here")
    ctx = _RecordingCtx()
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path))
    await tool.handler(ctx, skill_name="a")

    loaded = [e for e in ctx.events if isinstance(e, SkillLoaded)]
    assert len(loaded) == 1
    event = loaded[0]
    assert event.event_type == "skill.loaded"
    assert event.skill_name == "a"
    assert event.instructions_length == len("instructions here")
    assert event.resources_materialized == 0
    assert event.parent_correlation_id == "tool-call-1"


async def test_load_skill_event_counts_materialized_resources(tmp_path):
    skill_dir = _write_skill(tmp_path, "a", "a", "Skill A")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "x.py").write_text("x", encoding="utf-8")
    (skill_dir / "ref.md").write_text("ref", encoding="utf-8")

    sandbox = _FakeSandbox()
    ctx = _RecordingCtx(sandbox=sandbox)
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path), sandbox=sandbox)
    await tool.handler(ctx, skill_name="a")

    event = next(e for e in ctx.events if isinstance(e, SkillLoaded))
    assert event.resources_materialized == 2


async def test_load_skill_unknown_emits_no_event(tmp_path):
    _write_skill(tmp_path, "a", "a", "Skill A")
    ctx = _RecordingCtx()
    tool = make_load_skill_tool(resolve_skills(["a"], tmp_path))
    await tool.handler(ctx, skill_name="missing")
    assert [e for e in ctx.events if isinstance(e, SkillLoaded)] == []


# --- agent wiring ------------------------------------------------------------


def test_skill_is_exported():
    assert ExportedSkill is Skill


def test_agent_registers_loader_and_injects_catalog(tmp_path):
    _write_skill(tmp_path, "a", "pdf", "Extract PDFs")
    _write_skill(tmp_path, "b", "sql", "Run SQL")

    agent = Agent(
        name="t",
        model="openai/gpt-4o-mini",
        instructions="Base instructions.",
        skills=["pdf"],
        skills_dir=tmp_path,
    )

    assert "load_skill" in agent.tools
    assert set(agent._skills) == {"pdf"}

    prompt = agent._compose_system_prompt()
    assert "Base instructions." in prompt
    assert "<skills>" in prompt
    assert "- pdf: Extract PDFs" in prompt
    assert "sql" not in prompt  # unselected skill never reaches context


def test_agent_without_skills_is_unchanged():
    agent = Agent(name="t", model="openai/gpt-4o-mini", instructions="Base instructions.")
    assert "load_skill" not in agent.tools
    assert agent._skills == {}
    assert agent._compose_system_prompt() == "Base instructions."
