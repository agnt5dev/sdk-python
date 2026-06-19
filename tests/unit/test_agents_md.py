"""Unit tests for AGENTS.md always-on guidance loading and agent wiring."""

from agnt5 import Agent, discover_agents_md, load_agents_md
from agnt5.agent.agents_md import render_guidance


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_none_is_empty():
    assert load_agents_md(None) == ""


def test_load_file_path(tmp_path):
    f = tmp_path / "AGENTS.md"
    _write(f, "Be concise.")
    assert load_agents_md(f) == "Be concise."


def test_load_directory_uses_agents_md(tmp_path):
    _write(tmp_path / "AGENTS.md", "Repo rules.")
    assert load_agents_md(tmp_path) == "Repo rules."


def test_load_list_concatenates_in_order(tmp_path):
    _write(tmp_path / "AGENTS.md", "Root rules.")
    _write(tmp_path / "sub" / "AGENTS.md", "Sub rules.")
    out = load_agents_md([tmp_path / "AGENTS.md", tmp_path / "sub" / "AGENTS.md"])
    assert out == "Root rules.\n\nSub rules."


def test_load_skips_missing(tmp_path):
    _write(tmp_path / "AGENTS.md", "Only one.")
    out = load_agents_md([tmp_path / "AGENTS.md", tmp_path / "missing" / "AGENTS.md"])
    assert out == "Only one."


def test_render_guidance_empty_is_blank():
    assert render_guidance("") == ""


def test_render_guidance_wraps():
    block = render_guidance("Rules.")
    assert block.startswith("<project-guidance>")
    assert block.endswith("</project-guidance>")
    assert "Rules." in block


def test_discover_walks_up_to_git_boundary(tmp_path):
    (tmp_path / ".git").mkdir()
    _write(tmp_path / "AGENTS.md", "root")
    _write(tmp_path / "a" / "b" / "AGENTS.md", "leaf")
    # A parent above the repo root must not be collected
    _write(tmp_path.parent / "AGENTS.md", "outside-repo")

    found = discover_agents_md(tmp_path / "a" / "b")

    names = [str(p) for p in found]
    assert names[0].endswith("/AGENTS.md")  # outermost (root) first
    assert found[0] == tmp_path / "AGENTS.md"
    assert found[-1] == tmp_path / "a" / "b" / "AGENTS.md"  # most specific last
    assert (tmp_path.parent / "AGENTS.md") not in found


# --- agent wiring ------------------------------------------------------------


def test_agent_injects_guidance_before_skills(tmp_path):
    _write(tmp_path / "AGENTS.md", "Follow repo conventions.")
    skills_dir = tmp_path / "skills"
    _write(skills_dir / "pdf" / "SKILL.md", "---\nname: pdf\ndescription: Extract PDFs\n---\nbody")

    agent = Agent(
        name="t",
        model="openai/gpt-4o-mini",
        instructions="Base.",
        agents_md=tmp_path,
        skills=["pdf"],
        skills_dir=skills_dir,
    )

    prompt = agent._compose_system_prompt()
    assert "Base." in prompt
    assert "<project-guidance>" in prompt
    assert "Follow repo conventions." in prompt
    assert "<skills>" in prompt
    # ambient guidance precedes the on-demand skills catalog
    assert prompt.index("<project-guidance>") < prompt.index("<skills>")


def test_agent_without_guidance_unchanged():
    agent = Agent(name="t", model="openai/gpt-4o-mini", instructions="Base.")
    assert agent._agents_md_guidance == ""
    assert agent._compose_system_prompt() == "Base."
