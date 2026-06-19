"""AGENTS.md - always-on project/area guidance for agents.

``AGENTS.md`` is the open format for ambient operating instructions ("how to
work in this repo/area"). Unlike skills, it has no trigger metadata and is not
loaded on demand — its content sits in the agent's context at all times.

It is hierarchical: a root ``AGENTS.md`` plus more specific ones deeper in the
tree, where the more specific guidance wins. This module loads explicit
file/directory sources and offers a bounded upward discovery helper.

This pairs with on-demand skills (see :mod:`agent.skills`): guidance is the
always-on layer, skills are the on-demand layer. Both feed the same system
prompt composition in :mod:`agent.core`.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

AGENTS_FILE = "AGENTS.md"

AgentsMdSource = Union[str, Path, Sequence[Union[str, Path]]]

__all__ = ["discover_agents_md", "load_agents_md", "render_guidance"]


def discover_agents_md(start_dir: Union[str, Path], *, stop_at_git: bool = True) -> List[Path]:
    """Walk upward from ``start_dir`` collecting ``AGENTS.md`` files.

    Returned outermost-first so the most specific file (closest to
    ``start_dir``) comes last and therefore wins on concatenation. Bounded by
    the repo root (a directory containing ``.git``) when ``stop_at_git`` is set,
    otherwise by the filesystem root — never an unbounded walk.
    """
    start = Path(start_dir).resolve()
    found: List[Path] = []
    for d in [start, *start.parents]:
        candidate = d / AGENTS_FILE
        if candidate.is_file():
            found.append(candidate)
        if stop_at_git and (d / ".git").exists():
            break
    found.reverse()  # outermost first, most specific last
    return found


def load_agents_md(source: Optional[AgentsMdSource]) -> str:
    """Load and concatenate ``AGENTS.md`` content from one or more sources.

    Each source may be a file path or a directory (which uses its
    ``AGENTS.md``). A sequence is loaded in order, so later entries are treated
    as more specific. Missing files are skipped. Returns ``""`` when nothing is
    found, leaving skill-less/guidance-less agents unchanged.
    """
    if source is None:
        return ""

    items: Sequence[Union[str, Path]]
    if isinstance(source, (str, Path)):
        items = [source]
    else:
        items = source

    parts: List[str] = []
    for item in items:
        p = Path(item)
        f = p / AGENTS_FILE if p.is_dir() else p
        if f.is_file():
            text = f.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def render_guidance(text: str) -> str:
    """Wrap loaded guidance in the always-on ``<project-guidance>`` block.

    Returns ``""`` for empty text so callers can append unconditionally.
    """
    if not text:
        return ""
    return f"<project-guidance>\n{text}\n</project-guidance>"
