"""Skill-specific events for journal and observability.

Emitted when an agent loads a skill on demand, so the run timeline shows which
skills were pulled into context and whether their bundled resources were
materialized into the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..events import Completed, ComponentType


@dataclass(kw_only=True)
class SkillLoaded(Completed):
    """A skill's instructions were loaded into the agent context."""

    _event_type: ClassVar[str] = "skill.loaded"
    component_type: ComponentType = field(default=ComponentType.TOOL, init=False)
    skill_name: str = ""
    instructions_length: int = 0
    resources_materialized: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self._event_type)
