"""Third-party framework capture integrations.

Auto-attached at worker boot (see ``auto_enable``); each integration
soft-imports its target library and no-ops when it is absent. Gates:

- ``AGNT5_CAPTURE=off`` — master kill switch
- ``AGNT5_CAPTURE_OPENAI=0`` / ``AGNT5_CAPTURE_OPENAI_AGENTS=0`` /
  ``AGNT5_CAPTURE_GOOGLE_ADK=0`` — per library

All default on. Capture activates on library presence, not on the
``agnt5[openai]`` / ``agnt5[openai-agents]`` / ``agnt5[google-adk]`` extras
(those declare supported compatibility bands; the development lock pins the
versions exercised by CI).
"""

from __future__ import annotations

import logging

from ._common import library_capture_enabled, master_capture_enabled

logger = logging.getLogger(__name__)

_auto_enabled = False


def enable_openai_capture() -> bool:
    """Patch the raw ``openai`` client to emit lm.* journal events."""
    from . import openai as openai_integration

    return openai_integration.enable()


def enable_openai_agents_capture() -> bool:
    """Register a trace processor on the OpenAI Agents SDK."""
    from . import openai_agents

    return openai_agents.enable()


def enable_google_adk_capture() -> bool:
    """Attach the Google ADK capture plugin to every Runner."""
    from . import google_adk

    return google_adk.enable()


def auto_enable() -> None:
    """Enable capture for every supported library present in the process.

    Called at worker boot before user modules are imported. Never raises.
    """
    global _auto_enabled
    if _auto_enabled:
        return
    _auto_enabled = True

    if not master_capture_enabled():
        logger.debug("capture disabled via AGNT5_CAPTURE")
        return

    for flag, enabler in (
        ("AGNT5_CAPTURE_OPENAI", enable_openai_capture),
        ("AGNT5_CAPTURE_OPENAI_AGENTS", enable_openai_agents_capture),
        ("AGNT5_CAPTURE_GOOGLE_ADK", enable_google_adk_capture),
    ):
        if not library_capture_enabled(flag):
            logger.debug("capture disabled via %s", flag)
            continue
        try:
            enabler()
        except Exception:
            logger.warning("capture auto-enable failed for %s", flag, exc_info=True)


__all__ = [
    "auto_enable",
    "enable_google_adk_capture",
    "enable_openai_agents_capture",
    "enable_openai_capture",
]
