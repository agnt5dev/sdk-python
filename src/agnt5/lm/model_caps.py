"""What a model accepts, decided from its id.

Mirrors ``lm::model_caps`` in sdk-core. Newer reasoning models reject sampling
parameters (``temperature``, ``top_p``) with a 400 instead of ignoring them:
OpenAI gpt-5 and later and the o-series, and Claude models after Opus 4.6 /
Sonnet 4.6 (AGNT5-1403, AGNT5-1456).
"""

from __future__ import annotations

import re

_BEDROCK_VERSION_SUFFIX = re.compile(r"-v\d+:\d+$")
_OPENAI_GPT = re.compile(r"^gpt-(\d+)")
_OPENAI_O_SERIES = re.compile(r"^o\d+(-|$)")
_CLAUDE_FAMILIES = frozenset({"opus", "sonnet", "haiku", "instant"})


def _bare_model(model: str) -> str:
    """Strip ``provider/`` prefixes, Bedrock region/vendor prefixes, a Vertex
    ``@version`` suffix and a Bedrock ``-v1:0`` suffix."""
    name = model.strip().rsplit("/", 1)[-1]
    if "anthropic." in name:
        name = name.split("anthropic.", 1)[1]
    name = name.split("@", 1)[0]
    return _BEDROCK_VERSION_SUFFIX.sub("", name)


def is_openai_reasoning_model(model: str) -> bool:
    """gpt-5 and every later ``gpt-N``, and the o-series (o1, o3, o4, ...)."""
    name = _bare_model(model)
    match = _OPENAI_GPT.match(name)
    if match:
        return int(match.group(1)) >= 5
    return bool(_OPENAI_O_SERIES.match(name))


def claude_rejects_sampling_params(model: str) -> bool:
    """Claude models that reject ``temperature``/``top_p``: everything after
    Opus 4.6 / Sonnet 4.6 / Haiku 4.5 (the cutoff is per family), including
    Fable. New or unrecognised Claude models count as rejecting."""
    name = _bare_model(model)
    if not name.startswith("claude-"):
        return False

    version: list[int] = []
    family = None
    for token in re.split(r"[-.]", name[len("claude-"):]):
        if token.isdigit() and len(token) <= 2:
            version.append(int(token))
            continue
        if not version and token not in _CLAUDE_FAMILIES:
            return True
        if version:
            break
        family = token

    if not version:
        return True
    # Newest accepting version per family: Haiku 4.5, Opus/Sonnet 4.6.
    # Version-first ids (claude-3-5-haiku) are all 3.x or older.
    last_accepting = (4, 5) if family == "haiku" else (4, 6)
    minor = version[1] if len(version) > 1 else 0
    return (version[0], minor) > last_accepting


def rejects_sampling_params(model: str) -> bool:
    """Whether ``model`` returns a 400 for ``temperature`` or ``top_p``."""
    return is_openai_reasoning_model(model) or claude_rejects_sampling_params(model)
