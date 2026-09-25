"""SDK-core-owned assertion contract through the installed native binding."""

import json
from pathlib import Path

import pytest

from agnt5.eval import ScorerInput, structured_assertions
from agnt5.eval.types import ScorerRequest
from agnt5.scorer import RESERVED_BUILTIN_SCORER_NAMES, run_scorer

CASES = json.loads((Path(__file__).parents[1] / "fixtures/structured_assertions.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_structured_assertions(case):
    payload = case["input"]
    result = structured_assertions(
        ScorerInput(
            output=payload["output"], input=payload.get("input"), expected=payload.get("expected")
        ),
        payload["config"],
    )
    for key in ("score", "passed", "label"):
        assert getattr(result, key) == case["expect"][key]


@pytest.mark.asyncio
async def test_builtin_dispatch_requires_no_user_registration():
    assert "structured_assertions" in RESERVED_BUILTIN_SCORER_NAMES
    result = await run_scorer(
        "structured_assertions",
        ScorerRequest(output=[1, 2], config={"assertions": [{"expr": "unique(output)"}]}),
    )
    assert result.passed
    assert result.metadata["assertions"][0]["passed"] is True
