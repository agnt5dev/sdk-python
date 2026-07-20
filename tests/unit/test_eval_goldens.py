"""Cross-language golden parity tests for builtin scorers.

Reads `sdk-core/test-fixtures/eval/builtin_goldens.json` (shared with the
Rust and TypeScript SDKs) and asserts each row produces the same
`(score, passed)` here as in the other two SDKs. Rows with a `label`
field also enforce label equality.

If a row fails here but passes in Rust / TS, the Python binding has
drifted — fix Python, not the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agnt5.eval import (
    ScorerInput,
    contains,
    exact_match,
    json_schema,
    json_valid,
    levenshtein,
    numeric_range,
    regex_match,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2].parent
    / "sdk-core"
    / "test-fixtures"
    / "eval"
    / "builtin_goldens.json"
)


def _load_goldens() -> list[dict]:
    with FIXTURE_PATH.open() as f:
        data = json.load(f)
    return data["cases"]


def _scorer_input(payload: dict) -> ScorerInput:
    out = payload.get("output")
    si = ScorerInput(output=out)
    if "expected" in payload:
        si = ScorerInput(output=out, expected=payload["expected"])
    if "input" in payload:
        # ScorerInput's constructor signature is (output, expected=None, input=None, trace=None)
        si = ScorerInput(
            output=out,
            expected=payload.get("expected"),
            input=payload["input"],
        )
    return si


def _run(case: dict):
    scorer = case["scorer"]
    payload = case["input"]
    cfg = payload.get("config") or {}
    si = _scorer_input(payload)

    if scorer == "exact_match":
        return exact_match(si, case_sensitive=cfg.get("case_sensitive"))
    if scorer == "contains":
        return contains(
            si,
            pattern=cfg.get("pattern", ""),
            case_sensitive=cfg.get("case_sensitive"),
        )
    if scorer == "json_valid":
        return json_valid(si)
    if scorer == "json_schema":
        return json_schema(si, schema=cfg.get("schema"))
    if scorer == "numeric_range":
        return numeric_range(
            si,
            min=cfg.get("min"),
            max=cfg.get("max"),
            inclusive=cfg.get("inclusive"),
        )
    if scorer == "regex_match":
        return regex_match(si, pattern=cfg.get("pattern", ""))
    if scorer == "levenshtein":
        return levenshtein(si, threshold=cfg.get("threshold"))
    raise ValueError(f"unknown scorer in goldens: {scorer!r}")


@pytest.mark.parametrize("case", _load_goldens(), ids=lambda c: c["name"])
def test_builtin_scorer_golden(case: dict) -> None:
    expected = case["expect"]
    result = _run(case)

    assert result.score == pytest.approx(expected["score"]), (
        f"[{case['name']}] score mismatch: got {result.score}, expected {expected['score']}"
    )
    assert result.passed == expected["passed"], (
        f"[{case['name']}] passed mismatch: got {result.passed}, expected {expected['passed']}"
    )
    if "label" in expected:
        assert result.label == expected["label"], (
            f"[{case['name']}] label mismatch: got {result.label!r}, expected {expected['label']!r}"
        )
