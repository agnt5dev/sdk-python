"""Classification judges return usable labels or surface scorer errors."""

import importlib
import json
from types import SimpleNamespace

import pytest

from agnt5.eval.llm_judge import LLMJudgeConfig, llm_judge


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ({"score": 0.95, "passed": True}, "Pass"),
        ({"score": 0.1, "passed": False}, "Fail"),
        ({"passed": True}, "Pass"),
        ({"score": 0.95}, "Pass"),
        ({"label": "Fail", "passed": True, "score": 0.95}, "Fail"),
    ],
)
async def test_pass_fail_judge_infers_missing_label(monkeypatch, response, label):
    judge_module = importlib.import_module("agnt5.eval.llm_judge")
    captured = {}

    async def generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text=json.dumps(response))

    monkeypatch.setattr(judge_module, "_get_generate", lambda: generate)
    result = await llm_judge(
        "84",
        LLMJudgeConfig(criteria="Is the answer correct?", choice_scores={"Fail": 0, "Pass": 1}),
    )
    assert result.label == label
    assert result.score == (1 if label == "Pass" else 0)
    system_prompt = captured["messages"][0]["content"]
    assert '"label"' in system_prompt
    assert "Fail" in system_prompt and "Pass" in system_prompt


@pytest.mark.parametrize("response", [{}, {"label": "Maybe"}, {"passed": "true"}, "not json"])
async def test_unusable_classification_response_raises(monkeypatch, response):
    judge_module = importlib.import_module("agnt5.eval.llm_judge")

    async def generate(**kwargs):
        return SimpleNamespace(text=response if isinstance(response, str) else json.dumps(response))

    monkeypatch.setattr(judge_module, "_get_generate", lambda: generate)
    with pytest.raises(ValueError, match="[Jj]udge"):
        await llm_judge(
            "84",
            LLMJudgeConfig(criteria="Correct?", choice_scores={"Fail": 0, "Pass": 1}),
        )


async def test_classification_model_failure_raises(monkeypatch):
    judge_module = importlib.import_module("agnt5.eval.llm_judge")

    async def generate(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(judge_module, "_get_generate", lambda: generate)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await llm_judge(
            "84", LLMJudgeConfig(criteria="Correct?", choice_scores={"Fail": 0, "Pass": 1})
        )


@pytest.mark.parametrize(("score", "label"), [(0.8, "Good"), (0.6, "Partial"), (0.75, None)])
async def test_multiclass_missing_label_uses_unique_nearest_score(monkeypatch, score, label):
    judge_module = importlib.import_module("agnt5.eval.llm_judge")

    async def generate(**kwargs):
        return SimpleNamespace(text=json.dumps({"score": score}))

    monkeypatch.setattr(judge_module, "_get_generate", lambda: generate)
    config = LLMJudgeConfig(
        criteria="Quality?", choice_scores={"Bad": 0, "Partial": 0.5, "Good": 1}
    )
    if label is None:
        with pytest.raises(ValueError, match="Judge"):
            await llm_judge("84", config)
    else:
        assert (await llm_judge("84", config)).label == label


@pytest.mark.parametrize("choices", [{}, {"": 1}, {"Pass": 2}, {"Pass": float("nan")}])
async def test_invalid_classification_config_is_an_error(choices):
    with pytest.raises(ValueError, match="Judge"):
        await llm_judge("84", LLMJudgeConfig(criteria="Correct?", choice_scores=choices))
