"""Tests for AGNT5 Evaluation Framework.

Tests cover:
- Deterministic scorers (exact_match, contains, json_valid, regex_match, levenshtein)
- Trace assertions (glassbox testing)
- Custom scorers (@scorer decorator)
- LLM Judge (Python wrapper)
"""

import pytest

from agnt5.eval import (
    ScorerInput,
    ScorerResult,
    TraceAssertion,
    contains,
    exact_match,
    json_valid,
    levenshtein,
    regex_match,
    trace_scorer,
)
from agnt5.eval.scorer import (
    clear_custom_scorers,
    get_custom_scorer,
    get_scorer_info,
    is_scorer,
    list_custom_scorers,
    scorer,
)
from agnt5.eval.types import EvalContext, ScorerResultPy


@pytest.fixture(autouse=True)
def clear_scorers():
    """Clear custom scorer registry before and after each test."""
    clear_custom_scorers()
    yield
    clear_custom_scorers()


class TestScorerInput:
    """Test ScorerInput construction."""

    def test_basic_input(self):
        """Test creating a basic ScorerInput."""
        input_obj = ScorerInput(output="hello")
        assert input_obj is not None

    def test_input_with_expected(self):
        """Test ScorerInput with expected value."""
        input_obj = ScorerInput(output="hello", expected="hello")
        assert input_obj is not None

    def test_input_with_all_fields(self):
        """Test ScorerInput with all fields."""
        trace = [
            {
                "event_type": "run.started",
                "event_id": "1",
                "correlation_id": "a",
                "timestamp_ns": 1000,
                "data": {},
            }
        ]
        input_obj = ScorerInput(
            output="result",
            expected="expected",
            input="original input",
            trace=trace,
        )
        assert input_obj is not None


class TestExactMatch:
    """Test exact_match scorer."""

    def test_exact_match_pass(self):
        """Test exact match when strings are equal."""
        input_obj = ScorerInput(output="hello", expected="hello")
        result = exact_match(input_obj)

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "match"

    def test_exact_match_fail(self):
        """Test exact match when strings differ."""
        input_obj = ScorerInput(output="hello", expected="world")
        result = exact_match(input_obj)

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "mismatch"

    def test_exact_match_case_sensitive(self):
        """Test case sensitive exact match (default)."""
        input_obj = ScorerInput(output="Hello", expected="hello")
        result = exact_match(input_obj)

        assert result.score == 0.0
        assert result.passed is False

    def test_exact_match_case_insensitive(self):
        """Test case insensitive exact match."""
        input_obj = ScorerInput(output="Hello", expected="hello")
        result = exact_match(input_obj, case_sensitive=False)

        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_empty_strings(self):
        """Test exact match with empty strings."""
        input_obj = ScorerInput(output="", expected="")
        result = exact_match(input_obj)

        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_json_values(self):
        """Test exact match with JSON values."""
        input_obj = ScorerInput(output={"key": "value"}, expected={"key": "value"})
        result = exact_match(input_obj)

        assert result.score == 1.0
        assert result.passed is True


class TestContains:
    """Test contains scorer."""

    def test_contains_pass(self):
        """Test contains when pattern is found."""
        input_obj = ScorerInput(output="hello world")
        result = contains(input_obj, "world")

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "found"

    def test_contains_fail(self):
        """Test contains when pattern is not found."""
        input_obj = ScorerInput(output="hello world")
        result = contains(input_obj, "foo")

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "not_found"

    def test_contains_case_sensitive(self):
        """Test case sensitive contains (default)."""
        input_obj = ScorerInput(output="Hello World")
        result = contains(input_obj, "hello")

        assert result.score == 0.0
        assert result.passed is False

    def test_contains_case_insensitive(self):
        """Test case insensitive contains."""
        input_obj = ScorerInput(output="Hello World")
        result = contains(input_obj, "hello", case_sensitive=False)

        assert result.score == 1.0
        assert result.passed is True


class TestJsonValid:
    """Test json_valid scorer."""

    def test_json_valid_string(self):
        """Test json_valid with valid JSON string."""
        input_obj = ScorerInput(output='{"key": "value"}')
        result = json_valid(input_obj)

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "valid"

    def test_json_invalid_string(self):
        """Test json_valid with invalid JSON string."""
        input_obj = ScorerInput(output='{"key": value}')  # Missing quotes
        result = json_valid(input_obj)

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "invalid"

    def test_json_valid_object(self):
        """Test json_valid with Python dict (already valid JSON)."""
        input_obj = ScorerInput(output={"key": "value"})
        result = json_valid(input_obj)

        assert result.score == 1.0
        assert result.passed is True

    def test_json_valid_array(self):
        """Test json_valid with JSON array string."""
        input_obj = ScorerInput(output='[1, 2, 3]')
        result = json_valid(input_obj)

        assert result.score == 1.0
        assert result.passed is True


class TestRegexMatch:
    """Test regex_match scorer."""

    def test_regex_match_pass(self):
        """Test regex_match when pattern matches."""
        input_obj = ScorerInput(output="hello123world")
        result = regex_match(input_obj, r"\d+")

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "match"

    def test_regex_match_fail(self):
        """Test regex_match when pattern doesn't match."""
        input_obj = ScorerInput(output="hello world")
        result = regex_match(input_obj, r"\d+")

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "no_match"

    def test_regex_email_pattern(self):
        """Test regex_match with email pattern."""
        input_obj = ScorerInput(output="user@example.com")
        result = regex_match(input_obj, r"[\w.]+@[\w.]+\.\w+")

        assert result.score == 1.0
        assert result.passed is True

    def test_regex_invalid_pattern(self):
        """Test regex_match with invalid regex pattern."""
        input_obj = ScorerInput(output="test")
        result = regex_match(input_obj, r"[invalid")  # Unclosed bracket

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "error"
        assert "Invalid regex" in result.explanation


class TestLevenshtein:
    """Test levenshtein scorer."""

    def test_levenshtein_identical(self):
        """Test levenshtein with identical strings."""
        input_obj = ScorerInput(output="hello", expected="hello")
        result = levenshtein(input_obj)

        assert result.score == 1.0
        assert result.passed is True

    def test_levenshtein_similar(self):
        """Test levenshtein with similar strings."""
        input_obj = ScorerInput(output="hello", expected="hallo")
        result = levenshtein(input_obj)

        # Distance is 1, max length is 5, similarity = 1 - 1/5 = 0.8
        assert result.score == 0.8
        assert "Similarity" in result.explanation

    def test_levenshtein_different(self):
        """Test levenshtein with very different strings."""
        input_obj = ScorerInput(output="hello", expected="world")
        result = levenshtein(input_obj)

        assert result.score < 0.5

    def test_levenshtein_threshold_pass(self):
        """Test levenshtein with threshold that passes."""
        input_obj = ScorerInput(output="hello", expected="hallo")
        result = levenshtein(input_obj, threshold=0.7)

        assert result.score == 0.8
        assert result.passed is True

    def test_levenshtein_threshold_fail(self):
        """Test levenshtein with threshold that fails."""
        input_obj = ScorerInput(output="hello", expected="hallo")
        result = levenshtein(input_obj, threshold=0.9)

        assert result.score == 0.8
        assert result.passed is False

    def test_levenshtein_empty_strings(self):
        """Test levenshtein with empty strings."""
        input_obj = ScorerInput(output="", expected="")
        result = levenshtein(input_obj)

        assert result.score == 1.0


class TestTraceAssertions:
    """Test trace assertions for glassbox testing."""

    @pytest.fixture
    def sample_trace(self):
        """Create a sample trace for testing."""
        return [
            {
                "event_type": "run.started",
                "event_id": "1",
                "correlation_id": "a",
                "timestamp_ns": 1_000_000,
                "data": {},
            },
            {
                "event_type": "lm.call.completed",
                "event_id": "2",
                "correlation_id": "a",
                "timestamp_ns": 2_000_000,
                "data": {"total_tokens": 500},
            },
            {
                "event_type": "lm.call.completed",
                "event_id": "3",
                "correlation_id": "a",
                "timestamp_ns": 3_000_000,
                "data": {"total_tokens": 300},
            },
            {
                "event_type": "run.completed",
                "event_id": "4",
                "correlation_id": "a",
                "timestamp_ns": 4_000_000,
                "data": {},
            },
        ]

    def test_max_tokens_pass(self, sample_trace):
        """Test max_tokens assertion that passes."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.max_tokens(1000)]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 1.0
        assert result.passed is True

    def test_max_tokens_fail(self, sample_trace):
        """Test max_tokens assertion that fails."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.max_tokens(500)]  # Total is 800
        result = trace_scorer(input_obj, assertions)

        assert result.score == 0.0
        assert result.passed is False

    def test_max_lm_calls_pass(self, sample_trace):
        """Test max_lm_calls assertion that passes."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.max_lm_calls(5)]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 1.0
        assert result.passed is True

    def test_max_lm_calls_fail(self, sample_trace):
        """Test max_lm_calls assertion that fails."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.max_lm_calls(1)]  # Has 2 calls
        result = trace_scorer(input_obj, assertions)

        assert result.score == 0.0
        assert result.passed is False

    def test_no_errors_pass(self, sample_trace):
        """Test no_errors assertion that passes."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.no_errors()]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 1.0
        assert result.passed is True

    def test_no_errors_fail(self):
        """Test no_errors assertion that fails."""
        trace = [
            {
                "event_type": "run.started",
                "event_id": "1",
                "correlation_id": "a",
                "timestamp_ns": 1000,
                "data": {},
            },
            {
                "event_type": "run.failed",
                "event_id": "2",
                "correlation_id": "a",
                "timestamp_ns": 2000,
                "data": {"error": "Something went wrong"},
            },
        ]
        input_obj = ScorerInput(output="result", trace=trace)
        assertions = [TraceAssertion.no_errors()]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 0.0
        assert result.passed is False

    def test_duration_under_pass(self, sample_trace):
        """Test duration_under assertion that passes."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        # Duration is 3ms (4_000_000 - 1_000_000 ns)
        assertions = [TraceAssertion.duration_under(10)]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 1.0
        assert result.passed is True

    def test_duration_under_fail(self, sample_trace):
        """Test duration_under assertion that fails."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.duration_under(1)]  # 1ms, but actual is 3ms
        result = trace_scorer(input_obj, assertions)

        assert result.score == 0.0
        assert result.passed is False

    def test_event_sequence_pass(self, sample_trace):
        """Test event_sequence assertion that passes."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [TraceAssertion.event_sequence(["run.started", "run.completed"])]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 1.0
        assert result.passed is True

    def test_event_sequence_fail(self):
        """Test event_sequence assertion that fails."""
        trace = [
            {
                "event_type": "run.started",
                "event_id": "1",
                "correlation_id": "a",
                "timestamp_ns": 1000,
                "data": {},
            },
            {
                "event_type": "run.completed",
                "event_id": "2",
                "correlation_id": "a",
                "timestamp_ns": 2000,
                "data": {},
            },
        ]
        input_obj = ScorerInput(output="result", trace=trace)
        # Looking for workflow.step.completed which doesn't exist
        assertions = [TraceAssertion.event_sequence(["run.started", "workflow.step.completed"])]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 0.0
        assert result.passed is False

    def test_multiple_assertions(self, sample_trace):
        """Test multiple assertions together."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [
            TraceAssertion.max_tokens(1000),
            TraceAssertion.max_lm_calls(5),
            TraceAssertion.no_errors(),
        ]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 1.0
        assert result.passed is True

    def test_partial_pass(self, sample_trace):
        """Test when some assertions pass and some fail."""
        input_obj = ScorerInput(output="result", trace=sample_trace)
        assertions = [
            TraceAssertion.max_tokens(1000),  # Pass
            TraceAssertion.max_lm_calls(1),  # Fail (has 2)
            TraceAssertion.no_errors(),  # Pass
        ]
        result = trace_scorer(input_obj, assertions)

        # 2 out of 3 passed
        assert result.score == pytest.approx(2 / 3)
        assert result.passed is False

    def test_no_trace_provided(self):
        """Test trace_scorer when no trace is provided."""
        input_obj = ScorerInput(output="result")  # No trace
        assertions = [TraceAssertion.no_errors()]
        result = trace_scorer(input_obj, assertions)

        assert result.score == 0.0
        assert result.passed is False
        assert "No trace" in result.explanation


class TestCustomScorers:
    """Test custom scorer decorator and registry."""

    def test_scorer_decorator_basic(self):
        """Test basic @scorer decorator."""

        @scorer(name="test_scorer")
        def check_length(ctx: EvalContext) -> ScorerResultPy:
            length = len(str(ctx.output))
            return ScorerResultPy(
                score=min(1.0, length / 10),
                passed=length >= 5,
                explanation=f"Length: {length}",
            )

        # Check registration
        assert "test_scorer" in list_custom_scorers()
        assert get_custom_scorer("test_scorer") is not None

    def test_scorer_decorator_default_name(self):
        """Test @scorer decorator uses function name by default."""

        @scorer()
        def my_custom_scorer(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=1.0, passed=True)

        assert "my_custom_scorer" in list_custom_scorers()

    def test_custom_scorer_execution(self):
        """Test executing a custom scorer."""

        @scorer(name="format_checker")
        def check_format(ctx: EvalContext) -> ScorerResultPy:
            output = str(ctx.output)
            valid = output.startswith("Result:")
            return ScorerResultPy(
                score=1.0 if valid else 0.0,
                passed=valid,
                label="valid" if valid else "invalid",
            )

        scorer_fn = get_custom_scorer("format_checker")
        ctx = EvalContext(input="test", output="Result: success")
        result = scorer_fn(ctx)

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "valid"

    def test_custom_scorer_with_expected(self):
        """Test custom scorer accessing expected value."""

        @scorer(name="semantic_match")
        def semantic_check(ctx: EvalContext) -> ScorerResultPy:
            if ctx.expected is None:
                return ScorerResultPy(score=0.0, passed=False)

            # Simple check: both contain same key words
            output_words = set(str(ctx.output).lower().split())
            expected_words = set(str(ctx.expected).lower().split())
            overlap = len(output_words & expected_words)
            total = len(expected_words)

            score = overlap / total if total > 0 else 0.0
            return ScorerResultPy(score=score, passed=score >= 0.5)

        scorer_fn = get_custom_scorer("semantic_match")
        ctx = EvalContext(
            input="What is the capital?",
            output="The capital of France is Paris",
            expected="Paris is the capital",
        )
        result = scorer_fn(ctx)

        assert result.score > 0.5
        assert result.passed is True

    def test_is_scorer(self):
        """Test is_scorer function."""

        @scorer(name="test_fn")
        def scorer_func(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=1.0, passed=True)

        def regular_func():
            pass

        assert is_scorer(scorer_func) is True
        assert is_scorer(regular_func) is False

    def test_get_scorer_info(self):
        """Test get_scorer_info function."""

        @scorer(name="info_test", description="A test scorer")
        def test_fn(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=1.0, passed=True)

        info = get_scorer_info(test_fn)
        assert info is not None
        assert info["name"] == "info_test"
        assert info["description"] == "A test scorer"

    def test_get_scorer_info_non_scorer(self):
        """Test get_scorer_info with non-scorer function."""

        def regular_func():
            pass

        info = get_scorer_info(regular_func)
        assert info is None

    def test_list_custom_scorers(self):
        """Test list_custom_scorers function."""

        @scorer(name="scorer_a")
        def scorer_a(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=1.0, passed=True)

        @scorer(name="scorer_b")
        def scorer_b(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=0.5, passed=False)

        scorers = list_custom_scorers()
        assert "scorer_a" in scorers
        assert "scorer_b" in scorers

    def test_clear_custom_scorers(self):
        """Test clear_custom_scorers function."""

        @scorer(name="temp_scorer")
        def temp(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=1.0, passed=True)

        assert "temp_scorer" in list_custom_scorers()

        clear_custom_scorers()

        assert "temp_scorer" not in list_custom_scorers()
        assert get_custom_scorer("temp_scorer") is None


class TestEvalContext:
    """Test EvalContext helper methods."""

    def test_eval_context_basic(self):
        """Test basic EvalContext creation."""
        ctx = EvalContext(input="test input", output="test output")

        assert ctx.input == "test input"
        assert ctx.output == "test output"
        assert ctx.expected is None

    def test_eval_context_with_events(self):
        """Test EvalContext with events."""
        from agnt5.eval.types import TraceEvent

        events = [
            TraceEvent(
                event_type="lm.call.completed",
                event_id="1",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=1000,
                data={"total_tokens": 500},
            ),
            TraceEvent(
                event_type="lm.call.completed",
                event_id="2",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=2000,
                data={"total_tokens": 300},
            ),
        ]
        ctx = EvalContext(input="test", output="result", events=events)

        assert len(ctx.events) == 2

    def test_get_events_by_type(self):
        """Test get_events_by_type method."""
        from agnt5.eval.types import TraceEvent

        events = [
            TraceEvent(
                event_type="run.started",
                event_id="1",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=1000,
                data={},
            ),
            TraceEvent(
                event_type="lm.call.completed",
                event_id="2",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=2000,
                data={"total_tokens": 500},
            ),
        ]
        ctx = EvalContext(input="test", output="result", events=events)

        lm_events = ctx.get_events_by_type("lm.call.completed")
        assert len(lm_events) == 1
        assert lm_events[0].data["total_tokens"] == 500

    def test_get_lm_calls(self):
        """Test get_lm_calls method."""
        from agnt5.eval.types import TraceEvent

        events = [
            TraceEvent(
                event_type="run.started",
                event_id="1",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=1000,
                data={},
            ),
            TraceEvent(
                event_type="lm.call.completed",
                event_id="2",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=2000,
                data={"total_tokens": 500},
            ),
            TraceEvent(
                event_type="lm.call.completed",
                event_id="3",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=3000,
                data={"total_tokens": 300},
            ),
        ]
        ctx = EvalContext(input="test", output="result", events=events)

        lm_calls = ctx.get_lm_calls()
        assert len(lm_calls) == 2

    def test_get_total_tokens(self):
        """Test get_total_tokens method."""
        from agnt5.eval.types import TraceEvent

        events = [
            TraceEvent(
                event_type="lm.call.completed",
                event_id="1",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=1000,
                data={"total_tokens": 500},
            ),
            TraceEvent(
                event_type="lm.call.completed",
                event_id="2",
                correlation_id="a",
                parent_correlation_id=None,
                timestamp_ns=2000,
                data={"total_tokens": 300},
            ),
        ]
        ctx = EvalContext(input="test", output="result", events=events)

        total = ctx.get_total_tokens()
        assert total == 800


class TestLlmJudgeHelpers:
    """Test LLM Judge helper functions."""

    def test_extract_json_plain(self):
        """Test _extract_json with plain JSON."""
        from agnt5.eval.llm_judge import _extract_json

        result = _extract_json('{"score": 0.8, "passed": true}')
        assert result == '{"score": 0.8, "passed": true}'

    def test_extract_json_markdown_block(self):
        """Test _extract_json with markdown code block."""
        from agnt5.eval.llm_judge import _extract_json

        input_str = """```json
{"score": 0.9, "passed": true, "explanation": "Good answer"}
```"""
        result = _extract_json(input_str)
        assert '"score": 0.9' in result
        assert '"passed": true' in result

    def test_extract_json_with_text(self):
        """Test _extract_json with surrounding text."""
        from agnt5.eval.llm_judge import _extract_json

        input_str = """Here is my evaluation:
{"score": 0.7, "passed": true}
That's my assessment."""
        result = _extract_json(input_str)
        assert result == '{"score": 0.7, "passed": true}'

    def test_parse_llm_response_valid(self):
        """Test _parse_llm_response with valid JSON."""
        from agnt5.eval.llm_judge import _parse_llm_response

        result = _parse_llm_response('{"score": 0.85, "passed": true, "explanation": "Good"}')

        assert result.score == 0.85
        assert result.passed is True
        assert result.explanation == "Good"

    def test_parse_llm_response_clamps_score(self):
        """Test _parse_llm_response clamps score to [0, 1]."""
        from agnt5.eval.llm_judge import _parse_llm_response

        # Score > 1 should be clamped
        result = _parse_llm_response('{"score": 1.5, "passed": true}')
        assert result.score == 1.0

        # Score < 0 should be clamped
        result = _parse_llm_response('{"score": -0.5, "passed": false}')
        assert result.score == 0.0

    def test_parse_llm_response_infers_passed(self):
        """Test _parse_llm_response infers passed from score."""
        from agnt5.eval.llm_judge import _parse_llm_response

        # Score >= 0.7 should pass
        result = _parse_llm_response('{"score": 0.8}')
        assert result.passed is True

        # Score < 0.7 should fail
        result = _parse_llm_response('{"score": 0.5}')
        assert result.passed is False

    def test_parse_llm_response_invalid_json(self):
        """Test _parse_llm_response with invalid JSON."""
        from agnt5.eval.llm_judge import _parse_llm_response

        result = _parse_llm_response("This is not JSON at all")

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "parse_error"


class TestLlmJudgeConfig:
    """Test LlmJudgeConfig and LlmJudgeResult."""

    def test_config_defaults(self):
        """Test LlmJudgeConfig default values."""
        from agnt5.eval.llm_judge import LlmJudgeConfig

        config = LlmJudgeConfig(criteria="Is the answer helpful?")

        assert config.criteria == "Is the answer helpful?"
        assert config.model == "openai/gpt-4o-mini"
        assert config.system_prompt is None
        assert config.temperature == 0.0
        assert config.include_input is False

    def test_config_custom_values(self):
        """Test LlmJudgeConfig with custom values."""
        from agnt5.eval.llm_judge import LlmJudgeConfig

        config = LlmJudgeConfig(
            criteria="Is the response accurate?",
            model="anthropic/claude-3-haiku",
            temperature=0.1,
            include_input=True,
        )

        assert config.criteria == "Is the response accurate?"
        assert config.model == "anthropic/claude-3-haiku"
        assert config.temperature == 0.1
        assert config.include_input is True

    def test_result_dataclass(self):
        """Test LlmJudgeResult dataclass."""
        from agnt5.eval.llm_judge import LlmJudgeResult

        result = LlmJudgeResult(
            score=0.9,
            passed=True,
            explanation="Excellent answer",
            label="good",
            metadata={"raw": "data"},
        )

        assert result.score == 0.9
        assert result.passed is True
        assert result.explanation == "Excellent answer"
        assert result.label == "good"
        assert result.metadata == {"raw": "data"}


class TestScorerComponentType:
    """Test scorer as a component type."""

    def test_component_type_enum(self):
        """Test ComponentType includes SCORER."""
        from agnt5.events import ComponentType

        assert hasattr(ComponentType, "SCORER")
        assert ComponentType.SCORER.value == "scorer"

    def test_scorer_registry_get(self):
        """Test ScorerRegistry.get() method."""
        from agnt5.scorer import ScorerRegistry, scorer as scorer_dec
        from agnt5.eval.types import ScorerRequest, ScorerResult as ScorerResultType

        # Note: we need to use a unique name because the registry might not be cleared
        @scorer_dec(name="registry_test_scorer_unique")
        def test_scorer_fn(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        config = ScorerRegistry.get("registry_test_scorer_unique")
        assert config is not None
        assert config.name == "registry_test_scorer_unique"
        assert config.handler is not None

    def test_scorer_registry_all(self):
        """Test ScorerRegistry.all() returns all registered scorers."""
        from agnt5.scorer import ScorerRegistry, scorer as scorer_dec
        from agnt5.eval.types import ScorerRequest, ScorerResult as ScorerResultType

        @scorer_dec(name="all_test_scorer_unique_1")
        def scorer_fn_1(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        @scorer_dec(name="all_test_scorer_unique_2")
        def scorer_fn_2(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=0.5, passed=False)

        all_scorers = ScorerRegistry.all()
        assert "all_test_scorer_unique_1" in all_scorers
        assert "all_test_scorer_unique_2" in all_scorers


class TestScorerRequest:
    """Test ScorerRequest dataclass."""

    def test_scorer_request_basic(self):
        """Test basic ScorerRequest creation."""
        from agnt5.eval.types import ScorerRequest

        req = ScorerRequest(
            output={"result": "test"},
            expected={"expected": "value"},
            input={"input": "data"},  # Note: field is 'input' not 'input_data'
        )

        assert req.output == {"result": "test"}
        assert req.expected == {"expected": "value"}
        assert req.input == {"input": "data"}

    def test_scorer_request_minimal(self):
        """Test ScorerRequest with minimal fields."""
        from agnt5.eval.types import ScorerRequest

        req = ScorerRequest(output="test output")

        assert req.output == "test output"
        assert req.expected is None
        assert req.input is None
        assert req.trace is None


class TestScorerResult:
    """Test ScorerResult dataclass."""

    def test_scorer_result_full(self):
        """Test ScorerResult with all fields."""
        from agnt5.eval.types import ScorerResult as ScorerResultType

        result = ScorerResultType(
            score=0.85,
            passed=True,
            label="match",
            explanation="Strings are similar",
            metadata={"distance": 2},
        )

        assert result.score == 0.85
        assert result.passed is True
        assert result.label == "match"
        assert result.explanation == "Strings are similar"
        assert result.metadata == {"distance": 2}

    def test_scorer_result_minimal(self):
        """Test ScorerResult with minimal fields - passed defaults based on score."""
        from agnt5.eval.types import ScorerResult as ScorerResultType

        result = ScorerResultType(score=0.5)

        assert result.score == 0.5
        # Note: __post_init__ sets passed based on score if not provided
        assert result.passed is True  # score >= 0.5 -> passed
        assert result.label is None
        assert result.explanation is None

    def test_scorer_result_score_clamping(self):
        """Test ScorerResult clamps score to [0, 1]."""
        from agnt5.eval.types import ScorerResult as ScorerResultType

        # Score > 1 should be clamped
        result = ScorerResultType(score=1.5)
        assert result.score == 1.0

        # Score < 0 should be clamped
        result = ScorerResultType(score=-0.5)
        assert result.score == 0.0

    def test_scorer_result_passed_default(self):
        """Test ScorerResult passed defaults based on score threshold."""
        from agnt5.eval.types import ScorerResult as ScorerResultType

        # Score >= 0.5 should pass
        result = ScorerResultType(score=0.5)
        assert result.passed is True

        # Score < 0.5 should fail
        result = ScorerResultType(score=0.4)
        assert result.passed is False


class TestEvalResponse:
    """Test EvalResponse type for client.eval()."""

    def test_eval_response_creation(self):
        """Test EvalResponse creation."""
        from agnt5.responses import EvalResponse, ScorerResultSummary

        scores = [
            ScorerResultSummary(
                scorer="exact_match",
                score=1.0,
                passed=True,
                explanation="Exact match",
            ),
            ScorerResultSummary(
                scorer="contains",
                score=0.5,
                passed=False,
                explanation="Pattern not found",
            ),
        ]

        response = EvalResponse(
            output={"result": "test"},
            scores=scores,
            passed=False,  # One scorer failed
            run_id="test-run-123",
            trace_id="trace-456",
            duration_ms=100,
        )

        assert response.output == {"result": "test"}
        assert len(response.scores) == 2
        assert response.passed is False
        assert response.run_id == "test-run-123"
        assert response.trace_id == "trace-456"
        assert response.duration_ms == 100

    def test_eval_response_get_score(self):
        """Test EvalResponse.get_score() method returns ScorerResultSummary."""
        from agnt5.responses import EvalResponse, ScorerResultSummary

        scores = [
            ScorerResultSummary(
                scorer="exact_match", score=1.0, passed=True
            ),
            ScorerResultSummary(
                scorer="format_check", score=0.8, passed=True
            ),
        ]

        response = EvalResponse(
            output="test",
            scores=scores,
            passed=True,
            run_id="run-1",
        )

        # get_score returns the ScorerResultSummary, not just the score value
        exact_match = response.get_score("exact_match")
        assert exact_match is not None
        assert exact_match.score == 1.0
        assert exact_match.passed is True

        format_check = response.get_score("format_check")
        assert format_check is not None
        assert format_check.score == 0.8

        assert response.get_score("unknown") is None

    def test_eval_response_is_success_error(self):
        """Test EvalResponse.is_success and is_error properties."""
        from agnt5.responses import EvalResponse

        # Successful response
        success_response = EvalResponse(
            output="result",
            scores=[],
            passed=True,
            run_id="run-1",
        )
        assert success_response.is_success is True
        assert success_response.is_error is False

        # Error response
        error_response = EvalResponse(
            output=None,
            scores=[],
            passed=False,
            run_id="run-2",
            error="Execution failed",
        )
        assert error_response.is_success is False
        assert error_response.is_error is True

    def test_scorer_result_summary_dataclass(self):
        """Test ScorerResultSummary dataclass."""
        from agnt5.responses import ScorerResultSummary

        summary = ScorerResultSummary(
            scorer="test_scorer",
            score=0.75,
            passed=True,
            explanation="Good result",
            label="pass",
            metadata={"extra": "data"},
        )

        assert summary.scorer == "test_scorer"
        assert summary.score == 0.75
        assert summary.passed is True
        assert summary.explanation == "Good result"
        assert summary.label == "pass"
        assert summary.metadata == {"extra": "data"}
