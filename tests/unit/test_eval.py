"""Tests for AGNT5 Evaluation Framework.

Tests cover:
- Deterministic scorers (exact_match, contains, json_valid, json_schema,
  numeric_range, regex_match, levenshtein)
- Trace assertions (glassbox testing)
- Custom scorers (@scorer decorator)
- LLM Judge (Python wrapper)
"""

import asyncio
import importlib

import pytest

from agnt5.eval import (
    ScorerInput,
    TraceAssertion,
    contains,
    exact_match,
    json_schema,
    json_valid,
    levenshtein,
    numeric_range,
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
        input_obj = ScorerInput(output="[1, 2, 3]")
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


class TestJsonSchema:
    """Test json_schema scorer."""

    def test_valid_object_passes(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        input_obj = ScorerInput(output={"name": "Ada"})
        result = json_schema(input_obj, schema=schema)
        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "valid"

    def test_invalid_object_fails(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
        }
        input_obj = ScorerInput(output={"name": "Ada", "age": -5})
        result = json_schema(input_obj, schema=schema)
        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "invalid"
        assert result.explanation  # one-line summary present

    def test_string_output_is_parsed(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        input_obj = ScorerInput(output="[1, 2, 3]")
        result = json_schema(input_obj, schema=schema)
        assert result.score == 1.0

    def test_unparseable_string_is_parse_error(self):
        schema = {"type": "object"}
        input_obj = ScorerInput(output="{not json")
        result = json_schema(input_obj, schema=schema)
        assert result.score == 0.0
        assert result.label == "parse_error"

    def test_bad_schema_is_config_error(self):
        # Schemas are objects or booleans, not strings.
        input_obj = ScorerInput(output={"x": 1})
        result = json_schema(input_obj, schema="not a schema")
        assert result.label == "config_error"


class TestNumericRange:
    """Test numeric_range scorer."""

    def test_in_range_inclusive_default(self):
        input_obj = ScorerInput(output=5)
        result = numeric_range(input_obj, min=1, max=10)
        assert result.score == 1.0
        assert result.label == "in_range"

    def test_on_boundary_inclusive(self):
        for boundary in (1, 10):
            input_obj = ScorerInput(output=boundary)
            result = numeric_range(input_obj, min=1, max=10)
            assert result.score == 1.0, f"boundary {boundary} should pass inclusive"

    def test_on_boundary_exclusive(self):
        for boundary in (1, 10):
            input_obj = ScorerInput(output=boundary)
            result = numeric_range(input_obj, min=1, max=10, inclusive=False)
            assert result.score == 0.0, f"boundary {boundary} should fail exclusive"

    def test_out_of_range_below(self):
        input_obj = ScorerInput(output=0)
        result = numeric_range(input_obj, min=1, max=10)
        assert result.score == 0.0
        assert result.label == "out_of_range"

    def test_out_of_range_above(self):
        input_obj = ScorerInput(output=11)
        result = numeric_range(input_obj, min=1, max=10)
        assert result.score == 0.0

    def test_one_sided_min_only(self):
        input_obj = ScorerInput(output=100)
        result = numeric_range(input_obj, min=50)
        assert result.score == 1.0

    def test_one_sided_max_only(self):
        input_obj = ScorerInput(output=0.4)
        result = numeric_range(input_obj, max=0.5)
        assert result.score == 1.0

    def test_numeric_string_accepted(self):
        input_obj = ScorerInput(output="3.14")
        result = numeric_range(input_obj, min=0, max=5)
        assert result.score == 1.0

    def test_non_numeric_is_parse_error(self):
        input_obj = ScorerInput(output="not a number")
        result = numeric_range(input_obj, min=0, max=10)
        assert result.score == 0.0
        assert result.label == "parse_error"

    def test_missing_bounds_is_config_error(self):
        input_obj = ScorerInput(output=1)
        result = numeric_range(input_obj)
        assert result.label == "config_error"


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

        @scorer(name="info_test", description="A test scorer", scope="run")
        def test_fn(ctx: EvalContext) -> ScorerResultPy:
            return ScorerResultPy(score=1.0, passed=True)

        info = get_scorer_info(test_fn)
        assert info is not None
        assert info["name"] == "info_test"
        assert info["description"] == "A test scorer"
        assert info["scope"] == "run"

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

    def test_get_tool_calls_empty(self):
        """Tool-call helpers tolerate missing events."""
        ctx = EvalContext(input="test", output="result")

        assert ctx.get_tool_calls() == []
        assert ctx.get_tool_call_names() == []
        assert ctx.tool_trajectory_matches([], mode="exact") is True

    def test_get_tool_calls_from_normalized_session(self):
        """Tool calls are extracted from normalized session payloads."""
        from agnt5.eval.types import TraceEvent

        events = [
            TraceEvent(
                event_type="session.normalized",
                event_id="evt-1",
                correlation_id="span-root",
                timestamp_ns=1000,
                data={
                    "normalized_session": {
                        "tool_calls": [
                            {
                                "call_id": "call-1",
                                "span_id": "span-1",
                                "name": "search",
                                "started_at": 10,
                                "ended_at": 12,
                                "arguments_ref": "s3://args.json",
                                "arguments_hash": "hash-args",
                                "status": "completed",
                                "attributes_safe": {"source": "normalized"},
                            },
                            {
                                "call_id": "call-2",
                                "span_id": "span-2",
                                "name": "lookup",
                                "started_at": 13,
                                "ended_at": 14,
                            },
                            {"call_id": "malformed"},
                        ]
                    }
                },
            )
        ]

        ctx = EvalContext(input="test", output="result", events=events)
        calls = ctx.get_tool_calls()

        assert ctx.get_tool_call_names() == ["search", "lookup"]
        assert len(calls) == 2
        assert calls[0].call_id == "call-1"
        assert calls[0].span_id == "span-1"
        assert calls[0].arguments is None
        assert calls[0].metadata["arguments_ref"] == "s3://args.json"
        assert calls[0].metadata["attributes_safe"] == {"source": "normalized"}
        assert calls[1].started_at == 13

    def test_get_tool_calls_from_legacy_and_lm_events(self):
        """Tool calls are extracted from tool events and LM tool-call payloads."""
        from agnt5.eval.types import TraceEvent

        events = [
            TraceEvent(
                event_type="tool.call.started",
                event_id="evt-1",
                correlation_id="span-tool",
                timestamp_ns=1000,
                data={
                    "tool_name": "search",
                    "tool_call_id": "call-1",
                    "arguments": {"query": "weather"},
                },
            ),
            TraceEvent(
                event_type="tool.call.completed",
                event_id="evt-2",
                correlation_id="span-tool",
                timestamp_ns=2000,
                data={"tool_name": "search", "tool_call_id": "call-1", "duration_ms": 5},
            ),
            TraceEvent(
                event_type="lm.call.completed",
                event_id="evt-3",
                correlation_id="span-lm",
                timestamp_ns=3000,
                data={
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"id": 42}',
                            },
                        }
                    ]
                },
            ),
            TraceEvent(
                event_type="tool.call.completed",
                event_id="evt-4",
                correlation_id="span-bad",
                timestamp_ns=4000,
                data={"duration_ms": 1},
            ),
        ]

        ctx = EvalContext(input="test", output="result", events=events)
        calls = ctx.get_tool_calls()

        assert ctx.get_tool_call_names() == ["search", "lookup"]
        assert len(calls) == 2
        assert calls[0].call_id == "call-1"
        assert calls[0].arguments == {"query": "weather"}
        assert calls[0].status == "completed"
        assert calls[0].metadata["duration_ms"] == 5
        assert calls[1].call_id == "call-2"
        assert calls[1].arguments == {"id": 42}

    def test_tool_trajectory_helpers(self):
        """Trajectory helpers support exact, in-order, and any-order matching."""
        from agnt5.eval.types import (
            tool_trajectory_any_order,
            tool_trajectory_exact,
            tool_trajectory_in_order,
            tool_trajectory_matches,
        )

        actual = ["search", "lookup", "summarize", "search"]

        assert tool_trajectory_exact(actual, actual) is True
        assert tool_trajectory_exact(actual, ["search", "lookup"]) is False
        assert tool_trajectory_in_order(actual, ["search", "summarize"]) is True
        assert tool_trajectory_in_order(actual, ["summarize", "lookup"]) is False
        assert tool_trajectory_any_order(actual, ["lookup", "search", "search"]) is True
        assert tool_trajectory_any_order(actual, ["lookup", "lookup"]) is False
        assert tool_trajectory_matches(actual, ["summarize", "search"], mode="in_order") is True

    def test_scorer_request_tool_call_helpers(self):
        """ScorerRequest exposes the same tool-call helpers as EvalContext."""
        from agnt5.eval.types import ScorerRequest, TraceEvent

        request = ScorerRequest(
            output="result",
            trace=[
                TraceEvent(
                    event_type="tool.call.completed",
                    event_id="evt-1",
                    correlation_id="span-tool",
                    timestamp_ns=1000,
                    data={"tool_name": "search", "tool_call_id": "call-1"},
                )
            ],
        )

        assert request.get_tool_call_names() == ["search"]
        assert request.tool_trajectory_matches(["search"], mode="exact") is True


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


class TestLLMJudgeConfig:
    """Test LLMJudgeConfig and LLMJudgeResult."""

    def test_config_defaults(self):
        """Test LLMJudgeConfig default values."""
        from agnt5.eval.llm_judge import LLMJudgeConfig

        config = LLMJudgeConfig(criteria="Is the answer helpful?")

        assert config.criteria == "Is the answer helpful?"
        assert config.model == "openai/gpt-4o-mini"
        assert config.system_prompt is None
        assert config.temperature == 0.0
        assert config.include_input is False

    def test_config_custom_values(self):
        """Test LLMJudgeConfig with custom values."""
        from agnt5.eval.llm_judge import LLMJudgeConfig

        config = LLMJudgeConfig(
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
        """Test LLMJudgeResult dataclass."""
        from agnt5.eval.llm_judge import LLMJudgeResult

        result = LLMJudgeResult(
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


class TestLLMJudge:
    """Test LLMJudge class for client.eval() usage."""

    def test_llm_judge_defaults(self):
        """Test LLMJudge default values."""
        from agnt5.eval import LLMJudge

        judge = LLMJudge(criteria="Is the answer correct?")

        assert judge.criteria == "Is the answer correct?"
        assert judge.model == "openai/gpt-4o-mini"
        assert judge.include_input is False
        assert judge.temperature == 0.0

    def test_llm_judge_custom_values(self):
        """Test LLMJudge with custom values."""
        from agnt5.eval import LLMJudge

        judge = LLMJudge(
            criteria="Is the response helpful and accurate?",
            model="anthropic/claude-3-haiku",
            include_input=True,
            temperature=0.1,
        )

        assert judge.criteria == "Is the response helpful and accurate?"
        assert judge.model == "anthropic/claude-3-haiku"
        assert judge.include_input is True
        assert judge.temperature == 0.1

    def test_to_scorer_spec_with_provider_model(self):
        """Test to_scorer_spec with provider/model format."""
        from agnt5.eval import LLMJudge

        judge = LLMJudge(
            criteria="Check correctness",
            model="openai/gpt-4o",
        )

        spec = judge.to_scorer_spec()

        assert spec["name"] == "llm_judge"
        assert spec["config"]["provider"] == "openai"
        assert spec["config"]["model"] == "gpt-4o"
        assert spec["config"]["criteria"] == "Check correctness"
        assert spec["config"]["include_input"] is False
        assert spec["config"]["temperature"] == 0.0

    def test_to_scorer_spec_without_provider(self):
        """Test to_scorer_spec defaults to openai provider."""
        from agnt5.eval import LLMJudge

        judge = LLMJudge(
            criteria="Check quality",
            model="gpt-4o-mini",
        )

        spec = judge.to_scorer_spec()

        assert spec["name"] == "llm_judge"
        assert spec["config"]["provider"] == "openai"
        assert spec["config"]["model"] == "gpt-4o-mini"

    def test_to_scorer_spec_anthropic(self):
        """Test to_scorer_spec with Anthropic model."""
        from agnt5.eval import LLMJudge

        judge = LLMJudge(
            criteria="Evaluate response",
            model="anthropic/claude-3-5-sonnet-latest",
            include_input=True,
            temperature=0.2,
        )

        spec = judge.to_scorer_spec()

        assert spec["name"] == "llm_judge"
        assert spec["config"]["provider"] == "anthropic"
        assert spec["config"]["model"] == "claude-3-5-sonnet-latest"
        assert spec["config"]["include_input"] is True
        assert spec["config"]["temperature"] == 0.2

    def test_managed_judge_presets_to_scorer_spec(self):
        """Test named judge preset scorer specs."""
        from agnt5.eval import (
            EVALUATOR_PRESET_VERSION,
            Coherence,
            Conciseness,
            Correctness,
            Faithfulness,
            GoalSuccess,
            Harmfulness,
            Helpfulness,
            InstructionFollowing,
            Refusal,
            ResponseRelevance,
            Stereotyping,
        )

        presets = [
            (Correctness(answer_field="output.answer"), "correctness", "correctness"),
            (
                Faithfulness(context_fields=["input.context"], answer_field="output.answer"),
                "faithfulness",
                "faithfulness",
            ),
            (Helpfulness(), "helpfulness", "llm_judge"),
            (Coherence(), "coherence", "llm_judge"),
            (Conciseness(), "conciseness", "llm_judge"),
            (ResponseRelevance(), "response_relevance", "llm_judge"),
            (InstructionFollowing(), "instruction_following", "llm_judge"),
            (
                GoalSuccess(
                    session_fields=["session.goal"],
                    journal_event_fields=["journal.events"],
                ),
                "goal_success",
                "llm_judge",
            ),
            (Refusal(), "refusal", "llm_judge"),
            (Harmfulness(), "harmfulness", "llm_judge"),
            (Stereotyping(), "stereotyping", "llm_judge"),
        ]

        for preset, preset_name, scorer_name in presets:
            spec = preset.to_scorer_spec()

            assert spec["name"] == scorer_name
            assert spec["config"]["provider"] == "openai"
            assert spec["config"]["model"] == "gpt-4o-mini"
            assert spec["config"]["preset_name"] == preset_name
            assert spec["config"]["preset_version"] == EVALUATOR_PRESET_VERSION
            assert spec["config"]["prompt_version"] == f"agnt5.evaluator.{preset_name}.prompt.v1"
            assert spec["config"]["rubric_version"] == f"agnt5.evaluator.{preset_name}.rubric.v1"
            assert spec["config"]["output_schema"]["required"] == [
                "score",
                "passed",
                "label",
                "explanation",
            ]

        correctness_spec = presets[0][0].to_scorer_spec()
        faithfulness_spec = presets[1][0].to_scorer_spec()
        goal_success_spec = presets[7][0].to_scorer_spec()

        assert correctness_spec["config"]["answer_field"] == "output.answer"
        assert "criteria" not in correctness_spec["config"]
        assert faithfulness_spec["config"]["context_fields"] == ["input.context"]
        assert goal_success_spec["config"]["journal_event_fields"] == ["journal.events"]
        assert goal_success_spec["config"]["session_fields"] == ["session.goal"]

    def test_evaluator_preset_parse_result(self):
        """Named presets parse stable judge result fields."""
        from agnt5.eval import Helpfulness

        result = Helpfulness().parse_result(
            '{"label":"partial","explanation":"useful but incomplete","metadata":{"notes":1}}'
        )

        assert result.score == 0.5
        assert result.passed is False
        assert result.label == "partial"
        assert result.explanation == "useful but incomplete"
        assert result.metadata["notes"] == 1
        assert result.metadata["preset_name"] == "helpfulness"
        assert result.metadata["preset_version"] == "agnt5.evaluator_preset.v1"

    def test_evaluator_preset_evaluate_returns_stable_fields(self, monkeypatch):
        """Named presets run through llm_judge and return normalized fields."""
        from agnt5.eval import Correctness

        judge_module = importlib.import_module("agnt5.eval.llm_judge")
        captured = {}

        async def fake_generate(**kwargs):
            captured["messages"] = kwargs["messages"]

            class Response:
                text = '{"label":"pass","explanation":"matches reference","metadata":{"source":"fake"}}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)

        result = asyncio.run(
            Correctness().evaluate(
                output="4",
                expected="4",
                input_data={"question": "2+2?"},
            )
        )

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "pass"
        assert result.explanation == "matches reference"
        assert result.metadata["source"] == "fake"
        assert result.metadata["preset_name"] == "correctness"
        assert (
            "Choose exactly one label from: fail, partial, pass"
            in captured["messages"][1]["content"]
        )

    def test_custom_judge_prompt_template_to_scorer_spec(self):
        """Test custom judge prompt template scorer specs."""
        from agnt5.eval import LLMJudge

        judge = LLMJudge(
            prompt_template="Label {{output.answer}} against {{expected.answer}}",
            choice_scores={"correct": 1.0, "incorrect": 0.0},
            model="openai/gpt-test",
        )
        spec = judge.to_scorer_spec()

        assert spec["name"] == "llm_judge"
        assert (
            spec["config"]["prompt_template"]
            == "Label {{output.answer}} against {{expected.answer}}"
        )
        assert spec["config"]["choice_scores"] == {"correct": 1.0, "incorrect": 0.0}
        assert spec["config"]["model"] == "gpt-test"

    def test_custom_judge_prompt_template_and_choice_scores(self, monkeypatch):
        """Custom judge templates render fields and map labels to scores."""
        from agnt5.eval.llm_judge import LLMJudgeConfig, llm_judge

        judge_module = importlib.import_module("agnt5.eval.llm_judge")
        captured = {}

        async def fake_generate(**kwargs):
            captured["messages"] = kwargs["messages"]

            class Response:
                text = '{"label":"correct","explanation":"matches"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)

        result = asyncio.run(
            llm_judge(
                output={"answer": "4"},
                expected={"answer": "4"},
                input_data={"question": "2+2?"},
                config=LLMJudgeConfig(
                    criteria="",
                    model="openai/gpt-test",
                    prompt_template=(
                        "Question: {{input.question}}\n"
                        "Answer: {{output.answer}}\n"
                        "Reference: {{expected.answer}}"
                    ),
                    choice_scores={"correct": 1.0, "incorrect": 0.0},
                ),
            )
        )

        assert result.score == 1.0
        assert result.passed is True
        assert result.label == "correct"
        assert result.metadata["selected_label"] == "correct"
        assert "Question: 2+2?" in captured["messages"][1]["content"]
        assert (
            "Choose exactly one label from: correct, incorrect"
            in captured["messages"][1]["content"]
        )

    def test_custom_judge_prompt_template_without_output_appends_output(self, monkeypatch):
        """Criteria-like templates still include the target output."""
        from agnt5.eval.llm_judge import LLMJudgeConfig, llm_judge

        judge_module = importlib.import_module("agnt5.eval.llm_judge")
        captured = {}

        async def fake_generate(**kwargs):
            captured["messages"] = kwargs["messages"]

            class Response:
                text = '{"label":"actionable","explanation":"has recommendation"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)

        result = asyncio.run(
            llm_judge(
                output="Recommendation: buy below $100.",
                config=LLMJudgeConfig(
                    criteria="",
                    model="openai/gpt-test",
                    prompt_template="Judge whether the report is actionable.",
                    choice_scores={"actionable": 1.0, "not actionable": 0.0},
                ),
            )
        )

        assert result.score == 1.0
        assert "Judge whether the report is actionable." in captured["messages"][1]["content"]
        assert "## Output to Evaluate\nRecommendation: buy below $100." in captured["messages"][1]["content"]

    def test_custom_judge_invalid_label_fails_with_clear_label(self, monkeypatch):
        """Custom judge labels must be one of the configured choices."""
        from agnt5.eval.llm_judge import LLMJudgeConfig, llm_judge

        judge_module = importlib.import_module("agnt5.eval.llm_judge")

        async def fake_generate(**kwargs):
            class Response:
                text = '{"label":"maybe","explanation":"uncertain"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)

        result = asyncio.run(
            llm_judge(
                output="4",
                config=LLMJudgeConfig(
                    criteria="",
                    model="openai/gpt-test",
                    prompt_template="Score {{output}}",
                    choice_scores={"correct": 1.0, "incorrect": 0.0},
                ),
            )
        )

        assert result.passed is False
        assert result.label == "invalid_label"
        assert result.metadata["invalid_label"] == "maybe"
        assert result.metadata["allowed_labels"] == ["correct", "incorrect"]

    def test_custom_judge_choice_scores_infer_missing_label_from_score(self, monkeypatch):
        """Choice scores tolerate a missing label when the score maps uniquely."""
        from agnt5.eval.llm_judge import LLMJudgeConfig, llm_judge

        judge_module = importlib.import_module("agnt5.eval.llm_judge")

        async def fake_generate(**kwargs):
            class Response:
                text = '{"score":0,"explanation":"no recommendation"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)

        result = asyncio.run(
            llm_judge(
                output="Analysis without recommendation.",
                config=LLMJudgeConfig(
                    criteria="Actionability",
                    model="openai/gpt-test",
                    choice_scores={"actionable": 1.0, "not actionable": 0.0},
                ),
            )
        )

        assert result.score == 0.0
        assert result.passed is False
        assert result.label == "not actionable"
        assert result.metadata["selected_label"] == "not actionable"

    def test_custom_judge_missing_template_variable_reports_config_error(self):
        """Missing custom judge template variables fail before model calls."""
        from agnt5.eval.llm_judge import LLMJudgeConfig, llm_judge

        result = asyncio.run(
            llm_judge(
                output={"answer": "4"},
                config=LLMJudgeConfig(
                    criteria="",
                    model="openai/gpt-test",
                    prompt_template="Score {{output.missing}}",
                    choice_scores={"correct": 1.0, "incorrect": 0.0},
                ),
            )
        )

        assert result.passed is False
        assert result.label == "config_error"
        assert "output.missing" in (result.explanation or "")

    def test_faithfulness_builtin_handler_binds_context_fields(self, monkeypatch):
        """Smoke-test the managed faithfulness scorer with a fake judge model."""
        from agnt5.eval.types import ScorerRequest

        scorer_mod = importlib.import_module("agnt5.scorer")
        judge_module = importlib.import_module("agnt5.eval.llm_judge")
        captured = {}

        async def fake_generate(**kwargs):
            captured["messages"] = kwargs["messages"]

            class Response:
                text = '{"score":0.91,"passed":true,"explanation":"grounded","label":"pass"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)
        scorer_mod.ScorerRegistry.clear()
        scorer_mod._builtin_handlers_registered = False
        scorer_mod.register_builtin_scorer_handlers()

        request = ScorerRequest(
            output={"answer": "Paris is the capital of France."},
            input={"context": "France's capital city is Paris."},
            config={
                "context_fields": ["input.context"],
                "answer_field": "output.answer",
                "model": "gpt-test",
            },
        )
        result = asyncio.run(scorer_mod.run_scorer("faithfulness", request))

        assert result.passed is True
        assert result.explanation == "grounded"
        assert result.metadata["judge_preset"] == "faithfulness"
        assert result.metadata["context_fields"] == ["input.context"]
        assert "## Context" in captured["messages"][1]["content"]

    def test_correctness_builtin_handler_allows_reference_free_judging(self, monkeypatch):
        """Correctness can judge output against input without expected output."""
        from agnt5.eval.types import ScorerRequest

        scorer_mod = importlib.import_module("agnt5.scorer")
        judge_module = importlib.import_module("agnt5.eval.llm_judge")
        captured = {}

        async def fake_generate(**kwargs):
            captured["messages"] = kwargs["messages"]

            class Response:
                text = '{"score":0,"passed":false,"explanation":"not supported","label":"fail"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)
        scorer_mod.ScorerRegistry.clear()
        scorer_mod._builtin_handlers_registered = False
        scorer_mod.register_builtin_scorer_handlers()

        request = ScorerRequest(
            input={"question": "What did TSLA report last quarter?"},
            output={"answer": "TSLA reported fictional numbers."},
            config={"answer_field": "output.answer", "model": "gpt-test"},
        )
        result = asyncio.run(scorer_mod.run_scorer("correctness", request))

        assert result.passed is False
        assert result.label == "fail"
        assert result.metadata["judge_preset"] == "correctness"
        assert "## Input" in captured["messages"][1]["content"]
        assert "## Expected Output (Reference)" not in captured["messages"][1]["content"]

    def test_faithfulness_builtin_handler_reports_missing_context_field(self):
        """Missing configured context fields fail before calling a judge model."""
        from agnt5.eval.types import ScorerRequest

        scorer_mod = importlib.import_module("agnt5.scorer")
        scorer_mod.ScorerRegistry.clear()
        scorer_mod._builtin_handlers_registered = False
        scorer_mod.register_builtin_scorer_handlers()

        request = ScorerRequest(
            output={"answer": "Paris"},
            input={"other": "context"},
            config={"context_fields": ["input.context"]},
        )
        result = asyncio.run(scorer_mod.run_scorer("faithfulness", request))

        assert result.passed is False
        assert result.label == "config_error"
        assert "input.context" in (result.explanation or "")

    def test_agent_judge_builtin_handler_includes_trace_evidence(self, monkeypatch):
        """Agent judge inspects trace-eval context, tool evidence, and peer scores."""
        from agnt5.eval.types import ScorerRequest, TraceEvent

        scorer_mod = importlib.import_module("agnt5.scorer")
        judge_module = importlib.import_module("agnt5.eval.llm_judge")
        captured = {}

        async def fake_generate(**kwargs):
            captured["messages"] = kwargs["messages"]

            class Response:
                text = '{"score":0.82,"passed":true,"explanation":"evidence supports it","label":"pass"}'

            return Response()

        monkeypatch.setattr(judge_module, "_get_generate", lambda: fake_generate)
        scorer_mod.ScorerRegistry.clear()
        scorer_mod._builtin_handlers_registered = False
        scorer_mod.register_builtin_scorer_handlers()

        request = ScorerRequest(
            input={"question": "Summarize refund policy"},
            output={"answer": "Refunds are available within 30 days."},
            trace=[
                TraceEvent(
                    event_type="tool.call.completed",
                    event_id="event-1",
                    correlation_id="corr-1",
                    data={"tool_name": "web_search", "status": "ok"},
                )
            ],
            peer_scores=[{"scorer": "step_efficiency", "score": 0.7}],
            trace_eval_context={
                "schema_version": "agnt5.eval.trace_eval_context.v1",
                "features": {"tool_call_count": 1},
            },
            config={"model": "gpt-test", "allowed_tools": ["web_search"]},
        )
        result = asyncio.run(scorer_mod.run_scorer("agent_judge", request))

        prompt = captured["messages"][1]["content"]
        assert result.passed is True
        assert result.metadata["judge_preset"] == "agent_judge"
        assert result.metadata["agent_judge_version"] == "evidence_inspection_v1"
        assert "trace_eval_context" in result.metadata["evidence_sources"]
        assert "tool_calls" in result.metadata["evidence_sources"]
        assert "peer_scores" in result.metadata["evidence_sources"]
        assert "agent_judge_evidence" in prompt
        assert "web_search" in prompt
        assert "step_efficiency" in prompt

    def test_builtin_judge_scorers_are_not_custom_registry_entries(self):
        """Built-in judges are AGNT5-owned and separate from custom scorers."""
        scorer_mod = importlib.import_module("agnt5.scorer")
        scorer_mod.ScorerRegistry.clear()
        scorer_mod._BUILTIN_JUDGE_SCORER_REGISTRY.clear()
        scorer_mod._builtin_handlers_registered = False
        scorer_mod.register_builtin_scorer_handlers()

        assert scorer_mod.ScorerRegistry.get("llm_judge") is None
        assert scorer_mod.ScorerRegistry.get("correctness") is None
        assert scorer_mod.ScorerRegistry.get("faithfulness") is None
        assert scorer_mod.ScorerRegistry.get("agent_judge") is None
        assert scorer_mod.get_builtin_judge_scorer_config("llm_judge") is not None
        assert scorer_mod.get_builtin_judge_scorer_config("correctness") is not None
        assert scorer_mod.get_builtin_judge_scorer_config("faithfulness") is not None
        assert scorer_mod.get_builtin_judge_scorer_config("agent_judge") is not None

    def test_custom_scorer_cannot_use_builtin_names(self):
        """Only custom scorers may use component registration, and built-in names are reserved."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import scorer as scorer_dec

        with pytest.raises(ValueError, match="AGNT5 built-in scorer"):

            @scorer_dec(name="exact_match")
            def deterministic_collision(request: ScorerRequest) -> ScorerResultType:
                return ScorerResultType(score=1.0, passed=True)

        with pytest.raises(ValueError, match="AGNT5 built-in scorer"):

            @scorer_dec(name="step_efficiency")
            def trace_builtin_collision(request: ScorerRequest) -> ScorerResultType:
                return ScorerResultType(score=1.0, passed=True)

        with pytest.raises(ValueError, match="AGNT5 built-in scorer"):

            @scorer_dec(name="plan_quality")
            def plan_quality_collision(request: ScorerRequest) -> ScorerResultType:
                return ScorerResultType(score=1.0, passed=True)

        with pytest.raises(ValueError, match="AGNT5 built-in scorer"):

            @scorer_dec(name="plan_adherence")
            def plan_adherence_collision(request: ScorerRequest) -> ScorerResultType:
                return ScorerResultType(score=1.0, passed=True)

        with pytest.raises(ValueError, match="AGNT5 built-in scorer"):

            @scorer_dec(name="llm_judge")
            def judge_collision(request: ScorerRequest) -> ScorerResultType:
                return ScorerResultType(score=1.0, passed=True)

        with pytest.raises(ValueError, match="AGNT5 built-in scorer"):

            @scorer_dec(name="agent_judge")
            def agent_judge_collision(request: ScorerRequest) -> ScorerResultType:
                return ScorerResultType(score=1.0, passed=True)


class TestScorerComponentType:
    """Test scorer as a component type."""

    def test_component_type_enum(self):
        """Test ComponentType includes SCORER."""
        from agnt5.events import ComponentType

        assert hasattr(ComponentType, "SCORER")
        assert ComponentType.SCORER.value == "scorer"

    def test_scorer_registry_get(self):
        """Test ScorerRegistry.get() method."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import ScorerRegistry
        from agnt5.scorer import scorer as scorer_dec

        # Note: we need to use a unique name because the registry might not be cleared
        @scorer_dec(name="registry_test_scorer_unique")
        def test_scorer_fn(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        config = ScorerRegistry.get("registry_test_scorer_unique")
        assert config is not None
        assert config.name == "registry_test_scorer_unique"
        assert config.scope == "item"
        assert config.handler is not None

    def test_scorer_registry_scope(self):
        """Test scorer scope metadata."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import ScorerRegistry
        from agnt5.scorer import scorer as scorer_dec

        @scorer_dec(name="registry_test_run_scorer_unique", scope="run")
        def test_scorer_fn(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        config = ScorerRegistry.get("registry_test_run_scorer_unique")
        assert config is not None
        assert config.scope == "run"
        assert getattr(test_scorer_fn, "_scorer_scope") == "run"

    def test_scorer_registry_dependency_metadata(self):
        """Decorator stores dependencies for meta-evaluators."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import ScorerRegistry
        from agnt5.scorer import scorer as scorer_dec

        @scorer_dec(name="registry_test_meta_unique", depends_on=["primary_quality"])
        def test_scorer_fn(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        config = ScorerRegistry.get("registry_test_meta_unique")
        assert config is not None
        assert config.depends_on == ["primary_quality"]
        assert getattr(test_scorer_fn, "_scorer_depends_on") == ["primary_quality"]

    def test_scorer_registry_all(self):
        """Test ScorerRegistry.all() returns all registered scorers."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import ScorerRegistry
        from agnt5.scorer import scorer as scorer_dec

        @scorer_dec(name="all_test_scorer_unique_1")
        def scorer_fn_1(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        @scorer_dec(name="all_test_scorer_unique_2")
        def scorer_fn_2(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=0.5, passed=False)

        all_scorers = ScorerRegistry.all()
        assert "all_test_scorer_unique_1" in all_scorers
        assert "all_test_scorer_unique_2" in all_scorers

    def test_run_scorer_binds_structured_fields(self):
        """run_scorer should pass only the configured structured fields to handlers."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import run_scorer
        from agnt5.scorer import scorer as scorer_dec

        captured = {}

        @scorer_dec(name="field_binding_capture_unique")
        def capture_fields(request: ScorerRequest) -> ScorerResultType:
            captured["output"] = request.output
            captured["expected"] = request.expected
            captured["input"] = request.input
            captured["peer_scores"] = request.peer_scores
            return ScorerResultType(score=1.0, passed=True, metadata={"handler": "capture"})

        request = ScorerRequest(
            output={"final_output": {"answer": "Paris"}, "path_length": 3},
            expected={"answer": "Paris"},
            input={"question": "capital?"},
            peer_scores=[{"scorer": "primary_quality", "score": 1.0}],
            config={
                "output_field": "final_output.answer",
                "expected_field": "answer",
                "input_field": "question",
                "output_type": "string",
            },
        )

        result = asyncio.run(run_scorer("field_binding_capture_unique", request))

        assert captured == {
            "output": "Paris",
            "expected": "Paris",
            "input": "capital?",
            "peer_scores": [{"scorer": "primary_quality", "score": 1.0}],
        }
        assert result.passed is True
        assert result.metadata["handler"] == "capture"
        assert result.metadata["output_field"] == "final_output.answer"
        assert result.metadata["expected_field"] == "answer"
        assert result.metadata["input_field"] == "question"

    def test_run_scorer_reports_missing_field_binding(self):
        """Missing field bindings should fail with scorer and field diagnostics."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import run_scorer
        from agnt5.scorer import scorer as scorer_dec

        @scorer_dec(name="field_binding_missing_unique")
        def should_not_run(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        request = ScorerRequest(
            output={"other": "Paris"},
            expected="Paris",
            config={"output_field": "final_output.answer"},
        )
        result = asyncio.run(run_scorer("field_binding_missing_unique", request))

        assert result.passed is False
        assert result.label == "config_error"
        assert "field_binding_missing_unique" in (result.explanation or "")
        assert "final_output.answer" in (result.explanation or "")

    def test_run_scorer_reports_wrong_field_binding_type(self):
        """Type assertions should fail before the scorer handler runs."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import run_scorer
        from agnt5.scorer import scorer as scorer_dec

        @scorer_dec(name="field_binding_wrong_type_unique")
        def should_not_run(request: ScorerRequest) -> ScorerResultType:
            return ScorerResultType(score=1.0, passed=True)

        request = ScorerRequest(
            output={"tool_calls": {"name": "search"}},
            config={"output_field": "tool_calls", "output_type": "array"},
        )
        result = asyncio.run(run_scorer("field_binding_wrong_type_unique", request))

        assert result.passed is False
        assert result.label == "config_error"
        assert "expected array" in (result.explanation or "")

    def test_run_scorer_ignores_llm_judge_output_mode_as_field_binding_type(self):
        """UI judge output modes should not be interpreted as target output type assertions."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import run_scorer
        from agnt5.scorer import scorer as scorer_dec

        captured = {}

        @scorer_dec(name="field_binding_output_mode_unique")
        def capture_fields(request: ScorerRequest) -> ScorerResultType:
            captured["output"] = request.output
            return ScorerResultType(score=1.0, passed=True)

        request = ScorerRequest(
            output="A text report, not a numeric score.",
            config={"output_type": "score"},
        )
        result = asyncio.run(run_scorer("field_binding_output_mode_unique", request))

        assert result.passed is True
        assert captured["output"] == "A text report, not a numeric score."

    def test_scorer_context_peer_scores_available_to_meta_evaluator(self):
        """run_scorer passes prior item scores to context helpers."""
        from agnt5.eval.types import ScorerRequest
        from agnt5.eval.types import ScorerResult as ScorerResultType
        from agnt5.scorer import run_scorer
        from agnt5.scorer import scorer as scorer_dec

        captured = {}

        @scorer_dec(name="peer_score_meta_unique", depends_on=["quality"])
        def meta_score(ctx, request: ScorerRequest) -> ScorerResultType:
            captured["all"] = ctx.peer_scores()
            captured["quality"] = ctx.peer_scores("quality")
            captured["request"] = request.peer_scores
            return ScorerResultType(score=1.0, passed=True)

        request = ScorerRequest(
            output="ok",
            peer_scores=[
                {"scorer": "quality", "score": 0.75, "label": "pass"},
                {"name": "format", "score": 1.0, "label": "pass"},
            ],
        )
        result = asyncio.run(run_scorer("peer_score_meta_unique", request))

        assert result.passed is True
        assert captured["all"] == request.peer_scores
        assert captured["quality"] == [{"scorer": "quality", "score": 0.75, "label": "pass"}]
        assert captured["request"] == request.peer_scores


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
            ScorerResultSummary(scorer="exact_match", score=1.0, passed=True),
            ScorerResultSummary(scorer="format_check", score=0.8, passed=True),
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
