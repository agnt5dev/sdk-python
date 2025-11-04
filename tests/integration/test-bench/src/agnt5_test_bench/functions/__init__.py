"""
Test Functions

Minimal test functions covering core SDK functionality.
All functions are prefixed with fn_XX for easy identification.
"""

import importlib

# Import from numbered modules using importlib
_simple = importlib.import_module(".01_simple_functions", package=__name__)
fn_01_no_params = _simple.fn_01_no_params
fn_02_one_param = _simple.fn_02_one_param
fn_03_two_params = _simple.fn_03_two_params
fn_04_three_params = _simple.fn_04_three_params

_error = importlib.import_module(".02_error_functions", package=__name__)
fn_05_raises_value_error = _error.fn_05_raises_value_error
fn_06_raises_runtime_error = _error.fn_06_raises_runtime_error
fn_07_raises_custom_error = _error.fn_07_raises_custom_error

_retry = importlib.import_module(".03_retry_functions", package=__name__)
fn_08_retry_succeeds_on_attempt_2 = _retry.fn_08_retry_succeeds_on_attempt_2
fn_09_retry_exhausted = _retry.fn_09_retry_exhausted

_lm = importlib.import_module(".04_lm_functions", package=__name__)
fn_10_lm_simple_completion = _lm.fn_10_lm_simple_completion
fn_11_lm_structured_output = _lm.fn_11_lm_structured_output

_helpers = importlib.import_module(".05_test_helpers", package=__name__)
greet = _helpers.greet
long_task = _helpers.long_task
flaky_function = _helpers.flaky_function
generate_text = _helpers.generate_text

__all__ = [
    # Simple functions (01-04)
    "fn_01_no_params",
    "fn_02_one_param",
    "fn_03_two_params",
    "fn_04_three_params",
    # Error functions (05-07)
    "fn_05_raises_value_error",
    "fn_06_raises_runtime_error",
    "fn_07_raises_custom_error",
    # Retry functions (08-09)
    "fn_08_retry_succeeds_on_attempt_2",
    "fn_09_retry_exhausted",
    # LM functions (10-11)
    "fn_10_lm_simple_completion",
    "fn_11_lm_structured_output",
    # Test helpers (backward compatibility)
    "greet",
    "long_task",
    "flaky_function",
    "generate_text",
]
