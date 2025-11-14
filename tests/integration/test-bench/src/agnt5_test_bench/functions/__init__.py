"""Test Functions

Minimal test functions covering core SDK functionality.
All functions are prefixed with fn_XX for easy identification.
"""

from .simple_functions import (
    fn_01_no_params,
    fn_02_one_param,
    fn_03_two_params,
    fn_04_three_params,
)
from .error_functions import (
    fn_05_raises_value_error,
    fn_06_raises_runtime_error,
    fn_07_raises_custom_error,
)
from .retry_functions import (
    fn_08_retry_succeeds_on_attempt_2,
    fn_09_retry_exhausted,
)
from .lm_functions import (
    fn_10_lm_simple_completion,
    fn_11_lm_structured_output,
)
from .test_helpers import (
    greet,
    long_task,
    flaky_function,
    generate_text,
)

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
