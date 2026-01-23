//! PyO3 bindings for the AGNT5 evaluation framework.
//!
//! Provides Python access to Rust scorers for evaluating AI component outputs.

use agnt5_sdk_core::eval::{
    deterministic::{self, ContainsConfig, ExactMatchConfig, LevenshteinConfig, RegexConfig},
    trace::{self, TraceAssertion as RustTraceAssertion},
    ScorerInput as RustScorerInput, ScorerResult as RustScorerResult,
    TraceEvent as RustTraceEvent,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde_json::Value;

/// Convert a Python object to serde_json::Value
fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }

    // Try to extract as string first
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Try bool (must be before int since Python bool is a subclass of int)
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    // Try int
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }

    // Try float
    if let Ok(f) = obj.extract::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(Value::Number(n));
        }
    }

    // Try list
    #[allow(deprecated)]
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::new();
        for item in list.iter() {
            arr.push(py_to_value(py, &item)?);
        }
        return Ok(Value::Array(arr));
    }

    // Try dict
    #[allow(deprecated)]
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            let val = py_to_value(py, &value)?;
            map.insert(key_str, val);
        }
        return Ok(Value::Object(map));
    }

    // Fallback: try to convert to string
    if let Ok(s) = obj.str() {
        return Ok(Value::String(s.to_string()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Cannot convert object to JSON value",
    ))
}

/// Convert serde_json::Value to Python object
fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<Py<PyAny>> {
    use pyo3::conversion::IntoPyObject;

    match val {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => {
            // For bool, we need to get the bound object and convert to PyObject
            let py_bool = b.into_pyobject(py)?;
            Ok(py_bool.to_owned().into_any().unbind())
        }
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                let py_int = i.into_pyobject(py)?;
                Ok(py_int.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                let py_float = f.into_pyobject(py)?;
                Ok(py_float.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        Value::String(s) => {
            let py_str = s.clone().into_pyobject(py)?;
            Ok(py_str.into_any().unbind())
        }
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Python wrapper for ScorerResult
#[pyclass(name = "ScorerResult")]
#[derive(Clone)]
pub struct PyScorerResult {
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub passed: Option<bool>,
    #[pyo3(get)]
    pub label: Option<String>,
    #[pyo3(get)]
    pub explanation: Option<String>,
    metadata_value: Option<Value>,
}

#[pymethods]
impl PyScorerResult {
    #[new]
    #[pyo3(signature = (score, passed=None, label=None, explanation=None, metadata=None))]
    fn new(
        score: f64,
        passed: Option<bool>,
        label: Option<String>,
        explanation: Option<String>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let metadata_value = if let Some(m) = metadata {
            Some(py_to_value(m.py(), m)?)
        } else {
            None
        };
        Ok(Self {
            score,
            passed,
            label,
            explanation,
            metadata_value,
        })
    }

    /// Get metadata as a Python dict
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.metadata_value {
            Some(v) => Ok(Some(value_to_py(py, v)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScorerResult(score={}, passed={:?}, label={:?})",
            self.score, self.passed, self.label
        )
    }
}

impl From<RustScorerResult> for PyScorerResult {
    fn from(r: RustScorerResult) -> Self {
        PyScorerResult {
            score: r.score,
            passed: r.passed,
            label: r.label,
            explanation: r.explanation,
            metadata_value: r.metadata,
        }
    }
}

/// Python wrapper for ScorerInput
#[pyclass(name = "ScorerInput")]
#[derive(Clone)]
pub struct PyScorerInput {
    inner: RustScorerInput,
}

#[pymethods]
impl PyScorerInput {
    #[new]
    #[pyo3(signature = (output, expected=None, input=None, trace=None))]
    fn new(
        py: Python<'_>,
        output: &Bound<'_, PyAny>,
        expected: Option<&Bound<'_, PyAny>>,
        input: Option<&Bound<'_, PyAny>>,
        trace: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Self> {
        let output_val = py_to_value(py, output)?;
        let expected_val = expected.map(|e| py_to_value(py, e)).transpose()?;
        let input_val = input.map(|i| py_to_value(py, i)).transpose()?;

        let trace_events = trace
            .map(|t| {
                t.iter()
                    .filter_map(|item| {
                        // Try to extract as dict and convert to TraceEvent
                        #[allow(deprecated)]
                        if let Ok(dict) = item.downcast::<PyDict>() {
                            parse_trace_event(py, dict).ok()
                        } else {
                            None
                        }
                    })
                    .collect()
            });

        Ok(PyScorerInput {
            inner: RustScorerInput {
                output: output_val,
                expected: expected_val,
                input: input_val,
                trace: trace_events,
            },
        })
    }

    /// Get the output value
    #[getter]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        value_to_py(py, &self.inner.output)
    }

    /// Get the expected value
    #[getter]
    fn expected(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.expected {
            Some(v) => Ok(Some(value_to_py(py, v)?)),
            None => Ok(None),
        }
    }

    /// Get the input value
    #[getter]
    fn input_value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match &self.inner.input {
            Some(v) => Ok(Some(value_to_py(py, v)?)),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("ScorerInput(output={:?})", self.inner.output)
    }
}

/// Parse a Python dict into a TraceEvent
fn parse_trace_event(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<RustTraceEvent> {
    let event_type = dict
        .get_item("event_type")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_default();

    let event_id = dict
        .get_item("event_id")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_default();

    let correlation_id = dict
        .get_item("correlation_id")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_default();

    let parent_correlation_id = dict
        .get_item("parent_correlation_id")?
        .and_then(|v| {
            if v.is_none() {
                None
            } else {
                v.extract::<String>().ok()
            }
        });

    let timestamp_ns = dict
        .get_item("timestamp_ns")?
        .map(|v| v.extract::<i64>())
        .transpose()?
        .unwrap_or(0);

    let data = dict
        .get_item("data")?
        .map(|v| py_to_value(py, &v))
        .transpose()?
        .unwrap_or(Value::Object(Default::default()));

    let name = dict
        .get_item("name")?
        .and_then(|v| {
            if v.is_none() {
                None
            } else {
                v.extract::<String>().ok()
            }
        });

    Ok(RustTraceEvent {
        event_type,
        event_id,
        correlation_id,
        parent_correlation_id,
        timestamp_ns,
        data,
        name,
    })
}

// ============================================================================
// Deterministic Scorers
// ============================================================================

/// Check if output exactly matches expected value.
///
/// Args:
///     input: ScorerInput with output and expected values
///     case_sensitive: Whether to compare case-sensitively (default: True)
///
/// Returns:
///     ScorerResult with score 1.0 for match, 0.0 for mismatch
#[pyfunction]
#[pyo3(signature = (input, case_sensitive=None))]
fn exact_match(input: &PyScorerInput, case_sensitive: Option<bool>) -> PyScorerResult {
    let config = ExactMatchConfig { case_sensitive };
    deterministic::exact_match(&input.inner, &config).into()
}

/// Check if output contains a specific pattern.
///
/// Args:
///     input: ScorerInput with output value
///     pattern: String to search for
///     case_sensitive: Whether to search case-sensitively (default: True)
///
/// Returns:
///     ScorerResult with score 1.0 if found, 0.0 if not found
#[pyfunction]
#[pyo3(signature = (input, pattern, case_sensitive=None))]
fn contains(input: &PyScorerInput, pattern: String, case_sensitive: Option<bool>) -> PyScorerResult {
    let config = ContainsConfig {
        pattern,
        case_sensitive,
    };
    deterministic::contains(&input.inner, &config).into()
}

/// Check if output is valid JSON.
///
/// Args:
///     input: ScorerInput with output value
///
/// Returns:
///     ScorerResult with score 1.0 if valid, 0.0 if invalid
#[pyfunction]
fn json_valid(input: &PyScorerInput) -> PyScorerResult {
    deterministic::json_valid(&input.inner).into()
}

/// Check if output matches a regex pattern.
///
/// Args:
///     input: ScorerInput with output value
///     pattern: Regex pattern to match
///
/// Returns:
///     ScorerResult with score 1.0 if matches, 0.0 if not
#[pyfunction]
fn regex_match(input: &PyScorerInput, pattern: String) -> PyScorerResult {
    let config = RegexConfig { pattern };
    deterministic::regex_match(&input.inner, &config).into()
}

/// Calculate Levenshtein similarity between output and expected.
///
/// Args:
///     input: ScorerInput with output and expected values
///     threshold: Minimum similarity threshold (0.0-1.0, default: 0.0)
///
/// Returns:
///     ScorerResult with similarity score
#[pyfunction]
#[pyo3(signature = (input, threshold=None))]
fn levenshtein(input: &PyScorerInput, threshold: Option<f64>) -> PyScorerResult {
    let config = LevenshteinConfig { threshold };
    deterministic::levenshtein(&input.inner, &config).into()
}

// ============================================================================
// Trace Assertions
// ============================================================================

/// Python wrapper for TraceAssertion
#[pyclass(name = "TraceAssertion")]
#[derive(Clone)]
pub struct PyTraceAssertion {
    inner: RustTraceAssertion,
}

#[pymethods]
impl PyTraceAssertion {
    /// Assert total tokens used is at most `max`
    #[staticmethod]
    fn max_tokens(max: u64) -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::MaxTokens { max },
        }
    }

    /// Assert number of LLM calls is at most `max`
    #[staticmethod]
    fn max_lm_calls(max: u32) -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::MaxLmCalls { max },
        }
    }

    /// Assert events occur in the specified order
    #[staticmethod]
    fn event_sequence(events: Vec<String>) -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::EventSequence { events },
        }
    }

    /// Assert a specific step was memoized
    #[staticmethod]
    fn step_memoized(step_name: String) -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::StepMemoized { step_name },
        }
    }

    /// Assert no error events occurred
    #[staticmethod]
    fn no_errors() -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::NoErrors,
        }
    }

    /// Assert total duration is under `max_ms` milliseconds
    #[staticmethod]
    fn duration_under(max_ms: u64) -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::DurationUnder { max_ms },
        }
    }

    /// Assert a specific event type occurred at least `min` times
    #[staticmethod]
    fn event_count(event_type: String, min: u32) -> Self {
        PyTraceAssertion {
            inner: RustTraceAssertion::EventCount { event_type, min },
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RustTraceAssertion::MaxTokens { max } => format!("TraceAssertion.max_tokens({})", max),
            RustTraceAssertion::MaxLmCalls { max } => {
                format!("TraceAssertion.max_lm_calls({})", max)
            }
            RustTraceAssertion::EventSequence { events } => {
                format!("TraceAssertion.event_sequence({:?})", events)
            }
            RustTraceAssertion::StepMemoized { step_name } => {
                format!("TraceAssertion.step_memoized({:?})", step_name)
            }
            RustTraceAssertion::NoErrors => "TraceAssertion.no_errors()".into(),
            RustTraceAssertion::DurationUnder { max_ms } => {
                format!("TraceAssertion.duration_under({})", max_ms)
            }
            RustTraceAssertion::EventCount { event_type, min } => {
                format!("TraceAssertion.event_count({:?}, {})", event_type, min)
            }
            RustTraceAssertion::Custom { name, .. } => {
                format!("TraceAssertion.custom({:?})", name)
            }
        }
    }
}

/// Score a trace against multiple assertions.
///
/// Args:
///     input: ScorerInput containing the trace to evaluate
///     assertions: List of TraceAssertion objects
///
/// Returns:
///     ScorerResult with aggregate score (proportion passed)
#[pyfunction]
fn trace_scorer(input: &PyScorerInput, assertions: Vec<PyTraceAssertion>) -> PyScorerResult {
    let rust_assertions: Vec<RustTraceAssertion> =
        assertions.into_iter().map(|a| a.inner).collect();
    trace::trace_score(&input.inner, &rust_assertions).into()
}

// ============================================================================
// Module Registration
// ============================================================================

/// Register eval submodule with Python
pub fn register_eval(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let eval_module = PyModule::new(py, "eval")?;

    // Classes
    eval_module.add_class::<PyScorerResult>()?;
    eval_module.add_class::<PyScorerInput>()?;
    eval_module.add_class::<PyTraceAssertion>()?;

    // Deterministic scorers
    eval_module.add_function(wrap_pyfunction!(exact_match, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(contains, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(json_valid, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(regex_match, &eval_module)?)?;
    eval_module.add_function(wrap_pyfunction!(levenshtein, &eval_module)?)?;

    // Trace scorer
    eval_module.add_function(wrap_pyfunction!(trace_scorer, &eval_module)?)?;

    parent.add_submodule(&eval_module)?;
    Ok(())
}
