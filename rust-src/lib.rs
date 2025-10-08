use agnt5_sdk_core::telemetry::{
    create_tool_execution_span, record_tool_error, record_tool_success,
};
use opentelemetry::global::BoxedSpan;
use pyo3::prelude::*;
use std::sync::Mutex;

mod adk;
mod language_model;
mod types;
mod worker;
use types::{
    PyComponentInfo, PyExecuteComponentRequest, PyExecuteComponentResponse, PyStateTransition,
    PyStateUpdate, PyStepCheckpoint,
};
use worker::{PyWorker, PyWorkerConfig};

/// Wrapper for OpenTelemetry span that can be passed to Python
#[pyclass]
struct PyToolSpan {
    span: Mutex<Option<BoxedSpan>>,
}

#[pymethods]
impl PyToolSpan {
    /// End the span (called automatically by context manager)
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// End the span when exiting context manager
    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let mut span_guard = self.span.lock().unwrap();
        if let Some(mut span) = span_guard.take() {
            // Check if there was an exception
            if let Some(exc) = exc_value {
                let error_str = format!("{}", exc);
                record_tool_error(&mut span, &error_str);
            }
            // Span will be ended when dropped
        }
        Ok(false) // Don't suppress exceptions
    }

    /// Record successful tool execution with optional result
    fn record_success(&self, result: Option<String>) -> PyResult<()> {
        let mut span_guard = self.span.lock().unwrap();
        if let Some(ref mut span) = *span_guard {
            record_tool_success(span, result.as_deref());
        }
        Ok(())
    }

    /// Record tool execution error
    fn record_error(&self, error_msg: String) -> PyResult<()> {
        let mut span_guard = self.span.lock().unwrap();
        if let Some(ref mut span) = *span_guard {
            record_tool_error(span, &error_msg);
        }
        Ok(())
    }
}

/// Create a span for tool execution following OpenTelemetry Gen AI semantic conventions
#[pyfunction]
fn create_tool_span(
    tool_name: String,
    tool_call_id: Option<String>,
    tool_description: Option<String>,
    arguments: Option<String>,
) -> PyResult<PyToolSpan> {
    let span = create_tool_execution_span(
        &tool_name,
        tool_call_id.as_deref(),
        tool_description.as_deref(),
        arguments.as_deref(),
    );

    Ok(PyToolSpan {
        span: Mutex::new(Some(span)),
    })
}

/// Forward Python logs to Rust tracing system for OpenTelemetry integration
#[pyfunction]
fn log_from_python(
    level: &str,
    message: String,
    target: Option<String>,
    module_path: Option<String>,
    filename: Option<String>,
    line: Option<u32>,
) -> PyResult<()> {
    // Create a span with Python metadata, inheriting from current span if available
    let current_span = tracing::Span::current();
    let span = if current_span.is_none() || current_span == tracing::Span::none() {
        // No current span, create standalone span
        tracing::info_span!(
            "python_log",
            python.module = module_path.as_deref(),
            python.filename = filename.as_deref(),
            python.line = line,
            python.target = target.as_deref(),
            message = %message,
        )
    } else {
        // Create child span that inherits fields from current span (including invocation.id)
        tracing::info_span!(
            parent: &current_span,
            "python_log",
            python.module = module_path.as_deref(),
            python.filename = filename.as_deref(),
            python.line = line,
            python.target = target.as_deref(),
            message = %message,
        )
    };
    let _enter = span.enter();

    // Emit log at appropriate level through Rust tracing
    // Use agnt5_sdk_python target to ensure logs match the agnt5=info filter
    // Include message in both span field (for OpenTelemetry attributes) and log event
    match level.to_uppercase().as_str() {
        "DEBUG" => tracing::debug!(target: "agnt5_sdk_python", "{}", message),
        "INFO" => tracing::info!(target: "agnt5_sdk_python", "{}", message),
        "WARNING" | "WARN" => tracing::warn!(target: "agnt5_sdk_python", "{}", message),
        "ERROR" => tracing::error!(target: "agnt5_sdk_python", "{}", message),
        "CRITICAL" => tracing::error!(target: "agnt5_sdk_python", "[CRITICAL] {}", message),
        _ => tracing::info!(target: "agnt5_sdk_python", "[{}] {}", level, message),
    }

    Ok(())
}

/// The Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize PyO3-log to bridge Rust logs to Python
    pyo3_log::init();

    // Worker-related classes
    m.add_class::<PyWorkerConfig>()?;
    m.add_class::<PyWorker>()?;
    m.add_class::<PyExecuteComponentRequest>()?;
    m.add_class::<PyExecuteComponentResponse>()?;
    m.add_class::<PyComponentInfo>()?;
    m.add_class::<PyStepCheckpoint>()?;
    m.add_class::<PyStateTransition>()?;
    m.add_class::<PyStateUpdate>()?;

    // Language Model classes
    language_model::register_language_model(m)?;

    // ADK scaffolding
    adk::register_adk(m)?;

    // Telemetry classes
    m.add_class::<PyToolSpan>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(log_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(create_tool_span, m)?)?;

    Ok(())
}
