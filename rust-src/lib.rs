use agnt5_sdk_core::telemetry::{
    create_tool_execution_span, flush_telemetry, record_tool_error, record_tool_success,
};
use agnt5_sdk_core::RuntimeContext;
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::Span;
use opentelemetry::Context;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

mod adk;
mod entity_state;
mod language_model;
mod types;
mod worker;
use entity_state::EntityStateManager;
use types::{
    PyComponentInfo, PyExecuteComponentRequest, PyExecuteComponentResponse, PyStateTransition,
    PyStateUpdate, PyStepCheckpoint,
};
use worker::{PyWorker, PyWorkerConfig};

/// Generic wrapper for OpenTelemetry spans that can be passed to Python
/// Used for all component types (tasks, workflows, agents, tools, etc.)
#[pyclass]
struct PySpan {
    span: Mutex<Option<BoxedSpan>>,
}

#[pymethods]
impl PySpan {
    /// Context manager entry - returns self
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit - ends the span
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
                span.set_attribute(opentelemetry::KeyValue::new("error", true));
                span.set_attribute(opentelemetry::KeyValue::new("error.message", error_str.clone()));
                span.set_status(opentelemetry::trace::Status::error(error_str));
            } else {
                span.set_status(opentelemetry::trace::Status::Ok);
            }
            // Span will be ended when dropped
        }
        Ok(false) // Don't suppress exceptions
    }

    /// Set a span attribute
    fn set_attribute(&self, key: String, value: String) -> PyResult<()> {
        let mut span_guard = self.span.lock().unwrap();
        if let Some(ref mut span) = *span_guard {
            span.set_attribute(opentelemetry::KeyValue::new(key, value));
        }
        Ok(())
    }

    /// Record an exception on the span
    fn record_exception(&self, exception: String) -> PyResult<()> {
        let mut span_guard = self.span.lock().unwrap();
        if let Some(ref mut span) = *span_guard {
            span.set_attribute(opentelemetry::KeyValue::new("error", true));
            span.set_attribute(opentelemetry::KeyValue::new("error.message", exception.clone()));
            span.set_status(opentelemetry::trace::Status::error(exception));
        }
        Ok(())
    }

    /// Manually end the span with a status
    fn end(&self, status: Option<String>) -> PyResult<()> {
        let mut span_guard = self.span.lock().unwrap();
        if let Some(mut span) = span_guard.take() {
            if let Some(status_str) = status {
                if status_str.to_lowercase() == "error" {
                    span.set_status(opentelemetry::trace::Status::error(""));
                } else {
                    span.set_status(opentelemetry::trace::Status::Ok);
                }
            } else {
                span.set_status(opentelemetry::trace::Status::Ok);
            }
            // Span will be ended when dropped
        }
        Ok(())
    }
}

/// Wrapper for OpenTelemetry span that can be passed to Python
/// Kept for backwards compatibility with existing tool span API
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

/// Python wrapper for RuntimeContext providing access to correlation IDs
///
/// This wrapper exposes the Rust RuntimeContext to Python, providing access to
/// run_id, trace_id, and span_id for logging correlation and trace propagation.
#[pyclass]
#[derive(Clone)]
pub struct PyRuntimeContext {
    pub(crate) inner: Arc<RuntimeContext>,
}

#[pymethods]
impl PyRuntimeContext {
    /// Get the run_id (unique identifier for this execution)
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.clone()
    }

    /// Get the OpenTelemetry trace_id for distributed tracing correlation
    #[getter]
    fn trace_id(&self) -> Option<String> {
        self.inner.trace_id.clone()
    }

    /// Get the OpenTelemetry span_id for logging correlation
    #[getter]
    fn span_id(&self) -> Option<String> {
        self.inner.span_id.clone()
    }

    /// Get the service name
    #[getter]
    fn service_name(&self) -> String {
        self.inner.service_name.clone()
    }

    /// Get the component name
    #[getter]
    fn component_name(&self) -> String {
        self.inner.component_name.clone()
    }
}

impl PyRuntimeContext {
    /// Internal method to get OpenTelemetry context for span propagation
    pub(crate) fn get_otel_context(&self) -> Option<Context> {
        self.inner.otel_context.clone()
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

/// Create a generic span for any component type (task, workflow, agent, etc.)
///
/// This is the main span creation function that Python code should use for
/// instrumentation. It creates spans via the Rust OpenTelemetry system with
/// proper parent-child span relationships when RuntimeContext is provided.
///
/// # Arguments
/// * `name` - Span name (e.g., "fetch_data")
/// * `component_type` - Component type (e.g., "task", "workflow", "agent", "function")
/// * `runtime_context` - Optional RuntimeContext providing trace context for span linkage
/// * `attributes` - Optional key-value attributes for the span
///
/// # Returns
/// A PySpan that can be used as a context manager in Python
#[pyfunction]
fn create_span(
    name: String,
    component_type: String,
    runtime_context: Option<&PyRuntimeContext>,
    attributes: Option<std::collections::HashMap<String, String>>,
) -> PyResult<PySpan> {
    let metadata = attributes.unwrap_or_default();

    // Extract parent context and metadata from RuntimeContext if available
    let (parent_context, service_name, run_id) = if let Some(ctx) = runtime_context {
        (
            ctx.get_otel_context(),
            ctx.inner.service_name.as_str(),
            ctx.inner.run_id.as_str(),
        )
    } else {
        (None, "", "")
    };

    let span = agnt5_sdk_core::create_component_span(
        &name,
        &component_type,
        service_name,
        "", // worker_id - not needed for Python-initiated spans
        run_id,
        parent_context, // ✅ Linked to parent span if runtime_context provided!
        Some(&metadata),
    );

    Ok(PySpan {
        span: Mutex::new(Some(span)),
    })
}

/// Flush all pending telemetry data (spans and logs)
///
/// This should be called before worker shutdown to ensure batched spans are exported.
/// The batch span processor buffers spans with a 5-second timeout by default.
#[pyfunction]
fn flush_telemetry_py() -> PyResult<()> {
    flush_telemetry().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to flush telemetry: {}", e))
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
    trace_id: Option<String>,
    span_id: Option<String>,
    run_id: Option<String>,
) -> PyResult<()> {
    // Create a span with Python metadata, inheriting from current span if available
    // Include correlation IDs (trace_id, span_id, run_id) as span attributes for OTLP export
    let current_span = tracing::Span::current();
    let span = if current_span.is_none() || current_span == tracing::Span::none() {
        // No current span, create standalone span with correlation attributes
        tracing::info_span!(
            "python_log",
            python.module = module_path.as_deref(),
            python.filename = filename.as_deref(),
            python.line = line,
            python.target = target.as_deref(),
            message = %message,
            otel.trace_id = trace_id.as_deref(),
            otel.span_id = span_id.as_deref(),
            run.id = run_id.as_deref(),
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
            otel.trace_id = trace_id.as_deref(),
            otel.span_id = span_id.as_deref(),
            run.id = run_id.as_deref(),
        )
    };
    let _enter = span.enter();

    // Emit log at appropriate level through Rust tracing
    // Use agnt5_sdk_python target to ensure logs match the agnt5=info filter
    // Include correlation IDs as log event fields (not span fields) so they appear in OTLP log records
    match level.to_uppercase().as_str() {
        "DEBUG" => tracing::debug!(
            target: "agnt5_sdk_python",
            trace_id = trace_id.as_deref(),
            span_id = span_id.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "INFO" => tracing::info!(
            target: "agnt5_sdk_python",
            trace_id = trace_id.as_deref(),
            span_id = span_id.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "WARNING" | "WARN" => tracing::warn!(
            target: "agnt5_sdk_python",
            trace_id = trace_id.as_deref(),
            span_id = span_id.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "ERROR" => tracing::error!(
            target: "agnt5_sdk_python",
            trace_id = trace_id.as_deref(),
            span_id = span_id.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "CRITICAL" => tracing::error!(
            target: "agnt5_sdk_python",
            trace_id = trace_id.as_deref(),
            span_id = span_id.as_deref(),
            run_id = run_id.as_deref(),
            "[CRITICAL] {}",
            message
        ),
        _ => tracing::info!(
            target: "agnt5_sdk_python",
            trace_id = trace_id.as_deref(),
            span_id = span_id.as_deref(),
            run_id = run_id.as_deref(),
            "[{}] {}",
            level,
            message
        ),
    }

    Ok(())
}

/// The Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize PyO3-log to bridge Rust logs to Python
    // TODO: Re-enable once pyo3-log supports pyo3 0.27
    // pyo3_log::init();

    // Worker-related classes
    m.add_class::<PyWorkerConfig>()?;
    m.add_class::<PyWorker>()?;
    m.add_class::<PyExecuteComponentRequest>()?;
    m.add_class::<PyExecuteComponentResponse>()?;
    m.add_class::<PyComponentInfo>()?;
    m.add_class::<PyStepCheckpoint>()?;
    m.add_class::<PyStateTransition>()?;
    m.add_class::<PyStateUpdate>()?;

    // Entity state management
    m.add_class::<EntityStateManager>()?;

    // Language Model classes
    language_model::register_language_model(m)?;

    // ADK scaffolding
    adk::register_adk(m)?;

    // Telemetry classes
    m.add_class::<PySpan>()?;
    m.add_class::<PyToolSpan>()?;
    m.add_class::<PyRuntimeContext>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(log_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(create_span, m)?)?;
    m.add_function(wrap_pyfunction!(create_tool_span, m)?)?;
    m.add_function(wrap_pyfunction!(flush_telemetry_py, m)?)?;

    Ok(())
}
