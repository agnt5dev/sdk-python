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

/// Helper function to get runtime context data from Python's contextvar
///
/// Attempts to retrieve the current context from the _current_context contextvar
/// and extract trace context information from its _runtime_context attribute.
///
/// # Arguments
/// * `py` - Python GIL token
///
/// # Returns
/// Tuple of (otel_context Option, service_name String, run_id String) or None
pub(crate) fn get_runtime_context_from_contextvar(py: Python) -> PyResult<Option<(Option<opentelemetry::Context>, String, String)>> {
    // Import the context module
    let context_module = py.import("agnt5.context")?;

    // Get the get_current_context function
    let get_current_context = context_module.getattr("get_current_context")?;

    // Call get_current_context() to get the current Context object
    let current_context = get_current_context.call0()?;

    // Check if we got None
    if current_context.is_none() {
        return Ok(None);
    }

    // Try to get _runtime_context attribute (which is a PyRuntimeContext)
    if let Ok(runtime_context_obj) = current_context.getattr("_runtime_context") {
        if !runtime_context_obj.is_none() {
            // Try to extract as PyRuntimeContext to get full OpenTelemetry context
            if let Ok(py_runtime_ctx) = runtime_context_obj.extract::<Py<PyRuntimeContext>>() {
                let runtime_ctx = py_runtime_ctx.borrow(py);

                // Extract OpenTelemetry context for proper parent-child span linking
                let otel_ctx = runtime_ctx.get_otel_context();
                let service_name = runtime_ctx.inner.service_name.clone();
                let run_id = runtime_ctx.inner.run_id.clone();

                return Ok(Some((otel_ctx, service_name, run_id)));
            }

            // Fallback: If extraction fails, try accessing attributes directly
            let service_name: String = runtime_context_obj.getattr("service_name")?.extract()?;
            let run_id: String = runtime_context_obj.getattr("run_id")?.extract()?;

            // No OpenTelemetry context available in fallback path
            return Ok(Some((None, service_name, run_id)));
        }
    }

    Ok(None)
}

/// Create a generic span for any component type (task, workflow, agent, etc.)
///
/// This is the main span creation function that Python code should use for
/// instrumentation. It creates spans via the Rust OpenTelemetry system with
/// proper parent-child span relationships when RuntimeContext is provided.
///
/// If runtime_context is not provided, this function will attempt to retrieve
/// it from the Python context variable (_current_context) for automatic trace linking.
///
/// # Arguments
/// * `name` - Span name (e.g., "fetch_data")
/// * `component_type` - Component type (e.g., "task", "workflow", "agent", "function")
/// * `runtime_context` - Optional RuntimeContext providing trace context for span linkage
/// * `attributes` - Optional key-value attributes for the span
/// * `py` - Python GIL token (automatically provided by PyO3)
///
/// # Returns
/// A PySpan that can be used as a context manager in Python
#[pyfunction]
fn create_span(
    py: Python,
    name: String,
    component_type: String,
    runtime_context: Option<&PyRuntimeContext>,
    attributes: Option<std::collections::HashMap<String, String>>,
) -> PyResult<PySpan> {
    let metadata = attributes.unwrap_or_default();

    // Extract parent context and metadata from RuntimeContext
    // If not provided, try to get from Python contextvar
    // We need to bind contextvar result to ensure strings live long enough
    let contextvar_result = if runtime_context.is_none() {
        match get_runtime_context_from_contextvar(py) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Warning: Failed to get runtime_context from contextvar: {}", e);
                None
            }
        }
    } else {
        None
    };

    let (parent_context, service_name, run_id) = if let Some(ctx) = runtime_context {
        (
            ctx.get_otel_context(),
            ctx.inner.service_name.as_str(),
            ctx.inner.run_id.as_str(),
        )
    } else if let Some((otel_ctx, ref svc_name, ref run_id_str)) = contextvar_result {
        (otel_ctx, svc_name.as_str(), run_id_str.as_str())
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
    // When there's an active span, emit logs within that span's context.
    // The tracing_opentelemetry layer will automatically extract the OpenTelemetry
    // trace context from the active span and populate the OTLP log record's trace_id/span_id.
    //
    // We still include trace_id/span_id/run_id as log attributes for:
    // 1. Backwards compatibility with existing observability queries
    // 2. Additional correlation context in log viewers
    // 3. Cases where the span context might not be available
    //
    // The key difference: we DON'T create a span just for logging. We emit the log
    // in the current span context, and OpenTelemetry handles the rest.

    // CRITICAL FIX: If we have trace_id and span_id from Python, create an OpenTelemetry
    // context and attach it so the opentelemetry_appender_tracing layer can extract it
    let _cx_guard = if let (Some(tid_str), Some(sid_str)) = (&trace_id, &span_id) {
        use opentelemetry::trace::{TraceId, SpanId, SpanContext, TraceFlags, TraceContextExt};

        // Parse hex strings to bytes
        if let (Ok(tid_bytes), Ok(sid_bytes)) = (hex::decode(tid_str), hex::decode(sid_str)) {
            if tid_bytes.len() == 16 && sid_bytes.len() == 8 {
                let trace_id = TraceId::from_bytes(tid_bytes.try_into().unwrap());
                let span_id = SpanId::from_bytes(sid_bytes.try_into().unwrap());

                let span_context = SpanContext::new(
                    trace_id,
                    span_id,
                    TraceFlags::SAMPLED,
                    false,
                    Default::default(),
                );

                // Create a minimal context with this span
                let cx = opentelemetry::Context::current().with_remote_span_context(span_context);
                Some(cx.attach())
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Emit log at appropriate level through Rust tracing
    // The opentelemetry_appender_tracing layer will now extract trace_id/span_id
    // from the attached OpenTelemetry context above
    match level.to_uppercase().as_str() {
        "DEBUG" => tracing::debug!(
            target: "agnt5_sdk_python",
            python_module = module_path.as_deref(),
            python_filename = filename.as_deref(),
            python_line = line,
            python_target = target.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "INFO" => tracing::info!(
            target: "agnt5_sdk_python",
            python_module = module_path.as_deref(),
            python_filename = filename.as_deref(),
            python_line = line,
            python_target = target.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "WARNING" | "WARN" => tracing::warn!(
            target: "agnt5_sdk_python",
            python_module = module_path.as_deref(),
            python_filename = filename.as_deref(),
            python_line = line,
            python_target = target.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "ERROR" => tracing::error!(
            target: "agnt5_sdk_python",
            python_module = module_path.as_deref(),
            python_filename = filename.as_deref(),
            python_line = line,
            python_target = target.as_deref(),
            run_id = run_id.as_deref(),
            "{}",
            message
        ),
        "CRITICAL" => tracing::error!(
            target: "agnt5_sdk_python",
            python_module = module_path.as_deref(),
            python_filename = filename.as_deref(),
            python_line = line,
            python_target = target.as_deref(),
            run_id = run_id.as_deref(),
            "[CRITICAL] {}",
            message
        ),
        _ => tracing::info!(
            target: "agnt5_sdk_python",
            python_module = module_path.as_deref(),
            python_filename = filename.as_deref(),
            python_line = line,
            python_target = target.as_deref(),
            run_id = run_id.as_deref(),
            "[{}] {}",
            level,
            message
        ),
    }

    Ok(())
}

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
