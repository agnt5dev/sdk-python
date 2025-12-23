use agnt5_sdk_core::telemetry::{
    create_tool_execution_span, record_tool_error, record_tool_success,
};
use agnt5_sdk_core::RuntimeContext;
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::Span;
use opentelemetry::Context;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Check if span export is enabled (cached for performance)
/// Default: false (spans not exported to journal by default)
fn is_span_export_enabled() -> bool {
    static SPAN_EXPORT_ENABLED: OnceLock<bool> = OnceLock::new();
    *SPAN_EXPORT_ENABLED.get_or_init(|| {
        std::env::var("AGNT5_SPAN_EXPORT_ENABLED")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false) // Default: disabled
    })
}

mod adk;
mod checkpoint_client;
mod entity_state;
mod language_model;
mod memory;
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
    // Fields for journal export on streaming requests
    run_id: Option<String>,
    tenant_id: Option<String>,
    is_streaming: bool,
    name: String,
    component_type: String,
    start_time_ns: i64,
    // Parent span ID for maintaining trace hierarchy in journal export
    parent_span_id: Option<String>,
    // This span's trace_id and span_id for Python contextvar propagation
    trace_id: String,
    span_id: String,
}

#[pymethods]
impl PySpan {
    /// Get the trace ID for this span (for Python contextvar propagation)
    #[getter]
    fn trace_id(&self) -> &str {
        &self.trace_id
    }

    /// Get the span ID for this span (for Python contextvar propagation)
    #[getter]
    fn span_id(&self) -> &str {
        &self.span_id
    }

    /// Get the parent span ID (if any)
    #[getter]
    fn parent_span_id(&self) -> Option<&str> {
        self.parent_span_id.as_deref()
    }

    /// Context manager entry - returns self (Python wrapper handles contextvar)
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager exit - ends the span (Python wrapper handles contextvar)
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let end_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);

        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;

        // Get span IDs before processing (needed for journal export)
        let (trace_id_str, span_id_str) = if let Some(ref span) = *span_guard {
            let span_ctx = span.span_context();
            (span_ctx.trace_id().to_string(), span_ctx.span_id().to_string())
        } else {
            (String::new(), String::new())
        };

        let mut status_code = "ok";

        if let Some(mut span) = span_guard.take() {
            // Check if there was an exception
            if let Some(exc) = exc_value {
                // Check if this is WaitingForUserInputException (HITL pause - not an error)
                let is_hitl_pause = if let Some(exc_type_obj) = exc_type {
                    // Get the exception type name
                    if let Ok(exc_type_name) = exc_type_obj.getattr("__name__") {
                        if let Ok(name_str) = exc_type_name.extract::<String>() {
                            name_str == "WaitingForUserInputException"
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if is_hitl_pause {
                    // HITL pause is normal workflow behavior, not an error
                    span.set_attribute(opentelemetry::KeyValue::new("hitl.pause", true));
                    span.set_attribute(opentelemetry::KeyValue::new("hitl.question", format!("{}", exc)));
                    span.set_status(opentelemetry::trace::Status::Ok);
                } else {
                    // Regular exception - mark as error
                    status_code = "error";
                    let error_str = format!("{}", exc);
                    span.set_attribute(opentelemetry::KeyValue::new("error", true));
                    span.set_attribute(opentelemetry::KeyValue::new("error.message", error_str.clone()));
                    span.set_status(opentelemetry::trace::Status::error(error_str));
                }
            } else {
                span.set_status(opentelemetry::trace::Status::Ok);
            }
            // Span will be ended when dropped
        }

        // Queue span export for background flush if streaming AND span export are enabled
        // Uses the global SpanExportQueue which is flushed by a background task in the Worker's Tokio runtime
        // Note: Span export is disabled by default (AGNT5_SPAN_EXPORT_ENABLED=false)
        let span_export_enabled = is_span_export_enabled();
        tracing::debug!(
            "🔍 CHILD-SPAN-DEBUG: PySpan.__exit__ called, is_streaming={}, span_export_enabled={}, run_id={:?}, name={}",
            self.is_streaming,
            span_export_enabled,
            self.run_id,
            self.name
        );
        if self.is_streaming && span_export_enabled {
            if let Some(ref run_id) = self.run_id {
                // Get the global span export queue (set by Worker on initialization)
                if let Some(queue) = crate::worker::get_span_export_queue() {
                    use agnt5_sdk_core::span_export_queue::SpanExportRequest;

                    let request = SpanExportRequest {
                        run_id: run_id.clone(),
                        tenant_id: self.tenant_id.clone(),
                        trace_id: trace_id_str.clone(),
                        span_id: span_id_str.clone(),
                        parent_span_id: self.parent_span_id.clone(),
                        name: self.name.clone(),
                        kind: self.component_type.clone(),
                        start_time_ns: self.start_time_ns,
                        end_time_ns,
                        status_code: status_code.to_string(),
                        status_description: None,
                        attributes: None, // TODO: capture span attributes
                        queued_at: std::time::Instant::now(),
                    };

                    if let Err(e) = queue.push(request) {
                        tracing::warn!(
                            "Failed to queue child span for journal export: {}",
                            e
                        );
                    } else {
                        tracing::debug!(
                            "🔍 CHILD-SPAN-DEBUG: Queued span '{}' for journal export (queue_size={})",
                            self.name,
                            queue.len()
                        );
                    }
                } else {
                    tracing::debug!(
                        "🔍 CHILD-SPAN-DEBUG: Span export queue not initialized, skipping child span export"
                    );
                }
            }
        }

        Ok(false) // Don't suppress exceptions
    }

    /// Set a span attribute
    fn set_attribute(&self, key: String, value: String) -> PyResult<()> {
        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;
        if let Some(ref mut span) = *span_guard {
            span.set_attribute(opentelemetry::KeyValue::new(key, value));
        }
        Ok(())
    }

    /// Record an exception on the span
    fn record_exception(&self, exception: String) -> PyResult<()> {
        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;
        if let Some(ref mut span) = *span_guard {
            span.set_attribute(opentelemetry::KeyValue::new("error", true));
            span.set_attribute(opentelemetry::KeyValue::new("error.message", exception.clone()));
            span.set_status(opentelemetry::trace::Status::error(exception));
        }
        Ok(())
    }

    /// Manually end the span with a status
    fn end(&self, status: Option<String>) -> PyResult<()> {
        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;
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
        exc_type: Option<&Bound<'_, PyAny>>,
        exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;
        if let Some(mut span) = span_guard.take() {
            // Check if there was an exception
            if let Some(exc) = exc_value {
                // Check if this is WaitingForUserInputException (HITL pause - not an error)
                let is_hitl_pause = if let Some(exc_type_obj) = exc_type {
                    // Get the exception type name
                    if let Ok(exc_type_name) = exc_type_obj.getattr("__name__") {
                        if let Ok(name_str) = exc_type_name.extract::<String>() {
                            name_str == "WaitingForUserInputException"
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if is_hitl_pause {
                    // HITL pause is normal workflow behavior, not an error
                    span.set_attribute(opentelemetry::KeyValue::new("hitl.pause", true));
                    span.set_attribute(opentelemetry::KeyValue::new("hitl.question", format!("{}", exc)));
                    span.set_status(opentelemetry::trace::Status::Ok);
                } else {
                    // Regular exception - mark as error
                    let error_str = format!("{}", exc);
                    record_tool_error(&mut span, &error_str);
                }
            }
            // Span will be ended when dropped
        }
        Ok(false) // Don't suppress exceptions
    }

    /// Record successful tool execution with optional result
    fn record_success(&self, result: Option<String>) -> PyResult<()> {
        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;
        if let Some(ref mut span) = *span_guard {
            record_tool_success(span, result.as_deref());
        }
        Ok(())
    }

    /// Record tool execution error
    fn record_error(&self, error_msg: String) -> PyResult<()> {
        let mut span_guard = self.span.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Span mutex poisoned: {}", e))
        })?;
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

    /// Get the is_streaming flag (whether this is a streaming SSE request)
    #[getter]
    fn is_streaming(&self) -> bool {
        self.inner.is_streaming
    }

    /// Get the tenant_id
    #[getter]
    fn tenant_id(&self) -> String {
        self.inner.tenant_id.clone()
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
/// * `parent_trace_id` - Optional parent trace ID (from Python contextvar, for proper nesting)
/// * `parent_span_id` - Optional parent span ID (from Python contextvar, for proper nesting)
/// * `py` - Python GIL token (automatically provided by PyO3)
///
/// # Returns
/// A PySpan that can be used as a context manager in Python
#[pyfunction]
#[pyo3(signature = (name, component_type, runtime_context=None, attributes=None, parent_trace_id=None, parent_span_id=None))]
fn create_span(
    py: Python,
    name: String,
    component_type: String,
    runtime_context: Option<&PyRuntimeContext>,
    attributes: Option<std::collections::HashMap<String, String>>,
    parent_trace_id: Option<String>,
    parent_span_id: Option<String>,
) -> PyResult<PySpan> {
    use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};

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

    // Extract streaming context for journal export
    let (is_streaming, run_id_opt, tenant_id_opt) = if let Some(ctx) = runtime_context {
        let streaming = ctx.inner.is_streaming;
        (
            streaming,
            Some(ctx.inner.run_id.clone()),
            Some(ctx.inner.tenant_id.clone()),
        )
    } else {
        (false, None, None)
    };

    // PRIORITY ORDER for parent context:
    // 1. Explicit parent_trace_id/parent_span_id (from Python contextvar - async-safe!)
    // 2. RuntimeContext.otel_context (initial context from worker)
    // 3. Python contextvar fallback
    let parent_context_from_ids = if let (Some(ref trace_id_str), Some(ref span_id_str)) = (&parent_trace_id, &parent_span_id) {
        // Parse trace_id and span_id from hex strings
        let trace_id = TraceId::from_hex(trace_id_str).unwrap_or(TraceId::INVALID);
        let span_id = SpanId::from_hex(span_id_str).unwrap_or(SpanId::INVALID);

        if trace_id != TraceId::INVALID && span_id != SpanId::INVALID {
            // Create a SpanContext from the IDs
            let span_context = SpanContext::new(
                trace_id,
                span_id,
                TraceFlags::SAMPLED,
                true, // is_remote
                TraceState::default(),
            );
            // Create a Context with this remote span context for proper parent-child linking
            Some(Context::new().with_remote_span_context(span_context))
        } else {
            None
        }
    } else {
        None
    };

    let from_contextvar = parent_context_from_ids.is_some();

    let (parent_context, service_name, run_id) = if let Some(ctx) = parent_context_from_ids {
        // Use the context from Python contextvar - ensures proper async-safe nesting
        let svc = runtime_context.map(|c| c.inner.service_name.as_str()).unwrap_or("");
        let rid = runtime_context.map(|c| c.inner.run_id.as_str()).unwrap_or("");
        (Some(ctx), svc, rid)
    } else if let Some(ctx) = runtime_context {
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

    // Use provided parent_span_id or extract from parent context
    let parent_span_id_for_journal = parent_span_id.clone().or_else(|| {
        parent_context.as_ref().and_then(|ctx| {
            let span = ctx.span();
            let span_ctx = span.span_context();
            if span_ctx.is_valid() {
                Some(span_ctx.span_id().to_string())
            } else {
                None
            }
        })
    });

    tracing::debug!(
        "🔍 CHILD-SPAN-DEBUG: create_span called, name={}, parent_span_id={:?}, from_contextvar={}, is_streaming={}",
        name,
        parent_span_id_for_journal,
        from_contextvar,
        is_streaming
    );

    let span = agnt5_sdk_core::create_component_span(
        &name,
        &component_type,
        service_name,
        "", // worker_id - not needed for Python-initiated spans
        run_id,
        parent_context, // ✅ Linked to proper parent span!
        Some(&metadata),
    );

    // Extract this span's trace_id and span_id for Python contextvar propagation
    let (new_trace_id, new_span_id) = {
        let span_ctx = span.span_context();
        if span_ctx.is_valid() {
            (span_ctx.trace_id().to_string(), span_ctx.span_id().to_string())
        } else {
            (String::new(), String::new())
        }
    };

    // Capture start time for journal span data
    let start_time_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);

    Ok(PySpan {
        span: Mutex::new(Some(span)),
        run_id: run_id_opt,
        tenant_id: tenant_id_opt,
        is_streaming,
        name: name.clone(),
        component_type: component_type.clone(),
        start_time_ns,
        parent_span_id: parent_span_id_for_journal,
        trace_id: new_trace_id,
        span_id: new_span_id,
    })
}

/// Forward Python logs to Rust tracing system for OpenTelemetry integration
///
/// When is_streaming is true, logs are also exported to the journal for real-time SSE delivery.
#[pyfunction]
#[pyo3(signature = (level, message, target=None, module_path=None, filename=None, line=None, trace_id=None, span_id=None, run_id=None, is_streaming=None, tenant_id=None, deployment_id=None, attributes=None))]
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
    is_streaming: Option<bool>,
    tenant_id: Option<String>,
    deployment_id: Option<String>,
    attributes: Option<HashMap<String, String>>,
) -> PyResult<()> {
    // Get effective tenant_id and deployment_id from parameter or global config
    let effective_tenant_id = tenant_id
        .or_else(|| agnt5_sdk_core::telemetry::get_tenant_id().map(|s| s.to_string()));
    let effective_deployment_id = deployment_id
        .or_else(|| agnt5_sdk_core::telemetry::get_deployment_id().map(|s| s.to_string()));
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
                // Safe unwrap: we just checked the length
                let trace_id = TraceId::from_bytes(tid_bytes.try_into().expect("trace_id length verified"));
                let span_id = SpanId::from_bytes(sid_bytes.try_into().expect("span_id length verified"));

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

    // Serialize attributes to JSON for structured logging
    // Tracing macros require compile-time known fields, so we serialize to a single JSON string
    let attrs_json = attributes.as_ref().map(|attrs| {
        serde_json::to_string(attrs).unwrap_or_else(|_| "{}".to_string())
    });

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
            tenant_id = effective_tenant_id.as_deref(),
            deployment_id = effective_deployment_id.as_deref(),
            log_attributes = attrs_json.as_deref(),
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
            tenant_id = effective_tenant_id.as_deref(),
            deployment_id = effective_deployment_id.as_deref(),
            log_attributes = attrs_json.as_deref(),
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
            tenant_id = effective_tenant_id.as_deref(),
            deployment_id = effective_deployment_id.as_deref(),
            log_attributes = attrs_json.as_deref(),
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
            tenant_id = effective_tenant_id.as_deref(),
            deployment_id = effective_deployment_id.as_deref(),
            log_attributes = attrs_json.as_deref(),
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
            tenant_id = effective_tenant_id.as_deref(),
            deployment_id = effective_deployment_id.as_deref(),
            log_attributes = attrs_json.as_deref(),
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
            tenant_id = effective_tenant_id.as_deref(),
            deployment_id = effective_deployment_id.as_deref(),
            log_attributes = attrs_json.as_deref(),
            "[{}] {}",
            level,
            message
        ),
    }

    // Queue log export for background flush if streaming is enabled
    // Uses the global LogExportQueue which is flushed by a background task in the Worker's Tokio runtime
    if is_streaming.unwrap_or(false) {
        if let Some(ref rid) = run_id {
            // Get the global log export queue (set by Worker on initialization)
            if let Some(queue) = crate::worker::get_log_export_queue() {
                use agnt5_sdk_core::span_export_queue::LogExportRequest;

                // Get current timestamp
                let timestamp_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);

                let request = LogExportRequest {
                    run_id: rid.clone(),
                    tenant_id: effective_tenant_id.clone(),
                    deployment_id: effective_deployment_id.clone(),
                    trace_id: trace_id.clone().unwrap_or_default(),
                    span_id: span_id.clone().unwrap_or_default(),
                    timestamp_ns,
                    severity: level.to_string(),
                    body: message.clone(),
                    attributes: attributes.clone(),
                    queued_at: std::time::Instant::now(),
                };

                if let Err(e) = queue.push(request) {
                    tracing::warn!(
                        "Failed to queue log for journal export: {}",
                        e
                    );
                } else {
                    tracing::debug!(
                        "Queued log for journal export (queue_size={})",
                        queue.len()
                    );
                }
            } else {
                // Log export queue not initialized - this can happen during testing or
                // when Python calls this function before worker initialization.
                // Log a debug message but don't fail - tracing output above still works.
                tracing::debug!(
                    "Log export queue not initialized, skipping journal export for run_id={}",
                    rid
                );
            }
        }
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

    // Checkpoint client for step-level memoization (Phase 3)
    checkpoint_client::register_checkpoint_client(m)?;

    // Semantic memory
    memory::register_memory(m)?;

    // Telemetry classes
    m.add_class::<PySpan>()?;
    m.add_class::<PyToolSpan>()?;
    m.add_class::<PyRuntimeContext>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(log_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(create_span, m)?)?;
    m.add_function(wrap_pyfunction!(create_tool_span, m)?)?;
    Ok(())
}
