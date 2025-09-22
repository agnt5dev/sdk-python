use pyo3::prelude::*;

mod types;
mod worker;
use types::{
    PyComponentInfo,
    PyExecuteComponentRequest,
    PyExecuteComponentResponse,
    PyStateTransition,
    PyStateUpdate,
    PyStepCheckpoint,
};
use worker::{PyWorker, PyWorkerConfig};

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
    // Include message in both span field (for OpenTelemetry attributes) and log event (for VictoriaMetrics)
    match level.to_uppercase().as_str() {
        "DEBUG" => tracing::debug!("{}", message),
        "INFO" => tracing::info!("{}", message),
        "WARNING" | "WARN" => tracing::warn!("{}", message),
        "ERROR" => tracing::error!("{}", message),
        "CRITICAL" => tracing::error!("[CRITICAL] {}", message),
        _ => tracing::info!("[{}] {}", level, message),
    }

    Ok(())
}

/// The Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize PyO3-log to bridge Rust logs to Python
    pyo3_log::init();

    m.add_class::<PyWorkerConfig>()?;
    m.add_class::<PyWorker>()?;
    m.add_class::<PyExecuteComponentRequest>()?;
    m.add_class::<PyExecuteComponentResponse>()?;
    m.add_class::<PyComponentInfo>()?;
    m.add_class::<PyStepCheckpoint>()?;
    m.add_class::<PyStateTransition>()?;
    m.add_class::<PyStateUpdate>()?;
    m.add_function(wrap_pyfunction!(log_from_python, m)?)?;
    Ok(())
}
