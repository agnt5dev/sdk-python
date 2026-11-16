//! The Python context each inbound run executes in.
//!
//! A run arrives from the platform, not from Python code, so there is no caller
//! context to inherit. Each run gets its own fresh `contextvars.Context`
//! instead of borrowing whatever context the Tokio thread that dispatches it
//! last held. Values one run sets then stay with that run, and two runs
//! dispatched from the same thread never see each other's values.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

static CONTEXT_TYPE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static TRACE_METADATA_VAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static IS_AWAITABLE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Builds the context for one run, with `agnt5.worker._trace_metadata` set to
/// the run's execution-span trace headers. LLM calls read that variable to
/// parent their spans when no span is active.
pub(crate) fn new_run_context<'py>(
    py: Python<'py>,
    trace_metadata: HashMap<String, String>,
) -> PyResult<Bound<'py, PyAny>> {
    match TRACE_METADATA_VAR.import(py, "agnt5.worker", "_trace_metadata") {
        Ok(var) => context_with(py, var, trace_metadata),
        Err(err) => {
            // Tracing must not stop the run: fall back to a context without it.
            log::warn!("Failed to read _trace_metadata contextvar: {}", err);
            CONTEXT_TYPE.import(py, "contextvars", "Context")?.call0()
        }
    }
}

/// Calls `callable(arg)` inside `context`, so any contextvars it reads or sets
/// belong to that context rather than the calling thread's.
pub(crate) fn run_in<'py>(
    context: &Bound<'py, PyAny>,
    callable: &Bound<'py, PyAny>,
    arg: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    context.call_method1("run", (callable, arg))
}

pub(crate) fn is_awaitable(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    IS_AWAITABLE
        .import(value.py(), "inspect", "isawaitable")?
        .call1((value,))?
        .extract()
}

fn context_with<'py>(
    py: Python<'py>,
    var: &Bound<'py, PyAny>,
    value: HashMap<String, String>,
) -> PyResult<Bound<'py, PyAny>> {
    let context = CONTEXT_TYPE.import(py, "contextvars", "Context")?.call0()?;
    context.call_method1("run", (var.getattr("set")?, value))?;
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(py: Python<'_>) -> Bound<'_, PyAny> {
        py.import("contextvars")
            .unwrap()
            .getattr("ContextVar")
            .unwrap()
            .call1(("trace",))
            .unwrap()
    }

    fn get<'py>(
        context: &Bound<'py, PyAny>,
        var: &Bound<'py, PyAny>,
    ) -> Option<HashMap<String, String>> {
        let py = context.py();
        let getter = var.getattr("get").unwrap();
        context
            .call_method1("run", (getter, py.None()))
            .unwrap()
            .extract()
            .unwrap()
    }

    #[test]
    fn each_run_context_holds_only_its_own_value() {
        Python::initialize();
        Python::attach(|py| {
            let var = var(py);
            let a = context_with(
                py,
                &var,
                HashMap::from([("traceparent".into(), "a".into())]),
            )
            .unwrap();
            let b = context_with(
                py,
                &var,
                HashMap::from([("traceparent".into(), "b".into())]),
            )
            .unwrap();

            assert_eq!(get(&a, &var).unwrap()["traceparent"], "a");
            assert_eq!(get(&b, &var).unwrap()["traceparent"], "b");
            // The dispatching thread's own context is never touched.
            let ambient: Option<HashMap<String, String>> = var
                .call_method1("get", (py.None(),))
                .unwrap()
                .extract()
                .unwrap();
            assert!(ambient.is_none());
        });
    }

    #[test]
    fn a_run_does_not_inherit_the_dispatching_threads_values() {
        Python::initialize();
        Python::attach(|py| {
            let var = var(py);
            // Something earlier on this thread left a value behind.
            var.call_method1("set", (HashMap::from([("traceparent", "stale")]),))
                .unwrap();
            let fresh = context_with(py, &var, HashMap::new()).unwrap();
            assert!(get(&fresh, &var).unwrap().is_empty());
        });
    }

    #[test]
    fn callables_run_inside_the_given_context() {
        Python::initialize();
        Python::attach(|py| {
            let var = var(py);
            let context =
                context_with(py, &var, HashMap::from([("k".into(), "v".into())])).unwrap();
            let seen = run_in(
                &context,
                &var.getattr("get").unwrap(),
                py.None().into_bound(py),
            )
            .unwrap();
            let seen: HashMap<String, String> = seen.extract().unwrap();
            assert_eq!(seen["k"], "v");
        });
    }
}
