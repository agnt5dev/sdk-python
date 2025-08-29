use pyo3::prelude::*;

mod worker;
mod types;
use worker::{PyWorker, PyWorkerConfig};
use types::{PyInvokeFunctionRequest, PyInvokeFunctionResponse, PyComponentInfo};

/// The Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize PyO3-log to bridge Rust logs to Python
    pyo3_log::init();
    
    m.add_class::<PyWorkerConfig>()?;
    m.add_class::<PyWorker>()?;
    m.add_class::<PyInvokeFunctionRequest>()?;
    m.add_class::<PyInvokeFunctionResponse>()?;
    m.add_class::<PyComponentInfo>()?;
    Ok(())
}