use pyo3::prelude::*;

mod worker;
use worker::PyWorker;

/// The Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorker>()?;
    Ok(())
}