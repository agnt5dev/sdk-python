//! Waiting on async Rust work from synchronous Python entry points.
//!
//! Every synchronous binding that needs a future's result goes through
//! [`block_on_sync`], so the work always runs on the one shared
//! pyo3-async-runtimes Tokio runtime. A temporary runtime would cancel any
//! task the future spawns (gRPC stream forwarders, response listeners) as
//! soon as it is dropped, leaving whatever the future built unusable.

use std::future::Future;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::{Handle, RuntimeFlavor};

/// Blocks the calling thread until `future` completes on the shared runtime,
/// with the GIL released so Python threads and runtime tasks that need the
/// GIL keep running.
///
/// Outside any Tokio runtime this is a plain `block_on`. On a multi-thread
/// runtime worker, a bare `block_on` panics, so the wait runs inside
/// `block_in_place`, which hands the worker's queued tasks to another thread.
/// On a current-thread runtime, blocking would starve the tasks the future
/// depends on, so it raises `RuntimeError` instead of deadlocking.
pub(crate) fn block_on_sync<F>(py: Python<'_>, future: F) -> PyResult<F::Output>
where
    F: Future + Send,
    F::Output: Send,
{
    py.detach(|| wait(future))
}

fn wait<F: Future>(future: F) -> PyResult<F::Output> {
    let runtime = pyo3_async_runtimes::tokio::get_runtime();
    match Handle::try_current() {
        Err(_) => Ok(runtime.block_on(future)),
        Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
            Ok(tokio::task::block_in_place(|| runtime.block_on(future)))
        }
        Ok(_) => Err(PyRuntimeError::new_err(
            "cannot wait synchronously inside a current-thread Tokio runtime",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::wait;
    use std::time::Duration;

    #[test]
    fn tasks_spawned_by_the_future_outlive_the_wait() {
        // The ADK runtime client spawns its response listener while
        // connecting; it must keep running after the connect call returns.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = wait(async move {
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                let _ = tx.send(42);
            })
        })
        .unwrap();
        assert!(!handle.is_finished());
        assert_eq!(wait(rx).unwrap().unwrap(), 42);
    }

    #[test]
    fn waits_from_inside_a_multi_thread_runtime() {
        let outer = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let value = outer
            .block_on(async { tokio::spawn(async { wait(async { 7 }) }).await })
            .unwrap()
            .unwrap();
        assert_eq!(value, 7);
    }

    #[test]
    fn refuses_to_wait_inside_a_current_thread_runtime() {
        let outer = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = outer.block_on(async { wait(async { 7 }) });
        assert!(result.is_err());
    }
}
