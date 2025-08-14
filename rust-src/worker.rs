use pyo3::prelude::*;
use agnt5_sdk_core::Worker;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Simple Python worker wrapper
#[pyclass]
pub struct PyWorker {
    coordinator_endpoint: String,
    service_name: String,
    service_version: String,
    service_type: String,
    runtime: Arc<Mutex<Option<tokio::runtime::Runtime>>>,
    shutdown_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
}

#[pymethods]
impl PyWorker {
    /// Create a new PyWorker
    #[new]
    fn new(
        coordinator_endpoint: String,
        service_name: String,
        service_version: String,
        service_type: String,
    ) -> Self {
        Self {
            coordinator_endpoint,
            service_name,
            service_version,
            service_type,
            runtime: Arc::new(Mutex::new(None)),
            shutdown_tx: Arc::new(Mutex::new(None)),
        }
    }

    /// Get a worker ID (creates a new worker each time for simplicity)
    fn worker_id(&self) -> String {
        let worker = Worker::new(
            self.coordinator_endpoint.clone(),
            self.service_name.clone(),
            self.service_version.clone(),
            self.service_type.clone(),
        );
        worker.worker_id().to_string()
    }

    /// Get the coordinator endpoint
    fn get_endpoint(&self) -> String {
        self.coordinator_endpoint.clone()
    }

    /// Start the worker in the background
    fn start(&self) -> PyResult<()> {
        let mut runtime_guard = self.runtime.lock().unwrap();
        let mut shutdown_guard = self.shutdown_tx.lock().unwrap();

        if runtime_guard.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Worker is already running"
            ));
        }

        // Create new runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create runtime: {}", e)
            ))?;

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        // Clone data for the background task
        let coordinator_endpoint = self.coordinator_endpoint.clone();
        let service_name = self.service_name.clone();
        let service_version = self.service_version.clone();
        let service_type = self.service_type.clone();

        // Spawn the worker task
        rt.spawn(async move {
            let worker = Worker::new(
                coordinator_endpoint,
                service_name,
                service_version,
                service_type,
            );

            // Simple message handler for now
            let worker_task = worker.run(|_runtime_message| async {
                // TODO: Forward to Python callback in future
                Ok::<Option<agnt5_sdk_core::pb::ServiceMessage>, agnt5_sdk_core::SdkError>(None)
            });

            tokio::select! {
                result = worker_task => {
                    if let Err(e) = result {
                        eprintln!("Worker error: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    println!("Worker shutdown requested");
                }
            }
        });

        // Store runtime and shutdown sender
        *runtime_guard = Some(rt);
        *shutdown_guard = Some(shutdown_tx);

        Ok(())
    }

    /// Stop the worker
    fn stop(&self) -> PyResult<()> {
        let mut runtime_guard = self.runtime.lock().unwrap();
        let mut shutdown_guard = self.shutdown_tx.lock().unwrap();

        if let Some(shutdown_tx) = shutdown_guard.take() {
            // Send shutdown signal
            if let Err(_) = shutdown_tx.try_send(()) {
                // Channel might be closed, that's okay
            }
        }

        if let Some(rt) = runtime_guard.take() {
            // Shutdown the runtime
            rt.shutdown_background();
        }

        Ok(())
    }

    /// Check if worker is running
    fn is_running(&self) -> bool {
        let runtime_guard = self.runtime.lock().unwrap();
        runtime_guard.is_some()
    }

}