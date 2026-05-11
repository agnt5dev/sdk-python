//! Checkpoint client for Python SDK
//!
//! This module provides a Python-accessible client for step-level memoization.
//! It enables synchronous checkpoint calls with memoization responses.

use agnt5_sdk_core::client::{CheckpointResult, WorkerCoordinatorClient};
use agnt5_sdk_core::pb::CheckpointType;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Python-accessible checkpoint client for step-level memoization
///
/// This client provides synchronous checkpoint calls that can return
/// memoized results, enabling durable step execution with replay support.
#[pyclass]
pub struct PyCheckpointClient {
    client: Arc<Mutex<Option<WorkerCoordinatorClient>>>,
    endpoint: String,
}

/// Result of a checkpoint operation exposed to Python
#[pyclass]
#[derive(Clone)]
pub struct PyCheckpointResult {
    /// Whether the checkpoint was processed successfully
    #[pyo3(get)]
    pub success: bool,
    /// Error message if the checkpoint failed
    #[pyo3(get)]
    pub error_message: Option<String>,
    /// Whether the step was already memoized (for STEP_STARTED checkpoints)
    #[pyo3(get)]
    pub memoized: bool,
    /// Cached output if memoized (as bytes)
    #[pyo3(get)]
    pub cached_output: Option<Vec<u8>>,
}

impl From<CheckpointResult> for PyCheckpointResult {
    fn from(result: CheckpointResult) -> Self {
        Self {
            success: result.success,
            error_message: result.error_message,
            memoized: result.memoized,
            cached_output: result.cached_output,
        }
    }
}

impl PyCheckpointClient {
    async fn connected_client(
        client_arc: Arc<Mutex<Option<WorkerCoordinatorClient>>>,
    ) -> PyResult<WorkerCoordinatorClient> {
        let guard = client_arc.lock().await;
        guard.as_ref().cloned().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Checkpoint client not connected. Call connect() first.",
            )
        })
    }
}

#[pymethods]
impl PyCheckpointClient {
    /// Create a new checkpoint client
    ///
    /// # Arguments
    /// * `endpoint` - The Worker Coordinator endpoint URL (e.g., "http://localhost:34186")
    #[new]
    #[pyo3(signature = (endpoint = None))]
    pub fn new(endpoint: Option<String>) -> Self {
        let endpoint = endpoint.unwrap_or_else(|| {
            std::env::var("AGNT5_COORDINATOR_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:34186".to_string())
        });

        Self {
            client: Arc::new(Mutex::new(None)),
            endpoint,
        }
    }

    /// Connect to the Worker Coordinator
    ///
    /// This must be called before making checkpoint calls.
    fn connect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let client_arc = self.client.clone();
        let endpoint = self.endpoint.clone();

        future_into_py(py, async move {
            let client = WorkerCoordinatorClient::connect(endpoint)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyConnectionError::new_err(format!(
                        "Failed to connect to coordinator: {}",
                        e
                    ))
                })?;

            let mut guard = client_arc.lock().await;
            *guard = Some(client);

            Ok(())
        })
    }

    /// Send a step started checkpoint and check for memoized result
    ///
    /// Call this before executing a step. If the step is memoized,
    /// the result will contain the cached output and you can skip execution.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant that owns the run (required for engine lookups)
    /// * `run_id` - The workflow run ID
    /// * `step_key` - Unique key for this step (e.g., "step:greet:0")
    /// * `step_name` - Human-readable step name
    /// * `step_type` - Type of step (e.g., "function", "activity", "llm_call")
    /// * `input_payload` - Input data for the step (optional, for logging)
    ///
    /// # Returns
    /// PyCheckpointResult with memoized=True and cached_output if step was memoized
    fn step_started<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        run_id: String,
        step_key: String,
        step_name: String,
        step_type: String,
        input_payload: Option<Vec<u8>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client_arc = self.client.clone();

        future_into_py(py, async move {
            let mut client = PyCheckpointClient::connected_client(client_arc).await?;

            let result = client
                .checkpoint(
                    tenant_id,
                    run_id,
                    step_key,
                    step_name,
                    step_type,
                    CheckpointType::StepStarted,
                    input_payload,
                    None,
                    None,
                    None,
                )
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Checkpoint call failed: {}",
                        e
                    ))
                })?;

            Ok(PyCheckpointResult::from(result))
        })
    }

    /// Send a step completed checkpoint
    ///
    /// Call this after successfully executing a step to record the result
    /// for future memoization.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant that owns the run
    /// * `run_id` - The workflow run ID
    /// * `step_key` - Unique key for this step
    /// * `step_name` - Human-readable step name
    /// * `step_type` - Type of step
    /// * `output_payload` - Output data from the step
    /// * `latency_ms` - Step execution latency in milliseconds
    fn step_completed<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        run_id: String,
        step_key: String,
        step_name: String,
        step_type: String,
        output_payload: Vec<u8>,
        latency_ms: Option<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client_arc = self.client.clone();

        future_into_py(py, async move {
            let mut client = PyCheckpointClient::connected_client(client_arc).await?;

            let result = client
                .checkpoint(
                    tenant_id,
                    run_id,
                    step_key,
                    step_name,
                    step_type,
                    CheckpointType::StepCompleted,
                    Some(output_payload),
                    None,
                    None,
                    latency_ms,
                )
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Checkpoint call failed: {}",
                        e
                    ))
                })?;

            Ok(PyCheckpointResult::from(result))
        })
    }

    /// Send a step failed checkpoint
    ///
    /// Call this when a step fails to record the error for debugging
    /// and to prevent re-execution on replay.
    ///
    /// # Arguments
    /// * `tenant_id` - Tenant that owns the run
    /// * `run_id` - The workflow run ID
    /// * `step_key` - Unique key for this step
    /// * `step_name` - Human-readable step name
    /// * `step_type` - Type of step
    /// * `error_message` - Error message
    /// * `error_type` - Error type/class name
    fn step_failed<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        run_id: String,
        step_key: String,
        step_name: String,
        step_type: String,
        error_message: String,
        error_type: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client_arc = self.client.clone();

        future_into_py(py, async move {
            let mut client = PyCheckpointClient::connected_client(client_arc).await?;

            let result = client
                .checkpoint(
                    tenant_id,
                    run_id,
                    step_key,
                    step_name,
                    step_type,
                    CheckpointType::StepFailed,
                    None,
                    Some(error_message),
                    Some(error_type),
                    None,
                )
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Checkpoint call failed: {}",
                        e
                    ))
                })?;

            Ok(PyCheckpointResult::from(result))
        })
    }

    /// Check if a step result is memoized without sending a full checkpoint.
    ///
    /// Uses EngineService.FindByStepKey under the hood (replaces the legacy
    /// WorkerCoordinatorService.GetMemoizedStep RPC). The tenant id is part
    /// of the engine's (tenant_id, run_id) cache key and must be supplied by
    /// the caller.
    ///
    /// # Arguments
    /// * `tenant_id` - The tenant that owns the run
    /// * `run_id` - The workflow run ID
    /// * `step_key` - Unique key for this step
    ///
    /// # Returns
    /// The cached output bytes if memoized, None otherwise
    fn get_memoized_step<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        run_id: String,
        step_key: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client_arc = self.client.clone();

        future_into_py(py, async move {
            let mut client = PyCheckpointClient::connected_client(client_arc).await?;

            let result = client
                .get_memoized_step(tenant_id, run_id, step_key)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "FindByStepKey call failed: {}",
                        e
                    ))
                })?;

            Ok(result)
        })
    }
}

/// Register checkpoint client classes with the Python module
pub fn register_checkpoint_client(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCheckpointClient>()?;
    m.add_class::<PyCheckpointResult>()?;
    Ok(())
}
