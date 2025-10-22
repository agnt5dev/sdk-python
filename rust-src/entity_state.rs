/// Entity state management with gRPC-based persistence
///
/// This module provides EntityStateManager for persisting entity state to the platform.
/// It handles request/response routing through the worker's bidirectional gRPC stream.

use agnt5_sdk_core::pb::{
    runtime_service_request, runtime_service_response, service_message, EntityStateLoadRequest,
    EntityStateSaveRequest, RuntimeServiceRequest, RuntimeServiceResponse, ServiceMessage,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex, RwLock};

/// Result type for entity state operations
#[derive(Debug)]
pub struct EntityLoadResult {
    pub found: bool,
    pub state_json: Vec<u8>,
    pub version: i64,
}

#[derive(Debug)]
pub struct EntitySaveResult {
    pub new_version: i64,
}

/// Manages entity state persistence via gRPC
///
/// EntityStateManager coordinates with the platform to load and save entity state.
/// It uses the worker's bidirectional gRPC stream to send requests and receive responses.
#[pyclass]
pub struct EntityStateManager {
    /// Tenant ID for multi-tenancy
    pub(crate) tenant_id: String,

    /// Pending requests awaiting responses (request_id -> oneshot sender)
    pub(crate) pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<RuntimeServiceResponse>>>>,

    /// Channel for sending requests to the worker stream
    pub(crate) request_sender: Arc<Mutex<Option<agnt5_sdk_core::flume::Sender<ServiceMessage>>>>,
}

impl EntityStateManager {
    /// Create a new EntityStateManager
    pub fn new(tenant_id: String) -> Self {
        Self {
            tenant_id,
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            request_sender: Arc::new(Mutex::new(None)),
        }
    }

    /// Set the request sender (called by worker when stream is established)
    pub async fn set_request_sender(&self, sender: agnt5_sdk_core::flume::Sender<ServiceMessage>) {
        let mut request_sender = self.request_sender.lock().await;
        *request_sender = Some(sender);
        log::info!("EntityStateManager: Request sender configured");
    }

    /// Handle incoming RuntimeServiceResponse from the platform
    pub async fn handle_response(&self, response: RuntimeServiceResponse) {
        let request_id = response.request_id.clone();

        log::debug!(
            "EntityStateManager: Received response for request {}",
            request_id
        );

        // Find pending request
        let sender = {
            let mut pending = self.pending_requests.write().await;
            pending.remove(&request_id)
        };

        if let Some(sender) = sender {
            // Send response to waiting caller
            if sender.send(response).is_err() {
                log::warn!(
                    "EntityStateManager: Failed to deliver response for request {} (receiver dropped)",
                    request_id
                );
            }
        } else {
            log::warn!(
                "EntityStateManager: Received response for unknown request: {}",
                request_id
            );
        }
    }

    /// Send a request and wait for response
    async fn send_request(&self, operation: runtime_service_request::Operation) -> PyResult<RuntimeServiceResponse> {
        // Generate unique request ID
        let request_id = uuid::Uuid::new_v4().to_string();

        // Create oneshot channel for response
        let (response_tx, response_rx) = oneshot::channel();

        // Register pending request
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(request_id.clone(), response_tx);
        }

        // Build request message
        let request = RuntimeServiceRequest {
            request_id: request_id.clone(),
            tenant_id: self.tenant_id.clone(),
            session_id: String::new(), // Not needed for entity operations
            operation: Some(operation),
        };

        // Wrap in ServiceMessage with RuntimeService
        let service_message = ServiceMessage {
            worker_id: String::new(), // Will be set by worker
            message_type: Some(service_message::MessageType::RuntimeService(request)),
        };

        // Send request via worker stream
        {
            let sender = self.request_sender.lock().await;
            if let Some(ref sender) = *sender {
                sender
                    .send_async(service_message)
                    .await
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to send entity state request: {}",
                            e
                        ))
                    })?;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "EntityStateManager not connected to worker stream",
                ));
            }
        }

        log::debug!(
            "EntityStateManager: Sent request {} and waiting for response",
            request_id
        );

        // Wait for response with timeout
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            response_rx
        )
        .await
        .map_err(|_| {
            // Remove from pending on timeout
            let request_id = request_id.clone();
            let pending = self.pending_requests.clone();
            tokio::spawn(async move {
                let mut pending = pending.write().await;
                pending.remove(&request_id);
            });

            PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                "Entity state request timed out after 10 seconds"
            )
        })?
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Entity state request cancelled"
            )
        })?;

        // Check if response indicates error
        if !response.success {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Entity state operation failed: {}",
                response.error_message
            )));
        }

        Ok(response)
    }

    /// Load entity state from platform
    pub async fn load_state(
        &self,
        entity_type: String,
        entity_key: String,
    ) -> PyResult<EntityLoadResult> {
        log::info!(
            "EntityStateManager: Loading state for {}:{}",
            entity_type,
            entity_key
        );

        let operation = runtime_service_request::Operation::EntityStateLoad(
            EntityStateLoadRequest {
                entity_type,
                entity_key,
            },
        );

        let response = self.send_request(operation).await?;

        // Extract load result
        match response.result {
            Some(runtime_service_response::Result::EntityStateLoad(result)) => {
                log::info!(
                    "EntityStateManager: Load result - found: {}, version: {}",
                    result.found,
                    result.version
                );

                Ok(EntityLoadResult {
                    found: result.found,
                    state_json: result.state_json,
                    version: result.version,
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Unexpected response type for entity load",
            )),
        }
    }

    /// Save entity state to platform
    pub async fn save_state(
        &self,
        entity_type: String,
        entity_key: String,
        state_json: Vec<u8>,
        expected_version: i64,
    ) -> PyResult<EntitySaveResult> {
        log::info!(
            "EntityStateManager: Saving state for {}:{} (expected version: {})",
            entity_type,
            entity_key,
            expected_version
        );

        let operation = runtime_service_request::Operation::EntityStateSave(
            EntityStateSaveRequest {
                entity_type,
                entity_key,
                state_json,
                expected_version,
            },
        );

        let response = self.send_request(operation).await?;

        // Extract save result
        match response.result {
            Some(runtime_service_response::Result::EntityStateSave(result)) => {
                log::info!(
                    "EntityStateManager: Save successful - new version: {}",
                    result.new_version
                );

                Ok(EntitySaveResult {
                    new_version: result.new_version,
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Unexpected response type for entity save",
            )),
        }
    }
}

#[pymethods]
impl EntityStateManager {
    #[new]
    pub fn py_new(tenant_id: String) -> PyResult<Self> {
        Ok(Self::new(tenant_id))
    }

    /// Load entity state (Python-facing async method)
    pub fn py_load_state<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_key: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = manager.load_state(entity_type, entity_key).await?;

            // Return tuple: (found, state_json, version)
            Ok((result.found, result.state_json, result.version))
        })
    }

    /// Save entity state (Python-facing async method)
    pub fn py_save_state<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_key: String,
        state_json: Vec<u8>,
        expected_version: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = manager
                .save_state(entity_type, entity_key, state_json, expected_version)
                .await?;

            // Return new_version
            Ok(result.new_version)
        })
    }
}

// Helper to clone as Arc for async use
impl EntityStateManager {
    fn clone_arc(&self) -> Arc<Self> {
        Arc::new(Self {
            tenant_id: self.tenant_id.clone(),
            pending_requests: self.pending_requests.clone(),
            request_sender: self.request_sender.clone(),
        })
    }
}
