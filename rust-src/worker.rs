use crate::types::{PyComponentInfo, PyExecuteComponentRequest, PyExecuteComponentResponse};
use agnt5_sdk_core::pb::{
    runtime_message, ComponentInfo, ExecuteComponentResponse, RuntimeMessage, ServiceMessage,
};
use agnt5_sdk_core::worker::{Worker, WorkerConfig};
use anyhow;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::{Arc, Mutex};
use tracing;
// Removed baggage import - using span inheritance instead
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct PyWorkerConfig {
    pub service_name: String,
    pub service_version: String,
    pub service_type: String,
}

#[pymethods]
impl PyWorkerConfig {
    #[new]
    pub fn new(service_name: String, service_version: String, service_type: String) -> Self {
        Self {
            service_name,
            service_version,
            service_type,
        }
    }
}

impl From<PyWorkerConfig> for WorkerConfig {
    fn from(config: PyWorkerConfig) -> Self {
        WorkerConfig::new(
            config.service_name,
            config.service_version,
            config.service_type,
        )
    }
}

#[pyclass]
pub struct PyWorker {
    config: PyWorkerConfig,
    worker: Arc<Mutex<Option<Worker>>>,
    message_handler: Arc<Mutex<Option<PyObject>>>,
    components: Arc<Mutex<Vec<ComponentInfo>>>,
    service_metadata: Arc<Mutex<HashMap<String, String>>>,
}

#[pymethods]
impl PyWorker {
    /// Create a new PyWorker
    #[new]
    #[pyo3(signature = (config))]
    fn new(config: PyWorkerConfig) -> PyResult<Self> {
        Ok(Self {
            config,
            worker: Arc::new(Mutex::new(None)),
            message_handler: Arc::new(Mutex::new(None)),
            components: Arc::new(Mutex::new(Vec::new())),
            service_metadata: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Set the message handler callback
    fn set_message_handler(&self, handler: PyObject) -> PyResult<()> {
        log::info!("Setting message handler for PyWorker");

        let mut handler_guard = self.message_handler.lock().map_err(|e| {
            let err_msg = format!("Failed to lock message handler: {}", e);
            log::error!("{}", err_msg);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
        })?;
        *handler_guard = Some(handler);

        log::info!("Message handler set successfully");
        Ok(())
    }

    /// Set components for the worker
    fn set_components(&self, py_components: Vec<PyComponentInfo>) -> PyResult<()> {
        log::info!("Setting {} components for PyWorker", py_components.len());

        let components: Vec<ComponentInfo> = py_components
            .into_iter()
            .map(|py_comp| py_comp.into())
            .collect();

        let mut components_guard = self.components.lock().map_err(|e| {
            let err_msg = format!("Failed to lock components: {}", e);
            log::error!("{}", err_msg);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
        })?;
        *components_guard = components.clone();

        if let Ok(mut worker_guard) = self.worker.lock() {
            if let Some(worker) = worker_guard.as_mut() {
                worker.set_components(components);
            }
        }

        log::info!("Components set successfully");
        Ok(())
    }

    /// Set service-level metadata
    fn set_service_metadata(&self, metadata: HashMap<String, String>) -> PyResult<()> {
        let mut guard = self.service_metadata.lock().map_err(|e| {
            let err_msg = format!("Failed to lock service metadata: {}", e);
            log::error!("{}", err_msg);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
        })?;
        *guard = metadata.clone();

        if let Ok(mut worker_guard) = self.worker.lock() {
            if let Some(worker) = worker_guard.as_mut() {
                worker.set_metadata(metadata);
            }
        }
        Ok(())
    }

    /// Initialize the worker with components
    fn initialize(&self) -> PyResult<()> {
        log::info!(
            "Initializing PyWorker for service: {}",
            self.config.service_name
        );

        let worker_config: WorkerConfig = self.config.clone().into();

        let components = self
            .components
            .lock()
            .map_err(|e| {
                let err_msg = format!("Failed to lock components: {}", e);
                log::error!("{}", err_msg);
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
            })?
            .clone();

        let metadata = self
            .service_metadata
            .lock()
            .map_err(|e| {
                let err_msg = format!("Failed to lock service metadata: {}", e);
                log::error!("{}", err_msg);
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
            })?
            .clone();

        let worker = Worker::new(worker_config, components, metadata);

        let mut worker_guard = self.worker.lock().map_err(|e| {
            let err_msg = format!("Failed to lock worker: {}", e);
            log::error!("{}", err_msg);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
        })?;
        *worker_guard = Some(worker);

        log::info!("PyWorker initialized successfully");
        Ok(())
    }

    /// Run the worker
    fn run<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        log::info!(
            "Starting PyWorker run for service: {}",
            self.config.service_name
        );

        // Initialize worker if not already done
        if self.worker.lock().unwrap().is_none() {
            log::debug!("Worker not initialized, initializing now");
            self.initialize()?;
        }

        let worker_arc = self.worker.clone();
        let handler_arc = self.message_handler.clone();
        let service_name = self.config.service_name.clone();

        future_into_py(py, async move {
            let worker = {
                let worker_guard = worker_arc.lock().map_err(|e| {
                    let err_msg = format!("Failed to lock worker: {}", e);
                    log::error!("{}", err_msg);
                    pyo3::exceptions::PyRuntimeError::new_err(err_msg)
                })?;
                worker_guard
                    .as_ref()
                    .ok_or_else(|| {
                        log::error!("Worker not initialized");
                        pyo3::exceptions::PyRuntimeError::new_err("Worker not initialized")
                    })?
                    .clone()
            };

            // Create message handler that calls Python callback
            let message_handler = move |runtime_message: RuntimeMessage| {
                let handler_arc_inner = handler_arc.clone();
                async move { Self::handle_runtime_message(handler_arc_inner, runtime_message).await }
            };

            log::info!("Starting worker event loop for service: {}", service_name);

            match worker.run(message_handler).await {
                Ok(()) => {
                    log::info!(
                        "Worker event loop ended normally for service: {}",
                        service_name
                    );
                    Ok(())
                }
                Err(e) => {
                    let err_msg = format!("Worker run failed: {}", e);
                    log::error!("{}", err_msg);
                    Err(pyo3::exceptions::PyRuntimeError::new_err(err_msg))
                }
            }
        })
    }
}

impl PyWorker {
    /// Handle runtime message by calling Python callback
    async fn handle_runtime_message(
        handler_arc: Arc<Mutex<Option<PyObject>>>,
        runtime_message: RuntimeMessage,
    ) -> Result<Option<ServiceMessage>, agnt5_sdk_core::error::SdkError> {
        
        // Get the Python handler by cloning it properly
        let handler = {
            let handler_guard = handler_arc.lock().map_err(|e| {
                let err_msg = format!("Failed to lock message handler: {}", e);
                log::error!("{}", err_msg);
                agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
            })?;

            if let Some(handler) = handler_guard.as_ref() {
                Python::with_gil(|py| handler.clone_ref(py))
            } else {
                log::error!("No message handler set");
                return Err(agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(
                    "No message handler set"
                )));
            }
        };

        // Handle the message based on type
        match runtime_message.message_data {
            Some(runtime_message::MessageData::ExecuteComponent(invoke_request)) => {
                // Create tracing span with invocation_id that will be inherited by all logs
                let invocation_span = tracing::info_span!(
                    "execute_component",
                    invocation.id = %invoke_request.invocation_id,
                    service.name = %invoke_request.service_name,
                    component.name = %invoke_request.component_name,
                    worker.id = %runtime_message.worker_id,
                );
                let _guard = invocation_span.enter();

                log::info!(
                    "Received function invocation request - Data size: {} bytes",
                    invoke_request.input_data.len()
                );
                log::debug!("Request metadata: {:?}", invoke_request.metadata);

                // Extract trace context from request metadata
                let parent_context =
                    agnt5_sdk_core::extract_context_from_runtime_message(&invoke_request.metadata);

                // Note: invocation.id will be handled by tracing span and Python log forwarding

                // Create span for function execution
                let mut span = agnt5_sdk_core::create_function_span(
                    &invoke_request.component_name,
                    &invoke_request.service_name,
                    &runtime_message.worker_id,
                    &invoke_request.invocation_id,
                    Some(parent_context),
                    Some(&invoke_request.metadata),
                );

                // Convert to Python types
                let py_request = PyExecuteComponentRequest::from(invoke_request);

                // Call Python handler with GIL
                let result = Python::with_gil(
                    |py| -> Result<Option<ServiceMessage>, agnt5_sdk_core::error::SdkError> {
                        match handler.call1(py, (py_request,)) {
                            Ok(py_result) => {
                                // Extract PyExecuteComponentResponse from Python result
                                match py_result.extract::<PyExecuteComponentResponse>(py) {
                                    Ok(py_response) => {
                                        log::debug!(
                                            "Python handler returned response successfully"
                                        );

                                        // Record span result based on success/error
                                        if py_response.success {
                                            agnt5_sdk_core::record_span_success(
                                                &mut span,
                                                py_response.output_data.len(),
                                            );
                                        } else {
                                            let error_msg = py_response
                                                .error_message
                                                .as_deref()
                                                .unwrap_or("Unknown error");
                                            agnt5_sdk_core::record_span_error(&mut span, error_msg);
                                        }

                                        // Convert back to Rust types
                                        let rust_response: ExecuteComponentResponse =
                                            py_response.into();

                                        // Create ServiceMessage
                                        let service_message = ServiceMessage {
                                        worker_id: runtime_message.worker_id.clone(),
                                        message_type: Some(
                                            agnt5_sdk_core::pb::service_message::MessageType::FunctionResponse(rust_response)
                                        ),
                                    };

                                        Ok(Some(service_message))
                                    }
                                    Err(e) => {
                                        let err_msg =
                                            format!("Failed to extract Python response: {}", e);
                                        log::error!("{}", err_msg);
                                        agnt5_sdk_core::record_span_error(&mut span, &err_msg);
                                        Err(agnt5_sdk_core::error::SdkError::Other(
                                            anyhow::anyhow!(err_msg),
                                        ))
                                    }
                                }
                            }
                            Err(e) => {
                                let err_msg = format!("Python handler failed: {}", e);
                                log::error!("{}", err_msg);
                                agnt5_sdk_core::record_span_error(&mut span, &err_msg);
                                Err(agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(
                                    err_msg
                                )))
                            }
                        }
                    },
                )?;

                // End the span
                agnt5_sdk_core::end_span(span);

                Ok(result)
            }
            Some(runtime_message::MessageData::RegisterServiceResponse(_)) => {
                log::debug!(
                    "Ignoring RegisterServiceResponse message (type: {})",
                    runtime_message.message_type as i32
                );
                Ok(None)
            }
            Some(runtime_message::MessageData::GetComponents(_)) => {
                log::debug!(
                    "Ignoring GetComponents message (type: {})",
                    runtime_message.message_type as i32
                );
                Ok(None)
            }
            Some(runtime_message::MessageData::HealthResponse(_)) => {
                Ok(None)
            }
            None => {
                log::debug!(
                    "Ignoring message with no data (type: {})",
                    runtime_message.message_type as i32
                );
                Ok(None)
            }
        }
    }
}
