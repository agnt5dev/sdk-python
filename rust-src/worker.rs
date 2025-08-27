use pyo3::prelude::*;
use agnt5_sdk_core::{Worker, init_logging, get_error_buffer, clear_error_buffer};
use agnt5_sdk_core::worker::ConnectionState;
use agnt5_sdk_core::pb::{ServiceMessage, InvokeFunctionResponse, ComponentInfo, ComponentType};
use agnt5_sdk_core::pb::runtime_message::MessageData;
use agnt5_sdk_core::pb::service_message::MessageType;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Simple Python worker wrapper
#[pyclass]
pub struct PyWorker {
    coordinator_endpoint: String,
    service_name: String,
    service_version: String,
    service_type: String,
    tenant_id: Option<String>,
    deployment_id: Option<String>,
    runtime: Arc<Mutex<Option<tokio::runtime::Runtime>>>,
    shutdown_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
    worker_instance: Arc<Mutex<Option<Worker>>>,
}

#[pymethods]
impl PyWorker {
    /// Create a new PyWorker
    #[new]
    #[pyo3(signature = (coordinator_endpoint, service_name, service_version, service_type, tenant_id=None, deployment_id=None))]
    fn new(
        coordinator_endpoint: String,
        service_name: String,
        service_version: String,
        service_type: String,
        tenant_id: Option<String>,
        deployment_id: Option<String>,
    ) -> Self {
        Self {
            coordinator_endpoint,
            service_name,
            service_version,
            service_type,
            tenant_id,
            deployment_id,
            runtime: Arc::new(Mutex::new(None)),
            shutdown_tx: Arc::new(Mutex::new(None)),
            worker_instance: Arc::new(Mutex::new(None)),
        }
    }

    /// Get a worker ID (creates a new worker each time for simplicity)
    fn worker_id(&self) -> String {
        let worker = self.create_worker();
        worker.worker_id().to_string()
    }

    /// Get the coordinator endpoint
    fn get_endpoint(&self) -> String {
        self.coordinator_endpoint.clone()
    }

    /// Get the tenant ID
    fn tenant_id(&self) -> Option<String> {
        self.tenant_id.clone()
    }

    /// Get the deployment ID
    fn deployment_id(&self) -> Option<String> {
        self.deployment_id.clone()
    }

    /// Initialize logging for error capture
    #[staticmethod]
    fn init_logging() -> PyResult<()> {
        init_logging().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to initialize logging: {}", e)
        ))?;
        Ok(())
    }

    /// Get recent error messages from Rust core
    #[staticmethod]
    fn get_errors() -> Vec<String> {
        get_error_buffer()
    }

    /// Clear the error buffer
    #[staticmethod]
    fn clear_errors() {
        clear_error_buffer();
    }

    /// Get connection state as string
    fn get_connection_state(&self) -> String {
        if let Some(worker) = self.worker_instance.lock().unwrap().as_ref() {
            match worker.connection_state() {
                ConnectionState::Disconnected => "disconnected".to_string(),
                ConnectionState::Connecting => "connecting".to_string(),
                ConnectionState::Connected => "connected".to_string(),
                ConnectionState::Error(msg) => format!("error: {}", msg),
            }
        } else {
            "not_started".to_string()
        }
    }

    /// Start the worker and wait for connection acknowledgment
    fn start(&self) -> PyResult<()> {
        let mut runtime_guard = self.runtime.lock().unwrap();
        let mut shutdown_guard = self.shutdown_tx.lock().unwrap();

        if runtime_guard.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Worker is already running"
            ));
        }

        // Initialize logging if not already done
        let _ = init_logging();

        // Collect registered functions from Python decorators
        let components = Python::with_gil(|py| -> PyResult<Vec<ComponentInfo>> {
            let decorators = py.import("agnt5.decorators")?;
            let get_registered_functions = decorators.getattr("get_registered_functions")?;
            let registered_functions = get_registered_functions.call0()?;
            
            let mut components = Vec::new();
            
            // Convert Python dict to iterator
            if let Ok(items) = registered_functions.call_method0("items") {
                if let Ok(iter) = items.try_iter() {
                    for item in iter {
                        if let Ok(item) = item {
                            if let Ok((name, _func)) = item.extract::<(String, PyObject)>() {
                                // Create ComponentInfo for each registered function
                                components.push(ComponentInfo {
                                    name: name.clone(),
                                    component_type: ComponentType::Function as i32,
                                    input_schema: None,
                                    output_schema: None,
                                    config: std::collections::HashMap::new(),
                                    metadata: std::collections::HashMap::new(),
                                });
                                println!("  📝 Registered handler: {}", name);
                            }
                        }
                    }
                }
            }
            
            println!("  ✅ Collected {} handlers from Python decorators", components.len());
            Ok(components)
        })?;

        // Create new runtime
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create runtime: {}", e)
            ))?;

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        // Create connection result channel to wait for connection acknowledgment
        let (connection_tx, connection_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        // Clone data for the background task
        let coordinator_endpoint = self.coordinator_endpoint.clone();
        let service_name = self.service_name.clone();
        let service_version = self.service_version.clone();
        let service_type = self.service_type.clone();
        let tenant_id = self.tenant_id.clone();
        let deployment_id = self.deployment_id.clone();
        let worker_instance_ref = self.worker_instance.clone();

        // Spawn the worker task
        rt.spawn(async move {
            let mut connection_tx = Some(connection_tx);
            
            let worker = match (&tenant_id, &deployment_id) {
                (Some(tenant), Some(deployment)) => {
                    // Create worker with explicit tenant/deployment and add components
                    let mut worker = Worker::new_with_tenant(
                        coordinator_endpoint,
                        service_name,
                        service_version,
                        service_type,
                        tenant.clone(),
                        deployment.clone(),
                    );
                    worker.set_components(components);
                    worker
                },
                _ => {
                    // Use environment-based constructor with components
                    Worker::new_with_components(
                        coordinator_endpoint,
                        service_name,
                        service_version,
                        service_type,
                        components,
                    )
                }
            };

            // Store the worker instance for connection state tracking
            *worker_instance_ref.lock().unwrap() = Some(worker.clone());

            // First attempt to connect and register to verify connection
            match worker.connect_and_run_once().await {
                Ok(()) => {
                    // Connection successful, report back to main thread
                    if let Some(tx) = connection_tx.take() {
                        let _ = tx.send(Ok(()));
                    }
                }
                Err(e) => {
                    // Connection failed, report error and return
                    if let Some(tx) = connection_tx.take() {
                        let _ = tx.send(Err(format!("Connection failed: {}", e)));
                    }
                    return;
                }
            }

            // Now run the main worker loop with retries
            let worker_task = worker.run(|runtime_message| async {
                match runtime_message.message_data {
                    Some(MessageData::InvokeFunction(invocation)) => {
                        println!("📨 Received function invocation: {}", invocation.handler_name);
                        
                        // Call Python function via the decorators module
                        let result = Python::with_gil(|py| -> PyResult<Vec<u8>> {
                            // Import the decorators module
                            let decorators = py.import("agnt5.decorators")?;
                            
                            // The input_data contains the raw JSON bytes (already decoded from base64 by the coordinator)
                            let json_bytes = &invocation.input_data;
                            
                            // Call invoke_function from decorators
                            let invoke_function = decorators.getattr("invoke_function")?;
                            let result = invoke_function.call1((
                                &invocation.handler_name,
                                json_bytes.as_slice(),
                            ))?;
                            
                            // Convert Python bytes result to Vec<u8>
                            let bytes: Vec<u8> = result.extract()?;
                            Ok(bytes)
                        });
                        
                        // Create response message
                        let response = match result {
                            Ok(output_data) => {
                                println!("✅ Function '{}' executed successfully", invocation.handler_name);
                                InvokeFunctionResponse {
                                    invocation_id: invocation.invocation_id,
                                    success: true,
                                    output_data,
                                    error_message: String::new(),
                                    metadata: std::collections::HashMap::new(),
                                }
                            }
                            Err(err) => {
                                eprintln!("❌ Function '{}' failed: {}", invocation.handler_name, err);
                                InvokeFunctionResponse {
                                    invocation_id: invocation.invocation_id,
                                    success: false,
                                    output_data: Vec::new(),
                                    error_message: err.to_string(),
                                    metadata: std::collections::HashMap::new(),
                                }
                            }
                        };
                        
                        // Return ServiceMessage with function response
                        Ok(Some(ServiceMessage {
                            worker_id: worker.worker_id().to_string(),
                            message_type: Some(MessageType::FunctionResponse(response)),
                        }))
                    }
                    _ => {
                        // Other message types - no response needed
                        Ok(None)
                    }
                }
            });

            tokio::select! {
                result = worker_task => {
                    match result {
                        Ok(()) => {
                            // Worker completed successfully - this is normal for reconnection scenarios
                            println!("Worker completed successfully");
                        }
                        Err(e) => {
                            let error_msg = format!("Worker error: {}", e);
                            eprintln!("{}", error_msg);
                            
                            // Log via eprintln for now since we can't directly access tracing from here
                            // The error is already printed above and will be visible
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    println!("Worker shutdown requested");
                    if let Some(tx) = connection_tx.take() {
                        let _ = tx.send(Err("Worker shutdown before connection".to_string()));
                    }
                }
            }
        });

        // Store runtime and shutdown sender
        *runtime_guard = Some(rt);
        *shutdown_guard = Some(shutdown_tx);

        // Get a handle to the runtime before dropping the guard
        let rt_handle = {
            let guard = self.runtime.lock().unwrap();
            guard.as_ref().unwrap().handle().clone()
        };

        // Wait for connection result with timeout using the runtime handle
        let connection_result = match rt_handle.block_on(
            tokio::time::timeout(
                std::time::Duration::from_secs(30),
                connection_rx
            )
        ) {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Connection channel closed unexpectedly"
            )),
            Err(_) => return Err(PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(
                "Connection timeout: Worker could not connect to coordinator within 30 seconds"
            )),
        };

        match connection_result {
            Ok(()) => Ok(()),
            Err(error_msg) => Err(PyErr::new::<pyo3::exceptions::PyConnectionError, _>(
                format!("Failed to connect to coordinator: {}", error_msg)
            )),
        }
    }

    /// Stop the worker
    fn stop(&self) -> PyResult<()> {
        let mut runtime_guard = self.runtime.lock().unwrap();
        let mut shutdown_guard = self.shutdown_tx.lock().unwrap();
        let mut worker_guard = self.worker_instance.lock().unwrap();

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

        // Clean up worker instance
        *worker_guard = None;

        Ok(())
    }

    /// Check if worker is running
    fn is_running(&self) -> bool {
        let runtime_guard = self.runtime.lock().unwrap();
        runtime_guard.is_some()
    }

    /// Check if worker is connected (alias for is_running for clarity)
    fn is_connected(&self) -> bool {
        self.is_running()
    }
}

impl PyWorker {
    /// Helper method to create a Worker instance
    fn create_worker(&self) -> Worker {
        match (&self.tenant_id, &self.deployment_id) {
            (Some(tenant), Some(deployment)) => {
                Worker::new_with_tenant(
                    self.coordinator_endpoint.clone(),
                    self.service_name.clone(),
                    self.service_version.clone(),
                    self.service_type.clone(),
                    tenant.clone(),
                    deployment.clone(),
                )
            },
            _ => {
                Worker::new(
                    self.coordinator_endpoint.clone(),
                    self.service_name.clone(),
                    self.service_version.clone(),
                    self.service_type.clone(),
                )
            }
        }
    }
}