use pyo3::prelude::*;
use agnt5_sdk_core::Worker;
use agnt5_sdk_core::pb::{RuntimeMessage, ServiceMessage, InvokeFunctionResponse, ComponentInfo, ComponentType};
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

        // Clone data for the background task
        let coordinator_endpoint = self.coordinator_endpoint.clone();
        let service_name = self.service_name.clone();
        let service_version = self.service_version.clone();
        let service_type = self.service_type.clone();

        // Spawn the worker task
        rt.spawn(async move {
            let worker = Worker::new_with_components(
                coordinator_endpoint,
                service_name,
                service_version,
                service_type,
                components,
            );

            // Message handler that processes function invocations  
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