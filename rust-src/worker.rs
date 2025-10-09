use crate::types::{PyComponentInfo, PyExecuteComponentRequest, PyExecuteComponentResponse};
use agnt5_sdk_core::pb::{
    runtime_message, ComponentInfo, ComponentType, ExecuteComponentResponse, RuntimeMessage,
    ServiceMessage,
};
use agnt5_sdk_core::worker::{Worker, WorkerConfig};
use anyhow;
use futures::future::poll_fn;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::future::Future;
use std::sync::{Arc, Mutex};
use opentelemetry::trace::TraceContextExt;
use opentelemetry::global;
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
            let message_handler =
                move |runtime_message: RuntimeMessage,
                      tx: agnt5_sdk_core::flume::Sender<ServiceMessage>| {
                    let handler_arc_inner = handler_arc.clone();
                    async move {
                        Self::handle_runtime_message(handler_arc_inner, runtime_message, tx).await
                    }
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
        tx: agnt5_sdk_core::flume::Sender<ServiceMessage>,
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

                // Create RuntimeContext with extracted OTel context for Python access
                // This provides Python with run_id, trace_id, span_id for logging correlation
                let runtime_context = agnt5_sdk_core::RuntimeContext::with_trace_context(
                    invoke_request.invocation_id.clone(), // run_id
                    invoke_request.service_name.clone(),
                    invoke_request.component_name.clone(),
                    String::new(), // tenant_id - not available in this message, TODO: pass from worker registration
                    String::new(), // deployment_id - not available in this message, TODO: pass from worker registration
                    invoke_request.metadata.clone(),
                    parent_context.clone(),
                    std::sync::Arc::new(agnt5_sdk_core::DummyStateManager),
                );

                log::info!(
                    "Created RuntimeContext with run_id={}, trace_id={:?}, span_id={:?}",
                    runtime_context.run_id,
                    runtime_context.trace_id,
                    runtime_context.span_id
                );

                // Note: invocation.id will be handled by tracing span and Python log forwarding

                // Convert component_type from i32 to string for telemetry
                let component_type_str =
                    match ComponentType::try_from(invoke_request.component_type)
                        .unwrap_or(ComponentType::Function)
                    {
                        ComponentType::Function => "function",
                        ComponentType::Flow => "flow",
                        ComponentType::Object => "object",
                        ComponentType::Entity => "entity",
                        ComponentType::Task => "task",
                        ComponentType::Workflow => "workflow",
                        ComponentType::Agent => "agent",
                        ComponentType::Tool => "tool",
                        ComponentType::Mcp => "mcp",
                        ComponentType::Unspecified => "unspecified",
                    };

                // Create OpenTelemetry span for component execution
                // Clone parent_context since we need it both for span creation and context attachment
                let otel_span = agnt5_sdk_core::create_component_span(
                    &invoke_request.component_name,
                    component_type_str,
                    &invoke_request.service_name,
                    &runtime_message.worker_id,
                    &invoke_request.invocation_id,
                    Some(parent_context.clone()),
                    Some(&invoke_request.metadata),
                );

                // IMPORTANT: Attach the span to the parent context so Python code executes inside it
                // This ensures Python logs and operations are children of this span
                // and maintains the distributed trace link to the gateway/coordinator
                // Using parent_context.with_span() instead of Context::current_with_span()
                // is critical - it preserves the extracted trace context from the gateway
                {
                    let parent_span_ref = parent_context.span();
                    let parent_span_ctx = parent_span_ref.span_context();
                    log::info!(
                        "Trace propagation: BEFORE attach trace_id={}, span_id={}, valid={}",
                        parent_span_ctx.trace_id().to_string(),
                        parent_span_ctx.span_id().to_string(),
                        parent_span_ctx.is_valid()
                    );
                }
                let cx = parent_context.with_span(otel_span);

                {
                    let span_ref = cx.span();
                    let span_ctx = span_ref.span_context();
                    log::info!(
                        "Trace propagation: AFTER attach trace_id={}, span_id={}, valid={}",
                        span_ctx.trace_id().to_string(),
                        span_ctx.span_id().to_string(),
                        span_ctx.is_valid()
                    );
                }

                // Convert to Python types
                let mut py_request = PyExecuteComponentRequest::from(invoke_request);

                // CRITICAL: Inject trace context into request metadata for Python LM calls
                // This enables trace propagation across the Python async boundary
                {
                    use opentelemetry::propagation::Injector;

                    // Helper struct to inject into HashMap
                    struct HashMapInjector<'a>(&'a mut HashMap<String, String>);
                    impl<'a> Injector for HashMapInjector<'a> {
                        fn set(&mut self, key: &str, value: String) {
                            self.0.insert(key.to_string(), value);
                        }
                    }

                    let mut injector = HashMapInjector(&mut py_request.metadata);
                    global::get_text_map_propagator(|propagator| {
                        propagator.inject_context(&cx, &mut injector);
                    });

                    log::info!(
                        "Trace propagation: Injected into metadata - traceparent: {:?}",
                        py_request.metadata.get("traceparent")
                    );
                }

                // Attach RuntimeContext to request for Python access to correlation IDs
                py_request.runtime_context = Some(crate::PyRuntimeContext {
                    inner: std::sync::Arc::new(runtime_context),
                });

                log::info!("Attached RuntimeContext to Python request");

                // Call Python handler and get coroutine
                // Clone tx for potential use inside async context
                let tx_clone = tx.clone();
                let worker_id = runtime_message.worker_id.clone();

                // Get Python coroutine and convert to future within context guard
                let py_future = {
                    let _span_guard = cx.clone().attach();

                    // Get Python coroutine
                    let py_coroutine = Python::with_gil(|py| {
                        handler.call1(py, (py_request,))
                    })
                    .map_err(|e| {
                        let err_msg = format!("Failed to call Python handler: {}", e);
                        log::error!("{}", err_msg);
                        agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                    })?;

                    // Convert Python coroutine to Rust future
                    Python::with_gil(|py| {
                        pyo3_async_runtimes::tokio::into_future(py_coroutine.bind(py).clone())
                    })
                    .map_err(|e| {
                        let err_msg = format!("Failed to convert Python coroutine to future: {}", e);
                        log::error!("{}", err_msg);
                        agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                    })?
                    // _span_guard is dropped here before the await; we reattach below
                };

                // Await the Python coroutine while reattaching context on each poll
                let result_future = {
                    let mut py_future = Box::pin(py_future);
                    let cx_for_future = cx.clone();
                    poll_fn(move |task_cx| {
                        let _span_guard = cx_for_future.clone().attach();
                        {
                            let span_ref = cx_for_future.span();
                            let span_ctx = span_ref.span_context();
                            log::debug!(
                                "Trace propagation: INSIDE attach trace_id={}, span_id={}, valid={}",
                                span_ctx.trace_id().to_string(),
                                span_ctx.span_id().to_string(),
                                span_ctx.is_valid()
                            );
                        }
                        py_future.as_mut().poll(task_cx)
                    })
                };

                let result = result_future.await.map_err(|e| {
                    let err_msg = format!("Python handler execution failed: {}", e);
                    log::error!("{}", err_msg);
                    agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                })?;

                // Process the result
                let result = Python::with_gil(
                    |py| -> Result<Option<ServiceMessage>, agnt5_sdk_core::error::SdkError> {
                        let py_result = result.bind(py);
                        // Try to extract as list first (streaming case)
                        if let Ok(py_list) = py_result.extract::<Vec<PyExecuteComponentResponse>>() {
                                    log::info!(
                                        "🔥 STREAMING: Python returned {} chunks",
                                        py_list.len()
                                    );

                                    // Send each response via tx
                                    for (idx, py_response) in py_list.into_iter().enumerate() {
                                        log::info!(
                                            "🔥 STREAMING: Sending chunk {} to coordinator",
                                            idx
                                        );

                                        // Convert to Rust types
                                        let rust_response: ExecuteComponentResponse =
                                            py_response.into();

                                        // Create ServiceMessage
                                        let service_message = ServiceMessage {
                                            worker_id: worker_id.clone(),
                                            message_type: Some(
                                                agnt5_sdk_core::pb::service_message::MessageType::FunctionResponse(rust_response)
                                            ),
                                        };

                                        // Send via channel (blocking send)
                                        if let Err(e) = tx_clone.send(service_message) {
                                            log::error!(
                                                "Failed to send streaming chunk {}: {}",
                                                idx,
                                                e
                                            );
                                            return Err(agnt5_sdk_core::error::SdkError::Other(
                                                anyhow::anyhow!(
                                                    "Failed to send streaming chunk: {}",
                                                    e
                                                ),
                                            ));
                                        }
                                    }

                                    // Return None to indicate we handled sending ourselves
                                    log::debug!("All streaming chunks sent successfully");
                                    return Ok(None);
                        }

                        // Not a list, try single response (non-streaming case)
                        match py_result.extract::<PyExecuteComponentResponse>() {
                                    Ok(py_response) => {
                                        log::debug!("Python handler returned single response");

                                        // Record span result based on success/error
                                        let span = cx.span();
                                        if py_response.success {
                                            span.set_attribute(opentelemetry::KeyValue::new(
                                                "function.status",
                                                "success",
                                            ));
                                            span.set_attribute(opentelemetry::KeyValue::new(
                                                "function.output_size",
                                                py_response.output_data.len() as i64,
                                            ));
                                            span.set_status(opentelemetry::trace::Status::Ok);
                                        } else {
                                            let error_msg = py_response
                                                .error_message
                                                .as_deref()
                                                .unwrap_or("Unknown error");
                                            span.set_attribute(opentelemetry::KeyValue::new(
                                                "function.status",
                                                "error",
                                            ));
                                            span.set_attribute(opentelemetry::KeyValue::new(
                                                "function.error",
                                                error_msg.to_string(),
                                            ));
                                            span.set_status(opentelemetry::trace::Status::error(
                                                error_msg.to_string(),
                                            ));
                                        }

                                        // Convert back to Rust types
                                        let rust_response: ExecuteComponentResponse =
                                            py_response.into();

                                        // Create ServiceMessage
                                        let service_message = ServiceMessage {
                                        worker_id: worker_id.clone(),
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
                                let span = cx.span();
                                span.set_attribute(opentelemetry::KeyValue::new(
                                    "function.status",
                                    "error",
                                ));
                                span.set_attribute(opentelemetry::KeyValue::new(
                                    "function.error",
                                    err_msg.clone(),
                                ));
                                span.set_status(opentelemetry::trace::Status::error(
                                    err_msg.clone(),
                                ));
                                Err(agnt5_sdk_core::error::SdkError::Other(
                                    anyhow::anyhow!(err_msg),
                                ))
                            }
                        }
                    },
                )?;

                // Span will be automatically ended when _span_guard is dropped
                // This ensures proper span lifecycle management

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
            Some(runtime_message::MessageData::RuntimeServiceResponse(_)) => {
                log::debug!(
                    "Ignoring RuntimeServiceResponse message (type: {})",
                    runtime_message.message_type as i32
                );
                Ok(None)
            }
            Some(runtime_message::MessageData::HealthResponse(_)) => Ok(None),
            Some(runtime_message::MessageData::CancelExecution(_)) => {
                log::debug!(
                    "Ignoring CancelExecution message (type: {})",
                    runtime_message.message_type as i32
                );
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
