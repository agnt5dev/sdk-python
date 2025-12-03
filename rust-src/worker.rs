use crate::entity_state::EntityStateManager;
use crate::types::{PyComponentInfo, PyExecuteComponentRequest, PyExecuteComponentResponse};
use agnt5_sdk_core::pb::{
    runtime_message, ComponentInfo, ComponentType, ExecuteComponentResponse, RuntimeMessage,
    ServiceMessage,
};
use agnt5_sdk_core::worker::{Worker, WorkerConfig};
use anyhow;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_async_runtimes::{TaskLocals, into_future_with_locals};
use std::sync::{Arc, Mutex};
use tokio::sync::Mutex as AsyncMutex;
use opentelemetry::trace::{TraceContextExt, Span};
use opentelemetry::global;
use tracing;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
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
    message_handler: Arc<Mutex<Option<Py<PyAny>>>>,
    components: Arc<Mutex<Vec<ComponentInfo>>>,
    service_metadata: Arc<Mutex<HashMap<String, String>>>,
    // TaskLocals stores reference to Python event loop for concurrent async execution
    // This enables using into_future_with_locals instead of spawn_blocking + asyncio.run()
    event_loop_locals: Arc<Mutex<Option<TaskLocals>>>,
    // Entity state manager for database persistence
    entity_state_manager: Arc<AsyncMutex<Option<Arc<EntityStateManager>>>>,
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
            event_loop_locals: Arc::new(Mutex::new(None)),
            entity_state_manager: Arc::new(AsyncMutex::new(None)),
        })
    }

    /// Set entity state manager for database persistence
    fn set_entity_state_manager(&self, manager: Py<EntityStateManager>) -> PyResult<()> {
        let rust_manager = Python::attach(|py| -> PyResult<Arc<EntityStateManager>> {
            let manager_ref = manager.borrow(py);
            Ok(Arc::new(EntityStateManager {
                tenant_id: manager_ref.tenant_id.clone(),
                cache: manager_ref.cache.clone(),
                pending_requests: manager_ref.pending_requests.clone(),
                request_sender: manager_ref.request_sender.clone(),
                max_retries: manager_ref.max_retries,
                base_delay_ms: manager_ref.base_delay_ms,
            }))
        })?;

        // Store manager - use try_lock since we're not in async context
        // This is safe because we're called during initialization before any concurrent access
        match self.entity_state_manager.try_lock() {
            Ok(mut manager_guard) => {
                *manager_guard = Some(rust_manager);
                log::info!("Entity state manager configured for worker");
                Ok(())
            }
            Err(_) => {
                let err_msg = "Failed to set entity state manager: lock already held";
                log::error!("{}", err_msg);
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg))
            }
        }
    }

    /// Set the message handler callback
    fn set_message_handler(&self, handler: Py<PyAny>) -> PyResult<()> {
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

    /// Queue a workflow checkpoint for progressive durability
    ///
    /// This method is called from Python during workflow execution to send
    /// checkpoints (state changes, step completions) to the platform in real-time.
    ///
    /// # Arguments
    /// * `invocation_id` - Workflow run ID
    /// * `checkpoint_type` - Event type ("workflow.state.changed", etc.)
    /// * `checkpoint_data` - JSON payload as string
    /// * `sequence_number` - Monotonic sequence for ordering
    /// * `metadata` - Dictionary of metadata (tenant_id, deployment_id, etc.)
    fn queue_workflow_checkpoint(
        &self,
        invocation_id: String,
        checkpoint_type: String,
        checkpoint_data: String,
        sequence_number: i64,
        metadata: HashMap<String, String>,
    ) -> PyResult<()> {
        // Convert checkpoint_data string to bytes
        let data_bytes = checkpoint_data.into_bytes();

        // Access worker and queue checkpoint
        let worker_guard = self.worker.lock().map_err(|e| {
            let err_msg = format!("Failed to lock worker: {}", e);
            log::error!("{}", err_msg);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
        })?;

        if let Some(ref worker) = *worker_guard {
            worker.queue_checkpoint(
                invocation_id,
                checkpoint_type,
                data_bytes,
                sequence_number,
                metadata,
            ).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to queue checkpoint: {}", e)
                )
            })?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Worker not initialized"
            ));
        }

        Ok(())
    }

    /// Flush all buffered workflow checkpoints by waiting for background task
    ///
    /// This method waits 200ms for the background flush task (100ms interval)
    /// to send all queued checkpoints. This ensures checkpoints arrive at the
    /// platform BEFORE the workflow completion response is sent.
    ///
    /// Returns the number of checkpoints that were initially queued.
    fn flush_workflow_checkpoints(&self, py: Python) -> PyResult<usize> {
        let worker_guard = self.worker.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to lock worker: {}", e)
            )
        })?;

        let worker = worker_guard.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Worker not initialized")
        })?;

        // Get initial queue size
        let (initial_queued, _, _, _) = worker.checkpoint_metrics();

        if initial_queued == 0 {
            return Ok(0); // No checkpoints to flush
        }

        log::debug!("Waiting 200ms to flush {} queued checkpoints before response", initial_queued);

        // Release the lock while waiting
        drop(worker_guard);

        // Simple approach: Wait 200ms (2x flush interval) to ensure checkpoints are sent
        // The background flush task runs every 100ms, so 200ms guarantees at least 2 flushes
        py.detach(|| {
            std::thread::sleep(std::time::Duration::from_millis(200));
        });

        // Log final metrics
        let worker_guard = self.worker.lock().ok();
        if let Some(worker_guard) = worker_guard {
            if let Some(ref worker) = *worker_guard {
                let (queued, sent, _, _) = worker.checkpoint_metrics();
                log::debug!(
                    "Checkpoint flush complete: {} still queued, {} sent total",
                    queued,
                    sent
                );
            }
        }

        Ok(initial_queued as usize)
    }

    /// Set the Python event loop for concurrent async execution
    ///
    /// This method stores a reference to the Python event loop via TaskLocals.
    /// The stored event loop will be used to execute Python async functions
    /// concurrently without the overhead of spawn_blocking + asyncio.run().
    ///
    /// This enables true concurrent execution of Python async functions on the
    /// same event loop, allowing 100+ concurrent requests without thread pool
    /// bottlenecks.
    ///
    /// # Arguments
    /// * `event_loop` - The Python asyncio event loop object
    fn set_event_loop(&self, py: Python, event_loop: Py<PyAny>) -> PyResult<()> {
        log::info!("Setting Python event loop for concurrent async execution");

        // Create TaskLocals from the event loop
        // TaskLocals stores a reference to the Python event loop that can be
        // used across Rust async tasks via into_future_with_locals()
        let task_locals = TaskLocals::new(event_loop.bind(py).clone());

        let mut guard = self.event_loop_locals.lock().map_err(|e| {
            let err_msg = format!("Failed to lock event loop locals: {}", e);
            log::error!("{}", err_msg);
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err_msg)
        })?;
        *guard = Some(task_locals);

        log::info!("Event loop set successfully - concurrent Python async execution enabled");
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
        let needs_init = self.worker.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Worker mutex poisoned: {}", e))
        })?.is_none();

        if needs_init {
            log::debug!("Worker not initialized, initializing now");
            self.initialize()?;
        }

        let worker_arc = self.worker.clone();
        let handler_arc = self.message_handler.clone();
        let event_loop_locals_arc = self.event_loop_locals.clone();
        let entity_state_manager_arc = self.entity_state_manager.clone();
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
            // Now uses `Fn + Clone` to support concurrent execution
            let message_handler = {
                let handler_arc_clone = handler_arc.clone();
                let event_loop_locals_clone = event_loop_locals_arc.clone();
                let entity_state_manager_clone = entity_state_manager_arc.clone();
                move |runtime_message: RuntimeMessage,
                      tx: agnt5_sdk_core::flume::Sender<ServiceMessage>| {
                    let handler_arc_inner = handler_arc_clone.clone();
                    let event_loop_locals_inner = event_loop_locals_clone.clone();
                    let entity_state_manager_inner = entity_state_manager_clone.clone();
                    async move {
                        Self::handle_runtime_message(
                            handler_arc_inner,
                            event_loop_locals_inner,
                            entity_state_manager_inner,
                            runtime_message,
                            tx
                        ).await
                    }
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
        handler_arc: Arc<Mutex<Option<Py<PyAny>>>>,
        event_loop_locals_arc: Arc<Mutex<Option<TaskLocals>>>,
        entity_state_manager_arc: Arc<AsyncMutex<Option<Arc<EntityStateManager>>>>,
        runtime_message: RuntimeMessage,
        tx: agnt5_sdk_core::flume::Sender<ServiceMessage>,
    ) -> Result<Option<ServiceMessage>, agnt5_sdk_core::error::SdkError> {
        // On first message, set the sender on EntityStateManager if it exists
        {
            let manager_guard = entity_state_manager_arc.lock().await;
            if let Some(ref manager) = *manager_guard {
                // Check if sender is not already set
                let sender_guard = manager.request_sender.lock().await;
                if sender_guard.is_none() {
                    drop(sender_guard); // Release lock before calling set_request_sender
                    manager.set_request_sender(tx.clone()).await;
                    log::info!("Entity state manager connected to worker stream");
                }
            }
        }
        // Get the Python handler by cloning it properly
        let handler = {
            let handler_guard = handler_arc.lock().map_err(|e| {
                let err_msg = format!("Failed to lock message handler: {}", e);
                log::error!("{}", err_msg);
                agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
            })?;

            if let Some(handler) = handler_guard.as_ref() {
                Python::attach(|py| handler.clone_ref(py))
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

                // Extract trace context from request metadata
                let parent_context =
                    agnt5_sdk_core::extract_context_from_runtime_message(&invoke_request.metadata);

                // Extract tenant_id and deployment_id from request metadata
                // These are set by the Gateway and passed through Worker Coordinator
                let tenant_id = invoke_request
                    .metadata
                    .get("tenant_id")
                    .cloned()
                    .unwrap_or_default();
                let deployment_id = invoke_request
                    .metadata
                    .get("deployment_id")
                    .cloned()
                    .unwrap_or_default();

                // Clone tenant_id before it's moved into RuntimeContext (needed for journal export later)
                let tenant_id_for_journal_early = tenant_id.clone();

                // Create RuntimeContext with extracted OTel context for Python access
                // This provides Python with run_id, trace_id, span_id, tenant_id, deployment_id for logging correlation
                let runtime_context = agnt5_sdk_core::RuntimeContext::with_trace_context(
                    invoke_request.invocation_id.clone(), // run_id
                    invoke_request.service_name.clone(),
                    invoke_request.component_name.clone(),
                    tenant_id,
                    deployment_id,
                    invoke_request.metadata.clone(),
                    parent_context.clone(),
                    std::sync::Arc::new(agnt5_sdk_core::DummyStateManager),
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
                let mut otel_span = agnt5_sdk_core::create_component_span(
                    &invoke_request.component_name,
                    component_type_str,
                    &invoke_request.service_name,
                    &runtime_message.worker_id,
                    &invoke_request.invocation_id,
                    Some(parent_context.clone()),
                    Some(&invoke_request.metadata),
                );

                // Record execution request metric
                agnt5_sdk_core::record_execution_request(
                    &invoke_request.component_name,
                    component_type_str,
                );

                // Add input.data attribute before moving span into context
                otel_span.set_attribute(opentelemetry::KeyValue::new(
                    "input.data",
                    String::from_utf8_lossy(&invoke_request.input_data).to_string(),
                ));

                // Add is_streaming attribute for journal exporter filtering
                // Only spans with this attribute set to true will be exported to the journal
                // 🔍 STREAM-DEBUG: Log is_streaming value from request
                tracing::info!(
                    "🔍 STREAM-DEBUG: SDK received request with is_streaming={}",
                    invoke_request.is_streaming
                );
                if invoke_request.is_streaming {
                    tracing::info!(
                        "🔍 STREAM-DEBUG: Setting agnt5.is_streaming=true attribute on span"
                    );
                    otel_span.set_attribute(opentelemetry::KeyValue::new(
                        "agnt5.is_streaming",
                        true,
                    ));
                }

                // IMPORTANT: Attach the span to the parent context so Python code executes inside it
                // This ensures Python logs and operations are children of this span
                // and maintains the distributed trace link to the gateway/coordinator
                // Using parent_context.with_span() instead of Context::current_with_span()
                // is critical - it preserves the extracted trace context from the gateway
                let cx = parent_context.with_span(otel_span);

                // Clone fields needed for span before moving invoke_request
                let component_name = invoke_request.component_name.clone();
                let invocation_id = invoke_request.invocation_id.clone();

                // Capture fields needed for journal export (only used if is_streaming)
                let is_streaming_request = invoke_request.is_streaming;
                let tenant_id_for_journal = if tenant_id_for_journal_early.is_empty() { None } else { Some(tenant_id_for_journal_early) };
                let execution_start_nano = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as i64)
                    .unwrap_or(0);
                // Extract parent span ID from context if available
                let parent_span_id_for_journal: Option<String> = {
                    let parent_span = cx.span();
                    let parent_ctx = parent_span.span_context();
                    let parent_str = parent_ctx.span_id().to_string();
                    if parent_str == "0000000000000000" || parent_str.is_empty() {
                        None
                    } else {
                        Some(parent_str)
                    }
                };

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
                }

                // Attach RuntimeContext to request for Python access to correlation IDs
                py_request.runtime_context = Some(crate::PyRuntimeContext {
                    inner: std::sync::Arc::new(runtime_context),
                });

                // Call Python handler and get coroutine
                // Clone tx for potential use inside async context
                let tx_clone = tx.clone();
                let worker_id = runtime_message.worker_id.clone();

                // Create a tracing span for this execution
                // CRITICAL: Use OpenTelemetrySpanExt to explicitly set the OpenTelemetry context
                // This ensures logs emitted within this span have the correct trace_id/span_id
                let execution_span = tracing::info_span!(
                    "python_component_execution",
                    component.name = %component_name,
                    component.type = component_type_str,
                    run.id = %invocation_id
                );

                // CRITICAL: Set the OpenTelemetry context on the tracing span
                // This makes the OTel context available when the span is entered
                execution_span.set_parent(cx.clone());

                // Debug: Check if the OpenTelemetry span has trace_id
                let otel_span = cx.span();
                tracing::info!(
                    "Created execution span with OTel trace_id: {:?}",
                    otel_span.span_context().trace_id()
                );

                // Track execution start time for duration metrics
                let execution_start = std::time::Instant::now();

                // Execute Python handler using shared event loop for concurrent execution
                // This approach uses into_future_with_locals() to execute Python async functions
                // on the same event loop that was created at worker startup, enabling true
                // concurrent execution without spawn_blocking overhead.
                //
                // Benefits over spawn_blocking + asyncio.run():
                // - No thread pool bottleneck (was limited to ~8-16 concurrent executions)
                // - Lower overhead (no need to create new event loop per request)
                // - True concurrency (100+ async functions on same event loop)
                //
                // Use async move block and instrument it with the span to propagate trace context
                let result = async move {
                    // Get TaskLocals with the shared event loop
                    let task_locals: TaskLocals = Python::attach(|_py| -> Result<_, agnt5_sdk_core::error::SdkError> {
                        let guard = event_loop_locals_arc.lock().map_err(|e| {
                            let err_msg = format!("Failed to lock event loop locals: {}", e);
                            log::error!("{}", err_msg);
                            agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                        })?;

                        Ok(guard.as_ref().ok_or_else(|| {
                            let err_msg = "Event loop not initialized. Call set_event_loop() first.";
                            log::error!("{}", err_msg);
                            agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                        })?.clone())
                    })?;

                    // Call Python handler and execute on shared event loop
                    let rust_future = Python::attach(|py| -> Result<_, agnt5_sdk_core::error::SdkError> {
                        // Call the Python handler
                        let py_result = handler.call1(py, (py_request,))
                            .map_err(|e| {
                                let err_msg = format!("Failed to call Python handler: {}", e);
                                log::error!("{}", err_msg);
                                agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                            })?;

                        // Check if result is an awaitable (coroutine, Task, Future, or any __await__)
                        let inspect = py.import("inspect")
                            .map_err(|e| {
                                let err_msg = format!("Failed to import inspect: {}", e);
                                log::error!("{}", err_msg);
                                agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                            })?;

                        let is_awaitable = inspect.call_method1("isawaitable", (&py_result,))
                            .and_then(|result| result.extract::<bool>())
                            .unwrap_or(false);

                        let awaitable: Bound<PyAny> = if is_awaitable {
                            // It's an awaitable - use it directly
                            py_result.bind(py).clone()
                        } else {
                            // It's a regular return value (not awaitable)
                            // This shouldn't happen for async functions, but handle it gracefully
                            log::warn!("🔥 RUST: Handler returned NON-awaitable value (unexpected for async functions)");

                            // Wrap in a coroutine using asyncio.ensure_future with a coroutine wrapper
                            let asyncio = py.import("asyncio")
                                .map_err(|e| {
                                    let err_msg = format!("Failed to import asyncio: {}", e);
                                    log::error!("{}", err_msg);
                                    agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                                })?;

                            // Create a coroutine that returns the value
                            let coroutine = asyncio.call_method1("coroutine", (py_result,))
                                .map_err(|e| {
                                    let err_msg = format!("Failed to wrap in coroutine: {}", e);
                                    log::error!("{}", err_msg);
                                    agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                                })?;

                            coroutine
                        };

                        // Convert Python awaitable to Rust Future using shared event loop
                        let future = into_future_with_locals(&task_locals, awaitable)
                            .map_err(|e| {
                                let err_msg = format!("Failed to convert Python awaitable to Rust Future: {}", e);
                                log::error!("{}", err_msg);
                                agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                            })?;
                        Ok(future)
                    })?;

                    rust_future
                        .await
                        .map_err(|e| {
                            let err_msg = format!("Python async execution failed: {}", e);
                            log::error!("{}", err_msg);
                            agnt5_sdk_core::error::SdkError::Other(anyhow::anyhow!(err_msg))
                        })
                }
                .instrument(execution_span)
                .await?;

                // Process the result
                let result = Python::attach(
                    |py| -> Result<Option<ServiceMessage>, agnt5_sdk_core::error::SdkError> {
                        let py_result = result.bind(py);
                        // Try to extract as list first (streaming case)
                        if let Ok(py_list) = py_result.extract::<Vec<PyExecuteComponentResponse>>() {
                                    // Send each response via tx
                                    for py_response in py_list.into_iter() {

                                        // Convert to Rust types
                                        let rust_response: ExecuteComponentResponse =
                                            py_response.into();

                                        // Create ServiceMessage
                                        let service_message = ServiceMessage {
                                            worker_id: worker_id.clone(),
                                            metadata: std::collections::HashMap::new(),
                                            message_type: Some(
                                                agnt5_sdk_core::pb::service_message::MessageType::FunctionResponse(rust_response)
                                            ),
                                        };

                                        // Send via channel (blocking send)
                                        if let Err(e) = tx_clone.send(service_message) {
                                            log::error!("Failed to send streaming chunk: {}", e);
                                            return Err(agnt5_sdk_core::error::SdkError::Other(
                                                anyhow::anyhow!(
                                                    "Failed to send streaming chunk: {}",
                                                    e
                                                ),
                                            ));
                                        }
                                    }

                                    // Return None to indicate we handled sending ourselves
                                    return Ok(None);
                        }

                        // Not a list, try single response (non-streaming case)
                        match py_result.extract::<PyExecuteComponentResponse>() {
                                    Ok(py_response) => {
                                        // Calculate execution duration
                                        let execution_duration_ms = execution_start.elapsed().as_millis() as i64;

                                        // Record span result based on success/error
                                        let span = cx.span();

                                        // Always add execution duration and output data
                                        span.set_attribute(opentelemetry::KeyValue::new(
                                            "component.execution_ms",
                                            execution_duration_ms,
                                        ));
                                        span.set_attribute(opentelemetry::KeyValue::new(
                                            "output.data",
                                            String::from_utf8_lossy(&py_response.output_data).to_string(),
                                        ));

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

                                            // Add stack trace if available from metadata
                                            if let Some(stack_trace) = py_response.metadata.get("stack_trace") {
                                                span.set_attribute(opentelemetry::KeyValue::new(
                                                    "exception.stacktrace",
                                                    stack_trace.clone(),
                                                ));
                                            }
                                            if let Some(error_type) = py_response.metadata.get("error_type") {
                                                span.set_attribute(opentelemetry::KeyValue::new(
                                                    "exception.type",
                                                    error_type.clone(),
                                                ));
                                            }

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
                                        metadata: std::collections::HashMap::new(),
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

                // Export span to journal for real-time SSE streaming (if is_streaming)
                // This must happen BEFORE the span is dropped
                if is_streaming_request {
                    // Get span context for trace_id and span_id
                    let span = cx.span();
                    let span_context = span.span_context();
                    let trace_id = span_context.trace_id().to_string();
                    let span_id = span_context.span_id().to_string();

                    // Get current time as end time
                    let end_time_unix_nano = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as i64)
                        .unwrap_or(0);

                    // Determine status from result
                    let (status_code, status_description): (&str, Option<String>) = if let Some(ref msg) = result {
                        if let Some(agnt5_sdk_core::pb::service_message::MessageType::FunctionResponse(ref resp)) = msg.message_type {
                            if resp.success {
                                ("ok", None)
                            } else {
                                // error_message is a String (protobuf optional string becomes String with empty default)
                                let err_msg = if resp.error_message.is_empty() { None } else { Some(resp.error_message.clone()) };
                                ("error", err_msg)
                            }
                        } else {
                            ("ok", None)
                        }
                    } else {
                        ("ok", None) // Streaming case handled separately
                    };

                    // Create journal span data
                    let span_data = agnt5_sdk_core::create_journal_span_data(
                        &trace_id,
                        &span_id,
                        parent_span_id_for_journal.as_deref(),
                        &component_name,
                        "server", // span kind
                        execution_start_nano,
                        end_time_unix_nano,
                        status_code,
                        status_description.as_deref(),
                        None, // TODO: attributes
                    );

                    // Export component execution span to journal for SSE streaming
                    // Use timeout to prevent blocking indefinitely on network issues
                    // Best-effort: if export fails or times out, log warning and continue
                    let export_result = tokio::time::timeout(
                        std::time::Duration::from_millis(100),
                        agnt5_sdk_core::export_span_to_journal(
                            &invocation_id,
                            &span_data,
                            tenant_id_for_journal.as_deref(),
                        )
                    ).await;

                    match export_result {
                        Ok(Ok(())) => {
                            tracing::debug!("🔍 STREAM-DEBUG: Exported component span to journal");
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("🔍 STREAM-DEBUG: Failed to export span to journal: {}", e);
                        }
                        Err(_) => {
                            tracing::warn!("🔍 STREAM-DEBUG: Span export timed out after 100ms");
                        }
                    }
                }

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
            Some(runtime_message::MessageData::RuntimeServiceResponse(response)) => {
                // Route to EntityStateManager
                {
                    let manager_guard = entity_state_manager_arc.lock().await;
                    if let Some(ref manager) = *manager_guard {
                        log::debug!(
                            "Routing RuntimeServiceResponse to EntityStateManager (request_id: {})",
                            response.request_id
                        );
                        manager.handle_response(response).await;
                    } else {
                        log::debug!(
                            "Received RuntimeServiceResponse but no EntityStateManager configured (request_id: {})",
                            response.request_id
                        );
                    }
                }
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
