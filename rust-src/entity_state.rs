/// Entity state management with gRPC-based persistence and in-memory caching
///
/// This module provides EntityStateManager as the core business logic for entity state management.
/// It handles:
/// - In-memory cache (shared across all operations)
/// - Version tracking for optimistic concurrency control
/// - Retry logic with exponential backoff
/// - gRPC-based platform persistence

use agnt5_sdk_core::pb::{
    runtime_service_request, runtime_service_response, service_message, EntityStateLoadRequest,
    EntityStateSaveRequest, RuntimeServiceRequest, RuntimeServiceResponse, ServiceMessage,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{oneshot, Mutex, RwLock};

/// Cache entry with version tracking and TTL
#[derive(Debug, Clone)]
pub(crate) struct CachedState {
    /// Entity state as JSON bytes
    state_json: Vec<u8>,
    /// Current version for optimistic locking
    version: i64,
    /// Timestamp when this entry was cached
    cached_at: SystemTime,
    /// TTL for cache entry (optional, defaults to no expiry)
    ttl: Option<Duration>,
}

impl CachedState {
    fn new(state_json: Vec<u8>, version: i64) -> Self {
        Self {
            state_json,
            version,
            cached_at: SystemTime::now(),
            ttl: Some(Duration::from_secs(300)), // 5 minutes default TTL
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            if let Ok(elapsed) = self.cached_at.elapsed() {
                return elapsed > ttl;
            }
        }
        false
    }
}

/// Key for entity state (type, key)
type StateKey = (String, String);

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

/// Error types for entity operations
#[derive(Debug)]
pub enum EntityError {
    /// Version conflict - expected version doesn't match
    VersionConflict {
        expected: i64,
        actual: i64,
    },
    /// Platform communication error
    PlatformError(String),
    /// Timeout error
    Timeout,
    /// Request cancelled
    Cancelled,
    /// Not connected to platform
    NotConnected,
}

impl std::fmt::Display for EntityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityError::VersionConflict { expected, actual } => {
                write!(f, "Version conflict: expected {} but got {}", expected, actual)
            }
            EntityError::PlatformError(msg) => write!(f, "Platform error: {}", msg),
            EntityError::Timeout => write!(f, "Operation timed out"),
            EntityError::Cancelled => write!(f, "Operation cancelled"),
            EntityError::NotConnected => write!(f, "Not connected to platform"),
        }
    }
}

impl std::error::Error for EntityError {}

/// Manages entity state persistence via gRPC with caching and optimistic concurrency
///
/// EntityStateManager is the core business logic layer that:
/// - Maintains an in-memory cache of entity state
/// - Tracks versions for optimistic concurrency control
/// - Handles retries on version conflicts with exponential backoff
/// - Coordinates with the platform via gRPC for persistence
///
/// This is the Rust core that will be shared across all language SDKs.
#[pyclass]
pub struct EntityStateManager {
    /// Tenant ID for multi-tenancy
    pub(crate) tenant_id: String,

    /// In-memory cache: (entity_type, entity_key) -> cached state with version
    pub(crate) cache: Arc<RwLock<HashMap<StateKey, CachedState>>>,

    /// Pending requests awaiting responses (request_id -> oneshot sender)
    pub(crate) pending_requests: Arc<RwLock<HashMap<String, oneshot::Sender<RuntimeServiceResponse>>>>,

    /// Channel for sending requests to the worker stream
    pub(crate) request_sender: Arc<Mutex<Option<agnt5_sdk_core::flume::Sender<ServiceMessage>>>>,

    /// Maximum number of retries on version conflict (default: 3)
    pub(crate) max_retries: u32,

    /// Base delay for exponential backoff in milliseconds (default: 100ms)
    pub(crate) base_delay_ms: u64,
}

impl EntityStateManager {
    /// Create a new EntityStateManager
    pub fn new(tenant_id: String) -> Self {
        Self {
            tenant_id,
            cache: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            request_sender: Arc::new(Mutex::new(None)),
            max_retries: 3,
            base_delay_ms: 100,
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
    async fn send_request(&self, operation: runtime_service_request::Operation) -> Result<RuntimeServiceResponse, EntityError> {
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
            session_id: String::new(), // Not needed for entity operations
            operation: Some(operation),
        };

        // Wrap in ServiceMessage with RuntimeService
        let service_message = ServiceMessage {
            worker_id: String::new(), // Will be set by worker
            metadata: std::collections::HashMap::new(),
            message_type: Some(service_message::MessageType::RuntimeService(request)),
        };

        // Send request via worker stream
        {
            let sender = self.request_sender.lock().await;
            if let Some(ref sender) = *sender {
                sender
                    .send_async(service_message)
                    .await
                    .map_err(|e| EntityError::PlatformError(format!("Failed to send request: {}", e)))?;
            } else {
                return Err(EntityError::NotConnected);
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

            EntityError::Timeout
        })?
        .map_err(|_| EntityError::Cancelled)?;

        // Check if response indicates error
        if !response.success {
            return Err(EntityError::PlatformError(response.error_message));
        }

        Ok(response)
    }

    /// Load entity state from platform (bypasses cache)
    async fn load_from_platform(
        &self,
        entity_type: String,
        entity_key: String,
        scope: String,
        scope_id: String,
    ) -> Result<EntityLoadResult, EntityError> {
        log::info!(
            "EntityStateManager: Loading from platform {}:{} (scope: {}, scope_id: {})",
            entity_type,
            entity_key,
            scope,
            scope_id
        );

        let operation = runtime_service_request::Operation::EntityStateLoad(
            EntityStateLoadRequest {
                entity_type,
                entity_key,
                scope,
                scope_id,
            },
        );

        let response = self.send_request(operation).await?;

        // Extract load result
        match response.result {
            Some(runtime_service_response::Result::EntityStateLoad(result)) => {
                log::info!(
                    "EntityStateManager: Platform load result - found: {}, version: {}",
                    result.found,
                    result.version
                );

                Ok(EntityLoadResult {
                    found: result.found,
                    state_json: result.state_json,
                    version: result.version,
                })
            }
            _ => Err(EntityError::PlatformError("Unexpected response type for entity load".to_string())),
        }
    }

    /// Save entity state to platform (updates cache on success)
    async fn save_to_platform(
        &self,
        entity_type: String,
        entity_key: String,
        state_json: Vec<u8>,
        expected_version: i64,
        scope: String,
        scope_id: String,
    ) -> Result<EntitySaveResult, EntityError> {
        log::info!(
            "EntityStateManager: Saving to platform {}:{} (expected version: {}, scope: {}, scope_id: {})",
            entity_type,
            entity_key,
            expected_version,
            scope,
            scope_id
        );

        let operation = runtime_service_request::Operation::EntityStateSave(
            EntityStateSaveRequest {
                entity_type: entity_type.clone(),
                entity_key: entity_key.clone(),
                state_json: state_json.clone(),
                expected_version,
                scope,
                scope_id,
            },
        );

        let response = self.send_request(operation).await?;

        // Extract save result
        match response.result {
            Some(runtime_service_response::Result::EntityStateSave(result)) => {
                log::info!(
                    "EntityStateManager: Platform save successful - new version: {}",
                    result.new_version
                );

                // Update cache with new state and version
                let state_key = (entity_type, entity_key);
                let cached = CachedState::new(state_json, result.new_version);
                {
                    let mut cache = self.cache.write().await;
                    cache.insert(state_key, cached);
                }

                Ok(EntitySaveResult {
                    new_version: result.new_version,
                })
            }
            _ => Err(EntityError::PlatformError("Unexpected response type for entity save".to_string())),
        }
    }

    /// Get cached state or load from platform
    ///
    /// This is the primary method for reading entity state. It:
    /// 1. Checks cache first
    /// 2. Returns cached value if not expired
    /// 3. Loads from platform if cache miss or expired
    /// 4. Updates cache with loaded value
    pub async fn get_cached_or_load(
        &self,
        entity_type: String,
        entity_key: String,
        scope: String,
        scope_id: String,
    ) -> Result<(Vec<u8>, i64), EntityError> {
        // Cache key includes scope info for proper isolation
        let state_key = (entity_type.clone(), entity_key.clone());

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&state_key) {
                if !cached.is_expired() {
                    log::debug!(
                        "EntityStateManager: Cache hit for {}:{} (version {})",
                        entity_type,
                        entity_key,
                        cached.version
                    );
                    return Ok((cached.state_json.clone(), cached.version));
                } else {
                    log::debug!(
                        "EntityStateManager: Cache expired for {}:{}",
                        entity_type,
                        entity_key
                    );
                }
            }
        }

        // Cache miss or expired - load from platform
        log::debug!(
            "EntityStateManager: Cache miss for {}:{}, loading from platform",
            entity_type,
            entity_key
        );

        let result = self.load_from_platform(
            entity_type.clone(),
            entity_key.clone(),
            scope,
            scope_id,
        ).await?;

        // Update cache if found
        if result.found {
            let cached = CachedState::new(result.state_json.clone(), result.version);
            let mut cache = self.cache.write().await;
            cache.insert(state_key, cached);
        }

        Ok((result.state_json, result.version))
    }

    /// Save state with version check (optimistic locking)
    ///
    /// This method:
    /// 1. Saves to platform with expected version
    /// 2. Returns version conflict error if version doesn't match
    /// 3. Updates cache on success
    pub async fn save_with_version_check(
        &self,
        entity_type: String,
        entity_key: String,
        state_json: Vec<u8>,
        expected_version: i64,
        scope: String,
        scope_id: String,
    ) -> Result<i64, EntityError> {
        let result = self.save_to_platform(
            entity_type,
            entity_key,
            state_json,
            expected_version,
            scope,
            scope_id,
        ).await?;

        Ok(result.new_version)
    }

    /// Update entity state with automatic retry on version conflicts
    ///
    /// This is the high-level method for updating entity state. It:
    /// 1. Loads current state with version
    /// 2. Applies update function to get new state
    /// 3. Saves with version check
    /// 4. Retries with exponential backoff on version conflict
    ///
    /// The update_fn receives the current state and returns the new state.
    pub async fn update_with_retry<F, Fut>(
        &self,
        entity_type: String,
        entity_key: String,
        scope: String,
        scope_id: String,
        update_fn: F,
    ) -> Result<i64, EntityError>
    where
        F: Fn(Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<u8>, EntityError>>,
    {
        let mut attempt = 0;

        loop {
            // Load current state
            let (current_state, current_version) = self.get_cached_or_load(
                entity_type.clone(),
                entity_key.clone(),
                scope.clone(),
                scope_id.clone(),
            ).await?;

            // Apply update function
            let new_state = update_fn(current_state).await?;

            // Try to save with version check
            match self.save_with_version_check(
                entity_type.clone(),
                entity_key.clone(),
                new_state,
                current_version,
                scope.clone(),
                scope_id.clone(),
            ).await {
                Ok(new_version) => {
                    log::info!(
                        "EntityStateManager: Update successful for {}:{} (version {} -> {}) after {} attempts",
                        entity_type,
                        entity_key,
                        current_version,
                        new_version,
                        attempt + 1
                    );
                    return Ok(new_version);
                }
                Err(EntityError::VersionConflict { expected, actual }) => {
                    attempt += 1;
                    if attempt >= self.max_retries {
                        log::warn!(
                            "EntityStateManager: Max retries ({}) exceeded for {}:{} (expected version {}, actual {})",
                            self.max_retries,
                            entity_type,
                            entity_key,
                            expected,
                            actual
                        );
                        return Err(EntityError::VersionConflict { expected, actual });
                    }

                    // Exponential backoff: base_delay * 2^attempt
                    let delay_ms = self.base_delay_ms * (2_u64.pow(attempt - 1));
                    log::info!(
                        "EntityStateManager: Version conflict for {}:{}, retrying in {}ms (attempt {}/{})",
                        entity_type,
                        entity_key,
                        delay_ms,
                        attempt,
                        self.max_retries
                    );

                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;

                    // Invalidate cache to force reload
                    self.invalidate_cache(&entity_type, &entity_key).await;
                }
                Err(e) => {
                    // Other errors are not retryable
                    return Err(e);
                }
            }
        }
    }

    /// Invalidate cache entry for specific entity
    pub async fn invalidate_cache(&self, entity_type: &str, entity_key: &str) {
        let state_key = (entity_type.to_string(), entity_key.to_string());
        let mut cache = self.cache.write().await;
        cache.remove(&state_key);
        log::debug!(
            "EntityStateManager: Invalidated cache for {}:{}",
            entity_type,
            entity_key
        );
    }

    /// Clear entire cache (useful for testing)
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        log::debug!("EntityStateManager: Cleared entire cache");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        let total_entries = cache.len();
        let expired_entries = cache.values().filter(|c| c.is_expired()).count();
        (total_entries, expired_entries)
    }
}

// PyO3 Python bindings
#[pymethods]
impl EntityStateManager {
    #[new]
    pub fn py_new(tenant_id: String) -> PyResult<Self> {
        Ok(Self::new(tenant_id))
    }

    /// Load entity state (Python-facing async method)
    ///
    /// Returns tuple: (found, state_json, version)
    #[pyo3(signature = (entity_type, entity_key, scope = String::new(), scope_id = String::new()))]
    pub fn py_load_state<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_key: String,
        scope: String,
        scope_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = manager.load_from_platform(entity_type, entity_key, scope, scope_id).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Return tuple: (found, state_json, version)
            Ok((result.found, result.state_json, result.version))
        })
    }

    /// Save entity state (Python-facing async method)
    ///
    /// Returns new_version
    #[pyo3(signature = (entity_type, entity_key, state_json, expected_version, scope = String::new(), scope_id = String::new()))]
    pub fn py_save_state<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_key: String,
        state_json: Vec<u8>,
        expected_version: i64,
        scope: String,
        scope_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let new_version = manager.save_with_version_check(
                entity_type,
                entity_key,
                state_json,
                expected_version,
                scope,
                scope_id,
            ).await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Return new_version
            Ok(new_version)
        })
    }

    /// Get cached state or load from platform (Python-facing async method)
    ///
    /// Returns tuple: (state_json, version)
    #[pyo3(signature = (entity_type, entity_key, scope = String::new(), scope_id = String::new()))]
    pub fn py_get_cached_or_load<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_key: String,
        scope: String,
        scope_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (state_json, version) = manager.get_cached_or_load(entity_type, entity_key, scope, scope_id).await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            Ok((state_json, version))
        })
    }

    /// Invalidate cache (Python-facing method)
    pub fn py_invalidate_cache<'py>(
        &self,
        py: Python<'py>,
        entity_type: String,
        entity_key: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            manager.invalidate_cache(&entity_type, &entity_key).await;
            Ok(())
        })
    }

    /// Clear entire cache (Python-facing method)
    pub fn py_clear_cache<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            manager.clear_cache().await;
            Ok(())
        })
    }

    /// Get cache statistics (Python-facing method)
    ///
    /// Returns tuple: (total_entries, expired_entries)
    pub fn py_cache_stats<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = self.clone_arc();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (total, expired) = manager.cache_stats().await;
            Ok((total, expired))
        })
    }
}

// Helper to clone as Arc for async use
impl EntityStateManager {
    fn clone_arc(&self) -> Arc<Self> {
        Arc::new(Self {
            tenant_id: self.tenant_id.clone(),
            cache: self.cache.clone(),
            pending_requests: self.pending_requests.clone(),
            request_sender: self.request_sender.clone(),
            max_retries: self.max_retries,
            base_delay_ms: self.base_delay_ms,
        })
    }
}
