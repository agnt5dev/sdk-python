/// PyO3 bindings for memory.
///
/// Provides Python async access to the legacy SemanticMemory implementation
/// and the AGNT5-native MemoryService runtime.
use agnt5_sdk_core::lm::OpenAiEmbedder;
use agnt5_sdk_core::memory::{MemoryMetadata, MemoryResult, MemoryScope, SemanticMemory};
use agnt5_sdk_core::memory_service::{
    DeleteMemoryRequest as NativeDeleteMemoryRequest,
    MemoryFailurePolicy as NativeMemoryFailurePolicy,
    MemoryProviderKind as NativeMemoryProviderKind, MemoryRecord as NativeMemoryRecord,
    MemoryRuntime as NativeMemoryRuntime, MemoryRuntimeConfig as NativeMemoryRuntimeConfig,
    MemoryScope as NativeMemoryScope, MemoryScopeFilter as NativeMemoryScopeFilter,
    MemorySearchResult as NativeMemorySearchResult,
    MemoryServiceFactory as NativeMemoryServiceFactory,
    SaveMemoryRequest as NativeSaveMemoryRequest, SearchMemoryRequest as NativeSearchMemoryRequest,
};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Python-accessible memory scope enum
#[pyclass(name = "MemoryScope")]
#[derive(Clone)]
pub struct PyMemoryScope {
    inner: MemoryScope,
}

#[pymethods]
impl PyMemoryScope {
    /// Create a User scope
    #[staticmethod]
    fn user() -> Self {
        Self {
            inner: MemoryScope::User,
        }
    }

    /// Create a Tenant scope
    #[staticmethod]
    fn tenant() -> Self {
        Self {
            inner: MemoryScope::Tenant,
        }
    }

    /// Create an Agent scope
    #[staticmethod]
    fn agent() -> Self {
        Self {
            inner: MemoryScope::Agent,
        }
    }

    /// Create a Session scope
    #[staticmethod]
    fn session() -> Self {
        Self {
            inner: MemoryScope::Session,
        }
    }

    /// Create a Global scope
    #[staticmethod]
    #[pyo3(name = "global_")]
    fn global_scope() -> Self {
        Self {
            inner: MemoryScope::Global,
        }
    }

    /// Parse scope from string
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        MemoryScope::from_str(s)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid memory scope: '{}'. Valid values: user, tenant, agent, session, global",
                    s
                ))
            })
    }

    /// Get string representation
    fn as_str(&self) -> &'static str {
        self.inner.as_str()
    }

    fn __str__(&self) -> &'static str {
        self.inner.as_str()
    }

    fn __repr__(&self) -> String {
        format!("MemoryScope.{}", self.inner.as_str())
    }
}

/// Python-accessible memory metadata
#[pyclass(name = "MemoryMetadata")]
#[derive(Clone)]
pub struct PyMemoryMetadata {
    /// Source of the memory
    #[pyo3(get, set)]
    pub source: Option<String>,

    /// Timestamp when memory was created
    #[pyo3(get, set)]
    pub created_at: Option<String>,

    /// Additional custom metadata
    #[pyo3(get, set)]
    pub extra: HashMap<String, String>,
}

#[pymethods]
impl PyMemoryMetadata {
    #[new]
    #[pyo3(signature = (source=None, created_at=None, extra=None))]
    fn new(
        source: Option<String>,
        created_at: Option<String>,
        extra: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            source,
            created_at,
            extra: extra.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryMetadata(source={:?}, created_at={:?})",
            self.source, self.created_at
        )
    }
}

impl From<MemoryMetadata> for PyMemoryMetadata {
    fn from(m: MemoryMetadata) -> Self {
        Self {
            source: m.source,
            created_at: m.created_at,
            extra: m
                .extra
                .into_iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                .collect(),
        }
    }
}

impl From<PyMemoryMetadata> for MemoryMetadata {
    fn from(m: PyMemoryMetadata) -> Self {
        let mut metadata = MemoryMetadata::new();
        metadata.source = m.source;
        metadata.created_at = m.created_at;
        for (k, v) in m.extra {
            metadata.extra.insert(k, serde_json::Value::String(v));
        }
        metadata
    }
}

/// Python-accessible memory search result
#[pyclass(name = "MemoryResult")]
#[derive(Clone)]
pub struct PyMemoryResult {
    /// Unique identifier for this memory
    #[pyo3(get)]
    pub id: String,

    /// Original text content that was stored
    #[pyo3(get)]
    pub content: String,

    /// Similarity score (0.0 to 1.0, higher is more similar)
    #[pyo3(get)]
    pub score: f32,

    /// Associated metadata
    #[pyo3(get)]
    pub metadata: PyMemoryMetadata,
}

#[pymethods]
impl PyMemoryResult {
    fn __repr__(&self) -> String {
        format!(
            "MemoryResult(id='{}', score={:.4}, content='{}...')",
            self.id,
            self.score,
            &self.content[..self.content.len().min(50)]
        )
    }
}

impl From<MemoryResult> for PyMemoryResult {
    fn from(r: MemoryResult) -> Self {
        Self {
            id: r.id,
            content: r.content,
            score: r.score,
            metadata: r.metadata.into(),
        }
    }
}

/// Python-accessible AGNT5-native memory record.
#[pyclass(name = "Agnt5MemoryRecord")]
#[derive(Clone)]
pub struct PyAgnt5MemoryRecord {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub tenant_id: String,
    #[pyo3(get)]
    pub deployment_id: String,
    #[pyo3(get)]
    pub scope: String,
    #[pyo3(get)]
    pub scope_id: String,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub metadata_json: String,
    #[pyo3(get)]
    pub embedding_model: String,
    #[pyo3(get)]
    pub embedding_dim: u32,
    #[pyo3(get)]
    pub source_session_id: Option<String>,
    #[pyo3(get)]
    pub source_run_id: Option<String>,
    #[pyo3(get)]
    pub source_event_id: Option<String>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
}

impl From<NativeMemoryRecord> for PyAgnt5MemoryRecord {
    fn from(record: NativeMemoryRecord) -> Self {
        Self {
            id: record.id,
            tenant_id: record.tenant_id,
            deployment_id: record.deployment_id,
            scope: record.scope.as_str().to_string(),
            scope_id: record.scope_id,
            kind: record.kind,
            content: record.content,
            metadata_json: record.metadata.to_string(),
            embedding_model: record.embedding_model,
            embedding_dim: record.embedding_dim,
            source_session_id: record.source_session_id,
            source_run_id: record.source_run_id,
            source_event_id: record.source_event_id,
            created_at: record.created_at,
            updated_at: record.updated_at,
        }
    }
}

#[pymethods]
impl PyAgnt5MemoryRecord {
    fn __repr__(&self) -> String {
        format!(
            "Agnt5MemoryRecord(id='{}', scope='{}', scope_id='{}', kind='{}')",
            self.id, self.scope, self.scope_id, self.kind
        )
    }
}

/// Python-accessible AGNT5-native memory search result.
#[pyclass(name = "Agnt5MemorySearchResult")]
#[derive(Clone)]
pub struct PyAgnt5MemorySearchResult {
    #[pyo3(get)]
    pub record: PyAgnt5MemoryRecord,
    #[pyo3(get)]
    pub score: f32,
    #[pyo3(get)]
    pub distance: f32,
}

impl From<NativeMemorySearchResult> for PyAgnt5MemorySearchResult {
    fn from(result: NativeMemorySearchResult) -> Self {
        Self {
            record: result.record.into(),
            score: result.score,
            distance: result.distance,
        }
    }
}

#[pymethods]
impl PyAgnt5MemorySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "Agnt5MemorySearchResult(id='{}', score={:.4})",
            self.record.id, self.score
        )
    }
}

/// Python-accessible AGNT5-native memory runtime.
#[pyclass(name = "Agnt5MemoryRuntime")]
pub struct PyAgnt5MemoryRuntime {
    inner: Arc<NativeMemoryRuntime>,
}

#[pymethods]
impl PyAgnt5MemoryRuntime {
    /// Build the configured memory runtime from environment variables.
    #[staticmethod]
    fn from_env<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        future_into_py(py, async move {
            let runtime = build_native_memory_runtime_from_env().await?;
            Ok(Self {
                inner: Arc::new(runtime),
            })
        })
    }

    #[getter]
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    #[getter]
    fn unavailable_reason(&self) -> Option<String> {
        self.inner.unavailable_reason().map(str::to_string)
    }

    #[getter]
    fn provider_name(&self) -> Option<String> {
        self.inner.provider_name().map(str::to_string)
    }

    /// Save a semantic memory record.
    #[pyo3(signature = (
        tenant_id,
        deployment_id,
        scope,
        scope_id,
        content,
        kind=None,
        metadata_json=None,
        memory_id=None,
        source_session_id=None,
        source_run_id=None,
        source_event_id=None
    ))]
    fn save<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        deployment_id: String,
        scope: String,
        scope_id: String,
        content: String,
        kind: Option<String>,
        metadata_json: Option<String>,
        memory_id: Option<String>,
        source_session_id: Option<String>,
        source_run_id: Option<String>,
        source_event_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = parse_native_memory_scope(&scope)?;
        let metadata = parse_metadata_json(metadata_json)?;

        future_into_py(py, async move {
            let saved = inner
                .save_memory(NativeSaveMemoryRequest {
                    id: memory_id,
                    tenant_id,
                    deployment_id,
                    scope,
                    scope_id,
                    kind: kind.unwrap_or_else(|| "custom".to_string()),
                    content,
                    metadata,
                    source_session_id,
                    source_run_id,
                    source_event_id,
                })
                .await
                .map_err(py_runtime_error)?;

            Ok(saved.map(PyAgnt5MemoryRecord::from))
        })
    }

    /// Search semantic memory across one or more paired scope filters.
    #[pyo3(signature = (
        tenant_id,
        deployment_id,
        query,
        scope_filters,
        kinds=None,
        limit=10,
        min_score=None
    ))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        deployment_id: String,
        query: String,
        scope_filters: Vec<(String, String)>,
        kinds: Option<Vec<String>>,
        limit: u32,
        min_score: Option<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope_filters = parse_native_scope_filters(scope_filters)?;

        future_into_py(py, async move {
            let results = inner
                .search_memory(NativeSearchMemoryRequest {
                    tenant_id,
                    deployment_id,
                    query,
                    scope_filters,
                    kinds: kinds.unwrap_or_default(),
                    limit,
                    min_score,
                })
                .await
                .map_err(py_runtime_error)?;

            Ok(results
                .into_iter()
                .map(PyAgnt5MemorySearchResult::from)
                .collect::<Vec<_>>())
        })
    }

    /// Soft-delete semantic memory records by memory ID and/or scope.
    #[pyo3(signature = (
        tenant_id,
        deployment_id,
        memory_id=None,
        scope=None,
        scope_id=None,
        source_run_id=None
    ))]
    fn forget<'py>(
        &self,
        py: Python<'py>,
        tenant_id: String,
        deployment_id: String,
        memory_id: Option<String>,
        scope: Option<String>,
        scope_id: Option<String>,
        source_run_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope_filter = match (scope, scope_id) {
            (Some(scope), Some(scope_id)) => Some(NativeMemoryScopeFilter::new(
                parse_native_memory_scope(&scope)?,
                scope_id,
            )),
            (None, None) => None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "scope and scope_id must be provided together",
                ));
            }
        };

        future_into_py(py, async move {
            inner
                .delete_memory(NativeDeleteMemoryRequest {
                    tenant_id,
                    deployment_id,
                    memory_id,
                    scope_filter,
                    source_run_id,
                })
                .await
                .map_err(py_runtime_error)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Agnt5MemoryRuntime(available={}, provider={:?})",
            self.inner.is_available(),
            self.inner.provider_name()
        )
    }
}

/// Python-accessible SemanticMemory
///
/// Provides high-level semantic memory operations backed by vector database.
///
/// Example:
/// ```python
/// from agnt5._core import PySemanticMemory, PyMemoryScope
///
/// # Create memory from environment
/// memory = await PySemanticMemory.from_env(PyMemoryScope.user(), "user-123")
///
/// # Store content
/// memory_id = await memory.store("User prefers dark mode")
///
/// # Search for similar memories
/// results = await memory.search("color preferences", 5)
///
/// # Delete a memory
/// await memory.forget(memory_id)
/// ```
#[pyclass(name = "PySemanticMemory")]
pub struct PySemanticMemory {
    inner: Arc<Mutex<Option<SemanticMemory>>>,
    scope: MemoryScope,
    scope_id: String,
}

#[pymethods]
impl PySemanticMemory {
    /// Create a new SemanticMemory instance (lazy initialization)
    ///
    /// The actual connection to embedder and vector database happens on first use.
    #[new]
    fn new(scope: &PyMemoryScope, scope_id: String) -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            scope: scope.inner,
            scope_id,
        }
    }

    /// Create SemanticMemory from environment variables
    ///
    /// Auto-detects available embedder and vector database providers.
    #[staticmethod]
    fn from_env<'py>(
        py: Python<'py>,
        scope: &PyMemoryScope,
        scope_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope_inner = scope.inner;

        future_into_py(py, async move {
            let memory = SemanticMemory::from_env(scope_inner, &scope_id)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create SemanticMemory: {}",
                        e
                    ))
                })?;

            Ok(PySemanticMemory {
                inner: Arc::new(Mutex::new(Some(memory))),
                scope: scope_inner,
                scope_id,
            })
        })
    }

    /// Store content in semantic memory
    ///
    /// Returns the unique ID of the stored memory.
    fn store<'py>(&self, py: Python<'py>, content: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            mem.store(&content).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to store memory: {}", e))
            })
        })
    }

    /// Store content with custom metadata
    fn store_with_metadata<'py>(
        &self,
        py: Python<'py>,
        content: String,
        metadata: PyMemoryMetadata,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            mem.store_with_metadata(&content, metadata.into())
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to store memory: {}",
                        e
                    ))
                })
        })
    }

    /// Store multiple contents in batch (more efficient for RAG indexing)
    ///
    /// Returns the unique IDs of all stored memories.
    fn store_batch<'py>(
        &self,
        py: Python<'py>,
        contents: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
            mem.store_batch(&content_refs).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to store memory batch: {}",
                    e
                ))
            })
        })
    }

    /// Store multiple contents with metadata in batch
    ///
    /// Each content must have a corresponding metadata entry.
    fn store_batch_with_metadata<'py>(
        &self,
        py: Python<'py>,
        contents: Vec<String>,
        metadata: Vec<PyMemoryMetadata>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();
            let metadata_converted: Vec<MemoryMetadata> =
                metadata.into_iter().map(Into::into).collect();

            mem.store_batch_with_metadata(&content_refs, &metadata_converted)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to store memory batch: {}",
                        e
                    ))
                })
        })
    }

    /// Search for similar memories
    ///
    /// Returns memories ranked by similarity to the query text.
    #[pyo3(signature = (query, limit=10))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            let results = mem.search(&query, limit).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to search memories: {}",
                    e
                ))
            })?;

            let py_results: Vec<PyMemoryResult> = results.into_iter().map(Into::into).collect();
            Ok(py_results)
        })
    }

    /// Search with minimum score filter
    #[pyo3(signature = (query, limit=10, min_score=None))]
    fn search_with_options<'py>(
        &self,
        py: Python<'py>,
        query: String,
        limit: u32,
        min_score: Option<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            let results = mem
                .search_with_options(&query, limit, min_score)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to search memories: {}",
                        e
                    ))
                })?;

            let py_results: Vec<PyMemoryResult> = results.into_iter().map(Into::into).collect();
            Ok(py_results)
        })
    }

    /// Delete a memory by ID
    fn forget<'py>(&self, py: Python<'py>, memory_id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            mem.forget(&memory_id).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to forget memory: {}", e))
            })
        })
    }

    /// Get a specific memory by ID
    fn get<'py>(&self, py: Python<'py>, memory_id: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            let result = mem.get(&memory_id).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get memory: {}", e))
            })?;

            Ok(result.map(PyMemoryResult::from))
        })
    }

    /// Get the collection name being used
    #[getter]
    fn collection_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let scope = self.scope;
        let scope_id = self.scope_id.clone();

        future_into_py(py, async move {
            let memory = get_or_init_memory(inner, scope, &scope_id).await?;
            let guard = memory.lock().await;
            let mem = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("SemanticMemory not initialized")
            })?;

            Ok(mem.collection_name().to_string())
        })
    }

    /// Get the scope
    #[getter]
    fn scope(&self) -> PyMemoryScope {
        PyMemoryScope { inner: self.scope }
    }

    /// Get the scope ID
    #[getter]
    fn scope_id(&self) -> &str {
        &self.scope_id
    }

    fn __repr__(&self) -> String {
        format!(
            "PySemanticMemory(scope={}, scope_id='{}')",
            self.scope, self.scope_id
        )
    }
}

/// Helper function to get or initialize memory
async fn get_or_init_memory(
    inner: Arc<Mutex<Option<SemanticMemory>>>,
    scope: MemoryScope,
    scope_id: &str,
) -> PyResult<Arc<Mutex<Option<SemanticMemory>>>> {
    {
        let guard = inner.lock().await;
        if guard.is_some() {
            return Ok(inner.clone());
        }
    }

    // Initialize lazily
    let memory = SemanticMemory::from_env(scope, scope_id)
        .await
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to initialize SemanticMemory: {}. \
                 Ensure OPENAI_API_KEY is set for embeddings and \
                 QDRANT_URL, PINECONE_API_KEY+PINECONE_HOST, or POSTGRES_URL is set for vector database.",
                e
            ))
        })?;

    let mut guard = inner.lock().await;
    *guard = Some(memory);

    Ok(inner.clone())
}

async fn build_native_memory_runtime_from_env() -> PyResult<NativeMemoryRuntime> {
    let config = NativeMemoryRuntimeConfig::from_env().map_err(py_config_error)?;

    if !config.enabled || config.provider == NativeMemoryProviderKind::Disabled {
        return Ok(NativeMemoryRuntime::disabled(
            config,
            "memory disabled by configuration",
        ));
    }

    let embedder = match OpenAiEmbedder::from_env() {
        Ok(embedder) => Arc::new(embedder),
        Err(error) if config.failure_policy == NativeMemoryFailurePolicy::BestEffort => {
            return Ok(NativeMemoryRuntime::disabled(config, error.to_string()));
        }
        Err(error) => return Err(py_runtime_error(error)),
    };

    NativeMemoryServiceFactory::from_config(config, embedder)
        .await
        .map_err(py_runtime_error)
}

fn parse_native_scope_filters(
    raw_filters: Vec<(String, String)>,
) -> PyResult<Vec<NativeMemoryScopeFilter>> {
    raw_filters
        .into_iter()
        .map(|(scope, scope_id)| {
            Ok(NativeMemoryScopeFilter::new(
                parse_native_memory_scope(&scope)?,
                scope_id,
            ))
        })
        .collect()
}

fn parse_native_memory_scope(scope: &str) -> PyResult<NativeMemoryScope> {
    match scope.trim().to_ascii_lowercase().as_str() {
        "session" => Ok(NativeMemoryScope::Session),
        "user" => Ok(NativeMemoryScope::User),
        "app" | "global" | "shared" => Ok(NativeMemoryScope::App),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid memory scope `{other}`; expected session, user, or app"
        ))),
    }
}

fn parse_metadata_json(metadata_json: Option<String>) -> PyResult<JsonValue> {
    match metadata_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str(&raw).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid metadata_json: {e}"))
        }),
        _ => Ok(JsonValue::Object(Default::default())),
    }
}

fn py_config_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

fn py_runtime_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
}

/// Register memory module with Python
pub fn register_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryScope>()?;
    m.add_class::<PyMemoryMetadata>()?;
    m.add_class::<PyMemoryResult>()?;
    m.add_class::<PySemanticMemory>()?;
    m.add_class::<PyAgnt5MemoryRecord>()?;
    m.add_class::<PyAgnt5MemorySearchResult>()?;
    m.add_class::<PyAgnt5MemoryRuntime>()?;
    Ok(())
}
