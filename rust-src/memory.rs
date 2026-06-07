/// PyO3 bindings for memory.
///
/// Provides Python async access to the AGNT5-native MemoryService runtime.
use agnt5_sdk_core::lm::OpenAiEmbedder;
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
use std::sync::Arc;

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
    m.add_class::<PyAgnt5MemoryRecord>()?;
    m.add_class::<PyAgnt5MemorySearchResult>()?;
    m.add_class::<PyAgnt5MemoryRuntime>()?;
    Ok(())
}
