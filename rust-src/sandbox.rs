/// PyO3 bindings for the AGNT5 Sandbox module.
///
/// Exposes `PySandbox` — a Python class backed by either WasmSandbox (embedded)
/// or RemoteSandbox (HTTP client), selected via constructor arguments.
use agnt5_sdk_core::sandbox::{
    ExecuteCodeRequest, Language, RemoteSandbox, RemoteSandboxConfig, SandboxAuth,
    SandboxBackendKind, SandboxExecutor, SandboxWorkspace, WriteFileRequest,
};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "wasm-sandbox")]
use agnt5_sdk_core::sandbox::{WasmSandbox, WasmSandboxConfig};

/// Trait object wrapper so we can store either backend.
type BoxedExecutor = Arc<dyn SandboxExecutor>;
type BoxedWorkspace = Arc<dyn SandboxWorkspace>;

/// Python sandbox backed by the Rust sdk-core sandbox module.
///
/// Provides sandboxed code execution and workspace file operations.
/// Backend is selected at construction time:
///
/// - ``RustSandbox(endpoint="http://...")`` → RemoteSandbox (HTTP)
/// - ``RustSandbox(backend="wasm")`` → WasmSandbox (embedded, requires wasm-sandbox feature)
#[pyclass(name = "RustSandbox")]
pub struct PySandbox {
    executor: BoxedExecutor,
    workspace: BoxedWorkspace,
    backend_kind: SandboxBackendKind,
}

#[pymethods]
impl PySandbox {
    /// Create a new sandbox.
    ///
    /// Args:
    ///     backend: "remote" or "wasm". Default: auto-detect.
    ///     endpoint: HTTP endpoint for remote sandbox (required if backend="remote").
    ///     sandbox_id: Sandbox instance ID (default: "default").
    ///     api_key: API key for remote auth.
    ///     bearer_token: Bearer token for remote auth.
    ///     timeout_secs: Request timeout in seconds (default: 300).
    ///     quickjs_wasm_path: Path to QuickJS WASI binary (for wasm backend).
    #[new]
    #[pyo3(signature = (backend=None, endpoint=None, sandbox_id=None, api_key=None, bearer_token=None, timeout_secs=None, quickjs_wasm_path=None))]
    fn new(
        backend: Option<String>,
        endpoint: Option<String>,
        sandbox_id: Option<String>,
        api_key: Option<String>,
        bearer_token: Option<String>,
        timeout_secs: Option<u64>,
        quickjs_wasm_path: Option<String>,
    ) -> PyResult<Self> {
        let backend_str = backend.as_deref().unwrap_or("auto");
        let timeout = std::time::Duration::from_secs(timeout_secs.unwrap_or(300));

        match backend_str {
            "remote" => {
                let ep = endpoint.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "endpoint is required for remote backend",
                    )
                })?;
                let auth = if let Some(key) = api_key {
                    SandboxAuth::ApiKey(key)
                } else if let Some(token) = bearer_token {
                    SandboxAuth::BearerToken(token)
                } else {
                    SandboxAuth::None
                };
                let config = RemoteSandboxConfig {
                    endpoint: ep,
                    sandbox_id: sandbox_id.unwrap_or_else(|| "default".to_string()),
                    auth,
                    timeout,
                };
                let remote = RemoteSandbox::new(config).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create RemoteSandbox: {}",
                        e
                    ))
                })?;
                let arc: Arc<RemoteSandbox> = Arc::new(remote);
                Ok(PySandbox {
                    executor: arc.clone() as BoxedExecutor,
                    workspace: arc as BoxedWorkspace,
                    backend_kind: SandboxBackendKind::Remote,
                })
            }

            #[cfg(feature = "wasm-sandbox")]
            "wasm" => {
                let config = WasmSandboxConfig {
                    quickjs_wasm_path: quickjs_wasm_path.map(std::path::PathBuf::from),
                    ..Default::default()
                };
                let wasm = WasmSandbox::new(config).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create WasmSandbox: {}",
                        e
                    ))
                })?;
                let arc: Arc<WasmSandbox> = Arc::new(wasm);
                Ok(PySandbox {
                    executor: arc.clone() as BoxedExecutor,
                    workspace: arc as BoxedWorkspace,
                    backend_kind: SandboxBackendKind::Wasm,
                })
            }

            #[cfg(not(feature = "wasm-sandbox"))]
            "wasm" => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "WasmSandbox is not available. Rebuild the SDK with the wasm-sandbox feature.",
            )),

            "auto" => {
                // Auto-detect: use remote if endpoint provided, else try wasm
                if let Some(ep) = endpoint {
                    let auth = if let Some(key) = api_key {
                        SandboxAuth::ApiKey(key)
                    } else if let Some(token) = bearer_token {
                        SandboxAuth::BearerToken(token)
                    } else {
                        SandboxAuth::None
                    };
                    let config = RemoteSandboxConfig {
                        endpoint: ep,
                        sandbox_id: sandbox_id.unwrap_or_else(|| "default".to_string()),
                        auth,
                        timeout,
                    };
                    let remote = RemoteSandbox::new(config).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to create RemoteSandbox: {}",
                            e
                        ))
                    })?;
                    let arc: Arc<RemoteSandbox> = Arc::new(remote);
                    Ok(PySandbox {
                        executor: arc.clone() as BoxedExecutor,
                        workspace: arc as BoxedWorkspace,
                        backend_kind: SandboxBackendKind::Remote,
                    })
                } else {
                    #[cfg(feature = "wasm-sandbox")]
                    {
                        let config = WasmSandboxConfig {
                            quickjs_wasm_path: quickjs_wasm_path
                                .map(std::path::PathBuf::from),
                            ..Default::default()
                        };
                        let wasm = WasmSandbox::new(config).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "Failed to create WasmSandbox: {}",
                                e
                            ))
                        })?;
                        let arc: Arc<WasmSandbox> = Arc::new(wasm);
                        Ok(PySandbox {
                            executor: arc.clone() as BoxedExecutor,
                            workspace: arc as BoxedWorkspace,
                            backend_kind: SandboxBackendKind::Wasm,
                        })
                    }
                    #[cfg(not(feature = "wasm-sandbox"))]
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "No sandbox backend available. Provide endpoint for remote, or rebuild with wasm-sandbox feature.",
                    ))
                }
            }

            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown backend: '{}'. Use 'remote', 'wasm', or 'auto'.",
                other
            ))),
        }
    }

    /// Get the backend kind ("wasm" or "remote").
    #[getter]
    fn backend(&self) -> String {
        self.backend_kind.to_string()
    }

    /// Get sandbox capabilities as a dict.
    fn capabilities(&self, py: Python<'_>) -> PyResult<PyObject> {
        let caps = self.executor.capabilities();
        let dict = pyo3::types::PyDict::new(py);
        let languages: Vec<String> = caps.languages.iter().map(|l| l.to_string()).collect();
        dict.set_item("languages", languages)?;
        dict.set_item("supports_commands", caps.supports_commands)?;
        dict.set_item("supports_git", caps.supports_git)?;
        dict.set_item("supports_preview_url", caps.supports_preview_url)?;
        dict.set_item("supports_streaming", caps.supports_streaming)?;
        dict.set_item("supports_snapshots", caps.supports_snapshots)?;
        dict.set_item("has_network_access", caps.has_network_access)?;
        Ok(dict.into_any().unbind())
    }

    /// Execute code in the sandbox.
    ///
    /// Args:
    ///     code: Source code to execute.
    ///     language: Programming language ("javascript", "python", "bash").
    ///     timeout_ms: Execution timeout in milliseconds (default: 30000).
    ///     env: Environment variables as a dict.
    ///
    /// Returns:
    ///     Dict with stdout, stderr, exit_code, execution_time_ms, error.
    #[pyo3(signature = (code, language="javascript", timeout_ms=30000, env=None))]
    fn execute_code<'py>(
        &self,
        py: Python<'py>,
        code: String,
        language: &str,
        timeout_ms: u64,
        env: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let lang = parse_language(language)?;
        let executor = self.executor.clone();

        future_into_py(py, async move {
            let req = ExecuteCodeRequest {
                code,
                language: lang,
                timeout_ms,
                env,
                work_dir: None,
            };
            let result = executor.execute_code(req).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("execute_code failed: {}", e))
            })?;

            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("stdout", &result.stdout)?;
                dict.set_item("stderr", &result.stderr)?;
                dict.set_item("exit_code", result.exit_code)?;
                dict.set_item("execution_time_ms", result.execution_time_ms)?;
                dict.set_item("error", result.error)?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    /// Write a file to the sandbox workspace.
    ///
    /// Args:
    ///     path: File path relative to workspace root.
    ///     content: File content as bytes or string.
    ///
    /// Returns:
    ///     Dict with success, path, size, error.
    #[pyo3(signature = (path, content))]
    fn write_file<'py>(
        &self,
        py: Python<'py>,
        path: String,
        content: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workspace = self.workspace.clone();

        future_into_py(py, async move {
            let req = WriteFileRequest {
                path: path.clone(),
                content,
                mode: 0o644,
            };
            let result = workspace.write_file(req).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("write_file failed: {}", e))
            })?;

            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("success", result.success)?;
                dict.set_item("path", &result.path)?;
                dict.set_item("size", result.size)?;
                dict.set_item("error", result.error)?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    /// Read a file from the sandbox workspace.
    ///
    /// Args:
    ///     path: File path relative to workspace root.
    ///
    /// Returns:
    ///     Dict with path, content (bytes), size, is_dir, error.
    fn read_file<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let workspace = self.workspace.clone();

        future_into_py(py, async move {
            let result = workspace.read_file(&path).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("read_file failed: {}", e))
            })?;

            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("path", &result.path)?;
                dict.set_item("content", pyo3::types::PyBytes::new(py, &result.content))?;
                dict.set_item("size", result.size)?;
                dict.set_item("is_dir", result.is_dir)?;
                dict.set_item("error", result.error)?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    /// Delete a file or directory from the sandbox workspace.
    ///
    /// Args:
    ///     path: Path to delete.
    ///     recursive: If True, delete directories recursively.
    ///
    /// Returns:
    ///     True if the file was deleted.
    #[pyo3(signature = (path, recursive=false))]
    fn delete_file<'py>(
        &self,
        py: Python<'py>,
        path: String,
        recursive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workspace = self.workspace.clone();

        future_into_py(py, async move {
            let result = workspace.delete_file(&path, recursive).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("delete_file failed: {}", e))
            })?;
            Ok(result)
        })
    }

    /// List files in a directory within the sandbox workspace.
    ///
    /// Args:
    ///     path: Directory path to list (default: ".").
    ///     recursive: If True, list recursively.
    ///
    /// Returns:
    ///     Dict with path, total, files (list of dicts with name, path, size, is_dir).
    #[pyo3(signature = (path=".", recursive=false))]
    fn list_files<'py>(
        &self,
        py: Python<'py>,
        path: &str,
        recursive: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let workspace = self.workspace.clone();
        let path_owned = path.to_string();

        future_into_py(py, async move {
            let result = workspace
                .list_files(&path_owned, recursive)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "list_files failed: {}",
                        e
                    ))
                })?;

            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("path", &result.path)?;
                dict.set_item("total", result.total)?;

                let files_list = pyo3::types::PyList::empty(py);
                for f in &result.files {
                    let file_dict = pyo3::types::PyDict::new(py);
                    file_dict.set_item("name", &f.name)?;
                    file_dict.set_item("path", &f.path)?;
                    file_dict.set_item("size", f.size)?;
                    file_dict.set_item("is_dir", f.is_dir)?;
                    file_dict.set_item("mode", f.mode)?;
                    file_dict.set_item("mod_time", f.mod_time)?;
                    files_list.append(file_dict)?;
                }
                dict.set_item("files", files_list)?;
                dict.set_item("error", result.error)?;

                Ok(dict.into_any().unbind())
            })
        })
    }

    /// Check sandbox health.
    ///
    /// Returns:
    ///     Dict with status, sandbox_id, uptime_ms, backend_kind, error.
    fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let executor = self.executor.clone();

        future_into_py(py, async move {
            let result = executor.health().await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("health check failed: {}", e))
            })?;

            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("status", &result.status)?;
                dict.set_item("sandbox_id", &result.sandbox_id)?;
                dict.set_item("uptime_ms", result.uptime_ms)?;
                dict.set_item("backend_kind", result.backend_kind.to_string())?;
                dict.set_item("error", result.error)?;
                Ok(dict.into_any().unbind())
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("RustSandbox(backend='{}')", self.backend_kind)
    }
}

fn parse_language(s: &str) -> PyResult<Language> {
    match s.to_lowercase().as_str() {
        "javascript" | "js" => Ok(Language::Javascript),
        "python" | "py" => Ok(Language::Python),
        "bash" | "sh" => Ok(Language::Bash),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown language: '{}'. Use 'javascript', 'python', or 'bash'.",
            other
        ))),
    }
}

pub fn register_sandbox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySandbox>()?;
    Ok(())
}
