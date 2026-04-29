use agnt5_sdk_core::adk::{
    ContextHandle, ContextRuntimeConfig, DeterministicUtils, RuntimeControls, RuntimeServiceClient,
    SignalControls, TaskControls, TimerControls, ToolDefinition, ToolRegistry,
};
use agnt5_sdk_core::error::SdkError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::{Arc, OnceLock};

pub fn register_adk(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyContext>()?;
    module.add_class::<PyToolRegistry>()?;
    module.add_class::<PyToolDefinition>()?;
    module.add_class::<PyRuntimeControls>()?;
    module.add_class::<PySignalControls>()?;
    module.add_class::<PyTaskControls>()?;
    module.add_class::<PyTimerControls>()?;
    module.add_class::<PyDeterministicUtils>()?;
    module.add_function(wrap_pyfunction!(register_tool, module)?)?;
    module.add_function(wrap_pyfunction!(get_registered_tools, module)?)?;
    module.add_function(wrap_pyfunction!(clear_tools, module)?)?;
    Ok(())
}

static TOOL_REGISTRY: OnceLock<Arc<ToolRegistry>> = OnceLock::new();

fn global_tool_registry() -> &'static Arc<ToolRegistry> {
    TOOL_REGISTRY.get_or_init(ToolRegistry::new)
}

#[pyclass(name = "Context")]
#[derive(Clone)]
pub struct PyContext {
    inner: ContextHandle,
}

#[pymethods]
impl PyContext {
    #[new]
    fn new() -> Self {
        Self {
            inner: ContextHandle::new_placeholder(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (tenant_id, session_id, run_id, step_id, attempt, invocation_id=None))]
    fn with_runtime(
        tenant_id: String,
        session_id: String,
        run_id: String,
        step_id: String,
        attempt: i32,
        invocation_id: Option<String>,
    ) -> PyResult<Self> {
        if let Some(client) = runtime_client() {
            let mut config = ContextRuntimeConfig::new(
                Arc::clone(client),
                tenant_id,
                session_id,
                run_id,
                step_id,
                attempt,
            );
            if let Some(invocation_id) = invocation_id {
                config = config.with_invocation_id(invocation_id);
            }
            Ok(Self {
                inner: ContextHandle::new_runtime(config),
            })
        } else {
            Ok(Self {
                inner: ContextHandle::new_placeholder(),
            })
        }
    }

    fn runtime(&self) -> PyRuntimeControls {
        PyRuntimeControls {
            inner: self.inner.runtime(),
        }
    }

    fn signals(&self) -> PySignalControls {
        PySignalControls {
            inner: self.inner.signals(),
        }
    }

    fn tasks(&self) -> PyTaskControls {
        PyTaskControls {
            inner: self.inner.tasks(),
        }
    }

    fn timers(&self) -> PyTimerControls {
        PyTimerControls {
            inner: self.inner.timers(),
        }
    }

    fn utils(&self) -> PyDeterministicUtils {
        PyDeterministicUtils {
            inner: self.inner.utils(),
        }
    }
}

fn map_sdk_error(err: SdkError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
#[pyfunction]
fn register_tool(
    name: String,
    description: Option<String>,
    input_schema: Option<String>,
    output_schema: Option<String>,
) -> PyResult<()> {
    let registry = global_tool_registry();
    let definition = ToolDefinition {
        name,
        description,
        input_schema,
        output_schema,
    };
    registry
        .register(definition)
        .map(|_| ())
        .map_err(map_sdk_error)
}

#[pyfunction]
fn get_registered_tools() -> PyResult<Vec<PyToolDefinition>> {
    let registry = global_tool_registry();
    registry
        .list()
        .map(|defs| defs.into_iter().map(PyToolDefinition::from).collect())
        .map_err(map_sdk_error)
}

#[pyfunction]
fn clear_tools() -> PyResult<()> {
    global_tool_registry().clear().map_err(map_sdk_error)
}

#[pyclass(name = "RuntimeControls")]
#[derive(Clone)]
pub struct PyRuntimeControls {
    inner: RuntimeControls,
}

#[pymethods]
impl PyRuntimeControls {
    #[pyo3(signature = (note=None))]
    fn checkpoint(&self, note: Option<&str>) -> PyResult<()> {
        self.inner.checkpoint(note).map_err(map_sdk_error)
    }

    fn fail(&self, reason: &str) -> PyResult<()> {
        self.inner.fail(reason).map_err(map_sdk_error)
    }
}

#[pyclass(name = "SignalControls")]
#[derive(Clone)]
pub struct PySignalControls {
    inner: SignalControls,
}

#[pymethods]
impl PySignalControls {
    #[pyo3(signature = (name, correlation=None))]
    fn wait(&self, name: &str, correlation: Option<&str>) -> PyResult<()> {
        self.inner.wait(name, correlation).map_err(map_sdk_error)
    }
}

#[pyclass(name = "TaskControls")]
#[derive(Clone)]
pub struct PyTaskControls {
    inner: TaskControls,
}

#[pymethods]
impl PyTaskControls {
    fn spawn(&self, target: &str) -> PyResult<()> {
        self.inner.spawn(target).map_err(map_sdk_error)
    }
}

#[pyclass(name = "TimerControls")]
#[derive(Clone)]
pub struct PyTimerControls {
    inner: TimerControls,
}

#[pymethods]
impl PyTimerControls {
    fn sleep(&self, seconds: f64) -> PyResult<()> {
        if seconds.is_sign_negative() {
            return Err(PyRuntimeError::new_err(
                "sleep duration must be non-negative",
            ));
        }
        let duration = std::time::Duration::from_secs_f64(seconds);
        self.inner.sleep(duration).map_err(map_sdk_error)
    }
}

#[pyclass(name = "DeterministicUtils")]
#[derive(Clone)]
pub struct PyDeterministicUtils {
    inner: DeterministicUtils,
}

#[pyclass(name = "ToolRegistry")]
#[derive(Clone)]
pub struct PyToolRegistry {
    inner: Arc<ToolRegistry>,
}

#[pymethods]
impl PyToolRegistry {
    #[new]
    fn new() -> Self {
        Self {
            inner: ToolRegistry::new(),
        }
    }

    fn register(
        &self,
        name: String,
        description: Option<String>,
        input_schema: Option<String>,
        output_schema: Option<String>,
    ) -> PyResult<()> {
        let definition = ToolDefinition {
            name,
            description,
            input_schema,
            output_schema,
        };
        self.inner
            .register(definition)
            .map(|_| ())
            .map_err(map_sdk_error)
    }

    fn list(&self) -> PyResult<Vec<PyToolDefinition>> {
        self.inner
            .list()
            .map(|defs| defs.into_iter().map(PyToolDefinition::from).collect())
            .map_err(map_sdk_error)
    }

    fn clear(&self) -> PyResult<()> {
        self.inner.clear().map_err(map_sdk_error)
    }
}

#[pyclass(name = "ToolDefinition")]
#[derive(Clone)]
pub struct PyToolDefinition {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub description: Option<String>,
    #[pyo3(get)]
    pub input_schema: Option<String>,
    #[pyo3(get)]
    pub output_schema: Option<String>,
}

impl From<ToolDefinition> for PyToolDefinition {
    fn from(def: ToolDefinition) -> Self {
        Self {
            name: def.name,
            description: def.description,
            input_schema: def.input_schema,
            output_schema: def.output_schema,
        }
    }
}

#[pymethods]
impl PyDeterministicUtils {
    fn now(&self) -> PyResult<i64> {
        self.inner.now().map_err(map_sdk_error)
    }

    fn rand(&self) -> PyResult<f64> {
        self.inner.rand().map_err(map_sdk_error)
    }
}
static RUNTIME_CLIENT: OnceLock<Arc<RuntimeServiceClient>> = OnceLock::new();

fn runtime_client() -> Option<&'static Arc<RuntimeServiceClient>> {
    if let Some(client) = RUNTIME_CLIENT.get() {
        return Some(client);
    }

    match connect_runtime_client() {
        Ok(client) => {
            if RUNTIME_CLIENT.set(client).is_ok() {
                RUNTIME_CLIENT.get()
            } else {
                RUNTIME_CLIENT.get()
            }
        }
        Err(err) => {
            log::warn!("Runtime service client unavailable: {}", err);
            None
        }
    }
}

fn connect_runtime_client() -> Result<Arc<RuntimeServiceClient>, SdkError> {
    let fut = RuntimeServiceClient::connect_from_env();
    let client = if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // Use block_in_place to avoid "Cannot start a runtime from within a runtime" panic
        tokio::task::block_in_place(|| handle.block_on(fut))
    } else {
        tokio::runtime::Runtime::new()
            .map_err(|e| SdkError::Internal(format!("create runtime: {}", e)))?
            .block_on(fut)
    }?;

    Ok(Arc::new(client))
}
