use std::collections::HashMap;
use std::sync::Arc;

use agnt5_sdk_core::mcp::{McpClient, ServerConfig, SseConfig, StdioConfig, ToolContent};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use serde_json::{json, Value};
use tokio::sync::Mutex;

#[pyclass(name = "MCPClientCore")]
pub struct PyMcpClientCore {
    inner: Arc<Mutex<McpClient>>,
}

#[pymethods]
impl PyMcpClientCore {
    #[new]
    fn new(id: String) -> Self {
        Self {
            inner: Arc::new(Mutex::new(McpClient::new(id))),
        }
    }

    fn add_server(&self, name: String, config: &Bound<'_, PyAny>) -> PyResult<()> {
        let config_json = python_to_json(config)?;
        let server_config = parse_server_config(config_json)?;
        let mut client = self.inner.blocking_lock();
        client.add_server(name, server_config);
        Ok(())
    }

    #[pyo3(signature = (name, command, args=None, env=None, cwd=None))]
    fn add_stdio_server(
        &self,
        name: String,
        command: String,
        args: Option<Vec<String>>,
        env: Option<HashMap<String, String>>,
        cwd: Option<String>,
    ) {
        let mut client = self.inner.blocking_lock();
        client.add_server(
            name,
            ServerConfig::Stdio(StdioConfig {
                command,
                args: args.unwrap_or_default(),
                env: env.unwrap_or_default(),
                cwd,
            }),
        );
    }

    #[pyo3(signature = (name, url, headers=None, api_key=None))]
    fn add_sse_server(
        &self,
        name: String,
        url: String,
        headers: Option<HashMap<String, String>>,
        api_key: Option<String>,
    ) {
        let mut config = SseConfig::new(url);
        for (key, value) in headers.unwrap_or_default() {
            config = config.with_header(key, value);
        }
        if let Some(api_key) = api_key {
            config = config.with_api_key(api_key);
        }

        let mut client = self.inner.blocking_lock();
        client.add_server(name, ServerConfig::Sse(config));
    }

    fn connect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let client = inner.lock().await;
            client
                .connect()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }

    fn disconnect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let client = inner.lock().await;
            client
                .disconnect()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }

    fn list_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let client = inner.lock().await;
            let tools = client
                .list_tools()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let result = Python::with_gil(|py| -> PyResult<PyObject> {
                let items = tools
                    .into_iter()
                    .map(|item| {
                        json!({
                            "server": item.server,
                            "tool": {
                                "name": item.tool.name,
                                "description": item.tool.description,
                                "input_schema": item.tool.input_schema,
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(json_to_python(py, &Value::Array(items))?)
            })?;

            Ok(result)
        })
    }

    fn list_server_tools<'py>(&self, py: Python<'py>, server: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        future_into_py(py, async move {
            let client = inner.lock().await;
            let tools = client
                .list_server_tools(&server)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let result = Python::with_gil(|py| -> PyResult<PyObject> {
                let items = tools
                    .into_iter()
                    .map(|tool| {
                        json!({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        })
                    })
                    .collect::<Vec<_>>();
                Ok(json_to_python(py, &Value::Array(items))?)
            })?;

            Ok(result)
        })
    }

    #[pyo3(signature = (server, tool_name, arguments=None))]
    fn call_tool<'py>(
        &self,
        py: Python<'py>,
        server: String,
        tool_name: String,
        arguments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let arguments = arguments.map(python_to_json).transpose()?.unwrap_or(Value::Null);
        future_into_py(py, async move {
            let client = inner.lock().await;
            let result = client
                .call_tool(&server, &tool_name, arguments)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let result = Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(json_to_python(py, &call_tool_result_to_json(&result))?)
            })?;

            Ok(result)
        })
    }

    #[pyo3(signature = (tool_name, arguments=None))]
    fn call_tool_auto<'py>(
        &self,
        py: Python<'py>,
        tool_name: String,
        arguments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let arguments = arguments.map(python_to_json).transpose()?.unwrap_or(Value::Null);
        future_into_py(py, async move {
            let client = inner.lock().await;
            let result = client
                .call_tool_auto(&tool_name, arguments)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let result = Python::with_gil(|py| -> PyResult<PyObject> {
                Ok(json_to_python(py, &call_tool_result_to_json(&result))?)
            })?;

            Ok(result)
        })
    }

}

pub fn register_mcp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMcpClientCore>()?;
    Ok(())
}

fn parse_server_config(config: Value) -> PyResult<ServerConfig> {
    let obj = config
        .as_object()
        .ok_or_else(|| PyValueError::new_err("server config must be a dict"))?;

    if let Some(command) = obj.get("command").and_then(Value::as_str) {
        let args = obj
            .get("args")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let env = obj
            .get("env")
            .and_then(Value::as_object)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        let cwd = obj.get("cwd").and_then(Value::as_str).map(ToOwned::to_owned);

        return Ok(ServerConfig::Stdio(StdioConfig {
            command: command.to_string(),
            args,
            env,
            cwd,
        }));
    }

    if let Some(url) = obj.get("url").and_then(Value::as_str) {
        let mut config = SseConfig::new(url.to_string());
        if let Some(headers) = obj.get("headers").and_then(Value::as_object) {
            for (key, value) in headers {
                if let Some(value) = value.as_str() {
                    config = config.with_header(key.clone(), value.to_string());
                }
            }
        }
        return Ok(ServerConfig::Sse(config));
    }

    Err(PyValueError::new_err(
        "Invalid server config: must have 'command' or 'url'",
    ))
}

fn call_tool_result_to_json(result: &agnt5_sdk_core::mcp::CallToolResult) -> Value {
    let content = result
        .content
        .iter()
        .map(tool_content_to_json)
        .collect::<Vec<_>>();
    json!({
        "content": content,
        "is_error": result.is_error,
    })
}

fn tool_content_to_json(content: &ToolContent) -> Value {
    match content {
        ToolContent::Text { text } => json!({ "type": "text", "text": text }),
        ToolContent::Image { data, mime_type } => {
            json!({ "type": "image", "data": data, "mime_type": mime_type })
        }
        ToolContent::Resource { resource } => json!({
            "type": "resource",
            "resource": {
                "uri": resource.uri,
                "mime_type": resource.mime_type,
                "text": resource.text,
            }
        }),
    }
}

fn python_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = value.py();
    let json_module = py.import("json")?;
    let dumped: String = json_module.call_method1("dumps", (value,))?.extract()?;
    serde_json::from_str(&dumped).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn json_to_python(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    let json_module = py.import("json")?;
    let dumped = serde_json::to_string(value).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let loaded = json_module.call_method1("loads", (dumped,))?;
    Ok(loaded.unbind())
}
