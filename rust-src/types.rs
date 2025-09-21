use agnt5_sdk_core::pb::{
    ComponentInfo, ComponentType, ExecuteComponentRequest, ExecuteComponentResponse,
};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct PyExecuteComponentRequest {
    #[pyo3(get)]
    pub invocation_id: String,
    #[pyo3(get)]
    pub service_name: String,
    #[pyo3(get)]
    pub component_name: String,
    #[pyo3(get)]
    pub input_data: Vec<u8>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyExecuteComponentRequest {
    #[new]
    fn new(
        invocation_id: String,
        service_name: String,
        component_name: String,
        input_data: Vec<u8>,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            invocation_id,
            service_name,
            component_name,
            input_data,
            metadata: metadata.unwrap_or_default(),
        }
    }
}

impl From<ExecuteComponentRequest> for PyExecuteComponentRequest {
    fn from(req: ExecuteComponentRequest) -> Self {
        Self {
            invocation_id: req.invocation_id,
            service_name: req.service_name,
            component_name: req.component_name,
            input_data: req.input_data,
            metadata: req.metadata,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyExecuteComponentResponse {
    #[pyo3(get)]
    pub invocation_id: String,
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub output_data: Vec<u8>,
    #[pyo3(get)]
    pub error_message: Option<String>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyExecuteComponentResponse {
    #[new]
    fn new(
        invocation_id: String,
        success: bool,
        output_data: Vec<u8>,
        error_message: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            invocation_id,
            success,
            output_data,
            error_message,
            metadata: metadata.unwrap_or_default(),
        }
    }
}

impl From<PyExecuteComponentResponse> for ExecuteComponentResponse {
    fn from(resp: PyExecuteComponentResponse) -> Self {
        let result = if resp.success {
            Some(
                agnt5_sdk_core::pb::execute_component_response::Result::OutputData(
                    resp.output_data,
                ),
            )
        } else {
            None
        };

        Self {
            invocation_id: resp.invocation_id,
            success: resp.success,
            result,
            error_message: resp.error_message.unwrap_or_default(),
            metadata: resp.metadata,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyComponentInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub component_type: String,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
}

#[pymethods]
impl PyComponentInfo {
    #[new]
    fn new(
        name: String,
        component_type: String,
        metadata: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            name,
            component_type,
            metadata: metadata.unwrap_or_default(),
        }
    }
}

impl From<PyComponentInfo> for ComponentInfo {
    fn from(comp: PyComponentInfo) -> Self {
        let component_type = match comp.component_type.as_str() {
            "function" => ComponentType::Function as i32,
            "flow" => ComponentType::Flow as i32,
            "object" => ComponentType::Object as i32,
            "task" => ComponentType::Task as i32,
            "workflow" => ComponentType::Workflow as i32,
            "agent" => ComponentType::Agent as i32,
            "tool" => ComponentType::Tool as i32,
            "mcp" => ComponentType::Mcp as i32,
            _ => ComponentType::Unspecified as i32,
        };

        Self {
            name: comp.name,
            component_type,
            input_schema: None,
            output_schema: None,
            config: HashMap::new(),
            metadata: comp.metadata,
        }
    }
}
