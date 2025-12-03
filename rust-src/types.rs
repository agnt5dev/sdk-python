use agnt5_sdk_core::pb::{
    execute_component_response, ComponentInfo, ComponentSchema, ComponentType,
    ExecuteComponentRequest, ExecuteComponentResponse, StateTransition, StateUpdate,
    StepCheckpoint,
};
use pyo3::prelude::*;
use serde_json;
use std::collections::HashMap;

/// Convert a JSON schema string to ComponentSchema proto
fn parse_json_schema(json_str: &str) -> Option<ComponentSchema> {
    let json_value: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(val) => val,
        Err(_) => return None,
    };

    json_value_to_component_schema(&json_value)
}

/// Recursively convert a serde_json::Value to ComponentSchema
fn json_value_to_component_schema(json_value: &serde_json::Value) -> Option<ComponentSchema> {
    match json_value {
        serde_json::Value::Object(obj) => {
            let mut schema = ComponentSchema::default();

            // Set type
            if let Some(type_val) = obj.get("type") {
                if let serde_json::Value::String(type_str) = type_val {
                    schema.r#type = type_str.clone();
                }
            }

            // Set description
            if let Some(desc_val) = obj.get("description") {
                if let serde_json::Value::String(desc_str) = desc_val {
                    schema.description = Some(desc_str.clone());
                }
            }

            // Set format
            if let Some(format_val) = obj.get("format") {
                if let serde_json::Value::String(format_str) = format_val {
                    schema.format = Some(format_str.clone());
                }
            }

            // Set properties
            if let Some(props_val) = obj.get("properties") {
                if let serde_json::Value::Object(props_obj) = props_val {
                    for (key, value) in props_obj {
                        if let Some(prop_schema) = json_value_to_component_schema(value) {
                            schema.properties.insert(key.clone(), prop_schema);
                        }
                    }
                }
            }

            // Set required
            if let Some(req_val) = obj.get("required") {
                if let serde_json::Value::Array(req_array) = req_val {
                    for item in req_array {
                        if let serde_json::Value::String(req_str) = item {
                            schema.required.push(req_str.clone());
                        }
                    }
                }
            }

            // Set items (for arrays)
            if let Some(items_val) = obj.get("items") {
                if let Some(items_schema) = json_value_to_component_schema(items_val) {
                    schema.items = Some(Box::new(items_schema));
                }
            }

            // Set enum_values
            if let Some(enum_val) = obj.get("enum") {
                if let serde_json::Value::Array(enum_array) = enum_val {
                    for item in enum_array {
                        if let serde_json::Value::String(enum_str) = item {
                            schema.enum_values.push(enum_str.clone());
                        }
                    }
                }
            }

            Some(schema)
        }
        _ => None,
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyStepCheckpoint {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub key: Option<String>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub attempt: i32,
    #[pyo3(get)]
    pub updated_at: Option<String>,
    #[pyo3(get)]
    pub result: Vec<u8>,
}

#[pymethods]
impl PyStepCheckpoint {
    #[new]
    fn new(
        name: String,
        key: Option<String>,
        status: String,
        attempt: i32,
        updated_at: Option<String>,
        result: Option<Vec<u8>>,
    ) -> Self {
        Self {
            name,
            key,
            status,
            attempt,
            updated_at,
            result: result.unwrap_or_default(),
        }
    }
}

impl From<StepCheckpoint> for PyStepCheckpoint {
    fn from(checkpoint: StepCheckpoint) -> Self {
        Self {
            name: checkpoint.name,
            key: if checkpoint.key.is_empty() {
                None
            } else {
                Some(checkpoint.key)
            },
            status: checkpoint.status,
            attempt: checkpoint.attempt,
            updated_at: if checkpoint.updated_at.is_empty() {
                None
            } else {
                Some(checkpoint.updated_at)
            },
            result: checkpoint.result,
        }
    }
}

impl From<PyStepCheckpoint> for StepCheckpoint {
    fn from(py_checkpoint: PyStepCheckpoint) -> Self {
        Self {
            name: py_checkpoint.name,
            key: py_checkpoint.key.unwrap_or_default(),
            status: py_checkpoint.status,
            attempt: py_checkpoint.attempt,
            updated_at: py_checkpoint.updated_at.unwrap_or_default(),
            result: py_checkpoint.result,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyStateTransition {
    #[pyo3(get)]
    pub operation: String,
    #[pyo3(get)]
    pub key: String,
    #[pyo3(get)]
    pub old_value: Vec<u8>,
    #[pyo3(get)]
    pub new_value: Vec<u8>,
    #[pyo3(get)]
    pub timestamp: i64,
}

#[pymethods]
impl PyStateTransition {
    #[new]
    fn new(
        operation: String,
        key: String,
        old_value: Option<Vec<u8>>,
        new_value: Option<Vec<u8>>,
        timestamp: i64,
    ) -> Self {
        Self {
            operation,
            key,
            old_value: old_value.unwrap_or_default(),
            new_value: new_value.unwrap_or_default(),
            timestamp,
        }
    }
}

impl From<PyStateTransition> for StateTransition {
    fn from(transition: PyStateTransition) -> Self {
        Self {
            operation: transition.operation,
            key: transition.key,
            old_value: transition.old_value,
            new_value: transition.new_value,
            timestamp: transition.timestamp,
        }
    }
}

impl From<StateTransition> for PyStateTransition {
    fn from(transition: StateTransition) -> Self {
        Self {
            operation: transition.operation,
            key: transition.key,
            old_value: transition.old_value,
            new_value: transition.new_value,
            timestamp: transition.timestamp,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyStateUpdate {
    #[pyo3(get)]
    pub new_state: Vec<u8>,
    #[pyo3(get)]
    pub transitions: Vec<PyStateTransition>,
    #[pyo3(get)]
    pub output_data: Vec<u8>,
}

#[pymethods]
impl PyStateUpdate {
    #[new]
    fn new(
        new_state: Vec<u8>,
        transitions: Option<Vec<PyStateTransition>>,
        output_data: Option<Vec<u8>>,
    ) -> Self {
        Self {
            new_state,
            transitions: transitions.unwrap_or_default(),
            output_data: output_data.unwrap_or_default(),
        }
    }
}

impl From<PyStateUpdate> for StateUpdate {
    fn from(update: PyStateUpdate) -> Self {
        Self {
            new_state: update.new_state,
            transitions: update
                .transitions
                .into_iter()
                .map(StateTransition::from)
                .collect(),
            output_data: update.output_data,
        }
    }
}

impl From<StateUpdate> for PyStateUpdate {
    fn from(update: StateUpdate) -> Self {
        Self {
            new_state: update.new_state,
            transitions: update
                .transitions
                .into_iter()
                .map(PyStateTransition::from)
                .collect(),
            output_data: update.output_data,
        }
    }
}

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
    pub component_type: String,
    #[pyo3(get)]
    pub object_id: Option<String>,
    #[pyo3(get)]
    pub method_name: Option<String>,
    #[pyo3(get)]
    pub state_snapshot: Vec<u8>,
    #[pyo3(get)]
    pub journal_position: i64,
    #[pyo3(get)]
    pub flow_instance_id: Option<String>,
    #[pyo3(get)]
    pub flow_step: Option<i32>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    #[pyo3(get)]
    pub step_checkpoints: Vec<PyStepCheckpoint>,
    #[pyo3(get)]
    pub runtime_context: Option<crate::PyRuntimeContext>,
    #[pyo3(get)]
    pub session_id: Option<String>,
    #[pyo3(get)]
    pub user_id: Option<String>,
    // Retry orchestration
    #[pyo3(get)]
    pub attempt: i32,
    // Streaming execution flag - when true, SDK should send spans/logs to journal for real-time SSE
    #[pyo3(get)]
    pub is_streaming: bool,
}

#[pymethods]
impl PyExecuteComponentRequest {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        invocation_id: String,
        service_name: String,
        component_name: String,
        input_data: Vec<u8>,
        component_type: Option<String>,
        object_id: Option<String>,
        method_name: Option<String>,
        state_snapshot: Option<Vec<u8>>,
        journal_position: Option<i64>,
        metadata: Option<HashMap<String, String>>,
        step_checkpoints: Option<Vec<PyStepCheckpoint>>,
        flow_instance_id: Option<String>,
        flow_step: Option<i32>,
    ) -> Self {
        Self {
            invocation_id,
            service_name,
            component_name,
            input_data,
            component_type: component_type.unwrap_or_else(|| "function".to_string()),
            object_id,
            method_name,
            state_snapshot: state_snapshot.unwrap_or_default(),
            journal_position: journal_position.unwrap_or_default(),
            flow_instance_id,
            flow_step,
            metadata: metadata.unwrap_or_default(),
            step_checkpoints: step_checkpoints.unwrap_or_default(),
            runtime_context: None,
            session_id: None,
            user_id: None,
            attempt: 0,
            is_streaming: false,
        }
    }
}

impl From<ExecuteComponentRequest> for PyExecuteComponentRequest {
    fn from(req: ExecuteComponentRequest) -> Self {
        let component_type =
            match ComponentType::try_from(req.component_type).unwrap_or(ComponentType::Function) {
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
            }
            .to_string();

        Self {
            invocation_id: req.invocation_id,
            service_name: req.service_name,
            component_name: req.component_name,
            input_data: req.input_data,
            component_type,
            object_id: if req.object_id.is_empty() {
                None
            } else {
                Some(req.object_id)
            },
            method_name: if req.method_name.is_empty() {
                None
            } else {
                Some(req.method_name)
            },
            state_snapshot: req.state_snapshot,
            journal_position: req.journal_position,
            flow_instance_id: if req.flow_instance_id.is_empty() {
                None
            } else {
                Some(req.flow_instance_id)
            },
            flow_step: if req.flow_step == 0 {
                None
            } else {
                Some(req.flow_step)
            },
            metadata: req.metadata,
            step_checkpoints: req
                .step_checkpoints
                .into_iter()
                .map(PyStepCheckpoint::from)
                .collect(),
            runtime_context: None, // Will be set in worker.rs after context extraction
            session_id: if req.session_id.is_empty() {
                None
            } else {
                Some(req.session_id)
            },
            user_id: if req.user_id.is_empty() {
                None
            } else {
                Some(req.user_id)
            },
            attempt: req.attempt,
            is_streaming: req.is_streaming,
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
    pub state_update: Option<PyStateUpdate>,
    #[pyo3(get)]
    pub error_message: Option<String>,
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    // Streaming support (v1.1)
    #[pyo3(get)]
    pub is_chunk: bool,
    #[pyo3(get)]
    pub done: bool,
    #[pyo3(get)]
    pub chunk_index: i64,
    // Retry orchestration
    #[pyo3(get)]
    pub attempt: i32,
}

#[pymethods]
impl PyExecuteComponentResponse {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        invocation_id: String,
        success: bool,
        output_data: Vec<u8>,
        state_update: Option<PyStateUpdate>,
        error_message: Option<String>,
        metadata: Option<HashMap<String, String>>,
        is_chunk: Option<bool>,
        done: Option<bool>,
        chunk_index: Option<i64>,
        attempt: Option<i32>,
    ) -> Self {
        Self {
            invocation_id,
            success,
            output_data,
            state_update,
            error_message,
            metadata: metadata.unwrap_or_default(),
            is_chunk: is_chunk.unwrap_or(false),
            done: done.unwrap_or(true),
            chunk_index: chunk_index.unwrap_or(0),
            attempt: attempt.unwrap_or(0),
        }
    }
}

impl From<PyExecuteComponentResponse> for ExecuteComponentResponse {
    fn from(resp: PyExecuteComponentResponse) -> Self {
        let result = if resp.success {
            if let Some(update) = resp.state_update.clone() {
                Some(execute_component_response::Result::StateUpdate(
                    update.into(),
                ))
            } else {
                Some(execute_component_response::Result::OutputData(
                    resp.output_data.clone(),
                ))
            }
        } else {
            None
        };

        Self {
            invocation_id: resp.invocation_id,
            success: resp.success,
            result,
            error_message: resp.error_message.unwrap_or_default(),
            metadata: resp.metadata,
            is_chunk: resp.is_chunk,
            done: resp.done,
            chunk_index: resp.chunk_index,
            attempt: resp.attempt,
        }
    }
}

impl From<ExecuteComponentResponse> for PyExecuteComponentResponse {
    fn from(resp: ExecuteComponentResponse) -> Self {
        let (output_data, state_update) = match resp.result {
            Some(execute_component_response::Result::OutputData(data)) => (data, None),
            Some(execute_component_response::Result::StateUpdate(update)) => {
                (Vec::new(), Some(PyStateUpdate::from(update)))
            }
            Some(execute_component_response::Result::FlowContinuation(_)) => (Vec::new(), None),
            None => (Vec::new(), None),
        };

        Self {
            invocation_id: resp.invocation_id,
            success: resp.success,
            output_data,
            state_update,
            error_message: if resp.error_message.is_empty() {
                None
            } else {
                Some(resp.error_message)
            },
            metadata: resp.metadata,
            is_chunk: resp.is_chunk,
            done: resp.done,
            chunk_index: resp.chunk_index,
            attempt: resp.attempt,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyComponentInfo {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub component_type: String,
    #[pyo3(get, set)]
    pub metadata: HashMap<String, String>,
    #[pyo3(get, set)]
    pub config: HashMap<String, String>,
    #[pyo3(get, set)]
    pub input_schema: Option<String>, // JSON string
    #[pyo3(get, set)]
    pub output_schema: Option<String>, // JSON string
    #[pyo3(get, set)]
    pub definition: Option<String>, // Source code or definition
}

#[pymethods]
impl PyComponentInfo {
    #[new]
    fn new(
        name: String,
        component_type: String,
        metadata: Option<HashMap<String, String>>,
        config: Option<HashMap<String, String>>,
        input_schema: Option<String>,
        output_schema: Option<String>,
        definition: Option<String>,
    ) -> Self {
        Self {
            name,
            component_type,
            metadata: metadata.unwrap_or_default(),
            config: config.unwrap_or_default(),
            input_schema,
            output_schema,
            definition,
        }
    }
}

impl From<PyComponentInfo> for ComponentInfo {
    fn from(comp: PyComponentInfo) -> Self {
        let component_type = match comp.component_type.as_str() {
            "function" => ComponentType::Function as i32,
            "flow" => ComponentType::Flow as i32,
            "object" => ComponentType::Object as i32,
            "entity" => ComponentType::Entity as i32,
            "task" => ComponentType::Task as i32,
            "workflow" => ComponentType::Workflow as i32,
            "agent" => ComponentType::Agent as i32,
            "tool" => ComponentType::Tool as i32,
            "mcp" => ComponentType::Mcp as i32,
            _ => ComponentType::Unspecified as i32,
        };

        // Parse input schema if provided
        let input_schema = comp
            .input_schema
            .as_ref()
            .and_then(|json_str| parse_json_schema(json_str));

        // Parse output schema if provided
        let output_schema = comp
            .output_schema
            .as_ref()
            .and_then(|json_str| parse_json_schema(json_str));

        // Parse retry configuration from config dict
        let max_attempts = comp
            .config
            .get("max_attempts")
            .and_then(|v| v.parse::<i32>().ok());
        let initial_interval_ms = comp
            .config
            .get("initial_interval_ms")
            .and_then(|v| v.parse::<i32>().ok());
        let max_interval_ms = comp
            .config
            .get("max_interval_ms")
            .and_then(|v| v.parse::<i32>().ok());
        let backoff_type = comp.config.get("backoff_type").cloned();
        let backoff_multiplier = comp
            .config
            .get("backoff_multiplier")
            .and_then(|v| v.parse::<f64>().ok());

        Self {
            name: comp.name,
            component_type,
            input_schema,
            output_schema,
            config: comp.config,
            metadata: comp.metadata,
            definition: comp.definition,
            max_attempts,
            initial_interval_ms,
            max_interval_ms,
            backoff_type,
            backoff_multiplier,
        }
    }
}
