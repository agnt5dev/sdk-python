use std::collections::HashMap;
use std::env;
use std::sync::Mutex;

use agnt5_sdk_core::error::{Result as SdkResult, SdkError};
use opentelemetry::Context as OtelContext;
use agnt5_sdk_core::lm::{
    AnthropicProvider, AzureOpenAiProvider, BedrockProvider, GenerateRequest, GenerateResponse,
    GenerationConfig, GroqProvider, JsonSchemaFormat, Message, MessageRole, OpenAiProvider,
    OpenRouterProvider, ResponseFormat, StreamChunk, StreamHandle, StreamRequest, TokenUsage,
    ToolChoice, ToolDefinition,
};
use futures::StreamExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList};
use pyo3_async_runtimes::TaskLocals;
use serde::Deserialize;
use serde_json::{self, Value};

/// Internal configuration representation for the Python wrapper.
#[derive(Clone, Default)]
pub(crate) struct ConfigData {
    default_model: Option<String>,
    default_provider: Option<String>,
}

/// Python wrapper for language model configuration.
#[pyclass(name = "LanguageModelConfig")]
pub struct PyLanguageModelConfig {
    pub(crate) inner: ConfigData,
}

impl Clone for PyLanguageModelConfig {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[pymethods]
impl PyLanguageModelConfig {
    #[new]
    #[pyo3(signature = (default_model=None, default_provider=None))]
    fn new(default_model: Option<String>, default_provider: Option<String>) -> Self {
        Self {
            inner: ConfigData {
                default_model,
                default_provider,
            },
        }
    }

    #[staticmethod]
    fn from_environment() -> PyResult<Self> {
        let default_model = env::var("LM_MODEL")
            .or_else(|_| env::var("LM_DEFAULT_MODEL"))
            .ok();
        let default_provider = env::var("LM_PROVIDER").ok();

        Ok(Self {
            inner: ConfigData {
                default_model,
                default_provider,
            },
        })
    }
}

#[derive(Clone)]
enum ProviderKind {
    OpenAi(OpenAiProvider),
    Azure(AzureOpenAiProvider),
    Bedrock(BedrockProvider),
    Anthropic(AnthropicProvider),
    Groq(GroqProvider),
    OpenRouter(OpenRouterProvider),
}

impl ProviderKind {
    async fn generate(&self, request: GenerateRequest) -> SdkResult<GenerateResponse> {
        match self {
            ProviderKind::OpenAi(provider) => provider.generate(request).await,
            ProviderKind::Azure(provider) => provider.generate(request).await,
            ProviderKind::Bedrock(provider) => provider.generate(request).await,
            ProviderKind::Anthropic(provider) => provider.generate(request).await,
            ProviderKind::Groq(provider) => provider.generate(request).await,
            ProviderKind::OpenRouter(provider) => provider.generate(request).await,
        }
    }

    async fn stream(&self, request: StreamRequest) -> SdkResult<StreamHandle> {
        match self {
            ProviderKind::OpenAi(provider) => provider.stream(request).await,
            ProviderKind::Azure(provider) => provider.stream(request).await,
            ProviderKind::Bedrock(provider) => provider.stream(request).await,
            ProviderKind::Anthropic(provider) => provider.stream(request).await,
            ProviderKind::Groq(provider) => provider.stream(request).await,
            ProviderKind::OpenRouter(provider) => provider.stream(request).await,
        }
    }
}

/// Python wrapper for the simplified LanguageModel API.
#[pyclass(name = "LanguageModel")]
pub struct PyLanguageModel {
    default_model: Option<String>,
    default_provider: Option<String>,
    providers: Mutex<HashMap<String, ProviderKind>>,
}

#[pymethods]
impl PyLanguageModel {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyLanguageModelConfig>) -> PyResult<Self> {
        let (default_model, default_provider) = config
            .map(|cfg| {
                (
                    cfg.inner.default_model.clone(),
                    cfg.inner.default_provider.clone(),
                )
            })
            .unwrap_or((None, None));

        Ok(Self {
            default_model,
            default_provider,
            providers: Mutex::new(HashMap::new()),
        })
    }

    #[pyo3(signature = (prompt, **kwargs))]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt: PyObject,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs_ref = kwargs.as_ref();

        let model_kw = get_optional_string(kwargs_ref, "model")?;
        let provider_kw = get_optional_string(kwargs_ref, "provider")?;
        let system_prompt_kw = get_optional_string(kwargs_ref, "system_prompt")?;
        let temperature_kw = get_optional_f32(kwargs_ref, "temperature")?;
        let top_p_kw = get_optional_f32(kwargs_ref, "top_p")?;
        let max_tokens_kw = get_optional_u32(kwargs_ref, "max_tokens")?;
        let response_format_kw = get_optional_string(kwargs_ref, "response_format")?;
        let response_schema_kw = get_optional_string(kwargs_ref, "response_schema")?;
        let tools_kw = get_optional_string(kwargs_ref, "tools")?;
        let tool_choice_kw = get_optional_string(kwargs_ref, "tool_choice")?;
        let response_format =
            parse_response_format(response_format_kw.as_deref(), response_schema_kw.as_deref())?;
        let tools = parse_tools_json(tools_kw.as_deref())?;
        let tool_choice = parse_tool_choice_json(tool_choice_kw.as_deref())?;
        let user_kw = get_optional_string(kwargs_ref, "user")?;

        let (model, provider_name) = self.resolve_model_and_provider(model_kw, provider_kw)?;

        let mut request = build_request(
            py,
            &model,
            &prompt,
            system_prompt_kw,
            temperature_kw,
            top_p_kw,
            max_tokens_kw,
            response_format,
            user_kw,
        )?;

        if let Some(tool_defs) = tools {
            request = request.tools(tool_defs);
        }

        if let Some(choice) = tool_choice {
            request = request.tool_choice(Some(choice));
        }

        let provider = self.get_or_init_provider(&provider_name)?;

        // IMPORTANT: Extract trace context from Python's contextvar
        // The Rust worker injects traceparent into request.metadata, and Python worker
        // stores it in a contextvar. We read it here and reconstruct the OtelContext
        // to enable proper trace propagation across the async boundary.
        let otel_context = extract_context_from_python(py)?;

        // Use pyo3-async-runtimes with proper runtime context
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let mut request = request;
            request.otel_context = Some(otel_context);

            let response = provider.generate(request).await.map_err(sdk_error_to_py)?;
            Ok(PyResponse { inner: response })
        })
    }

    #[pyo3(signature = (prompt, **kwargs))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        prompt: PyObject,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs_ref = kwargs.as_ref();

        let model_kw = get_optional_string(kwargs_ref, "model")?;
        let provider_kw = get_optional_string(kwargs_ref, "provider")?;
        let system_prompt_kw = get_optional_string(kwargs_ref, "system_prompt")?;
        let temperature_kw = get_optional_f32(kwargs_ref, "temperature")?;
        let top_p_kw = get_optional_f32(kwargs_ref, "top_p")?;
        let max_tokens_kw = get_optional_u32(kwargs_ref, "max_tokens")?;
        let response_format_kw = get_optional_string(kwargs_ref, "response_format")?;
        let response_schema_kw = get_optional_string(kwargs_ref, "response_schema")?;
        let tools_kw = get_optional_string(kwargs_ref, "tools")?;
        let tool_choice_kw = get_optional_string(kwargs_ref, "tool_choice")?;
        let response_format =
            parse_response_format(response_format_kw.as_deref(), response_schema_kw.as_deref())?;
        let tools = parse_tools_json(tools_kw.as_deref())?;
        let tool_choice = parse_tool_choice_json(tool_choice_kw.as_deref())?;
        let user_kw = get_optional_string(kwargs_ref, "user")?;

        let (model, provider_name) = self.resolve_model_and_provider(model_kw, provider_kw)?;

        let mut request = build_request(
            py,
            &model,
            &prompt,
            system_prompt_kw,
            temperature_kw,
            top_p_kw,
            max_tokens_kw,
            response_format,
            user_kw,
        )?;

        if let Some(tool_defs) = tools {
            request = request.tools(tool_defs);
        }

        if let Some(choice) = tool_choice {
            request = request.tool_choice(Some(choice));
        }

        let provider = self.get_or_init_provider(&provider_name)?;
        let model_for_delta = model.clone();

        // IMPORTANT: Extract trace context from Python's contextvar
        // The Rust worker injects traceparent into request.metadata, and Python worker
        // stores it in a contextvar. We read it here and reconstruct the OtelContext
        // to enable proper trace propagation across the async boundary.
        let otel_context = extract_context_from_python(py)?;

        // Use pyo3-async-runtimes with proper runtime context for streaming
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let mut request = request;
            request.otel_context = Some(otel_context);

            let mut handle = provider.stream(request).await.map_err(sdk_error_to_py)?;
            let mut chunks = Vec::new();

            while let Some(item) = handle.next().await {
                let chunk = item.map_err(sdk_error_to_py)?;
                match chunk {
                    StreamChunk::Delta { content } => {
                        chunks.push(PyStreamChunk::delta(model_for_delta.clone(), content));
                    }
                    StreamChunk::Completed(response) => {
                        chunks.push(PyStreamChunk::from_response(response));
                    }
                }
            }

            Ok(chunks)
        })
    }

    fn list_providers(&self) -> Vec<String> {
        let mut providers = Vec::new();

        if env::var("OPENAI_API_KEY").is_ok() {
            providers.push("openai".to_string());
        }
        if env::var("AZURE_OPENAI_API_KEY").is_ok() && env::var("AZURE_OPENAI_ENDPOINT").is_ok() {
            providers.push("azure".to_string());
        }
        if env::var("AWS_ACCESS_KEY_ID").is_ok() && env::var("AWS_SECRET_ACCESS_KEY").is_ok() {
            providers.push("bedrock".to_string());
        }
        if env::var("ANTHROPIC_API_KEY").is_ok() {
            providers.push("anthropic".to_string());
        }
        if env::var("GROQ_API_KEY").is_ok() {
            providers.push("groq".to_string());
        }
        if env::var("OPENROUTER_API_KEY").is_ok() {
            providers.push("openrouter".to_string());
        }

        providers
    }
}

impl PyLanguageModel {
    fn resolve_model_and_provider(
        &self,
        model_kw: Option<String>,
        provider_kw: Option<String>,
    ) -> PyResult<(String, String)> {
        let model = model_kw
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| {
                PyValueError::new_err("A model must be provided via argument or configuration")
            })?;

        let mut provider = provider_kw.or_else(|| self.default_provider.clone());

        if provider.is_none() {
            if let Some((prefix, _)) = model.split_once('/') {
                provider = Some(prefix.to_string());
            }
        }

        let provider = provider.ok_or_else(|| {
            PyValueError::new_err(
                "Unable to determine provider. Pass `provider` explicitly or prefix the model (e.g. `openai/gpt-4o`).",
            )
        })?;

        // Gateway providers like OpenRouter can handle models with different prefixes
        // (e.g., openrouter provider with anthropic/claude-3.5-haiku model)
        // So we skip the validation for these gateway providers
        let gateway_providers = ["openrouter", "litellm"];
        let is_gateway = gateway_providers.contains(&provider.to_lowercase().as_str());

        if !is_gateway {
            if let Some((prefix, _)) = model.split_once('/') {
                if provider.to_lowercase() != prefix.to_lowercase() {
                    return Err(PyValueError::new_err(format!(
                        "Provider `{provider}` does not match model prefix `{prefix}`"
                    )));
                }
            }
        }

        Ok((model, provider.to_lowercase()))
    }

    fn get_or_init_provider(&self, provider: &str) -> PyResult<ProviderKind> {
        {
            let cache = self
                .providers
                .lock()
                .expect("provider cache mutex poisoned");
            if let Some(existing) = cache.get(provider) {
                return Ok(existing.clone());
            }
        }

        let created = instantiate_provider(provider).map_err(sdk_error_to_py)?;
        let mut cache = self
            .providers
            .lock()
            .expect("provider cache mutex poisoned");
        cache.insert(provider.to_string(), created.clone());
        Ok(created)
    }
}

fn instantiate_provider(provider: &str) -> SdkResult<ProviderKind> {
    match provider {
        "openai" => Ok(ProviderKind::OpenAi(OpenAiProvider::from_env()?)),
        "azure" => Ok(ProviderKind::Azure(AzureOpenAiProvider::from_env()?)),
        "bedrock" => Ok(ProviderKind::Bedrock(BedrockProvider::from_env()?)),
        "anthropic" => Ok(ProviderKind::Anthropic(AnthropicProvider::from_env()?)),
        "groq" => Ok(ProviderKind::Groq(GroqProvider::from_env()?)),
        "openrouter" => Ok(ProviderKind::OpenRouter(OpenRouterProvider::from_env()?)),
        other => Err(SdkError::Configuration(format!(
            "Unsupported provider `{other}`"
        ))),
    }
}

fn build_request(
    py: Python<'_>,
    model: &str,
    prompt: &PyObject,
    system_prompt_kw: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    response_format: Option<ResponseFormat>,
    user_id: Option<String>,
) -> PyResult<GenerateRequest> {
    let pairs = parse_prompt(py, prompt)?;

    let mut system_prompt = system_prompt_kw;
    let mut messages = Vec::new();

    for (role, content) in pairs {
        match role {
            MessageRole::System => {
                if system_prompt.is_none() {
                    system_prompt = Some(content);
                } else {
                    messages.push(Message::system(content));
                }
            }
            MessageRole::User => messages.push(Message::user(content)),
            MessageRole::Assistant => messages.push(Message::assistant(content)),
        }
    }

    let mut request = GenerateRequest::new(model.to_string());

    if let Some(system) = system_prompt {
        request = request.system_prompt(system);
    }

    for message in messages {
        request = request.message(message);
    }

    if let Some(user) = user_id {
        request = request.user_id(user);
    }

    request = request.configure(|config: &mut GenerationConfig| {
        if let Some(temp) = temperature {
            config.temperature = Some(temp);
        }
        if let Some(tp) = top_p {
            config.top_p = Some(tp);
        }
        if let Some(max) = max_tokens {
            config.max_output_tokens = Some(max);
        }
    });

    if let Some(format) = response_format {
        request.config.response_format = format;
    }

    Ok(request)
}

fn parse_prompt(py: Python<'_>, prompt: &PyObject) -> PyResult<Vec<(MessageRole, String)>> {
    if let Ok(text) = prompt.extract::<String>(py) {
        return Ok(vec![(MessageRole::User, text)]);
    }

    if let Ok(list) = prompt.downcast_bound::<PyList>(py) {
        let mut messages = Vec::with_capacity(list.len());
        for item in list {
            if let Ok(dict) = item.downcast::<PyDict>() {
                let role_value = dict
                    .get_item("role")?
                    .ok_or_else(|| PyValueError::new_err("Chat message missing 'role' field"))?;
                let role: String = role_value.extract()?;
                let content_value = dict
                    .get_item("content")?
                    .ok_or_else(|| PyValueError::new_err("Chat message missing 'content' field"))?;
                let content: String = content_value.extract()?;

                let role = match role.to_lowercase().as_str() {
                    "system" => MessageRole::System,
                    "user" => MessageRole::User,
                    "assistant" => MessageRole::Assistant,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported message role `{other}`"
                        )))
                    }
                };

                messages.push((role, content));
            } else if let Ok(text) = item.extract::<String>() {
                messages.push((MessageRole::User, text));
            } else {
                return Err(PyValueError::new_err(
                    "Each chat message must be a dict with 'role' and 'content' or a string",
                ));
            }
        }

        return Ok(messages);
    }

    Err(PyValueError::new_err(
        "Prompt must be a string or a list of message dictionaries",
    ))
}

fn get_optional_string(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<String>> {
    if let Some(kwargs) = kwargs {
        if let Some(value) = kwargs.get_item(key)? {
            if value.is_none() {
                return Ok(None);
            }
            return value
                .extract::<String>()
                .map(Some)
                .map_err(|_| PyValueError::new_err(format!("Expected `{key}` to be a string")));
        }
    }
    Ok(None)
}

fn get_optional_f32(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<f32>> {
    if let Some(kwargs) = kwargs {
        if let Some(value) = kwargs.get_item(key)? {
            if value.is_none() {
                return Ok(None);
            }
            return value
                .extract::<f32>()
                .map(Some)
                .map_err(|_| PyValueError::new_err(format!("Expected `{key}` to be a float")));
        }
    }
    Ok(None)
}

fn get_optional_u32(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<u32>> {
    if let Some(kwargs) = kwargs {
        if let Some(value) = kwargs.get_item(key)? {
            if value.is_none() {
                return Ok(None);
            }
            return value
                .extract::<u32>()
                .map(Some)
                .map_err(|_| PyValueError::new_err(format!("Expected `{key}` to be an integer")));
        }
    }
    Ok(None)
}

fn parse_response_format(
    format: Option<&str>,
    schema_json: Option<&str>,
) -> PyResult<Option<ResponseFormat>> {
    let schema = parse_schema_json(schema_json)?;

    match (format.map(|f| f.trim().to_lowercase()), schema) {
        (None, None) => Ok(None),
        (None, Some(schema)) => Ok(Some(ResponseFormat::JsonSchema(schema))),
        (Some(f), schema_opt) => match f.as_str() {
            "" => Ok(None),
            "text" | "plain" | "default" => {
                if schema_opt.is_some() {
                    Err(PyValueError::new_err(
                        "response_schema provided but response_format is text",
                    ))
                } else {
                    Ok(Some(ResponseFormat::Text))
                }
            }
            "json" | "json_object" => {
                if let Some(schema) = schema_opt {
                    Ok(Some(ResponseFormat::JsonSchema(schema)))
                } else {
                    Ok(Some(ResponseFormat::Json))
                }
            }
            "json_schema" | "schema" => {
                let schema = schema_opt.ok_or_else(|| {
                    PyValueError::new_err(
                        "response_schema is required when response_format='json_schema'",
                    )
                })?;
                Ok(Some(ResponseFormat::JsonSchema(schema)))
            }
            other => Err(PyValueError::new_err(format!(
                "Unsupported response_format `{other}`"
            ))),
        },
    }
}

fn parse_schema_json(schema_json: Option<&str>) -> PyResult<Option<JsonSchemaFormat>> {
    match schema_json {
        None => Ok(None),
        Some(raw) => {
            let value: Value = serde_json::from_str(raw).map_err(|err| {
                PyValueError::new_err(format!("Failed to parse response_schema: {err}"))
            })?;

            let mut name = "response".to_string();
            let mut strict = true;
            let schema_value = if let Some(obj) = value.as_object() {
                if let Some(schema_field) = obj.get("schema") {
                    if let Some(name_field) = obj.get("name").and_then(|v| v.as_str()) {
                        name = name_field.to_string();
                    }
                    if let Some(strict_field) = obj.get("strict").and_then(|v| v.as_bool()) {
                        strict = strict_field;
                    }
                    schema_field.clone()
                } else {
                    value
                }
            } else {
                value
            };

            let format = JsonSchemaFormat::new(name, schema_value).with_strict(strict);
            Ok(Some(format))
        }
    }
}

#[derive(Deserialize)]
struct ToolSpec {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<Value>,
    #[serde(default)]
    strict: Option<bool>,
}

fn parse_tools_json(json: Option<&str>) -> PyResult<Option<Vec<ToolDefinition>>> {
    let raw = match json {
        None => return Ok(None),
        Some(raw) if raw.trim().is_empty() => return Ok(None),
        Some(raw) => raw,
    };

    let specs: Vec<ToolSpec> = serde_json::from_str(raw)
        .map_err(|err| PyValueError::new_err(format!("Failed to parse tools: {err}")))?;

    let defs = specs
        .into_iter()
        .map(|spec| {
            let mut tool = ToolDefinition::new(spec.name);
            if let Some(description) = spec.description {
                tool = tool.description(description);
            }
            if let Some(parameters) = spec.parameters {
                tool = tool.parameters(parameters);
            }
            if let Some(strict) = spec.strict {
                tool = tool.strict(strict);
            }
            tool
        })
        .collect();

    Ok(Some(defs))
}

fn parse_tool_choice_json(json: Option<&str>) -> PyResult<Option<ToolChoice>> {
    let raw = match json {
        None => return Ok(None),
        Some(raw) if raw.trim().is_empty() => return Ok(None),
        Some(raw) => raw,
    };

    let value: Value = serde_json::from_str(raw)
        .map_err(|err| PyValueError::new_err(format!("Failed to parse tool_choice: {err}")))?;

    match value {
        Value::Null => Ok(None),
        Value::String(s) => match s.as_str() {
            "auto" => Ok(Some(ToolChoice::Auto)),
            "none" => Ok(Some(ToolChoice::None)),
            name => Ok(Some(ToolChoice::Tool {
                name: name.to_string(),
            })),
        },
        Value::Object(mut obj) => {
            if let Some(Value::String(type_field)) = obj.remove("type") {
                if type_field == "function" {
                    if let Some(Value::Object(mut func)) = obj.remove("function") {
                        if let Some(Value::String(name)) = func.remove("name") {
                            return Ok(Some(ToolChoice::Tool { name }));
                        }
                    }
                    return Err(PyValueError::new_err(
                        "tool_choice.function.name must be provided",
                    ));
                }
            }

            if let Some(Value::String(name)) = obj.remove("name") {
                Ok(Some(ToolChoice::Tool { name }))
            } else {
                Err(PyValueError::new_err(
                    "Unsupported tool_choice structure; expected string or function definition",
                ))
            }
        }
        _ => Err(PyValueError::new_err(
            "tool_choice must be a string or object",
        )),
    }
}

fn sdk_error_to_py(err: SdkError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Python wrapper around the core generate response.
#[pyclass(name = "Response")]
pub struct PyResponse {
    inner: GenerateResponse,
}

#[pymethods]
impl PyResponse {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn content(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn created(&self) -> Option<u64> {
        self.inner.created
    }

    #[getter]
    fn usage(&self) -> Option<PyUsage> {
        self.inner.usage.as_ref().map(|usage| PyUsage {
            inner: usage.clone(),
        })
    }

    #[getter]
    fn finish_reason(&self) -> Option<String> {
        self.inner.finish_reason.clone()
    }

    #[getter]
    fn object(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.inner
            .object
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn raw(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.inner
            .raw
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }
}

/// Wrapper for streaming chunks.
#[pyclass(name = "StreamChunk")]
pub struct PyStreamChunk {
    text: String,
    model: String,
    usage: Option<TokenUsage>,
    finished: bool,
    finish_reason: Option<String>,
    object: Option<Value>,
    raw: Option<Value>,
}

impl PyStreamChunk {
    fn delta(model: String, text: String) -> Self {
        Self {
            text,
            model,
            usage: None,
            finished: false,
            finish_reason: None,
            object: None,
            raw: None,
        }
    }

    fn from_response(response: GenerateResponse) -> Self {
        Self {
            text: response.text,
            model: response.model,
            usage: response.usage,
            finished: true,
            finish_reason: response.finish_reason,
            object: response.object,
            raw: response.raw,
        }
    }
}

#[pymethods]
impl PyStreamChunk {
    #[getter]
    fn text(&self) -> String {
        self.text.clone()
    }

    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    #[getter]
    fn finished(&self) -> bool {
        self.finished
    }

    #[getter]
    fn finish_reason(&self) -> Option<String> {
        self.finish_reason.clone()
    }

    #[getter]
    fn usage(&self) -> Option<PyUsage> {
        self.usage.as_ref().map(|usage| PyUsage {
            inner: usage.clone(),
        })
    }

    #[getter]
    fn object(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.object
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn raw(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.raw
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }
}

/// Python wrapper around token usage information.
#[pyclass(name = "Usage")]
pub struct PyUsage {
    inner: TokenUsage,
}

#[pymethods]
impl PyUsage {
    #[getter]
    fn prompt_tokens(&self) -> Option<u32> {
        self.inner.prompt_tokens
    }

    #[getter]
    fn input_tokens(&self) -> Option<u32> {
        self.inner.prompt_tokens
    }

    #[getter]
    fn completion_tokens(&self) -> Option<u32> {
        self.inner.completion_tokens
    }

    #[getter]
    fn output_tokens(&self) -> Option<u32> {
        self.inner.completion_tokens
    }

    #[getter]
    fn total_tokens(&self) -> Option<u32> {
        self.inner.total_tokens
    }
}

/// Extract OpenTelemetry context from Python's contextvar
///
/// The Rust worker injects traceparent into request metadata, and the Python worker
/// stores it in a contextvar named _trace_metadata. This function reads that contextvar
/// and reconstructs the OpenTelemetry context for trace propagation.
fn extract_context_from_python(py: Python<'_>) -> PyResult<OtelContext> {
    // Import agnt5.worker module to access _trace_metadata contextvar
    let worker_module = py.import("agnt5.worker")?;
    let trace_metadata_var = worker_module.getattr("_trace_metadata")?;

    // Call .get() on the contextvar to get the current value (dict)
    let metadata_dict = trace_metadata_var.call_method0("get")?;

    // Convert Python dict to Rust HashMap
    let metadata: HashMap<String, String> = metadata_dict.extract()?;

    // Use the sdk-core function to extract context from metadata
    let ctx = agnt5_sdk_core::extract_context_from_runtime_message(&metadata);

    // Log for debugging
    use opentelemetry::trace::TraceContextExt;
    let span = ctx.span();
    let span_ctx = span.span_context();
    log::info!(
        "Trace propagation: Extracted from Python contextvar - trace_id={}, span_id={}, valid={}",
        span_ctx.trace_id().to_string(),
        span_ctx.span_id().to_string(),
        span_ctx.is_valid()
    );

    Ok(ctx)
}

/// Register the language model bindings with the Python module.
pub fn register_language_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLanguageModelConfig>()?;
    m.add_class::<PyLanguageModel>()?;
    m.add_class::<PyResponse>()?;
    m.add_class::<PyStreamChunk>()?;
    m.add_class::<PyUsage>()?;
    Ok(())
}

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    let json_module = py.import("json")?;
    let serialized = serde_json::to_string(value)
        .map_err(|err| PyValueError::new_err(format!("Failed to serialize JSON value: {err}")))?;
    let obj = json_module.call_method1("loads", (serialized,))?;
    Ok(obj.unbind())
}
