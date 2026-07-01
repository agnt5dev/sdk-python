use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex};

use agnt5_sdk_core::error::{Result as SdkResult, SdkError};
use agnt5_sdk_core::lm::{
    AnthropicProvider, AzureOpenAiProvider, BasetenProvider, BedrockProvider, BuiltInTool,
    ContentBlockType, DeepSeekProvider, FireworksProvider, GenerateRequest, GenerateResponse,
    GenerationConfig, GoogleProvider, GroqProvider, HuggingFaceProvider, JsonSchemaFormat,
    LanguageModel, LeptonProvider, Message, MessageRole, MistralProvider, OllamaProvider,
    OpenAiProvider, OpenRouterProvider, PromptCacheConfig, ResponseFormat, StreamChunk,
    StreamHandle, StreamRequest, TogetherProvider, TokenUsage, ToolCall, ToolChoice,
    ToolDefinition, XaiProvider,
};
use futures::StreamExt;
use opentelemetry::Context as OtelContext;
use pyo3::exceptions::{PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList};
use pyo3_async_runtimes::TaskLocals;
use serde::Deserialize;
use serde_json::{self, Value};
use tokio::sync::mpsc;

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
    Baseten(BasetenProvider),
    DeepSeek(DeepSeekProvider),
    Fireworks(FireworksProvider),
    Google(GoogleProvider),
    Groq(GroqProvider),
    Lepton(LeptonProvider),
    Mistral(MistralProvider),
    Ollama(OllamaProvider),
    OpenRouter(OpenRouterProvider),
    Together(TogetherProvider),
    Xai(XaiProvider),
    HuggingFace(HuggingFaceProvider),
}

impl ProviderKind {
    async fn generate(&self, request: GenerateRequest) -> SdkResult<GenerateResponse> {
        match self {
            ProviderKind::OpenAi(provider) => provider.generate(request).await,
            ProviderKind::Azure(provider) => provider.generate(request).await,
            ProviderKind::Bedrock(provider) => provider.generate(request).await,
            ProviderKind::Anthropic(provider) => provider.generate(request).await,
            ProviderKind::Baseten(provider) => provider.generate(request).await,
            ProviderKind::DeepSeek(provider) => provider.generate(request).await,
            ProviderKind::Fireworks(provider) => provider.generate(request).await,
            ProviderKind::Google(provider) => provider.generate(request).await,
            ProviderKind::Groq(provider) => provider.generate(request).await,
            ProviderKind::Lepton(provider) => provider.generate(request).await,
            ProviderKind::Mistral(provider) => provider.generate(request).await,
            ProviderKind::Ollama(provider) => provider.generate(request).await,
            ProviderKind::OpenRouter(provider) => provider.generate(request).await,
            ProviderKind::Together(provider) => provider.generate(request).await,
            ProviderKind::Xai(provider) => provider.generate(request).await,
            ProviderKind::HuggingFace(provider) => provider.generate(request).await,
        }
    }

    async fn stream(&self, request: StreamRequest) -> SdkResult<StreamHandle> {
        match self {
            ProviderKind::OpenAi(provider) => provider.stream(request).await,
            ProviderKind::Azure(provider) => provider.stream(request).await,
            ProviderKind::Bedrock(provider) => provider.stream(request).await,
            ProviderKind::Anthropic(provider) => provider.stream(request).await,
            ProviderKind::Baseten(provider) => provider.stream(request).await,
            ProviderKind::DeepSeek(provider) => provider.stream(request).await,
            ProviderKind::Fireworks(provider) => provider.stream(request).await,
            ProviderKind::Google(provider) => provider.stream(request).await,
            ProviderKind::Groq(provider) => provider.stream(request).await,
            ProviderKind::Lepton(provider) => provider.stream(request).await,
            ProviderKind::Mistral(provider) => provider.stream(request).await,
            ProviderKind::Ollama(provider) => provider.stream(request).await,
            ProviderKind::OpenRouter(provider) => provider.stream(request).await,
            ProviderKind::Together(provider) => provider.stream(request).await,
            ProviderKind::Xai(provider) => provider.stream(request).await,
            ProviderKind::HuggingFace(provider) => provider.stream(request).await,
        }
    }

    async fn create_cached_content(
        &self,
        model: &str,
        system: Option<String>,
        contents: Vec<String>,
        ttl_seconds: Option<u32>,
    ) -> SdkResult<String> {
        match self {
            ProviderKind::Google(provider) => {
                provider
                    .create_cached_content(model, system, contents, ttl_seconds)
                    .await
            }
            _ => Err(SdkError::Configuration {
                message: "explicit context caching is only supported for Google Gemini".to_string(),
                field: Some("provider".to_string()),
            }),
        }
    }

    async fn delete_cached_content(&self, name: &str) -> SdkResult<()> {
        match self {
            ProviderKind::Google(provider) => provider.delete_cached_content(name).await,
            _ => Err(SdkError::Configuration {
                message: "explicit context cache deletion is only supported for Google Gemini"
                    .to_string(),
                field: Some("provider".to_string()),
            }),
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
        prompt: Py<PyAny>,
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
        let response_schema_kw = get_optional_string(kwargs_ref, "response_schema_kw")?;
        let tools_kw = get_optional_string(kwargs_ref, "tools")?;
        let tool_choice_kw = get_optional_string(kwargs_ref, "tool_choice")?;
        let previous_response_id_kw = get_optional_string(kwargs_ref, "previous_response_id")?;
        let built_in_tools_kw = get_optional_string(kwargs_ref, "built_in_tools")?;
        let prompt_cache_kw = get_optional_string(kwargs_ref, "prompt_cache")?;
        let cache_control_kw = get_optional_bool(kwargs_ref, "cache_control")?;
        let cache_ttl_kw = get_optional_string(kwargs_ref, "cache_ttl")?;
        let google_cached_content_kw = get_optional_string(kwargs_ref, "google_cached_content")?;
        let built_in_tools = parse_built_in_tools_json(built_in_tools_kw.as_deref())?;
        let prompt_cache = parse_prompt_cache_json(prompt_cache_kw.as_deref())?;
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

        // Set previous_response_id for OpenAI Responses API conversation continuation
        if let Some(prev_id) = previous_response_id_kw {
            request.previous_response_id = Some(prev_id);
        }

        if !built_in_tools.is_empty() {
            request.config.built_in_tools = built_in_tools;
        }

        apply_prompt_cache_config(
            &mut request.config,
            prompt_cache,
            cache_control_kw,
            cache_ttl_kw,
            google_cached_content_kw,
        );

        let provider = self.get_or_init_provider(&provider_name)?;

        // IMPORTANT: Extract trace context from Python's CURRENT OpenTelemetry span
        // This ensures LLM spans are children of the currently active span
        // (e.g., python_component_execution) rather than the original gateway span.
        let otel_context = match extract_current_span_context_from_python(py) {
            Ok(Some(ctx)) => Some(ctx),
            Ok(None) => {
                // Fallback to old methods if no current span
                if let Some((otel_ctx, _, _)) = crate::get_runtime_context_from_contextvar(py)? {
                    otel_ctx
                } else {
                    extract_context_from_python(py).ok()
                }
            }
            Err(_) => {
                // If extraction fails, try legacy methods
                if let Some((otel_ctx, _, _)) = crate::get_runtime_context_from_contextvar(py)? {
                    otel_ctx
                } else {
                    extract_context_from_python(py).ok()
                }
            }
        };

        // Use pyo3-async-runtimes with proper runtime context
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let mut request = request;
            request.otel_context = otel_context;

            let response = provider.generate(request).await.map_err(sdk_error_to_py)?;
            Ok(PyResponse { inner: response })
        })
    }

    #[pyo3(signature = (model, contents, system_prompt=None, ttl_seconds=None, provider=None))]
    fn create_cache<'py>(
        &self,
        py: Python<'py>,
        model: String,
        contents: String,
        system_prompt: Option<String>,
        ttl_seconds: Option<u32>,
        provider: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let contents: Vec<String> = serde_json::from_str(&contents)
            .map_err(|err| PyValueError::new_err(format!("Failed to parse contents: {err}")))?;
        let (model, provider_name) = self.resolve_model_and_provider(Some(model), provider)?;
        let provider = self.get_or_init_provider(&provider_name)?;

        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let name = provider
                .create_cached_content(&model, system_prompt, contents, ttl_seconds)
                .await
                .map_err(sdk_error_to_py)?;
            Ok(name)
        })
    }

    #[pyo3(signature = (name, provider=None))]
    fn delete_cache<'py>(
        &self,
        py: Python<'py>,
        name: String,
        provider: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider_name = provider
            .or_else(|| self.default_provider.clone())
            .unwrap_or_else(|| "google".to_string())
            .to_lowercase();
        let provider = self.get_or_init_provider(&provider_name)?;

        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            provider
                .delete_cached_content(&name)
                .await
                .map_err(sdk_error_to_py)?;
            Ok(())
        })
    }

    #[pyo3(signature = (prompt, **kwargs))]
    fn stream<'py>(
        &self,
        py: Python<'py>,
        prompt: Py<PyAny>,
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
        let response_schema_kw = get_optional_string(kwargs_ref, "response_schema_kw")?;
        let tools_kw = get_optional_string(kwargs_ref, "tools")?;
        let tool_choice_kw = get_optional_string(kwargs_ref, "tool_choice")?;
        let previous_response_id_kw = get_optional_string(kwargs_ref, "previous_response_id")?;
        let built_in_tools_kw = get_optional_string(kwargs_ref, "built_in_tools")?;
        let prompt_cache_kw = get_optional_string(kwargs_ref, "prompt_cache")?;
        let cache_control_kw = get_optional_bool(kwargs_ref, "cache_control")?;
        let cache_ttl_kw = get_optional_string(kwargs_ref, "cache_ttl")?;
        let google_cached_content_kw = get_optional_string(kwargs_ref, "google_cached_content")?;
        let built_in_tools = parse_built_in_tools_json(built_in_tools_kw.as_deref())?;
        let prompt_cache = parse_prompt_cache_json(prompt_cache_kw.as_deref())?;
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

        // Set previous_response_id for OpenAI Responses API conversation continuation
        if let Some(prev_id) = previous_response_id_kw {
            request.previous_response_id = Some(prev_id);
        }

        if !built_in_tools.is_empty() {
            request.config.built_in_tools = built_in_tools;
        }

        apply_prompt_cache_config(
            &mut request.config,
            prompt_cache,
            cache_control_kw,
            cache_ttl_kw,
            google_cached_content_kw,
        );

        let provider = self.get_or_init_provider(&provider_name)?;
        let model_for_delta = model.clone();

        // IMPORTANT: Extract trace context from Python's CURRENT OpenTelemetry span
        // This ensures LLM spans are children of the currently active span
        let otel_context = match extract_current_span_context_from_python(py) {
            Ok(Some(ctx)) => Some(ctx),
            Ok(None) => {
                if let Some((otel_ctx, _, _)) = crate::get_runtime_context_from_contextvar(py)? {
                    otel_ctx
                } else {
                    extract_context_from_python(py).ok()
                }
            }
            Err(_) => {
                if let Some((otel_ctx, _, _)) = crate::get_runtime_context_from_contextvar(py)? {
                    otel_ctx
                } else {
                    extract_context_from_python(py).ok()
                }
            }
        };

        // Use pyo3-async-runtimes with proper runtime context for streaming
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            let mut request = request;
            request.otel_context = otel_context;

            let mut handle = provider.stream(request).await.map_err(sdk_error_to_py)?;
            let mut chunks = Vec::new();

            while let Some(item) = handle.next().await {
                let chunk = item.map_err(sdk_error_to_py)?;
                match chunk {
                    StreamChunk::ContentBlockStart { index, block_type } => {
                        chunks.push(PyStreamChunk::content_block_start(
                            model_for_delta.clone(),
                            index,
                            block_type,
                        ));
                    }
                    StreamChunk::Delta {
                        content,
                        index,
                        block_type,
                    } => {
                        chunks.push(PyStreamChunk::delta(
                            model_for_delta.clone(),
                            content,
                            index,
                            block_type,
                        ));
                    }
                    StreamChunk::ContentBlockStop { index } => {
                        chunks.push(PyStreamChunk::content_block_stop(
                            model_for_delta.clone(),
                            index,
                        ));
                    }
                    StreamChunk::Completed(response) => {
                        chunks.push(PyStreamChunk::from_response(response));
                    }
                }
            }

            Ok(chunks)
        })
    }

    /// Stream generation with true async iteration.
    ///
    /// Returns an async iterator that yields chunks one at a time as they arrive,
    /// enabling real-time token-by-token streaming.
    ///
    /// Usage:
    /// ```python
    /// async for chunk in lm.stream_iter(prompt="Hello"):
    ///     print(chunk.text, end="", flush=True)
    /// ```
    #[pyo3(signature = (prompt, **kwargs))]
    fn stream_iter<'py>(
        &self,
        py: Python<'py>,
        prompt: Py<PyAny>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<PyAsyncStreamHandle> {
        // Parse kwargs
        let kwargs_ref = kwargs.as_ref();
        let model_kw = get_optional_string(kwargs_ref, "model")?;
        let provider_kw = get_optional_string(kwargs_ref, "provider")?;
        let system_prompt_kw = get_optional_string(kwargs_ref, "system_prompt")?;
        let temperature = get_optional_f32(kwargs_ref, "temperature")?;
        let top_p = get_optional_f32(kwargs_ref, "top_p")?;
        let max_tokens = get_optional_u32(kwargs_ref, "max_tokens")?;
        let user_id = get_optional_string(kwargs_ref, "user")?;
        let response_format_str = get_optional_string(kwargs_ref, "response_format")?;
        let response_schema = get_optional_string(kwargs_ref, "response_schema")?;
        let tools_kw = get_optional_string(kwargs_ref, "tools")?;
        let tool_choice_kw = get_optional_string(kwargs_ref, "tool_choice")?;
        let previous_response_id_kw = get_optional_string(kwargs_ref, "previous_response_id")?;
        let built_in_tools_kw = get_optional_string(kwargs_ref, "built_in_tools")?;
        let prompt_cache_kw = get_optional_string(kwargs_ref, "prompt_cache")?;
        let cache_control_kw = get_optional_bool(kwargs_ref, "cache_control")?;
        let cache_ttl_kw = get_optional_string(kwargs_ref, "cache_ttl")?;
        let google_cached_content_kw = get_optional_string(kwargs_ref, "google_cached_content")?;
        let built_in_tools = parse_built_in_tools_json(built_in_tools_kw.as_deref())?;
        let prompt_cache = parse_prompt_cache_json(prompt_cache_kw.as_deref())?;

        let response_format =
            parse_response_format(response_format_str.as_deref(), response_schema.as_deref())?;
        let tools = parse_tools_json(tools_kw.as_deref())?;
        let tool_choice = parse_tool_choice_json(tool_choice_kw.as_deref())?;

        let (model, provider_name) = self.resolve_model_and_provider(model_kw, provider_kw)?;

        let mut request = build_request(
            py,
            &model,
            &prompt,
            system_prompt_kw,
            temperature,
            top_p,
            max_tokens,
            response_format,
            user_id,
        )?;

        if let Some(tool_defs) = tools {
            request = request.tools(tool_defs);
        }
        if let Some(choice) = tool_choice {
            request = request.tool_choice(Some(choice));
        }

        // Set previous_response_id for OpenAI Responses API conversation continuation
        if let Some(prev_id) = previous_response_id_kw {
            request.previous_response_id = Some(prev_id);
        }

        if !built_in_tools.is_empty() {
            request.config.built_in_tools = built_in_tools;
        }

        apply_prompt_cache_config(
            &mut request.config,
            prompt_cache,
            cache_control_kw,
            cache_ttl_kw,
            google_cached_content_kw,
        );

        let provider = self.get_or_init_provider(&provider_name)?;
        let model_for_chunks = model.clone();

        // Extract trace context from Python's CURRENT OpenTelemetry span
        let otel_context = match extract_current_span_context_from_python(py) {
            Ok(Some(ctx)) => Some(ctx),
            Ok(None) => {
                if let Some((otel_ctx, _, _)) = crate::get_runtime_context_from_contextvar(py)? {
                    otel_ctx
                } else {
                    extract_context_from_python(py).ok()
                }
            }
            Err(_) => {
                if let Some((otel_ctx, _, _)) = crate::get_runtime_context_from_contextvar(py)? {
                    otel_ctx
                } else {
                    extract_context_from_python(py).ok()
                }
            }
        };

        // Create channel for streaming chunks
        let (tx, rx) = mpsc::channel::<Result<PyStreamChunk, String>>(32);
        let receiver = Arc::new(tokio::sync::Mutex::new(rx));
        let exhausted = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Spawn a tokio task to read from the stream and send to channel
        let handle = PyAsyncStreamHandle {
            receiver,
            exhausted,
        };

        // Spawn the streaming task on the pyo3-async-runtimes managed tokio runtime
        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let mut request = request;
            request.otel_context = otel_context;

            match provider.stream(request).await {
                Ok(mut stream_handle) => {
                    while let Some(item) = stream_handle.next().await {
                        let chunk_result = match item {
                            Ok(chunk) => {
                                let py_chunk = match chunk {
                                    StreamChunk::ContentBlockStart { index, block_type } => {
                                        PyStreamChunk::content_block_start(
                                            model_for_chunks.clone(),
                                            index,
                                            block_type,
                                        )
                                    }
                                    StreamChunk::Delta {
                                        content,
                                        index,
                                        block_type,
                                    } => PyStreamChunk::delta(
                                        model_for_chunks.clone(),
                                        content,
                                        index,
                                        block_type,
                                    ),
                                    StreamChunk::ContentBlockStop { index } => {
                                        PyStreamChunk::content_block_stop(
                                            model_for_chunks.clone(),
                                            index,
                                        )
                                    }
                                    StreamChunk::Completed(response) => {
                                        PyStreamChunk::from_response(response)
                                    }
                                };
                                Ok(py_chunk)
                            }
                            Err(err) => Err(err.to_string()),
                        };

                        // If send fails, receiver was dropped - stop streaming
                        if tx.send(chunk_result).await.is_err() {
                            break;
                        }
                    }
                }
                Err(err) => {
                    // Send error to channel
                    let _ = tx.send(Err(err.to_string())).await;
                }
            }
            // Channel will be closed when tx is dropped
        });

        Ok(handle)
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
        if env::var("BASETEN_API_KEY").is_ok() {
            providers.push("baseten".to_string());
        }
        if env::var("DEEPSEEK_API_KEY").is_ok() {
            providers.push("deepseek".to_string());
        }
        if env::var("FIREWORKS_API_KEY").is_ok() {
            providers.push("fireworks".to_string());
        }
        if env::var("GOOGLE_API_KEY").is_ok() || env::var("GEMINI_API_KEY").is_ok() {
            providers.push("google".to_string());
        }
        if env::var("GROQ_API_KEY").is_ok() {
            providers.push("groq".to_string());
        }
        if (env::var("LEPTON_API_KEY").is_ok() || env::var("LEPTON_API_TOKEN").is_ok())
            && (env::var("LEPTON_BASE_URL").is_ok() || env::var("LEPTON_API_BASE").is_ok())
        {
            providers.push("lepton".to_string());
        }
        if env::var("MISTRAL_API_KEY").is_ok() {
            providers.push("mistral".to_string());
        }
        // Ollama doesn't require an API key - check for OLLAMA_HOST or assume localhost
        if env::var("OLLAMA_HOST").is_ok() || env::var("OLLAMA_BASE_URL").is_ok() {
            providers.push("ollama".to_string());
        }
        if env::var("OPENROUTER_API_KEY").is_ok() {
            providers.push("openrouter".to_string());
        }
        if env::var("TOGETHER_API_KEY").is_ok() {
            providers.push("together".to_string());
        }
        if env::var("XAI_API_KEY").is_ok() {
            providers.push("xai".to_string());
        }
        if env::var("HUGGINGFACE_API_KEY").is_ok() || env::var("HF_TOKEN").is_ok() {
            providers.push("huggingface".to_string());
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
                let provider_lower = provider.to_lowercase();
                let prefix_lower = prefix.to_lowercase();

                // HuggingFace can use both "hf" and "huggingface" as provider names
                let is_hf_match = (provider_lower == "huggingface" || provider_lower == "hf")
                    && (prefix_lower == "huggingface" || prefix_lower == "hf");

                if provider_lower != prefix_lower && !is_hf_match {
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
        "baseten" => Ok(ProviderKind::Baseten(BasetenProvider::from_env()?)),
        "deepseek" => Ok(ProviderKind::DeepSeek(DeepSeekProvider::from_env()?)),
        "fireworks" => Ok(ProviderKind::Fireworks(FireworksProvider::from_env()?)),
        "google" | "gemini" => Ok(ProviderKind::Google(GoogleProvider::from_env()?)),
        "groq" => Ok(ProviderKind::Groq(GroqProvider::from_env()?)),
        "lepton" => Ok(ProviderKind::Lepton(LeptonProvider::from_env()?)),
        "mistral" => Ok(ProviderKind::Mistral(MistralProvider::from_env()?)),
        "ollama" => Ok(ProviderKind::Ollama(OllamaProvider::from_env()?)),
        "openrouter" => Ok(ProviderKind::OpenRouter(OpenRouterProvider::from_env()?)),
        "together" => Ok(ProviderKind::Together(TogetherProvider::from_env()?)),
        "xai" => Ok(ProviderKind::Xai(XaiProvider::from_env()?)),
        "huggingface" | "hf" => Ok(ProviderKind::HuggingFace(HuggingFaceProvider::from_env()?)),
        other => Err(SdkError::Configuration {
            message: format!("Unsupported provider `{other}`"),
            field: Some("provider".to_string()),
        }),
    }
}

fn build_request(
    py: Python<'_>,
    model: &str,
    prompt: &Py<PyAny>,
    system_prompt_kw: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    response_format: Option<ResponseFormat>,
    user_id: Option<String>,
) -> PyResult<GenerateRequest> {
    let parsed_messages = parse_prompt(py, prompt)?;

    let mut system_prompt = system_prompt_kw;
    let mut messages = Vec::new();

    for parsed in parsed_messages {
        match parsed.role {
            MessageRole::System => {
                if system_prompt.is_none() {
                    system_prompt = Some(parsed.content);
                } else {
                    messages.push(Message::system(parsed.content));
                }
            }
            MessageRole::User => {
                // Check if this is a tool result message
                if let Some(tool_call_id) = parsed.tool_call_id {
                    messages.push(Message::tool_result(tool_call_id, parsed.content));
                } else {
                    messages.push(Message::user(parsed.content));
                }
            }
            MessageRole::Assistant => {
                // Check if this is an assistant message with tool calls
                if let Some(tool_calls) = parsed.tool_calls {
                    messages.push(Message::assistant_with_tool_calls(
                        parsed.content,
                        tool_calls,
                    ));
                } else {
                    messages.push(Message::assistant(parsed.content));
                }
            }
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

/// Parsed message from Python including all fields for agentic workflows
struct ParsedMessage {
    role: MessageRole,
    content: String,
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}

fn parse_prompt(py: Python<'_>, prompt: &Py<PyAny>) -> PyResult<Vec<ParsedMessage>> {
    if let Ok(text) = prompt.extract::<String>(py) {
        return Ok(vec![ParsedMessage {
            role: MessageRole::User,
            content: text,
            tool_calls: None,
            tool_call_id: None,
        }]);
    }

    if let Ok(list) = prompt.cast_bound::<PyList>(py) {
        let mut messages = Vec::with_capacity(list.len());
        for item in list {
            if let Ok(dict) = item.cast::<PyDict>() {
                let role_value = dict
                    .get_item("role")?
                    .ok_or_else(|| PyValueError::new_err("Chat message missing 'role' field"))?;
                let role: String = role_value.extract()?;

                // Content can be optional for assistant messages with only tool_calls
                let content: String = dict
                    .get_item("content")?
                    .map(|v| v.extract::<String>())
                    .transpose()?
                    .unwrap_or_default();

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

                // Extract tool_calls if present (for assistant messages)
                let tool_calls: Option<Vec<ToolCall>> = if let Some(tc_value) =
                    dict.get_item("tool_calls")?
                {
                    if tc_value.is_none() {
                        None
                    } else {
                        // tool_calls is a list of dicts with id, name, arguments
                        #[allow(deprecated)]
                        let tc_list = tc_value
                            .downcast::<PyList>()
                            .map_err(|_| PyValueError::new_err("tool_calls must be a list"))?;
                        let mut calls = Vec::with_capacity(tc_list.len());
                        for tc_item in tc_list.iter() {
                            #[allow(deprecated)]
                            let tc_dict = tc_item.downcast::<PyDict>().map_err(|_| {
                                PyValueError::new_err("Each tool_call must be a dict")
                            })?;

                            let id: String = tc_dict
                                .get_item("id")?
                                .ok_or_else(|| PyValueError::new_err("tool_call missing 'id'"))?
                                .extract()?;
                            let name: String = tc_dict
                                .get_item("name")?
                                .ok_or_else(|| PyValueError::new_err("tool_call missing 'name'"))?
                                .extract()?;

                            // arguments can be string (JSON) or dict
                            let arguments_value =
                                tc_dict.get_item("arguments")?.ok_or_else(|| {
                                    PyValueError::new_err("tool_call missing 'arguments'")
                                })?;
                            let arguments: String =
                                if let Ok(s) = arguments_value.extract::<String>() {
                                    s
                                } else {
                                    // Convert dict/other to JSON string
                                    let json_mod = py.import("json")?;
                                    json_mod
                                        .call_method1("dumps", (arguments_value,))?
                                        .extract()?
                                };

                            calls.push(ToolCall {
                                id,
                                name,
                                arguments,
                            });
                        }
                        Some(calls)
                    }
                } else {
                    None
                };

                // Extract tool_call_id if present (for tool result messages)
                let tool_call_id: Option<String> =
                    if let Some(tcid_value) = dict.get_item("tool_call_id")? {
                        if tcid_value.is_none() {
                            None
                        } else {
                            Some(tcid_value.extract()?)
                        }
                    } else {
                        None
                    };

                messages.push(ParsedMessage {
                    role,
                    content,
                    tool_calls,
                    tool_call_id,
                });
            } else if let Ok(text) = item.extract::<String>() {
                messages.push(ParsedMessage {
                    role: MessageRole::User,
                    content: text,
                    tool_calls: None,
                    tool_call_id: None,
                });
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

fn get_optional_bool(kwargs: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<bool>> {
    if let Some(kwargs) = kwargs {
        if let Some(value) = kwargs.get_item(key)? {
            if value.is_none() {
                return Ok(None);
            }
            return value
                .extract::<bool>()
                .map(Some)
                .map_err(|_| PyValueError::new_err(format!("Expected `{key}` to be a boolean")));
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

#[derive(Deserialize)]
struct PromptCacheSpec {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    ttl: Option<String>,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    retention: Option<String>,
    #[serde(default)]
    resource: Option<String>,
}

fn parse_prompt_cache_json(json: Option<&str>) -> PyResult<Option<PromptCacheConfig>> {
    let raw = match json {
        None => return Ok(None),
        Some(raw) if raw.trim().is_empty() => return Ok(None),
        Some(raw) => raw,
    };

    let spec: PromptCacheSpec = serde_json::from_str(raw)
        .map_err(|err| PyValueError::new_err(format!("Failed to parse prompt_cache: {err}")))?;
    Ok(Some(PromptCacheConfig {
        enabled: spec.enabled.unwrap_or(true),
        ttl: spec.ttl,
        key: spec.key,
        retention: spec.retention,
        resource: spec.resource,
    }))
}

fn apply_prompt_cache_config(
    config: &mut GenerationConfig,
    prompt_cache: Option<PromptCacheConfig>,
    cache_control: Option<bool>,
    cache_ttl: Option<String>,
    google_cached_content: Option<String>,
) {
    if let Some(cache) = prompt_cache {
        config.prompt_cache = Some(cache);
    }

    if cache_control.is_some() || cache_ttl.is_some() || google_cached_content.is_some() {
        let mut cache = config
            .prompt_cache
            .take()
            .unwrap_or_else(PromptCacheConfig::enabled);
        if let Some(enabled) = cache_control {
            cache.enabled = enabled;
        }
        if let Some(ttl) = cache_ttl {
            cache.ttl = Some(ttl);
            cache.enabled = true;
        }
        if let Some(name) = google_cached_content {
            cache.resource = Some(name);
            cache.enabled = true;
        }
        config.prompt_cache = Some(cache);
    }
}

fn parse_built_in_tools_json(json: Option<&str>) -> PyResult<Vec<BuiltInTool>> {
    let raw = match json {
        None => return Ok(Vec::new()),
        Some(raw) if raw.trim().is_empty() => return Ok(Vec::new()),
        Some(raw) => raw,
    };
    let names: Vec<String> = serde_json::from_str(raw)
        .map_err(|err| PyValueError::new_err(format!("Failed to parse built_in_tools: {err}")))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        match BuiltInTool::from_provider_name(&name) {
            Some(tool) => out.push(tool),
            None => {
                return Err(PyValueError::new_err(format!(
                    "Unknown built_in_tool: {name}"
                )));
            }
        }
    }
    Ok(out)
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
            "required" => Ok(Some(ToolChoice::Required)),
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
    fn tool_calls(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        // Convert tool_calls from Rust to Python list of dicts
        if let Some(ref tool_calls) = self.inner.tool_calls {
            let py_list = pyo3::types::PyList::empty(py);
            for tool_call in tool_calls {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("id", &tool_call.id)?;
                dict.set_item("name", &tool_call.name)?;
                dict.set_item("arguments", &tool_call.arguments)?;
                py_list.append(dict)?;
            }
            Ok(Some(py_list.into()))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn object(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .object
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }

    #[getter]
    fn raw(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .raw
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }
}

/// Wrapper for streaming chunks.
///
/// Chunk types:
/// - "content_block_start": Start of a content block (text or thinking)
/// - "delta": Incremental content within a block
/// - "content_block_stop": End of a content block
/// - "completed": Final response with full text, usage, and metadata
#[pyclass(name = "StreamChunk")]
pub struct PyStreamChunk {
    /// Type of chunk: "content_block_start", "delta", "content_block_stop", "completed"
    chunk_type: String,
    /// Content text (for delta and completed chunks)
    text: String,
    /// Model name
    model: String,
    /// Block type: "text" or "thinking" (for start/delta chunks)
    block_type: Option<String>,
    /// Content block index (0-indexed)
    index: Option<u32>,
    /// Token usage (for completed chunks)
    usage: Option<TokenUsage>,
    /// Whether this is the final chunk (completed)
    finished: bool,
    /// Finish reason (for completed chunks)
    finish_reason: Option<String>,
    /// Parsed JSON object (for completed chunks with JSON response)
    object: Option<Value>,
    /// Raw API response (for completed chunks)
    raw: Option<Value>,
}

impl PyStreamChunk {
    fn content_block_start(model: String, index: u32, block_type: ContentBlockType) -> Self {
        Self {
            chunk_type: "content_block_start".to_string(),
            text: String::new(),
            model,
            block_type: Some(block_type_to_string(block_type)),
            index: Some(index),
            usage: None,
            finished: false,
            finish_reason: None,
            object: None,
            raw: None,
        }
    }

    fn delta(model: String, text: String, index: u32, block_type: ContentBlockType) -> Self {
        Self {
            chunk_type: "delta".to_string(),
            text,
            model,
            block_type: Some(block_type_to_string(block_type)),
            index: Some(index),
            usage: None,
            finished: false,
            finish_reason: None,
            object: None,
            raw: None,
        }
    }

    fn content_block_stop(model: String, index: u32) -> Self {
        Self {
            chunk_type: "content_block_stop".to_string(),
            text: String::new(),
            model,
            block_type: None,
            index: Some(index),
            usage: None,
            finished: false,
            finish_reason: None,
            object: None,
            raw: None,
        }
    }

    fn from_response(response: GenerateResponse) -> Self {
        Self {
            chunk_type: "completed".to_string(),
            text: response.text,
            model: response.model,
            block_type: None,
            index: None,
            usage: response.usage,
            finished: true,
            finish_reason: response.finish_reason,
            object: response.object,
            raw: response.raw,
        }
    }
}

fn block_type_to_string(block_type: ContentBlockType) -> String {
    match block_type {
        ContentBlockType::Text => "text".to_string(),
        ContentBlockType::Thinking => "thinking".to_string(),
    }
}

#[pymethods]
impl PyStreamChunk {
    /// Type of chunk: "content_block_start", "delta", "content_block_stop", "completed"
    #[getter]
    fn chunk_type(&self) -> String {
        self.chunk_type.clone()
    }

    /// Content text (for delta and completed chunks)
    #[getter]
    fn text(&self) -> String {
        self.text.clone()
    }

    /// Alias for text (backwards compatibility)
    #[getter]
    fn content(&self) -> String {
        self.text.clone()
    }

    /// Model name
    #[getter]
    fn model(&self) -> String {
        self.model.clone()
    }

    /// Block type: "text" or "thinking" (for start/delta chunks)
    #[getter]
    fn block_type(&self) -> Option<String> {
        self.block_type.clone()
    }

    /// Content block index (0-indexed)
    #[getter]
    fn index(&self) -> Option<u32> {
        self.index
    }

    /// Whether this is the final chunk (completed)
    #[getter]
    fn finished(&self) -> bool {
        self.finished
    }

    /// Finish reason (for completed chunks)
    #[getter]
    fn finish_reason(&self) -> Option<String> {
        self.finish_reason.clone()
    }

    /// Token usage (for completed chunks)
    #[getter]
    fn usage(&self) -> Option<PyUsage> {
        self.usage.as_ref().map(|usage| PyUsage {
            inner: usage.clone(),
        })
    }

    /// Parsed JSON object (for completed chunks with JSON response)
    #[getter]
    fn object(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.object
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }

    /// Raw API response (for completed chunks)
    #[getter]
    fn raw(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.raw
            .as_ref()
            .map(|value| json_to_py(py, value))
            .transpose()
    }

    /// Check if this is a thinking block
    fn is_thinking(&self) -> bool {
        self.block_type.as_deref() == Some("thinking")
    }

    /// Check if this is a text block
    fn is_text(&self) -> bool {
        self.block_type.as_deref() == Some("text")
    }
}

/// Async iterator for streaming LLM responses.
///
/// This class implements the Python async iterator protocol (`__aiter__` and `__anext__`),
/// enabling true token-by-token streaming instead of collecting all chunks before returning.
///
/// Usage:
/// ```python
/// async for chunk in lm.stream_iter(...):
///     print(chunk.text, end="", flush=True)
/// ```
#[pyclass(name = "AsyncStreamHandle")]
pub struct PyAsyncStreamHandle {
    /// Channel receiver for streaming chunks
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<Result<PyStreamChunk, String>>>>,
    /// Flag to track if stream is exhausted
    exhausted: Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl PyAsyncStreamHandle {
    /// Return self as the async iterator
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Get the next chunk from the stream
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = self.receiver.clone();
        let exhausted = self.exhausted.clone();

        // Use pyo3-async-runtimes to create a Python awaitable
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals, async move {
            // Check if already exhausted
            if exhausted.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(PyStopAsyncIteration::new_err("stream exhausted"));
            }

            let mut rx = receiver.lock().await;
            match rx.recv().await {
                Some(Ok(chunk)) => Ok(chunk),
                Some(Err(err)) => Err(PyValueError::new_err(err)),
                None => {
                    exhausted.store(true, std::sync::atomic::Ordering::SeqCst);
                    Err(PyStopAsyncIteration::new_err("stream exhausted"))
                }
            }
        })
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

    /// Input tokens served from the prompt cache (cache hits). Subset of
    /// `prompt_tokens`.
    #[getter]
    fn cached_tokens(&self) -> Option<u32> {
        self.inner.cached_tokens
    }

    /// Input tokens written to the prompt cache on this request (cache writes).
    /// Anthropic-style providers only.
    #[getter]
    fn cache_creation_tokens(&self) -> Option<u32> {
        self.inner.cache_creation_tokens
    }
}

/// Extract OpenTelemetry context from Python's contextvar (legacy)
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

    Ok(ctx)
}

/// Extract the CURRENT span context from Python's agnt5.tracing contextvar
///
/// This gets the currently active span from AGNT5's own contextvar system (not Python's
/// OpenTelemetry SDK), ensuring LLM spans are created as children of the current execution
/// span (e.g., agent.calculator) rather than the original gateway span.
///
/// The agnt5.tracing module uses a contextvar `_current_span` that tracks SpanInfo(trace_id, span_id)
/// which is properly set when entering span context managers created via create_span().
fn extract_current_span_context_from_python(py: Python<'_>) -> PyResult<Option<OtelContext>> {
    // Import agnt5.tracing module to access get_current_span_info()
    let tracing_module = py.import("agnt5.tracing")?;

    // Call get_current_span_info() which reads from the _current_span contextvar
    let span_info = tracing_module.call_method0("get_current_span_info")?;

    // Check if we got None (no current span)
    if span_info.is_none() {
        return Ok(None);
    }

    // Extract trace_id and span_id from SpanInfo dataclass
    let trace_id_str: String = span_info.getattr("trace_id")?.extract()?;
    let span_id_str: String = span_info.getattr("span_id")?.extract()?;

    // Check for empty strings (invalid span)
    if trace_id_str.is_empty() || span_id_str.is_empty() {
        return Ok(None);
    }

    // Create traceparent format: "00-{trace_id}-{span_id}-01"
    // trace_id is already 32 hex chars, span_id is already 16 hex chars
    let traceparent = format!("00-{}-{}-01", trace_id_str, span_id_str);

    // Create a metadata map with the traceparent
    let mut metadata = HashMap::new();
    metadata.insert("traceparent".to_string(), traceparent);

    // Use the sdk-core function to extract context from metadata
    let ctx = agnt5_sdk_core::extract_context_from_runtime_message(&metadata);

    Ok(Some(ctx))
}

/// Register the language model bindings with the Python module.
pub fn register_language_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLanguageModelConfig>()?;
    m.add_class::<PyLanguageModel>()?;
    m.add_class::<PyResponse>()?;
    m.add_class::<PyStreamChunk>()?;
    m.add_class::<PyAsyncStreamHandle>()?;
    m.add_class::<PyUsage>()?;
    Ok(())
}

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    use pyo3::conversion::IntoPyObject;

    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(value) => {
            let py_bool = value.into_pyobject(py)?;
            Ok(py_bool.to_owned().into_any().unbind())
        }
        Value::Number(value) => {
            if let Some(int_value) = value.as_i64() {
                let py_int = int_value.into_pyobject(py)?;
                Ok(py_int.into_any().unbind())
            } else if let Some(uint_value) = value.as_u64() {
                let py_int = uint_value.into_pyobject(py)?;
                Ok(py_int.into_any().unbind())
            } else if let Some(float_value) = value.as_f64() {
                let py_float = float_value.into_pyobject(py)?;
                Ok(py_float.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        Value::String(value) => {
            let py_str = value.as_str().into_pyobject(py)?;
            Ok(py_str.into_any().unbind())
        }
        Value::Array(values) => {
            let list = PyList::empty(py);
            for item in values {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, item) in values {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}
