use pyo3::prelude::*;
use std::collections::HashMap;

use agnt5_sdk_core::chat;

// ---------------------------------------------------------------------------
// Type wrappers
// ---------------------------------------------------------------------------

/// A chat platform (slack, discord, teams, telegram).
#[pyclass(name = "Platform")]
#[derive(Clone)]
pub struct PyPlatform {
    pub(crate) inner: chat::Platform,
}

#[pymethods]
impl PyPlatform {
    #[staticmethod]
    fn slack() -> Self {
        Self { inner: chat::Platform::Slack }
    }
    #[staticmethod]
    fn discord() -> Self {
        Self { inner: chat::Platform::Discord }
    }
    #[staticmethod]
    fn teams() -> Self {
        Self { inner: chat::Platform::Teams }
    }
    #[staticmethod]
    fn telegram() -> Self {
        Self { inner: chat::Platform::Telegram }
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let p: chat::Platform = s
            .parse()
            .map_err(|e: String| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner: p })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Platform('{}')", self.inner)
    }
}

/// A user on a chat platform.
#[pyclass(name = "ChatUser")]
#[derive(Clone)]
pub struct PyChatUser {
    pub(crate) inner: chat::ChatUser,
}

#[pymethods]
impl PyChatUser {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn platform(&self) -> PyPlatform {
        PyPlatform { inner: self.inner.platform }
    }

    fn __repr__(&self) -> String {
        format!("ChatUser(id='{}', name='{}')", self.inner.id, self.inner.name)
    }
}

/// A file attachment.
#[pyclass(name = "Attachment")]
#[derive(Clone)]
pub struct PyAttachment {
    pub(crate) inner: chat::Attachment,
}

#[pymethods]
impl PyAttachment {
    #[getter]
    fn filename(&self) -> &str {
        &self.inner.filename
    }
    #[getter]
    fn mime_type(&self) -> Option<&str> {
        self.inner.mime_type.as_deref()
    }
    #[getter]
    fn url(&self) -> Option<&str> {
        self.inner.url.as_deref()
    }
    #[getter]
    fn size_bytes(&self) -> Option<u64> {
        self.inner.size_bytes
    }
}

/// A normalized chat message from any platform.
#[pyclass(name = "ChatMessage")]
#[derive(Clone)]
pub struct PyChatMessage {
    pub(crate) inner: chat::ChatMessage,
}

#[pymethods]
impl PyChatMessage {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    #[getter]
    fn platform(&self) -> PyPlatform {
        PyPlatform { inner: self.inner.platform }
    }
    #[getter]
    fn channel_id(&self) -> &str {
        &self.inner.channel_id
    }
    #[getter]
    fn thread_id(&self) -> Option<&str> {
        self.inner.thread_id.as_deref()
    }
    #[getter]
    fn author(&self) -> PyChatUser {
        PyChatUser { inner: self.inner.author.clone() }
    }
    #[getter]
    fn content(&self) -> &str {
        &self.inner.content
    }
    #[getter]
    fn attachments(&self) -> Vec<PyAttachment> {
        self.inner
            .attachments
            .iter()
            .map(|a| PyAttachment { inner: a.clone() })
            .collect()
    }
    #[getter]
    fn is_mention(&self) -> bool {
        self.inner.is_mention
    }
    #[getter]
    fn is_dm(&self) -> bool {
        self.inner.is_dm
    }
    #[getter]
    fn raw_payload(&self) -> &[u8] {
        &self.inner.raw_payload
    }
    #[getter]
    fn metadata(&self) -> HashMap<String, String> {
        self.inner.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChatMessage(id='{}', channel='{}', content='{}')",
            self.inner.id,
            self.inner.channel_id,
            truncate(&self.inner.content, 50),
        )
    }
}

/// Normalized event from a chat platform.
#[pyclass(name = "ChatEvent")]
#[derive(Clone)]
pub struct PyChatEvent {
    pub(crate) inner: chat::ChatEvent,
}

#[pymethods]
impl PyChatEvent {
    /// Event type: "message", "mention", "reaction", "slash_command", "action", "url_verification"
    #[getter]
    fn event_type(&self) -> &str {
        match &self.inner {
            chat::ChatEvent::Message(_) => "message",
            chat::ChatEvent::Mention(_) => "mention",
            chat::ChatEvent::Reaction { .. } => "reaction",
            chat::ChatEvent::SlashCommand { .. } => "slash_command",
            chat::ChatEvent::Action { .. } => "action",
            chat::ChatEvent::UrlVerification { .. } => "url_verification",
        }
    }

    /// The message (for Message and Mention events). None for other event types.
    #[getter]
    fn message(&self) -> Option<PyChatMessage> {
        match &self.inner {
            chat::ChatEvent::Message(m) | chat::ChatEvent::Mention(m) => {
                Some(PyChatMessage { inner: m.clone() })
            }
            _ => None,
        }
    }

    /// Channel ID (if applicable).
    #[getter]
    fn channel_id(&self) -> Option<String> {
        self.inner.channel_id().map(|s| s.to_string())
    }

    /// Thread ID (if applicable).
    #[getter]
    fn thread_id(&self) -> Option<String> {
        self.inner.thread_id().map(|s| s.to_string())
    }

    /// User who triggered the event (if applicable).
    #[getter]
    fn user(&self) -> Option<PyChatUser> {
        self.inner.user().map(|u| PyChatUser { inner: u.clone() })
    }

    /// The challenge string (for url_verification events only).
    #[getter]
    fn challenge(&self) -> Option<String> {
        match &self.inner {
            chat::ChatEvent::UrlVerification { challenge } => Some(challenge.clone()),
            _ => None,
        }
    }

    /// Emoji name (for reaction events only).
    #[getter]
    fn emoji(&self) -> Option<String> {
        match &self.inner {
            chat::ChatEvent::Reaction { emoji, .. } => Some(emoji.clone()),
            _ => None,
        }
    }

    /// Slash command name (for slash_command events only).
    #[getter]
    fn command(&self) -> Option<String> {
        match &self.inner {
            chat::ChatEvent::SlashCommand { command, .. } => Some(command.clone()),
            _ => None,
        }
    }

    /// Slash command arguments (for slash_command events only).
    #[getter]
    fn args(&self) -> Option<String> {
        match &self.inner {
            chat::ChatEvent::SlashCommand { args, .. } => Some(args.clone()),
            _ => None,
        }
    }

    /// Action ID (for action events only).
    #[getter]
    fn action_id(&self) -> Option<String> {
        match &self.inner {
            chat::ChatEvent::Action { action_id, .. } => Some(action_id.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ChatEvent(type='{}')", self.event_type())
    }
}

/// HTTP request spec built by Rust, executed by the language SDK's HTTP client.
#[pyclass(name = "PlatformRequest")]
#[derive(Clone)]
pub struct PyPlatformRequest {
    pub(crate) inner: chat::PlatformRequest,
}

#[pymethods]
impl PyPlatformRequest {
    #[getter]
    fn url(&self) -> &str {
        &self.inner.url
    }
    #[getter]
    fn method(&self) -> &str {
        &self.inner.method
    }
    #[getter]
    fn headers(&self) -> HashMap<String, String> {
        self.inner.headers.clone()
    }
    #[getter]
    fn body(&self) -> &[u8] {
        &self.inner.body
    }

    fn __repr__(&self) -> String {
        format!("PlatformRequest({} {})", self.inner.method, self.inner.url)
    }
}

/// Streaming message buffer with markdown healing.
#[pyclass(name = "StreamingMessageBuffer")]
pub struct PyStreamingMessageBuffer {
    inner: chat::StreamingMessageBuffer,
}

#[pymethods]
impl PyStreamingMessageBuffer {
    #[new]
    #[pyo3(signature = (flush_interval_ms=500))]
    fn new(flush_interval_ms: u64) -> Self {
        Self {
            inner: chat::StreamingMessageBuffer::new(flush_interval_ms),
        }
    }

    /// Push a new token into the buffer.
    fn push(&mut self, token: &str) {
        self.inner.push(token);
    }

    /// Check whether enough time has passed to flush.
    fn should_flush(&self) -> bool {
        self.inner.should_flush()
    }

    /// Flush the buffer, returning markdown-healed content (or None if no new content).
    fn flush(&mut self) -> Option<String> {
        self.inner.flush()
    }

    /// Finalize the buffer, returning complete raw content.
    fn finalize(&mut self) -> String {
        self.inner.finalize()
    }

    /// Current accumulated content length.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if buffer is empty.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Verify a webhook signature for the given platform.
///
/// Returns True if valid, False if invalid signature.
/// Raises ValueError for unsupported platforms or missing headers.
#[pyfunction]
#[pyo3(signature = (platform, secret, headers, body))]
fn verify_webhook(
    platform: &PyPlatform,
    secret: &str,
    headers: HashMap<String, String>,
    body: &[u8],
) -> PyResult<bool> {
    chat::verify_webhook(&platform.inner, secret, &headers, body)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Parse a raw webhook body into a normalized ChatEvent.
///
/// Raises ValueError for unsupported platforms or malformed payloads.
#[pyfunction]
#[pyo3(signature = (platform, body))]
fn parse_event(platform: &PyPlatform, body: &[u8]) -> PyResult<PyChatEvent> {
    let event = chat::parse_event(&platform.inner, body)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyChatEvent { inner: event })
}

/// Build a Slack chat.postMessage request.
#[pyfunction]
#[pyo3(signature = (token, channel, text, thread_ts=None))]
fn slack_post_message(
    token: &str,
    channel: &str,
    text: &str,
    thread_ts: Option<&str>,
) -> PyPlatformRequest {
    PyPlatformRequest {
        inner: chat::request_builder::slack_post_message(token, channel, text, thread_ts),
    }
}

/// Build a Slack chat.update request (for streaming post-then-edit).
#[pyfunction]
fn slack_update_message(
    token: &str,
    channel: &str,
    message_ts: &str,
    text: &str,
) -> PyPlatformRequest {
    PyPlatformRequest {
        inner: chat::request_builder::slack_update_message(token, channel, message_ts, text),
    }
}

/// Build a Slack chat.postEphemeral request.
#[pyfunction]
#[pyo3(signature = (token, channel, user, text, thread_ts=None))]
fn slack_post_ephemeral(
    token: &str,
    channel: &str,
    user: &str,
    text: &str,
    thread_ts: Option<&str>,
) -> PyPlatformRequest {
    PyPlatformRequest {
        inner: chat::request_builder::slack_post_ephemeral(token, channel, user, text, thread_ts),
    }
}

/// Build a Slack reactions.add request.
#[pyfunction]
fn slack_add_reaction(
    token: &str,
    channel: &str,
    message_ts: &str,
    emoji: &str,
) -> PyPlatformRequest {
    PyPlatformRequest {
        inner: chat::request_builder::slack_add_reaction(token, channel, message_ts, emoji),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_chat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let chat_module = PyModule::new(m.py(), "chat")?;
    chat_module.add_class::<PyPlatform>()?;
    chat_module.add_class::<PyChatUser>()?;
    chat_module.add_class::<PyAttachment>()?;
    chat_module.add_class::<PyChatMessage>()?;
    chat_module.add_class::<PyChatEvent>()?;
    chat_module.add_class::<PyPlatformRequest>()?;
    chat_module.add_class::<PyStreamingMessageBuffer>()?;
    chat_module.add_function(wrap_pyfunction!(verify_webhook, &chat_module)?)?;
    chat_module.add_function(wrap_pyfunction!(parse_event, &chat_module)?)?;
    chat_module.add_function(wrap_pyfunction!(slack_post_message, &chat_module)?)?;
    chat_module.add_function(wrap_pyfunction!(slack_update_message, &chat_module)?)?;
    chat_module.add_function(wrap_pyfunction!(slack_post_ephemeral, &chat_module)?)?;
    chat_module.add_function(wrap_pyfunction!(slack_add_reaction, &chat_module)?)?;
    m.add_submodule(&chat_module)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
