//! Feature-gated Python bridge for durable activation V1.

use agnt5_sdk_core::client::EngineClient;
use agnt5_sdk_core::error::{ErrorCode, SdkError};
use agnt5_sdk_core::pb::{
    activation_payload, ActivationEvidence, ActivationExternalOutcomeCertainty, ActivationPayload,
    ActivationStatus, ActivationUsage, BeginActivationRequest, ChildActivationLinkage,
    CompleteActivationRequest, FailActivationRequest,
};
use agnt5_sdk_core::runtime_adapter::{ActivationAdapter, ActivationDecision};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::sync::Arc;
use tokio::sync::Mutex;

const NATIVE_ACTIVATION_ERROR_PREFIX: &str = "AGNT5_ACTIVATION_ERROR:";

fn inline_evidence(entries: Vec<(String, Vec<u8>, Vec<u8>)>) -> Vec<ActivationEvidence> {
    entries
        .into_iter()
        .map(|(evidence_type, payload, sha256)| ActivationEvidence {
            evidence_type,
            payload: Some(ActivationPayload {
                value: Some(activation_payload::Value::InlineData(payload)),
            }),
            sha256,
        })
        .collect()
}

fn activation_error_code(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::DurabilityUnavailable => "DURABILITY_UNAVAILABLE",
        ErrorCode::NondeterministicReplay => "NON_DETERMINISTIC_REPLAY",
        ErrorCode::StaleAuthority => "STALE_AUTHORITY",
        ErrorCode::ActivationCancelled => "CANCELLED",
        ErrorCode::ActivationContended => "CONTENDED",
        ErrorCode::PayloadConflict => "PAYLOAD_CONFLICT",
        ErrorCode::IllegalTransition => "ILLEGAL_TRANSITION",
        ErrorCode::StateVersionConflict => "STATE_VERSION_CONFLICT",
        ErrorCode::RequiredChildUnresolved => "REQUIRED_CHILD_UNRESOLVED",
        ErrorCode::InvalidInput | ErrorCode::InvalidMessage | ErrorCode::InvalidState => {
            "INVALID_ARGUMENT"
        }
        _ => "UNKNOWN_OUTCOME",
    }
}

fn bridge_error(
    code: &str,
    message: impl Into<String>,
    activation_id: &str,
    attempt: u32,
) -> PyErr {
    let payload = serde_json::json!({
        "code": code,
        "message": message.into(),
        "activationId": activation_id,
        "attempt": attempt,
    });
    pyo3::exceptions::PyRuntimeError::new_err(format!("{NATIVE_ACTIVATION_ERROR_PREFIX}{payload}"))
}

fn activation_error(error: SdkError) -> PyErr {
    let code = activation_error_code(error.code());
    let (message, activation_id, attempt) = match error {
        SdkError::Activation {
            message,
            activation_id,
            attempt,
            ..
        } => (
            message,
            activation_id.unwrap_or_default(),
            attempt.unwrap_or_default(),
        ),
        other => (other.to_string(), String::new(), 0),
    };
    bridge_error(code, message, &activation_id, attempt)
}

#[pyclass]
pub struct PyActivationClient {
    adapter: Arc<Mutex<Option<ActivationAdapter>>>,
    endpoint: String,
}

#[pyclass]
#[derive(Clone)]
pub struct PyActivationDecision {
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub activation_id: String,
    #[pyo3(get)]
    pub attempt: u32,
    #[pyo3(get)]
    pub accepted_journal_offset: u64,
    #[pyo3(get)]
    pub fence_token: Vec<u8>,
    #[pyo3(get)]
    pub replay_output: Option<Vec<u8>>,
    #[pyo3(get)]
    pub message: String,
}

#[pyclass]
#[derive(Clone)]
pub struct PyActivationCompletionReceipt {
    #[pyo3(get)]
    pub activation_id: String,
    #[pyo3(get)]
    pub attempt: u32,
    #[pyo3(get)]
    pub accepted_journal_offset: u64,
    #[pyo3(get)]
    pub replayed: bool,
}

#[pyclass]
#[derive(Clone)]
pub struct PyActivationFailureReceipt {
    #[pyo3(get)]
    pub activation_id: String,
    #[pyo3(get)]
    pub attempt: u32,
    #[pyo3(get)]
    pub accepted_journal_offset: u64,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub replayed: bool,
}

impl PyActivationClient {
    async fn connected_adapter(
        adapter: Arc<Mutex<Option<ActivationAdapter>>>,
        endpoint: String,
    ) -> PyResult<ActivationAdapter> {
        let mut guard = adapter.lock().await;
        if guard.is_none() {
            let client = EngineClient::connect(&endpoint)
                .await
                .map_err(activation_error)?;
            *guard = Some(ActivationAdapter::new(client));
        }
        Ok(guard
            .as_ref()
            .expect("activation adapter initialized")
            .clone())
    }
}

fn decision_to_python(decision: ActivationDecision) -> PyResult<PyActivationDecision> {
    match decision {
        ActivationDecision::Execute(receipt) => Ok(PyActivationDecision {
            kind: "EXECUTE".to_string(),
            activation_id: receipt.activation_id,
            attempt: receipt.attempt,
            accepted_journal_offset: receipt.accepted_journal_offset,
            fence_token: receipt.fence_token,
            replay_output: None,
            message: String::new(),
        }),
        ActivationDecision::Replay(receipt) => {
            let replay_output = match receipt.result.value {
                Some(activation_payload::Value::InlineData(value)) => value,
                Some(activation_payload::Value::Reference(_)) => {
                    return Err(bridge_error(
                        "REFERENCE_REQUIRED",
                        "Python activation replay does not yet support referenced payloads",
                        &receipt.activation_id,
                        receipt.attempt,
                    ));
                }
                None => {
                    return Err(bridge_error(
                        "UNKNOWN_OUTCOME",
                        "activation replay receipt has no canonical payload",
                        &receipt.activation_id,
                        receipt.attempt,
                    ));
                }
            };
            Ok(PyActivationDecision {
                kind: "REPLAY".to_string(),
                activation_id: receipt.activation_id,
                attempt: receipt.attempt,
                accepted_journal_offset: receipt.accepted_journal_offset,
                fence_token: Vec::new(),
                replay_output: Some(replay_output),
                message: String::new(),
            })
        }
        ActivationDecision::Wait {
            activation_id,
            receipt,
        } => Ok(PyActivationDecision {
            kind: "WAIT".to_string(),
            activation_id,
            attempt: receipt.attempt,
            accepted_journal_offset: receipt.accepted_journal_offset,
            fence_token: Vec::new(),
            replay_output: None,
            message: "activation attempt is owned by another worker session".to_string(),
        }),
        ActivationDecision::Conflict {
            activation_id,
            receipt,
        } => Ok(PyActivationDecision {
            kind: "CONFLICT".to_string(),
            activation_id,
            attempt: 0,
            accepted_journal_offset: 0,
            fence_token: Vec::new(),
            replay_output: None,
            message: receipt.message,
        }),
        ActivationDecision::Cancelled {
            activation_id,
            attempt,
            accepted_journal_offset,
        } => Ok(PyActivationDecision {
            kind: "CANCELLED".to_string(),
            activation_id,
            attempt,
            accepted_journal_offset,
            fence_token: Vec::new(),
            replay_output: None,
            message: "activation is cancelled".to_string(),
        }),
        ActivationDecision::UnknownOutcome {
            activation_id,
            receipt,
        } => Ok(PyActivationDecision {
            kind: "UNKNOWN_OUTCOME".to_string(),
            activation_id,
            attempt: 0,
            accepted_journal_offset: receipt.accepted_journal_offset,
            fence_token: Vec::new(),
            replay_output: None,
            message: receipt.error_code,
        }),
    }
}

fn status_name(status: ActivationStatus) -> &'static str {
    match status {
        ActivationStatus::Active => "ACTIVE",
        ActivationStatus::RetryReady => "RETRY_READY",
        ActivationStatus::Completed => "COMPLETED",
        ActivationStatus::Failed => "FAILED",
        ActivationStatus::Suspended => "SUSPENDED",
        ActivationStatus::Cancelled => "CANCELLED",
        ActivationStatus::UnknownOutcome => "UNKNOWN_OUTCOME",
        ActivationStatus::Unspecified => "UNSPECIFIED",
    }
}

#[pymethods]
impl PyActivationClient {
    #[new]
    #[pyo3(signature = (endpoint = None))]
    pub fn new(endpoint: Option<String>) -> Self {
        let endpoint = endpoint
            .or_else(|| std::env::var("AGNT5_ENGINE_URL").ok())
            .or_else(|| std::env::var("AGNT5_COORDINATOR_ENDPOINT").ok())
            .unwrap_or_else(|| "http://localhost:34186".to_string());
        Self {
            adapter: Arc::new(Mutex::new(None)),
            endpoint,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_activation<'py>(
        &self,
        py: Python<'py>,
        project_id: String,
        run_id: String,
        parent_activation_id: String,
        kind: i32,
        stable_key: String,
        input_digest: Vec<u8>,
        definition_digest: Vec<u8>,
        recovery_policy: i32,
        worker_session_id: String,
        run_authority: Vec<u8>,
        lease_authority: Vec<u8>,
        child: Option<(String, String, String, Vec<u8>, i32)>,
        display_name: String,
        input_data: Vec<u8>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let adapter = self.adapter.clone();
        let endpoint = self.endpoint.clone();
        future_into_py(py, async move {
            let mut adapter = Self::connected_adapter(adapter, endpoint).await?;
            let decision = adapter
                .begin(BeginActivationRequest {
                    project_id,
                    run_id,
                    parent_activation_id,
                    kind,
                    stable_key,
                    input_digest,
                    definition_digest,
                    recovery_policy,
                    worker_session_id,
                    run_authority,
                    lease_authority,
                    child: child.map(
                        |(
                            child_key,
                            child_run_id,
                            child_session_id,
                            child_definition_digest,
                            join_policy,
                        )| ChildActivationLinkage {
                            child_key,
                            child_run_id,
                            child_session_id,
                            child_definition_digest,
                            join_policy,
                        },
                    ),
                    display_name,
                    input_data,
                })
                .await
                .map_err(activation_error)?;
            decision_to_python(decision)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_activation<'py>(
        &self,
        py: Python<'py>,
        project_id: String,
        run_id: String,
        activation_id: String,
        attempt: u32,
        fence_token: Vec<u8>,
        output: Vec<u8>,
        output_digest: Vec<u8>,
        tokens_in: i64,
        tokens_out: i64,
        cost_usd: f64,
        latency_ms: i64,
        provider: String,
        model: String,
        cached_tokens: i64,
        evidence: Vec<(String, Vec<u8>, Vec<u8>)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let adapter = self.adapter.clone();
        let endpoint = self.endpoint.clone();
        future_into_py(py, async move {
            let mut adapter = Self::connected_adapter(adapter, endpoint).await?;
            let receipt = adapter
                .complete(CompleteActivationRequest {
                    project_id,
                    run_id,
                    activation_id,
                    attempt,
                    fence_token,
                    output: Some(ActivationPayload {
                        value: Some(activation_payload::Value::InlineData(output)),
                    }),
                    output_digest,
                    state_mutations: Vec::new(),
                    outbox_intents: Vec::new(),
                    usage: Some(ActivationUsage {
                        tokens_in,
                        tokens_out,
                        cost_usd,
                        latency_ms,
                        provider,
                        model,
                        cached_tokens,
                    }),
                    evidence: inline_evidence(evidence),
                })
                .await
                .map_err(activation_error)?;
            Ok(PyActivationCompletionReceipt {
                activation_id: receipt.activation_id,
                attempt: receipt.attempt,
                accepted_journal_offset: receipt.accepted_journal_offset,
                replayed: receipt.replayed,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn fail_activation<'py>(
        &self,
        py: Python<'py>,
        project_id: String,
        run_id: String,
        activation_id: String,
        attempt: u32,
        fence_token: Vec<u8>,
        error_code: String,
        error_data: Vec<u8>,
        retryable: bool,
        external_outcome_certainty: String,
        evidence: Vec<(String, Vec<u8>, Vec<u8>)>,
        latency_ms: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if external_outcome_certainty != "UNKNOWN" {
            return Err(bridge_error(
                "INVALID_ARGUMENT",
                "Python V1 failure bridge currently requires UNKNOWN external outcome certainty",
                &activation_id,
                attempt,
            ));
        }
        let adapter = self.adapter.clone();
        let endpoint = self.endpoint.clone();
        future_into_py(py, async move {
            let mut adapter = Self::connected_adapter(adapter, endpoint).await?;
            let receipt = adapter
                .fail(FailActivationRequest {
                    project_id,
                    run_id,
                    activation_id,
                    attempt,
                    fence_token,
                    error_code,
                    error_data: Some(ActivationPayload {
                        value: Some(activation_payload::Value::InlineData(error_data)),
                    }),
                    retryable,
                    external_outcome_certainty: ActivationExternalOutcomeCertainty::Unknown as i32,
                    evidence: inline_evidence(evidence),
                    latency_ms,
                })
                .await
                .map_err(activation_error)?;
            Ok(PyActivationFailureReceipt {
                activation_id: receipt.activation_id,
                attempt: receipt.attempt,
                accepted_journal_offset: receipt.accepted_journal_offset,
                status: status_name(receipt.status).to_string(),
                replayed: receipt.replayed,
            })
        })
    }
}

pub fn register_activation_client(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyActivationClient>()?;
    m.add_class::<PyActivationDecision>()?;
    m.add_class::<PyActivationCompletionReceipt>()?;
    m.add_class::<PyActivationFailureReceipt>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_message_does_not_expose_worker_session_authority() {
        let secret_session = "agnt5ws1.payload.signature";
        let decision = ActivationDecision::Wait {
            activation_id: "activation-1".to_string(),
            receipt: agnt5_sdk_core::pb::ActivationWaitReceipt {
                attempt: 2,
                active_worker_session_id: secret_session.to_string(),
                accepted_journal_offset: 7,
            },
        };

        let converted = decision_to_python(decision).expect("WAIT decision");

        assert_eq!(
            converted.message,
            "activation attempt is owned by another worker session"
        );
        assert!(!converted.message.contains(secret_session));
    }

    #[test]
    fn preserves_required_child_error_code() {
        assert_eq!(
            activation_error_code(ErrorCode::RequiredChildUnresolved),
            "REQUIRED_CHILD_UNRESOLVED"
        );
    }
}
