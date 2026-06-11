from types import SimpleNamespace

import pytest

from agnt5.worker._executors import (
    _ensure_input_dict,
    _resolve_session_user_ids,
    _truncate_input,
)


def test_request_session_user_ids_win_over_payload():
    request = SimpleNamespace(
        invocation_id="run-1",
        session_id="session-from-request",
        user_id="user-from-request",
    )

    session_id, user_id = _resolve_session_user_ids(
        request,
        {
            "session_id": "session-from-payload",
            "user_id": "user-from-payload",
        },
    )

    assert session_id == "session-from-request"
    assert user_id == "user-from-request"


def test_payload_session_user_ids_are_honored_for_legacy_calls():
    request = SimpleNamespace(invocation_id="run-1", session_id="", user_id="")

    session_id, user_id = _resolve_session_user_ids(
        request,
        {
            "session_id": "session-from-payload",
            "user_id": "user-from-payload",
        },
    )

    assert session_id == "session-from-payload"
    assert user_id == "user-from-payload"


def test_invocation_id_is_ephemeral_session_fallback():
    request = SimpleNamespace(invocation_id="run-1")

    session_id, user_id = _resolve_session_user_ids(request, {})

    assert session_id == "run-1"
    assert user_id is None


def test_resolve_session_user_ids_ignores_non_dict_payload():
    request = SimpleNamespace(invocation_id="run-1", session_id="", user_id="")

    session_id, user_id = _resolve_session_user_ids(request, "Alabama")

    assert session_id == "run-1"
    assert user_id is None


def test_truncate_input_handles_non_dict_payloads():
    assert _truncate_input("Alabama") == "Alabama"
    assert _truncate_input(["a", "b"]) == "['a', 'b']"
    assert _truncate_input(42) == "42"


@pytest.mark.parametrize("payload", ["Alabama", ["a", "b"], 42])
def test_ensure_input_dict_rejects_non_object_dataset_inputs(payload):
    with pytest.raises(ValueError) as exc_info:
        _ensure_input_dict(payload)

    assert "Component input must be a JSON object" in str(exc_info.value)
    assert type(payload).__name__ in str(exc_info.value)
