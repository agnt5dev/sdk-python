from types import SimpleNamespace

from agnt5.worker._executors import _resolve_session_user_ids


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
