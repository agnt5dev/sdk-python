import httpx

from agnt5.client import AsyncClient, Client


def test_clients_default_to_beta_ha_request_timeout():
    client = Client("http://gateway.test")
    try:
        assert client.timeout == 45.0
    finally:
        client.close()

    assert AsyncClient("http://gateway.test").timeout == 45.0


def test_sync_run_uses_client_timeout_when_call_timeout_is_omitted():
    seen_timeout = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_timeout.update(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={"run_id": "run-1", "status": "completed", "output": {"ok": True}},
        )

    client = Client("http://gateway.test", timeout=45.0)
    client.close()
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=client.timeout,
    )
    try:
        response = client.run("greet")
    finally:
        client.close()

    assert response.is_success
    assert set(seen_timeout.values()) == {45.0}
