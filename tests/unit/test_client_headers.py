from agnt5.client import AsyncClient, Client


def test_client_default_headers_include_ambient_deployment_id(monkeypatch):
    monkeypatch.setenv("AGNT5_DEPLOYMENT_ID", "dep-env")
    client = Client()
    try:
        headers = client._build_headers()
    finally:
        client.close()

    assert headers["X-DEPLOYMENT-ID"] == "dep-env"


def test_client_component_execution_headers_omit_ambient_deployment_id(monkeypatch):
    monkeypatch.setenv("AGNT5_DEPLOYMENT_ID", "dep-env")
    client = Client()
    try:
        headers = client._build_headers(include_ambient_deployment_id=False)
    finally:
        client.close()

    assert "X-DEPLOYMENT-ID" not in headers


def test_client_component_execution_headers_keep_explicit_deployment_id(monkeypatch):
    monkeypatch.setenv("AGNT5_DEPLOYMENT_ID", "dep-env")
    client = Client(deployment_id="dep-explicit")
    try:
        headers = client._build_headers(include_ambient_deployment_id=False)
    finally:
        client.close()

    assert headers["X-DEPLOYMENT-ID"] == "dep-explicit"


def test_client_component_execution_headers_allow_per_call_deployment_id(monkeypatch):
    monkeypatch.setenv("AGNT5_DEPLOYMENT_ID", "dep-env")
    client = Client()
    try:
        headers = client._build_headers(
            deployment_id="dep-call",
            include_ambient_deployment_id=False,
        )
    finally:
        client.close()

    assert headers["X-DEPLOYMENT-ID"] == "dep-call"


def test_async_client_component_execution_headers_omit_ambient_deployment_id(monkeypatch):
    monkeypatch.setenv("AGNT5_DEPLOYMENT_ID", "dep-env")
    client = AsyncClient()

    headers = client._build_headers(include_ambient_deployment_id=False)

    assert "X-DEPLOYMENT-ID" not in headers


def test_async_client_component_execution_headers_keep_explicit_deployment_id(monkeypatch):
    monkeypatch.setenv("AGNT5_DEPLOYMENT_ID", "dep-env")
    client = AsyncClient(deployment_id="dep-explicit")

    headers = client._build_headers(include_ambient_deployment_id=False)

    assert headers["X-DEPLOYMENT-ID"] == "dep-explicit"
