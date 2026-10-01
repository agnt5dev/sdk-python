"""Sampling parameters, reasoning effort and stop reasons on newer models.

gpt-6 and Claude after Opus 4.6 / Sonnet 4.6 reject ``temperature`` and
``top_p`` with a 400, so an Agent must not send its default temperature to
them (AGNT5-1403, AGNT5-1456). ``reasoning_effort`` used to be dropped between
the Python client and sdk-core, and ``finish_reason`` was hard-coded to None.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agnt5 import Agent
from agnt5.agent.decorator import agent as agent_decorator
from agnt5.lm import GenerateRequest, ReasoningEffort
from agnt5.lm.model_caps import (
    claude_rejects_sampling_params,
    is_openai_reasoning_model,
    rejects_sampling_params,
)


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5-mini",
        "openai/gpt-6-luna",
        "o3-mini",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-fable-5-1",
        "anthropic/claude-haiku-4-6",
        "bedrock/us-east-1/us.anthropic.claude-opus-4-7-20260115-v1:0",
    ],
)
def test_models_that_reject_sampling(model):
    assert rejects_sampling_params(model)


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "groq/openai/gpt-oss-120b",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-20250514",
        "anthropic/claude-3-5-sonnet-20241022",
        "claude-2.1",
    ],
)
def test_models_that_accept_sampling(model):
    assert not rejects_sampling_params(model)


def test_unknown_claude_family_defaults_to_rejecting():
    assert claude_rejects_sampling_params("anthropic/claude-newfamily-1")
    assert not is_openai_reasoning_model("anthropic/claude-opus-5")


@pytest.mark.parametrize("model", ["anthropic/claude-opus-5", "anthropic/claude-sonnet-5"])
def test_agent_omits_default_temperature_for_claude_that_rejects_it(model):
    agent = Agent(name="claude_agent", model=model, instructions="Test")
    request = GenerateRequest(model=agent.model)

    agent._apply_generation_config(request)

    assert request.config.temperature is None


def test_agent_keeps_default_temperature_for_older_claude():
    agent = Agent(name="haiku_agent", model="anthropic/claude-haiku-4-5", instructions="Test")
    request = GenerateRequest(model=agent.model)

    agent._apply_generation_config(request)

    assert request.config.temperature == 0.7


def test_agent_decorator_default_is_not_an_explicit_temperature():
    """@agent used to pass a plain 0.7, which counted as explicit and was sent."""

    @agent_decorator(
        name="decorated_claude_agent",
        model="anthropic/claude-opus-5",
        instructions="Test",
    )
    def decorated():
        pass

    from agnt5.agent import AgentRegistry

    registered = AgentRegistry.get("decorated_claude_agent")
    request = GenerateRequest(model=registered.model)
    registered._apply_generation_config(request)

    assert request.config.temperature is None


@pytest.mark.parametrize("effort", ["none", "low", ReasoningEffort.HIGH])
def test_agent_reasoning_effort_reaches_the_request(effort):
    agent = Agent(
        name="effort_agent",
        model="openai/gpt-6-luna",
        instructions="Test",
        reasoning_effort=effort,
    )
    request = GenerateRequest(model=agent.model)

    agent._apply_generation_config(request)

    assert request.config.reasoning_effort == ReasoningEffort(effort)
    assert agent._model_config_snapshot()["reasoning_effort"] == ReasoningEffort(effort).value


def test_agent_rejects_unknown_reasoning_effort():
    with pytest.raises(ValueError):
        Agent(name="bad_effort", model="openai/gpt-6-luna", instructions="Test", reasoning_effort="max")


class _CaptureHandler(BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        type(self).bodies.append(json.loads(self.rfile.read(length)))
        payload = json.dumps(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 1790000000,
                "status": "completed",
                "model": "gpt-6-luna",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "391", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def openai_capture(monkeypatch):
    _CaptureHandler.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
    yield _CaptureHandler.bodies
    server.shutdown()


async def test_reasoning_effort_is_sent_to_openai(openai_capture):
    from agnt5 import lm

    response = await lm.generate(
        model="openai/gpt-6-luna",
        prompt="What is 17 x 23?",
        temperature=0.7,
        reasoning_effort=ReasoningEffort.LOW,
    )

    body = openai_capture[0]
    assert body["reasoning"]["effort"] == "low"
    assert "temperature" not in body
    assert response.text == "391"
