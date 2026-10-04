"""Publishing MCPServer definitions with the deployment (AGNT5-1569)."""


import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agnt5 import MCPServer, Worker, function, workflow
from agnt5.function import FunctionContext, FunctionRegistry
from agnt5.mcp.publish import MCPServerRegistry
from agnt5.workflow import WorkflowContext, WorkflowRegistry


class Order(BaseModel):
    order_id: str
    total: float


@function(name="mcp_test_lookup_order")
async def lookup_order(ctx: FunctionContext, order_id: str) -> Order:
    """Look up an order by ID.

    Longer text that is not part of the tool description.
    """
    return Order(order_id=order_id, total=1.0)


@workflow(name="mcp_test_triage")
async def triage(ctx: WorkflowContext, account: str, max_tickets: int = 25) -> str:
    """Classify new tickets."""
    return "ok"


@pytest.fixture(autouse=True)
def clean_servers():
    MCPServerRegistry.clear()
    yield
    MCPServerRegistry.clear()


def support_server() -> MCPServer:
    server = MCPServer("support", instructions="Order and ticket tools.")
    server.add_function("lookup_order", lookup_order, annotations={"readOnlyHint": True})
    server.add_workflow("triage_ticket", triage, mode="background", annotations={"destructiveHint": True})
    server.add_agent("support_agent", SimpleNamespace(name="mcp_test_agent", instructions="Answer support questions.\nMore."))
    return server


def test_short_constructor_and_old_signature():
    new = MCPServer("support", instructions="Tools.")
    assert (new.info.id, new.info.name, new.info.version) == ("support", "support", "0.1.0")
    old = MCPServer("legacy-id", "Legacy Server", "2.0.0")
    assert (old.info.name, old.info.version) == ("Legacy Server", "2.0.0")
    old_kw = MCPServer(id="kw", name="Kw", version="1.0.0")
    assert old_kw.info.version == "1.0.0"


def test_definition_follows_the_platform_contract():
    definition = support_server().definition()
    assert definition["schema_version"] == 1
    assert definition["name"] == "support"
    assert definition["instructions"] == "Order and ticket tools."

    lookup, triage_tool, agent_tool = definition["tools"]
    assert lookup["component"] == {"type": "function", "name": "mcp_test_lookup_order"}
    assert lookup["description"] == "Look up an order by ID."
    assert lookup["annotations"] == {"readOnlyHint": True}
    assert lookup["input_schema"]["type"] == "object"
    assert "order_id" in lookup["input_schema"]["properties"]
    assert lookup["output_schema"]["type"] == "object", "a Pydantic return type is an object schema"
    assert "mode" not in lookup, "the platform fills in the default mode"

    assert triage_tool["component"] == {"type": "workflow", "name": "mcp_test_triage"}
    assert triage_tool["mode"] == "background"
    # Hints are explicit: unstated readOnlyHint means the tool writes.
    assert triage_tool["annotations"] == {"readOnlyHint": False, "destructiveHint": True}
    assert "output_schema" not in triage_tool, "a str result is not an object schema, so it is left out"
    assert triage_tool["input_schema"]["properties"]["max_tickets"].get("default") == 25

    assert agent_tool["component"] == {"type": "agent", "name": "mcp_test_agent"}
    assert agent_tool["input_schema"]["required"] == ["input"]
    assert agent_tool["description"] == "Answer support questions."
    json.dumps(definition)  # must serialize


def test_options_are_checked_where_they_are_written():
    server = MCPServer("support")
    with pytest.raises(ValueError, match="mode"):
        server.add_function("a", lookup_order, mode="eventually")
    with pytest.raises(ValueError, match="visibility"):
        server.add_function("a", lookup_order, visibility=["browser"])
    with pytest.raises(ValueError, match="unknown annotation"):
        server.add_function("a", lookup_order, annotations={"readonly": True})
    with pytest.raises(ValueError, match="True or False"):
        server.add_function("a", lookup_order, annotations={"readOnlyHint": "yes"})
    with pytest.raises(ValueError, match="reserved"):
        server.add_function("get_run", lookup_order)
    with pytest.raises(ValueError, match="1 to 128"):
        server.add_function("look up", lookup_order)
    with pytest.raises(TypeError, match="@function"):
        server.add_function("plain", lambda: None)
    server.add_function("lookup", lookup_order)
    with pytest.raises(ValueError, match="already has a tool"):
        server.add_workflow("lookup", triage)


def test_stdio_only_servers_are_not_published():
    stdio = MCPServer("legacy", "Legacy", "1.0.0", workflows={"plain": lambda **_: None})
    stdio.add_workflow("also_plain", lambda **_: None)
    assert not stdio.published


def test_worker_registers_published_servers_as_mcp_components():
    # Declared here: other test modules clear the global registries.
    @function(name="mcp_worker_test_lookup")
    async def lookup(ctx: FunctionContext, order_id: str) -> dict:
        """Look up an order."""
        return {}

    @workflow(name="mcp_worker_test_triage")
    async def triage_wf(ctx: WorkflowContext, account: str) -> str:
        return "ok"

    try:
        server = MCPServer("support")
        server.add_function("lookup_order", lookup)
        server.add_workflow("triage_ticket", triage_wf)
        MCPServer("stdio-only")  # nothing published: not registered
        components = Worker(service_name="py-worker")._discover_components()

        mcp = [c for c in components if c.component_type == "mcp"]
        assert [c.name for c in mcp] == ["support"]
        definition = json.loads(mcp[0].definition)
        assert [t["component"]["name"] for t in definition["tools"]] == [
            "mcp_worker_test_lookup",
            "mcp_worker_test_triage",
        ]
        names = {c.name for c in components}
        assert {"mcp_worker_test_lookup", "mcp_worker_test_triage"} <= names, (
            "the tools' components register alongside the server"
        )
    finally:
        FunctionRegistry.discard("mcp_worker_test_lookup")
        WorkflowRegistry.discard("mcp_worker_test_triage")


def teardown_module():
    FunctionRegistry.discard("mcp_test_lookup_order")
    WorkflowRegistry.discard("mcp_test_triage")
