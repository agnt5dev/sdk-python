"""Publishing MCPServer definitions with the deployment (AGNT5-1569)."""


import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agnt5 import MCPServer, Worker, function, workflow
from agnt5.function import FunctionContext, FunctionRegistry
from agnt5.mcp.publish import (
    MAX_VIEW_BYTES,
    MAX_VIEWS_BYTES,
    RUN_VIEW,
    MCPServerRegistry,
    check_views_budget,
    registration_size,
    valid_server_name,
)
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


def test_view_none_turns_the_run_card_off():
    server = MCPServer("support")
    server.add_workflow("triage_ticket", triage)
    server.add_workflow("quiet_triage", triage, view=None)
    server.add_agent("support_agent", SimpleNamespace(name="mcp_test_agent", instructions="Help."), view=None)
    server.add_function("lookup", lookup_order, mode="background", view=RUN_VIEW)
    tools = {tool["name"]: tool for tool in server.definition()["tools"]}

    assert "view" not in tools["triage_ticket"], "the default (the run card) is left to the platform"
    assert "view" not in tools["lookup"]
    assert tools["quiet_triage"]["view"] == "none"
    assert tools["support_agent"]["view"] == "none"

    with pytest.raises(ValueError, match="view must be"):
        server.add_workflow("custom", triage, view="board")


BOARD = "<!doctype html><title>Order</title><p>An order board ✓</p>"


def test_add_view_ships_a_view_tools_name(tmp_path):
    server = MCPServer("support")
    board = server.add_view("order", BOARD)
    assert (board.name, board.server, board.size) == ("order", "support", len(BOARD.encode()))
    assert board.sha256 == hashlib.sha256(BOARD.encode()).hexdigest()
    assert BOARD not in repr(board), "the bundle stays out of reprs"

    built = tmp_path / "receipt.html"
    built.write_text("<!doctype html><p>Receipt</p>", encoding="utf-8")
    receipt = server.add_view("receipt", path=built)

    # Any mode may show a view: sync tools too. By handle or by name.
    server.add_function("lookup", lookup_order, view=board)
    server.add_workflow("triage_ticket", triage, view="receipt")
    server.add_workflow("plain", triage)
    definition = server.definition()
    tools = {tool["name"]: tool for tool in definition["tools"]}
    assert tools["lookup"]["view"] == "order"
    assert tools["triage_ticket"]["view"] == "receipt"
    assert "view" not in tools["plain"]
    assert definition["views"] == [
        {"name": "order", "sha256": board.sha256, "size": board.size, "html": BOARD},
        {"name": "receipt", "sha256": receipt.sha256, "size": receipt.size, "html": receipt.html},
    ]
    assert server.views == {"order": board, "receipt": receipt}
    json.dumps(definition)


def test_add_view_is_checked_where_it_is_written(tmp_path):
    server = MCPServer("support")
    with pytest.raises(ValueError, match="lowercase"):
        server.add_view("Order Board", BOARD)
    for reserved in ("run", "none"):
        with pytest.raises(ValueError, match="reserved"):
            server.add_view(reserved, BOARD)
    with pytest.raises(ValueError, match="one of html= or path="):
        server.add_view("order")
    with pytest.raises(ValueError, match="one of html= or path="):
        server.add_view("order", BOARD, path=tmp_path / "x.html")
    with pytest.raises(ValueError, match="does not exist. Build it first"):
        server.add_view("order", path=tmp_path / "missing.html")
    with pytest.raises(ValueError, match="no HTML"):
        server.add_view("order", "  ")
    with pytest.raises(ValueError, match="the limit is 2097152"):
        server.add_view("order", "x" * (MAX_VIEW_BYTES + 1))
    server.add_view("order", BOARD)
    with pytest.raises(ValueError, match="already has a view"):
        server.add_view("order", BOARD)

    # A tool names one of its own server's views, added first.
    other = MCPServer("billing")
    invoice = other.add_view("invoice", BOARD)
    with pytest.raises(ValueError, match="belongs to MCP server 'billing'"):
        server.add_function("lookup", lookup_order, view=invoice)
    with pytest.raises(ValueError, match="view must be"):
        server.add_function("lookup", lookup_order, view="chart")


def test_views_are_budgeted_as_they_travel():
    # The definition is a JSON string inside a JSON webhook: escaped twice.
    for html in [BOARD, '<script>const a = "x\\y";</script>', "line\none\ttab\x01", "✓ é 注 😀"]:
        twice = json.dumps(json.dumps(html, ensure_ascii=False)[1:-1], ensure_ascii=False)[1:-1]
        assert registration_size(html) == len(twice.encode()), html

    # Within 2 MB, but four times that escaped twice: refused for the server.
    quotes = '"' * (1024 * 1024)
    server = MCPServer("one")
    with pytest.raises(ValueError, match="take 4194304 bytes of the registration"):
        server.add_view("a", quotes)
    assert MAX_VIEWS_BYTES < 4 * 1024 * 1024, "registrations are accepted up to 4 MB"


def test_the_worker_budgets_the_views_of_published_servers_only():
    big = "x" * MAX_VIEW_BYTES
    one, two, stdio = MCPServer("one"), MCPServer("two"), MCPServer("stdio-only")
    one.add_view("a", big)
    two.add_view("b", big)  # each server is within its own budget
    stdio.add_view("c", big)
    check_views_budget([one])  # unpublished servers' views don't count
    with pytest.raises(ValueError, match=r"take 4194304 bytes .* \(one/a 2097152, two/b 2097152\)"):
        check_views_budget([one, two])


def test_worker_refuses_to_register_past_the_views_budget():
    @function(name="mcp_budget_test_lookup")
    async def lookup(ctx: FunctionContext, order_id: str) -> dict:
        return {}

    try:
        for name in ("one", "two"):
            server = MCPServer(name)
            server.add_function("lookup", lookup, view=server.add_view("big", "x" * MAX_VIEW_BYTES))
        MCPServer("stdio-only").add_view("big", "x" * MAX_VIEW_BYTES)
        with pytest.raises(ValueError, match="bytes of its registration together"):
            Worker(service_name="py-worker")._discover_components()
        MCPServerRegistry.discard("two")
        Worker(service_name="py-worker")._discover_components()  # the unpublished one doesn't count
    finally:
        FunctionRegistry.discard("mcp_budget_test_lookup")


def test_html_with_unpaired_surrogates_is_refused():
    server = MCPServer("support")
    for html in ["<p>\ud800</p>", "<p>\udc00</p>"]:
        with pytest.raises(ValueError, match="unpaired UTF-16 surrogates"):
            server.add_view("order", html)
    assert server.add_view("order", "<p>😀</p>").size == len("<p>😀</p>".encode())


def test_names_must_match_whole(tmp_path):
    server = MCPServer("support")
    with pytest.raises(ValueError, match="lowercase"):
        server.add_view("order\n", BOARD)
    with pytest.raises(ValueError, match="1 to 128"):
        server.add_function("lookup\n", lookup_order)
    assert not valid_server_name("support\n")
    assert valid_server_name("support")


def test_oversized_files_are_refused_before_they_are_read(tmp_path, monkeypatch):
    huge = tmp_path / "huge.html"
    with open(huge, "wb") as f:
        f.truncate(MAX_VIEW_BYTES + 1)

    def no_read(*args, **kwargs):
        raise AssertionError("the file was opened")

    monkeypatch.setattr("builtins.open", no_read)
    with pytest.raises(ValueError, match="the limit is 2097152"):
        MCPServer("support").add_view("order", path=huge)


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
        board = server.add_view("board", BOARD)
        server.add_function("lookup_order", lookup, view=board)
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
        # The bundle travels with the registration, byte for byte.
        assert definition["views"][0]["html"] == BOARD
        assert definition["views"][0]["sha256"] == hashlib.sha256(BOARD.encode()).hexdigest()
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
