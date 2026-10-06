"""A worker given a list of workflows or functions serves only those (AGNT5-1401).

``@workflow`` and ``@function`` register a component process-wide as soon as
its module is imported. A worker given ``workflows=[...]`` used to register and
run every imported workflow anyway, so a cron workflow dropped from that list
kept its schedule, and kept running, after the deploy that dropped it.
``functions=[...]`` had the same gap: an unlisted function was still
registered, and callable through the API.
"""

from types import SimpleNamespace

import pytest

from agnt5 import Agent, Context, WorkflowContext, function, tool, workflow
from agnt5.agent import AgentRegistry
from agnt5.function import FunctionRegistry
from agnt5.tool import ToolRegistry
from agnt5.worker._core import Worker
from agnt5.worker._prompt_executor import PROMPT_EXECUTOR_COMPONENT_NAME
from agnt5.workflow import WorkflowRegistry


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    # Worker() sets AGNT5_WORKER_MODE; deleting it first restores its absence afterwards.
    monkeypatch.delenv("AGNT5_WORKER_MODE", raising=False)
    WorkflowRegistry.clear()
    FunctionRegistry.clear()
    AgentRegistry.clear()
    ToolRegistry.clear()
    yield
    WorkflowRegistry.clear()
    FunctionRegistry.clear()
    AgentRegistry.clear()
    ToolRegistry.clear()


def define_workflows():
    @workflow(cron="* * * * *")
    async def sync_inventory(ctx: WorkflowContext) -> str:
        return "synced"

    @workflow
    async def health_check(ctx: WorkflowContext) -> str:
        return "ok"

    return sync_inventory, health_check


def define_functions():
    @function
    async def lookup_order(ctx: Context) -> str:
        return "order"

    @function
    async def refund_order(ctx: Context) -> str:
        return "refunded"

    return lookup_order, refund_order


def registered(worker: Worker, component_type: str) -> dict:
    return {
        component.name: component
        for component in worker._discover_components()
        if component.component_type == component_type
    }


def request(component_name: str, component_type: str = "workflow") -> SimpleNamespace:
    return SimpleNamespace(
        invocation_id=f"run-{component_name}",
        component_name=component_name,
        component_type=component_type,
        input_data=b"{}",
        attempt=0,
    )


# --- workflows -----------------------------------------------------------------


def test_worker_registers_only_the_workflows_it_is_given(fake_native_core):
    _, health_check = define_workflows()

    worker = Worker(service_name="ops", workflows=[health_check])

    assert set(registered(worker, "workflow")) == {"health_check"}


def test_a_renamed_workflow_is_matched_by_identity_not_function_name(fake_native_core):
    @workflow(name="nightly_sync", cron="0 2 * * *")
    async def sync(ctx: WorkflowContext) -> str:
        return "synced"

    worker = Worker(service_name="ops", workflows=[sync])

    workflows = registered(worker, "workflow")
    assert set(workflows) == {"nightly_sync"}
    assert workflows["nightly_sync"].metadata["cron"] == "0 2 * * *"


async def test_a_workflow_the_worker_was_not_given_is_refused(fake_native_core, monkeypatch):
    _, health_check = define_workflows()
    worker = Worker(service_name="ops", workflows=[health_check])
    executed = []

    async def execute_workflow(config, input_data, request):
        executed.append(config.name)
        return "executed"

    monkeypatch.setattr(worker, "_execute_workflow", execute_workflow)
    handle = worker._create_message_handler()

    refused = await handle(request("sync_inventory"))
    assert refused.success is False
    assert refused.error_message == "Component 'sync_inventory' of type 'workflow' not found"
    assert await handle(request("health_check")) == "executed"
    assert executed == ["health_check"]


def test_register_components_adds_workflows_to_the_list(fake_native_core):
    sync_inventory, health_check = define_workflows()
    worker = Worker(service_name="ops", workflows=[health_check])

    worker.register_components(workflows=[sync_inventory])

    assert set(registered(worker, "workflow")) == {"sync_inventory", "health_check"}


def test_naming_workflows_after_construction_limits_the_worker_to_them(fake_native_core):
    _, health_check = define_workflows()
    worker = Worker(service_name="ops")

    worker.register_components(workflows=[health_check])

    assert set(registered(worker, "workflow")) == {"health_check"}


def test_an_empty_workflows_list_serves_no_workflows(fake_native_core):
    define_workflows()

    worker = Worker(service_name="ops", workflows=[])

    assert registered(worker, "workflow") == {}


def test_a_worker_given_no_workflows_list_serves_every_imported_workflow(fake_native_core):
    # Unchanged, and what the serverless entrypoint does with workflows=None.
    define_workflows()

    worker = Worker(service_name="ops")

    assert set(registered(worker, "workflow")) == {"sync_inventory", "health_check"}


# --- functions -----------------------------------------------------------------


def test_worker_registers_only_the_functions_it_is_given(fake_native_core):
    lookup_order, _ = define_functions()

    worker = Worker(service_name="ops", functions=[lookup_order])

    # The built-in prompt executor is always served.
    assert set(registered(worker, "function")) == {"lookup_order", PROMPT_EXECUTOR_COMPONENT_NAME}


async def test_a_function_the_worker_was_not_given_is_refused(fake_native_core, monkeypatch):
    lookup_order, _ = define_functions()
    worker = Worker(service_name="ops", functions=[lookup_order])
    executed = []

    async def execute_function(config, input_data, request):
        executed.append(config.name)
        return "executed"

    monkeypatch.setattr(worker, "_execute_function", execute_function)
    handle = worker._create_message_handler()

    refused = await handle(request("refund_order", "function"))
    assert refused.success is False
    assert refused.error_message == "Component 'refund_order' of type 'function' not found"
    assert await handle(request("lookup_order", "function")) == "executed"
    assert await handle(request(PROMPT_EXECUTOR_COMPONENT_NAME, "function")) == "executed"
    assert executed == ["lookup_order", PROMPT_EXECUTOR_COMPONENT_NAME]


def test_a_worker_given_no_functions_list_serves_every_imported_function(fake_native_core):
    define_functions()

    worker = Worker(service_name="ops")

    assert set(registered(worker, "function")) == {
        "lookup_order",
        "refund_order",
        PROMPT_EXECUTOR_COMPONENT_NAME,
    }


@pytest.mark.parametrize("kind", ["tool", "agent"])
@pytest.mark.parametrize("listed", [True, False])
async def test_unlisted_tool_or_agent_is_refused(fake_native_core, monkeypatch, kind, listed):
    @tool(name="served_tool")
    async def served_tool(ctx: Context) -> str:
        return "served"

    @tool(name="unlisted_tool")
    async def unlisted_tool(ctx: Context) -> str:
        return "unlisted"

    served_agent = Agent(name="served_agent", model="openai/gpt-test", instructions="Serve")
    Agent(name="unlisted_agent", model="openai/gpt-test", instructions="Unlisted")
    components = {"tools": [served_tool], "agents": [served_agent]} if listed else {}
    worker = Worker(service_name="restricted", **components)
    executed = []

    async def execute(component, input_data, request):
        executed.append(component.name)
        return "executed"

    monkeypatch.setattr(worker, f"_execute_{kind}", execute)
    handler = worker._create_message_handler()
    refused = await handler(request(f"unlisted_{kind}", kind))
    assert refused.success is False
    assert executed == []
    if listed:
        assert await handler(request(f"served_{kind}", kind)) == "executed"
        assert set(registered(worker, kind)) == {f"served_{kind}"}
