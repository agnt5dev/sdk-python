import json

import pytest

from agnt5.workflows import (
    FlowDefinition,
    WorkflowStep,
    StepType,
    task_step,
    wait_signal_step,
    wait_timer_step,
    register_workflow,
    get_registered_workflows,
    clear_workflows,
)


def test_flow_definition_serialization():
    clear_workflows()

    definition = FlowDefinition(
        steps=[
            task_step(
                name="plan",
                service_name="svc",
                handler_name="plan",
                input_data={"x": 1},
            ),
            wait_signal_step(
                name="await",
                signal_name="ready",
                dependencies=["plan"],
            ),
            wait_timer_step(
                name="retry",
                timer_key="backoff",
                dependencies=["await"],
                delay_ms=5000,
            ),
        ]
    )

    register_workflow("example", definition)

    workflows = get_registered_workflows()
    assert "example" in workflows
    stored = workflows["example"].to_dict()

    assert stored["steps"][0]["service_name"] == "svc"
    assert stored["steps"][1]["type"] == StepType.WAIT_SIGNAL.value
    assert stored["steps"][2]["timer"]["delay_ms"] == 5000

    serialized = definition.to_json()
    parsed = json.loads(serialized)
    assert len(parsed["steps"]) == 3
    assert parsed["steps"][0]["handler_name"] == "plan"


def test_register_workflow_validations():
    clear_workflows()

    definition = FlowDefinition(
        steps=[
            WorkflowStep(name="only", service_name="svc", handler_name="handler"),
        ]
    )

    register_workflow("unique", definition)

    with pytest.raises(ValueError):
        register_workflow("unique", definition)

    bad_definition = FlowDefinition(steps=[])
    with pytest.raises(ValueError):
        register_workflow("empty", bad_definition)
