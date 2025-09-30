"""Workflow definition and registration utilities for the AGNT5 Python SDK."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Supported step kinds matching the platform orchestrator."""

    TASK = "task"
    WAIT_SIGNAL = "wait_signal"
    WAIT_TIMER = "wait_timer"


@dataclass
class SignalConfig:
    """Signal wait configuration."""

    name: str
    timeout_ms: Optional[int] = None
    on_timeout: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"name": self.name}
        if self.timeout_ms is not None:
            data["timeout_ms"] = self.timeout_ms
        if self.on_timeout:
            data["on_timeout"] = self.on_timeout
        return data


@dataclass
class TimerConfig:
    """Timer wait configuration."""

    key: str
    delay_ms: Optional[int] = None
    cron_expr: Optional[str] = None
    max_retries: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"key": self.key}
        if self.delay_ms is not None:
            data["delay_ms"] = self.delay_ms
        if self.cron_expr:
            data["cron_expr"] = self.cron_expr
        if self.max_retries is not None:
            data["max_retries"] = self.max_retries
        return data


@dataclass
class WorkflowStep:
    """Single step in a workflow definition."""

    name: str
    step_type: StepType = StepType.TASK
    service_name: Optional[str] = None
    handler_name: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    object_keys: List[str] = field(default_factory=list)
    input_data: Optional[Dict[str, Any]] = None
    signal: Optional[SignalConfig] = None
    timer: Optional[TimerConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"name": self.name}
        if self.step_type != StepType.TASK:
            data["type"] = self.step_type.value

        if self.service_name:
            data["service_name"] = self.service_name
        if self.handler_name:
            data["handler_name"] = self.handler_name

        if self.dependencies:
            data["dependencies"] = list(self.dependencies)
        if self.object_keys:
            data["object_keys"] = list(self.object_keys)

        if self.input_data is not None:
            data["input_data"] = self.input_data

        if self.signal:
            data["signal"] = self.signal.to_dict()
        if self.timer:
            data["timer"] = self.timer.to_dict()

        return data


@dataclass
class FlowDefinition:
    """Complete workflow definition compatible with the platform orchestrator."""

    steps: List[WorkflowStep]

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


_workflow_registry: Dict[str, FlowDefinition] = {}


def register_workflow(name: str, definition: FlowDefinition) -> None:
    """Register a workflow definition for publication during worker startup."""

    if not name:
        raise ValueError("Workflow name must be provided")

    if name in _workflow_registry:
        raise ValueError(f"Workflow '{name}' already registered")

    if not isinstance(definition, FlowDefinition):
        raise TypeError("definition must be a FlowDefinition instance")

    _validate_flow_definition(definition)
    _workflow_registry[name] = definition
    logger.debug("Registered workflow definition: %s", name)


def workflow(
    name: Optional[str] = None,
) -> Callable[[Callable[[], FlowDefinition]], Callable[[], FlowDefinition]]:
    """Decorator to register a workflow definition provided by a factory function."""

    def decorator(factory: Callable[[], FlowDefinition]) -> Callable[[], FlowDefinition]:
        flow_name = name or factory.__name__
        definition = factory()
        register_workflow(flow_name, definition)
        return factory

    return decorator


def get_registered_workflows() -> Dict[str, FlowDefinition]:
    """Return a shallow copy of the registered workflows."""

    return dict(_workflow_registry)


def clear_workflows() -> None:
    """Clear the workflow registry (used for tests)."""

    _workflow_registry.clear()


def task_step(
    name: str,
    *,
    service_name: str,
    handler_name: str,
    dependencies: Optional[List[str]] = None,
    input_data: Optional[Dict[str, Any]] = None,
    object_keys: Optional[List[str]] = None,
) -> WorkflowStep:
    """Helper for creating a task step."""

    return WorkflowStep(
        name=name,
        step_type=StepType.TASK,
        service_name=service_name,
        handler_name=handler_name,
        dependencies=dependencies or [],
        object_keys=object_keys or [],
        input_data=input_data,
    )


def wait_signal_step(
    name: str,
    *,
    signal_name: str,
    dependencies: Optional[List[str]] = None,
    timeout_ms: Optional[int] = None,
    on_timeout: Optional[str] = None,
) -> WorkflowStep:
    """Helper for creating a signal wait step."""

    return WorkflowStep(
        name=name,
        step_type=StepType.WAIT_SIGNAL,
        dependencies=dependencies or [],
        signal=SignalConfig(name=signal_name, timeout_ms=timeout_ms, on_timeout=on_timeout),
    )


def wait_timer_step(
    name: str,
    *,
    timer_key: str,
    dependencies: Optional[List[str]] = None,
    delay_ms: Optional[int] = None,
    cron_expr: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> WorkflowStep:
    """Helper for creating a timer wait step."""

    return WorkflowStep(
        name=name,
        step_type=StepType.WAIT_TIMER,
        dependencies=dependencies or [],
        timer=TimerConfig(
            key=timer_key, delay_ms=delay_ms, cron_expr=cron_expr, max_retries=max_retries
        ),
    )


def _validate_flow_definition(definition: FlowDefinition) -> None:
    if not definition.steps:
        raise ValueError("Workflow must contain at least one step")

    seen_names: set[str] = set()
    for step in definition.steps:
        if not step.name:
            raise ValueError("Each workflow step must have a name")
        if step.name in seen_names:
            raise ValueError(f"Duplicate step name detected: {step.name}")
        seen_names.add(step.name)

        if step.step_type == StepType.TASK:
            if not step.service_name or not step.handler_name:
                raise ValueError(f"Task step '{step.name}' requires service_name and handler_name")
        elif step.step_type == StepType.WAIT_SIGNAL:
            if not step.signal:
                raise ValueError(f"Signal step '{step.name}' requires signal configuration")
        elif step.step_type == StepType.WAIT_TIMER:
            if not step.timer:
                raise ValueError(f"Timer step '{step.name}' requires timer configuration")

        for dependency in step.dependencies:
            if dependency not in seen_names:
                raise ValueError(
                    f"Step '{step.name}' depends on unknown or out-of-order step '{dependency}'"
                )
