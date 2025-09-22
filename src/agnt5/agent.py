"""Stateful agent decorator and registry for durable virtual objects."""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from .context import Context

logger = logging.getLogger(__name__)


class DurableAgent:
    """Base class for durable agents; override ``on_message`` in subclasses."""

    async def on_message(self, ctx: "AgentContext", message: Any) -> Any:  # pragma: no cover - interface only
        raise NotImplementedError("DurableAgent subclasses must implement on_message")


AgentContext = Context


KeyFn = Callable[..., str]


@dataclass
class StatefulAgentDefinition:
    name: str
    agent_cls: Type[DurableAgent]
    key_resolver: KeyFn
    module: str
    qualname: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "qualname": self.qualname,
        }


class StatefulAgentRegistry:
    """Thread-safe registry for stateful agents."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: Dict[str, StatefulAgentDefinition] = {}

    def register(self, definition: StatefulAgentDefinition) -> None:
        with self._lock:
            if definition.name in self._agents:
                existing = self._agents[definition.name]
                raise ValueError(
                    "Stateful agent '%s' already registered with handler '%s'."
                    % (definition.name, existing.qualname)
                )
            self._agents[definition.name] = definition
            logger.debug("Registered stateful agent '%s'", definition.name)

    def get(self, name: str) -> Optional[StatefulAgentDefinition]:
        with self._lock:
            return self._agents.get(name)

    def agents(self) -> Dict[str, StatefulAgentDefinition]:
        with self._lock:
            return dict(self._agents)

    def clear(self) -> None:
        with self._lock:
            self._agents.clear()


_AGENT_REGISTRY = StatefulAgentRegistry()


def get_agent_registry() -> StatefulAgentRegistry:
    return _AGENT_REGISTRY


def stateful(
    *,
    name: Optional[str] = None,
    key: Optional[KeyFn] = None,
) -> Callable[[Type[DurableAgent]], Type[DurableAgent]]:
    """Decorator used to register a durable agent class."""

    if key is None:
        raise ValueError("@agent.stateful requires a key callable to derive the object key")

    def decorator(cls: Type[DurableAgent]) -> Type[DurableAgent]:
        if not inspect.isclass(cls):
            raise TypeError("@agent.stateful can only decorate classes")
        if not issubclass(cls, DurableAgent):
            raise TypeError("Stateful agents must inherit from DurableAgent")

        effective_name = name or cls.__name__
        if not effective_name:
            raise ValueError("Stateful agents must have a non-empty name")

        definition = StatefulAgentDefinition(
            name=effective_name,
            agent_cls=cls,
            key_resolver=key,
            module=cls.__module__ or "<unknown>",
            qualname=cls.__qualname__,
        )

        get_agent_registry().register(definition)

        setattr(cls, "_agnt5_agent_definition", definition)
        logger.debug("Stateful agent '%s' registered", effective_name)
        return cls

    return decorator


def get_registered_agents() -> Dict[str, StatefulAgentDefinition]:
    return get_agent_registry().agents()


def clear_agents() -> None:
    get_agent_registry().clear()
    try:
        from .context import AgentClient

        AgentClient.clear_cache()
    except Exception:  # pragma: no cover - defensive guard during import cycles
        pass


__all__ = [
    "AgentContext",
    "DurableAgent",
    "StatefulAgentDefinition",
    "clear_agents",
    "get_agent_registry",
    "get_registered_agents",
    "stateful",
]
