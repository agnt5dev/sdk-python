"""Tool execution client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from ..context import Context


@dataclass
class _RegisteredTool:
    name: str
    callable: Callable


class ToolClient:
    """Lightweight registry for tools declared within a function run."""

    def __init__(self, context: Context) -> None:
        self._context = context
        self._registry: Dict[str, _RegisteredTool] = {}

    def register(
        self,
        name: str,
        *,
        handler: ToolHandler,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tool:
        if not callable(handler):
            raise TypeError("Tool handler must be callable")
        exists = name in self._registry
        tool = Tool(name=name, description=description, input_schema=schema)
        self._registry[name] = _RegisteredTool(tool=tool, handler=handler)
        if exists:
            self._context.log().warning(
                "Tool '%s' was re-registered; overriding previous handler", name
            )
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._registry[name].tool

    def handler(self, name: str) -> ToolHandler:
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._registry[name].handler

    def list(self) -> List[Tool]:
        return [entry.tool for entry in self._registry.values()]

    def clear(self) -> None:
        self._registry.clear()
