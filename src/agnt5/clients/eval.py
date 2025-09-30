"""Evaluation client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from ..context import Context


class EvalClient:
    """Stub evaluation client; hooks will integrate with runtime in later phases."""

    def __init__(self, context: Context) -> None:
        self._context = context

    @contextmanager
    def group(self, name: str):  # pragma: no cover - grouping helper
        self._context.log().debug("Starting eval group '%s'", name)
        try:
            yield self
        finally:
            self._context.log().debug("Finished eval group '%s'", name)

    def judge(self, name: str, **kwargs: Any) -> None:
        self._context.log().debug("Eval judge '%s' invoked with %s", name, kwargs)

    def metric(self, name: str, value: Union[int, float], **labels: Any) -> None:
        self._context.metrics().observe(f"eval.{name}", float(value), **labels)
