"""Checkpoint storage for durable step execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


class StepCheckpoint:
    name: str
    key: Optional[str]
    status: str
    attempt: int
    result: Any
    updated_at: datetime

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "StepCheckpoint":
        name = payload.get("name")
        if not name:
            raise ValueError("Checkpoint payload missing 'name'")
        key = payload.get("key")
        status = (payload.get("status") or "SUCCEEDED").upper()
        attempt = int(payload.get("attempt", 1))
        raw_result = payload.get("result")
        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                result = raw_result
        else:
            result = raw_result
        timestamp_raw = payload.get("updated_at")
        if isinstance(timestamp_raw, str):
            try:
                updated_at = datetime.fromisoformat(timestamp_raw)
            except ValueError:
                updated_at = datetime.now(timezone.utc)
        else:
            updated_at = datetime.now(timezone.utc)
        return cls(
            name=name, key=key, status=status, attempt=attempt, result=result, updated_at=updated_at
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "status": self.status,
            "attempt": self.attempt,
            "result": self.result,
            "updated_at": self.updated_at.replace(tzinfo=timezone.utc).isoformat(),
        }


class StepCheckpointStore:
    """Maintains step checkpoints for a single invocation."""

    def __init__(self, checkpoints: Sequence[StepCheckpoint]):
        self._completed: Dict[Tuple[str, str], StepCheckpoint] = {}
        self._new: Dict[Tuple[str, str], StepCheckpoint] = {}

        for checkpoint in checkpoints:
            key = (checkpoint.name, _normalise_key(checkpoint.name, checkpoint.key))
            if checkpoint.status == "SUCCEEDED":
                self._completed[key] = checkpoint

    def get_completed(self, name: str, key: Optional[str]) -> Optional[StepCheckpoint]:
        lookup_key = (name, _normalise_key(name, key))
        if lookup_key in self._new:
            return self._new[lookup_key]
        return self._completed.get(lookup_key)

    def record_success(
        self, name: str, key: Optional[str], result: Any, attempt: int
    ) -> StepCheckpoint:
        checkpoint = StepCheckpoint(
            name=name,
            key=key,
            status="SUCCEEDED",
            attempt=attempt,
            result=result,
            updated_at=datetime.now(timezone.utc),
        )
        lookup_key = (name, _normalise_key(name, key))
        self._new[lookup_key] = checkpoint
        return checkpoint

    def export_new(self) -> List[Dict[str, Any]]:
        if not self._new:
            return []
        return [checkpoint.to_payload() for checkpoint in self._new.values()]
