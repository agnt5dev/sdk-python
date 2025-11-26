"""Step-level checkpoint client for durable workflow execution.

This module provides the CheckpointClient class for step-level memoization,
enabling workflows to skip re-execution of completed steps after crashes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ._telemetry import setup_module_logger

logger = setup_module_logger(__name__)


@dataclass
class CheckpointResult:
    """Result of a checkpoint operation.

    Attributes:
        success: Whether the checkpoint was processed successfully
        error_message: Error message if the checkpoint failed
        memoized: Whether the step was already memoized (for step_started)
        cached_output: Cached output if memoized (as bytes)
    """

    success: bool
    error_message: Optional[str] = None
    memoized: bool = False
    cached_output: Optional[bytes] = None


class CheckpointClient:
    """Client for step-level checkpoint and memoization.

    This client provides synchronous checkpoint calls that enable
    durable step execution with platform-side memoization.

    The typical flow for a durable step:
    1. Call step_started() before execution
    2. If memoized, return the cached result
    3. Otherwise, execute the step
    4. Call step_completed() with the result

    Example:
        ```python
        async with CheckpointClient() as client:
            result = await client.step_started(run_id, "step:fetch_data:0", "fetch_data", "function")
            if result.memoized:
                return json.loads(result.cached_output)

            # Execute the step
            data = await fetch_data()

            # Record completion
            await client.step_completed(run_id, "step:fetch_data:0", "fetch_data", "function", json.dumps(data).encode())
            return data
        ```
    """

    def __init__(self, endpoint: Optional[str] = None):
        """Initialize checkpoint client.

        Args:
            endpoint: Worker Coordinator endpoint URL.
                     Defaults to AGNT5_COORDINATOR_ENDPOINT env var or http://localhost:34186
        """
        try:
            from ._core import PyCheckpointClient

            self._client = PyCheckpointClient(endpoint)
            self._connected = False
        except ImportError as e:
            logger.warning(f"Checkpoint client not available (Rust core not loaded): {e}")
            self._client = None
            self._connected = False

    async def connect(self) -> None:
        """Connect to the Worker Coordinator.

        Must be called before making checkpoint calls.
        """
        if self._client is None:
            raise RuntimeError("Checkpoint client not available (Rust core not loaded)")

        await self._client.connect()
        self._connected = True
        logger.debug("Checkpoint client connected to coordinator")

    async def __aenter__(self) -> "CheckpointClient":
        """Async context manager entry - connects to coordinator."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # No explicit disconnect needed - connection is cleaned up on drop
        pass

    async def step_started(
        self,
        run_id: str,
        step_key: str,
        step_name: str,
        step_type: str = "function",
        input_payload: Optional[bytes] = None,
    ) -> CheckpointResult:
        """Send a step started checkpoint and check for memoized result.

        Call this before executing a step. If the step is memoized,
        the result will contain the cached output.

        Args:
            run_id: The workflow run ID
            step_key: Unique key for this step (e.g., "step:greet:0")
            step_name: Human-readable step name
            step_type: Type of step (e.g., "function", "activity", "llm_call")
            input_payload: Input data for the step (optional, for logging)

        Returns:
            CheckpointResult with memoized=True and cached_output if step was memoized
        """
        if self._client is None:
            return CheckpointResult(success=False, error_message="Client not available")

        if not self._connected:
            await self.connect()

        result = await self._client.step_started(
            run_id, step_key, step_name, step_type, input_payload
        )

        return CheckpointResult(
            success=result.success,
            error_message=result.error_message,
            memoized=result.memoized,
            cached_output=result.cached_output,
        )

    async def step_completed(
        self,
        run_id: str,
        step_key: str,
        step_name: str,
        step_type: str,
        output_payload: bytes,
        latency_ms: Optional[int] = None,
    ) -> CheckpointResult:
        """Send a step completed checkpoint.

        Call this after successfully executing a step to record the result
        for future memoization.

        Args:
            run_id: The workflow run ID
            step_key: Unique key for this step
            step_name: Human-readable step name
            step_type: Type of step
            output_payload: Output data from the step
            latency_ms: Step execution latency in milliseconds

        Returns:
            CheckpointResult indicating success or failure
        """
        if self._client is None:
            return CheckpointResult(success=False, error_message="Client not available")

        if not self._connected:
            await self.connect()

        result = await self._client.step_completed(
            run_id, step_key, step_name, step_type, output_payload, latency_ms
        )

        return CheckpointResult(
            success=result.success,
            error_message=result.error_message,
            memoized=result.memoized,
            cached_output=result.cached_output,
        )

    async def step_failed(
        self,
        run_id: str,
        step_key: str,
        step_name: str,
        step_type: str,
        error_message: str,
        error_type: str,
    ) -> CheckpointResult:
        """Send a step failed checkpoint.

        Call this when a step fails to record the error.

        Args:
            run_id: The workflow run ID
            step_key: Unique key for this step
            step_name: Human-readable step name
            step_type: Type of step
            error_message: Error message
            error_type: Error type/class name

        Returns:
            CheckpointResult indicating success or failure
        """
        if self._client is None:
            return CheckpointResult(success=False, error_message="Client not available")

        if not self._connected:
            await self.connect()

        result = await self._client.step_failed(
            run_id, step_key, step_name, step_type, error_message, error_type
        )

        return CheckpointResult(
            success=result.success,
            error_message=result.error_message,
            memoized=result.memoized,
            cached_output=result.cached_output,
        )

    async def get_memoized_step(
        self, run_id: str, step_key: str
    ) -> Optional[bytes]:
        """Check if a step result is memoized.

        Use this for quick memoization lookups before executing expensive steps.

        Args:
            run_id: The workflow run ID
            step_key: Unique key for this step

        Returns:
            The cached output bytes if memoized, None otherwise
        """
        if self._client is None:
            return None

        if not self._connected:
            await self.connect()

        return await self._client.get_memoized_step(run_id, step_key)
