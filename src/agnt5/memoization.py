"""Memoization support for AGNT5 agents.

This module provides auto-memoization of LLM and tool calls within agent loops,
enabling deterministic replay on crash recovery while preserving first-run
non-determinism.

Uses hybrid hashing approach:
- step_key: Sequence-based for ordering (lm.0, lm.1, tool.search.0)
- content_hash: SHA256 of inputs for validation on replay

Integrates with platform's step-level checkpoint system via CheckpointClient
for durable memoization that survives process crashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context

logger = logging.getLogger(__name__)


class MemoizationManager:
    """
    Handles read-through caching via platform checkpoint system for LLM and tool calls.

    Uses hybrid approach:
    - step_key: Sequence-based for ordering (lm.0, lm.1, tool.search.0)
    - content_hash: SHA256 of inputs for validation on replay

    The step_key ensures correct replay order, while the content_hash detects
    if inputs changed between runs (e.g., developer modified code).

    Integrates with CheckpointClient for durable storage via platform's
    step checkpoint system (Checkpoint/GetMemoizedStep RPCs).

    Example:
        ```python
        memo = MemoizationManager(ctx)

        # For LLM calls
        step_key, content_hash = memo.lm_call_key(model, messages, config)
        cached = await memo.get_cached_lm_result(step_key, content_hash)
        if cached:
            return cached  # Skip execution
        # ... execute LLM call ...
        await memo.cache_lm_result(step_key, content_hash, result)
        ```
    """

    def __init__(self, ctx: "Context") -> None:
        """
        Initialize memoization manager.

        Args:
            ctx: Execution context for run_id and platform access
        """
        self._ctx = ctx
        self._lm_sequence = 0
        self._tool_sequences: Dict[str, int] = {}  # Per-tool sequence counters
        self._checkpoint_client = None
        self._connected = False

    async def _ensure_client(self) -> bool:
        """
        Lazily initialize and connect the checkpoint client.

        Returns:
            True if client is ready, False if unavailable
        """
        if self._connected and self._checkpoint_client is not None:
            return True

        if self._checkpoint_client is None:
            try:
                from .checkpoint import CheckpointClient
                self._checkpoint_client = CheckpointClient()
            except ImportError as e:
                logger.debug(f"CheckpointClient not available: {e}")
                return False
            except Exception as e:
                logger.debug(f"Failed to create CheckpointClient: {e}")
                return False

        if not self._connected:
            try:
                await self._checkpoint_client.connect()
                self._connected = True
            except Exception as e:
                logger.debug(f"Failed to connect CheckpointClient: {e}")
                return False

        return True

    def _content_hash(self, data: dict) -> str:
        """
        Generate SHA256 hash of content for validation.

        Uses first 16 characters of hex digest for compact storage.

        Args:
            data: Dictionary to hash (will be JSON serialized)

        Returns:
            16-character hex hash string
        """
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def lm_call_key(
        self,
        model: str,
        messages: List[Any],
        config: dict,
    ) -> Tuple[str, str]:
        """
        Generate (step_key, content_hash) for LLM call.

        Args:
            model: Model name/identifier
            messages: List of message objects or dicts
            config: Generation config (temperature, max_tokens, etc.)

        Returns:
            Tuple of (step_key, content_hash):
            - step_key: Sequence-based key for journal lookup (e.g., "lm.0")
            - content_hash: Hash of inputs for replay validation
        """
        step_key = f"lm.{self._lm_sequence}"
        self._lm_sequence += 1

        # Build hashable representation of messages
        messages_data = []
        for m in messages:
            if hasattr(m, 'role') and hasattr(m, 'content'):
                # Message object
                role = m.role.value if hasattr(m.role, 'value') else str(m.role)
                messages_data.append({"role": role, "content": m.content})
            elif isinstance(m, dict):
                # Dict format
                messages_data.append({
                    "role": m.get('role', 'user'),
                    "content": m.get('content', '')
                })
            else:
                # Fallback
                messages_data.append({"content": str(m)})

        content_hash = self._content_hash({
            "model": model,
            "messages": messages_data,
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
        })

        return step_key, content_hash

    def tool_call_key(self, tool_name: str, kwargs: dict) -> Tuple[str, str]:
        """
        Generate (step_key, content_hash) for tool call.

        Args:
            tool_name: Name of the tool being called
            kwargs: Tool arguments

        Returns:
            Tuple of (step_key, content_hash):
            - step_key: Sequence-based key for journal lookup (e.g., "tool.search.0")
            - content_hash: Hash of inputs for replay validation
        """
        seq = self._tool_sequences.get(tool_name, 0)
        step_key = f"tool.{tool_name}.{seq}"
        self._tool_sequences[tool_name] = seq + 1

        content_hash = self._content_hash({
            "tool": tool_name,
            "args": kwargs,
        })

        return step_key, content_hash

    async def get_cached_lm_result(
        self,
        step_key: str,
        content_hash: str,
    ) -> Optional[Any]:
        """
        Check platform checkpoint for cached LLM result.

        Args:
            step_key: Sequence-based key (e.g., "lm.0")
            content_hash: Hash of current inputs for validation

        Returns:
            Cached GenerateResponse if found and valid, None otherwise
        """
        if not await self._ensure_client():
            return None

        run_id = self._ctx.run_id

        try:
            # Use platform's GetMemoizedStep RPC
            cached_bytes = await self._checkpoint_client.get_memoized_step(
                run_id, step_key
            )

            if cached_bytes:
                # Deserialize cached result
                cached_data = json.loads(cached_bytes)

                # Validate content hash matches (warn if mismatch)
                stored_hash = cached_data.get("input_hash")
                if stored_hash and stored_hash != content_hash:
                    logger.warning(
                        f"Content mismatch on replay for {step_key}: "
                        f"stored={stored_hash}, current={content_hash}. "
                        "Inputs may have changed between runs."
                    )

                # Return cached output
                output_data = cached_data.get("output_data")
                if output_data:
                    logger.debug(f"Cache hit for LLM call {step_key}")
                    # Convert output_data back to GenerateResponse
                    from .lm import GenerateResponse
                    return GenerateResponse.from_dict(output_data)

        except Exception as e:
            logger.debug(f"Failed to lookup cached LLM result for {step_key}: {e}")

        return None

    async def cache_lm_result(
        self,
        step_key: str,
        content_hash: str,
        result: Any,
    ) -> None:
        """
        Write LLM result to platform checkpoint for future replay.

        Args:
            step_key: Sequence-based key (e.g., "lm.0")
            content_hash: Hash of inputs for validation on replay
            result: GenerateResponse to cache
        """
        if not await self._ensure_client():
            return

        run_id = self._ctx.run_id

        try:
            # Convert result to dict for storage
            output_data = result.to_dict() if hasattr(result, 'to_dict') else result

            # Build cache payload with hash for validation
            cache_payload = json.dumps({
                "input_hash": content_hash,
                "output_data": output_data,
            }).encode()

            # Use platform's Checkpoint RPC with step_completed
            await self._checkpoint_client.step_completed(
                run_id=run_id,
                step_key=step_key,
                step_name="lm_call",
                step_type="llm",
                output_payload=cache_payload,
            )
            logger.debug(f"Cached LLM result for {step_key}")

        except Exception as e:
            logger.warning(f"Failed to cache LLM result for {step_key}: {e}")

    async def get_cached_tool_result(
        self,
        step_key: str,
        content_hash: str,
    ) -> Tuple[bool, Optional[Any]]:
        """
        Check platform checkpoint for cached tool result.

        Args:
            step_key: Sequence-based key (e.g., "tool.search.0")
            content_hash: Hash of current inputs for validation

        Returns:
            Tuple of (found, result):
            - found: True if cache entry exists
            - result: Cached result if found, None otherwise
        """
        if not await self._ensure_client():
            return False, None

        run_id = self._ctx.run_id

        try:
            # Use platform's GetMemoizedStep RPC
            cached_bytes = await self._checkpoint_client.get_memoized_step(
                run_id, step_key
            )

            if cached_bytes:
                # Deserialize cached result
                cached_data = json.loads(cached_bytes)

                # Validate content hash matches (warn if mismatch)
                stored_hash = cached_data.get("input_hash")
                if stored_hash and stored_hash != content_hash:
                    logger.warning(
                        f"Content mismatch on replay for {step_key}: "
                        f"stored={stored_hash}, current={content_hash}. "
                        "Inputs may have changed between runs."
                    )

                # Return cached output
                output_data = cached_data.get("output_data")
                logger.debug(f"Cache hit for tool call {step_key}")
                return True, output_data

        except Exception as e:
            logger.debug(f"Failed to lookup cached tool result for {step_key}: {e}")

        return False, None

    async def cache_tool_result(
        self,
        step_key: str,
        content_hash: str,
        result: Any,
    ) -> None:
        """
        Write tool result to platform checkpoint for future replay.

        Args:
            step_key: Sequence-based key (e.g., "tool.search.0")
            content_hash: Hash of inputs for validation on replay
            result: Tool result to cache
        """
        if not await self._ensure_client():
            return

        run_id = self._ctx.run_id

        try:
            # Build cache payload with hash for validation
            cache_payload = json.dumps({
                "input_hash": content_hash,
                "output_data": result,
            }, default=str).encode()

            # Use platform's Checkpoint RPC with step_completed
            await self._checkpoint_client.step_completed(
                run_id=run_id,
                step_key=step_key,
                step_name="tool_call",
                step_type="tool",
                output_payload=cache_payload,
            )
            logger.debug(f"Cached tool result for {step_key}")

        except Exception as e:
            logger.warning(f"Failed to cache tool result for {step_key}: {e}")

    def reset(self) -> None:
        """
        Reset sequence counters.

        Call this when starting a new execution to ensure deterministic
        step_key generation.
        """
        self._lm_sequence = 0
        self._tool_sequences.clear()
