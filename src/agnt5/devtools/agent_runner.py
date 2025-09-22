"""CLI helper to execute stateful agents locally using ``ctx.agent.call``."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import uuid
from typing import Any, Dict, Mapping

from ..context import AgentCallResult, Context


def _parse_json(value: str, *, description: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - user input validation
        raise SystemExit(f"Invalid JSON for {description}: {exc}") from exc


def _normalise_output(result: AgentCallResult) -> Mapping[str, Any]:
    return {
        "object_id": result.object_id,
        "output": result.output,
        "state": result.state,
        "transitions": result.transitions,
    }


def _build_context(args: argparse.Namespace) -> Context:
    return Context(
        run_id=args.run_id,
        step_id=args.step_id,
        attempt=args.attempt,
        invocation_id=f"cli-{uuid.uuid4()}",
        service_name=args.service_name,
        component_name=args.component_name,
        metadata={"headers": {"x-cli-agent": "true"}},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a registered stateful agent locally")
    parser.add_argument("--module", required=True, help="Python module to import (registers the agent)")
    parser.add_argument("--agent", required=True, help="Registered agent name to execute")
    parser.add_argument(
        "--payload",
        default="{}",
        help="JSON payload to pass to the agent method (default: empty object)",
    )
    parser.add_argument(
        "--method",
        default="on_message",
        help="Agent method to invoke (default: on_message)",
    )
    parser.add_argument(
        "--object-id",
        dest="object_id",
        default=None,
        help="Override object identifier (otherwise derived via key resolver)",
    )
    parser.add_argument(
        "--run-id",
        default="dev-run",
        help="Run identifier injected into the local context",
    )
    parser.add_argument(
        "--step-id",
        default="dev-step",
        help="Step identifier injected into the local context",
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="Attempt number injected into the local context",
    )
    parser.add_argument(
        "--service-name",
        default="dev-service",
        help="Service name injected into the local context",
    )
    parser.add_argument(
        "--component-name",
        default="cli-agent",
        help="Component name injected into the local context",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args(argv)

    try:
        importlib.import_module(args.module)
    except Exception as exc:  # pragma: no cover - user feedback path
        raise SystemExit(f"Failed to import module '{args.module}': {exc}") from exc

    payload = _parse_json(args.payload, description="payload")

    context = _build_context(args)

    async def _run() -> AgentCallResult:
        return await context.agent.call(
            args.agent,
            payload,
            object_id=args.object_id,
            method=args.method,
        )

    try:
        result = asyncio.run(_run())
    except Exception as exc:  # pragma: no cover - runtime errors bubbled to CLI
        raise SystemExit(f"Agent execution failed: {exc}") from exc

    output = _normalise_output(result)
    json.dump(output, sys.stdout, indent=2 if args.pretty else None)
    if not args.pretty:
        sys.stdout.write("\n")
    sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
