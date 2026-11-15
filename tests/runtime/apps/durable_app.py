"""Worker app for the real-runtime tests.

Each step records an external side effect by appending a line to the file in
`SDK_TEST_EFFECTS_FILE`, standing in for a call such as a payment refund. The
tests read that file to check a step ran exactly once, even across a worker
crash, since the file lives outside any worker process.
"""

import asyncio
import os

from agnt5 import Worker, WorkflowContext, workflow

EFFECTS_FILE = os.environ["SDK_TEST_EFFECTS_FILE"]


def record(effect: str) -> str:
    with open(EFFECTS_FILE, "a") as handle:
        handle.write(effect + "\n")
    return effect


@workflow
async def refund_order(ctx: WorkflowContext, order_id: str) -> dict:
    """Three steps, like a refund: look up, refund, notify."""
    looked_up = await ctx.step("lookup_order", lambda: record(f"lookup:{order_id}"))
    refunded = await ctx.step("issue_refund", lambda: record(f"refund:{order_id}"))
    notified = await ctx.step("notify_customer", lambda: record(f"notify:{order_id}"))
    return {"steps": [looked_up, refunded, notified]}


@workflow
async def sleep_between_steps(ctx: WorkflowContext, run_key: str, seconds: float) -> dict:
    """A step, a durable sleep, then another step."""
    await ctx.step("before_sleep", lambda: record(f"before:{run_key}"))
    await ctx.sleep(seconds, name="pause_between_steps")
    await ctx.step("after_sleep", lambda: record(f"after:{run_key}"))
    return {"slept": seconds}


@workflow
async def ask_for_approval(ctx: WorkflowContext, run_key: str) -> dict:
    """A step, then a pause until a person answers."""
    await ctx.step("before_pause", lambda: record(f"before_pause:{run_key}"))
    answer = await ctx.wait_for_user(question="Approve the refund?", input_type="text")
    await ctx.step("after_pause", lambda: record(f"after_pause:{run_key}"))
    return {"answer": answer}


async def main() -> None:
    worker = Worker(
        service_name="sdk-runtime-tests",
        workflows=[refund_order, sleep_between_steps, ask_for_approval],
        worker_mode="pull",
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
