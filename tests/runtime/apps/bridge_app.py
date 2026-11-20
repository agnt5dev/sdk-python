"""Worker app for the real-runtime tests of the Python/Rust bridge.

The components report what they observed inside the worker: values they read
back, and timestamps written to the file in `SDK_TEST_EFFECTS_FILE`, one JSON
object per line. The tests read that file, which lives outside the worker
process, to check behavior the run's output alone can't show.
"""

import asyncio
import contextvars
import json
import os
import time

from agnt5 import Agent, StepContext, Worker, WorkflowContext, step, workflow
from agnt5.context import get_current_context
from agnt5.lm import GenerateRequest, GenerateResponse, LanguageModel, TokenUsage
from agnt5.lm.events import (
    LMCompleted,
    LMContentBlockCompleted,
    LMContentBlockDelta,
    LMContentBlockStarted,
)

EFFECTS_FILE = os.environ["SDK_TEST_EFFECTS_FILE"]

# Set by each run for itself. A run must never start with another run's value.
run_marker: contextvars.ContextVar[str] = contextvars.ContextVar("run_marker", default="unset")


def record(**fields) -> None:
    with open(EFFECTS_FILE, "a") as handle:
        handle.write(json.dumps(fields) + "\n")


@step
def report_run_id(ctx: StepContext) -> str:
    """A sync step runs in a thread; it must still see its run's context."""
    current = get_current_context()
    return current.run_id if current is not None else ""


@workflow
async def hold_a_context_value(ctx: WorkflowContext, key: str, hold_s: float) -> dict:
    """Read the marker, set this run's own value, wait so runs overlap, read again,
    then ask a sync step which run it belongs to."""
    from agnt5.worker import _trace_metadata

    seen_at_start = run_marker.get()
    run_marker.set(key)
    record(key=key, event="start", at=time.monotonic())
    await asyncio.sleep(hold_s)
    record(key=key, event="end", at=time.monotonic())
    return {
        "seen_at_start": seen_at_start,
        "seen_at_end": run_marker.get(),
        "traceparent": _trace_metadata.get().get("traceparent"),
        "step_run_id": await ctx.step(report_run_id, key=key),
        "run_id": ctx.run_id,
    }


class ScriptedModel(LanguageModel):
    """Streams a fixed number of text blocks, recording when each one starts.

    Only the model is scripted. The worker, gateway and runtime are real, so the
    gap between two blocks includes everything the worker does at a block
    boundary before it asks the model for more.
    """

    def __init__(self, blocks: int) -> None:
        self.blocks = blocks

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        text = " ".join(f"block-{i}" for i in range(self.blocks))
        return GenerateResponse(
            text=text,
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(self, request: GenerateRequest):
        stream_key = request.messages[-1].content if request.messages else ""
        for index in range(self.blocks):
            record(key=stream_key, event="block_start", index=index, at=time.monotonic())
            common = dict(
                name="scripted-model",
                correlation_id=f"block-{index}",
                parent_correlation_id="scripted",
                block_type="text",
                index=index,
            )
            yield LMContentBlockStarted(**common)
            yield LMContentBlockDelta(content=f"block-{index} ", **common)
            yield LMContentBlockCompleted(**common)
        yield LMCompleted(
            name="scripted-model",
            correlation_id="scripted",
            parent_correlation_id="scripted",
            model="scripted-model",
            provider="scripted",
            input_tokens=1,
            output_tokens=self.blocks,
            total_tokens=self.blocks + 1,
            output_data={"text": " ".join(f"block-{i}" for i in range(self.blocks))},
        )


streaming_reporter = Agent(
    name="streaming_reporter",
    model=ScriptedModel(blocks=6),
    instructions="Report in blocks.",
)


async def main() -> None:
    worker = Worker(
        service_name="sdk-runtime-bridge-tests",
        workflows=[hold_a_context_value],
        agents=[streaming_reporter],
        worker_mode="pull",
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
