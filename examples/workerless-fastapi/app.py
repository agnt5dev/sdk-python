from __future__ import annotations

import os

from fastapi import FastAPI

from agnt5 import function, workflow
from agnt5.serverless import serve

app = FastAPI(title="AGNT5 Workerless FastAPI Example")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@function
async def uppercase(text: str) -> dict[str, str]:
    return {"text": text.upper()}


@workflow
async def hello(ctx, name: str = "world") -> dict[str, str]:
    return {"message": f"hello {name}"}


@workflow
async def research(ctx, title: str = "AGNT5") -> dict[str, str | int]:
    page = await ctx.step("fetch", lambda: {"title": title, "fetch_count": 1})
    await ctx.yield_if_needed()
    return {"summary": f"summary:{page['title']}", "fetch_count": page["fetch_count"]}


agnt5_workerless = serve(
    service_name="agnt5-workerless-fastapi",
    service_version=os.getenv("GIT_SHA", "local"),
    signing_secret=lambda: os.getenv("AGNT5_SERVERLESS_SIGNING_SECRET"),
)
agnt5_workerless.mount_fastapi(app)
