"""LM Functions - Test OpenAI integration via lm.generate() and lm.stream()."""

from agnt5 import function, FunctionContext, lm


@function
async def fn_10_lm_generate(ctx: FunctionContext, prompt: str, model_name: str) -> dict:
    """Function that calls LLM for text completion using non-streaming generate().

    This waits for the complete response before returning.
    """
    ctx.logger.info(f"[lm.generate] Calling LM with prompt: {prompt[:50]}...")

    # Use non-streaming generate() - waits for complete response
    response = await lm.generate(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.text
    ctx.logger.info(f"[lm.generate] LM response: {result[:50]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model_name,
        "method": "generate",
    }


@function
async def fn_10_lm_stream(ctx: FunctionContext, prompt: str, model_name: str) -> dict:
    """Function that calls LLM for text completion using streaming stream().

    This streams tokens as they arrive, collecting them into a full response.
    Each chunk is logged as it arrives to demonstrate streaming behavior.
    """
    ctx.logger.info(f"[lm.stream] Calling LM with prompt: {prompt[:50]}...")

    # Use streaming stream() - yields chunks as they arrive
    chunks = []
    chunk_count = 0

    async for chunk in lm.stream(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    ):
        chunks.append(chunk)
        chunk_count += 1
        # Log each chunk at INFO level so it appears in SSE stream (streamv2 with X-Include-Traces)
        ctx.logger.info(f"[lm.stream] chunk[{chunk_count}]: '{chunk}'")

    # Combine all chunks into final response
    result = "".join(chunks)
    ctx.logger.info(f"[lm.stream] LM response ({chunk_count} chunks): {result[:50]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model_name,
        "method": "stream",
        "chunk_count": chunk_count,
    }


__all__ = ["fn_10_lm_generate", "fn_10_lm_stream"]
