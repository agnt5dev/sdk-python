"""LM Functions - Test OpenAI integration via ctx.lm()."""

from agnt5 import function, FunctionContext


@function
async def fn_10_lm_simple_completion(ctx: FunctionContext, prompt: str) -> dict:
    """Function that calls OpenAI for text completion."""
    ctx.logger.info(f"Calling LM with prompt: {prompt[:50]}...")

    # Use context's lm() method for OpenAI calls
    response = await ctx.lm(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    result = response["choices"][0]["message"]["content"]
    ctx.logger.info(f"LM response: {result[:50]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": "openai/gpt-4o-mini",
    }


@function
async def fn_11_lm_structured_output(ctx: FunctionContext, question: str) -> dict:
    """Function that gets structured output from LM."""
    ctx.logger.info(f"Getting structured answer for: {question}")

    response = await ctx.lm(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely in 1-2 sentences.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.5,
    )

    answer = response["choices"][0]["message"]["content"]

    return {
        "question": question,
        "answer": answer,
        "tokens": response.get("usage", {}),
    }


__all__ = ["fn_10_lm_simple_completion", "fn_11_lm_structured_output"]
