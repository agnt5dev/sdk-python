"""LM Functions - Test OpenAI integration via ctx.lm()."""

from agnt5 import function, FunctionContext, lm


@function
async def fn_10_lm_simple_completion(ctx: FunctionContext, prompt: str, model_name: str) -> dict:
    """Function that calls OpenAI for text completion."""
    ctx.logger.info(f"Calling LM with prompt: {prompt[:50]}...")

    # Use context's lm() method for OpenAI calls
    response = await lm.generate(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.text
    ctx.logger.info(f"LM response: {result[:50]}...")

    return {
        "prompt": prompt,
        "response": result,
        "model": model_name,
    }


@function
async def fn_11_lm_structured_output(ctx: FunctionContext, question: str, model_name: str) -> dict:
    """Function that gets structured output from LM."""
    ctx.logger.info(f"Getting structured answer for: {question}")

    response = await lm.generate(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely in 1-2 sentences.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.5,
    )

    return {
        "question": question,
        "answer": response.text,
        "tokens": response.usage,
    }


__all__ = ["fn_10_lm_simple_completion", "fn_11_lm_structured_output"]
