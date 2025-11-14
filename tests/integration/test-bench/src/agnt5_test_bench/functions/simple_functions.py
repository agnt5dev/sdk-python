"""Simple Functions - Test different parameter counts."""

from agnt5 import function, FunctionContext


@function(name="fn_01_no_params")
async def fn_01_no_params(ctx: FunctionContext) -> dict:
    """Function with no parameters."""
    ctx.logger.info("Executing no_params")
    return {"result": "success", "params": 0}


@function(name="fn_02_one_param")
async def fn_02_one_param(ctx: FunctionContext, value: int) -> dict:
    """Function with one parameter."""
    ctx.logger.info(f"Executing one_param with value={value}")
    return {"result": value * 2, "params": 1}


@function(name="fn_03_two_params")
async def fn_03_two_params(ctx: FunctionContext, a: int, b: int) -> dict:
    """Function with two parameters."""
    ctx.logger.info(f"Executing two_params with a={a}, b={b}")
    return {"result": a + b, "params": 2}


@function(name="fn_04_three_params")
async def fn_04_three_params(ctx: FunctionContext, x: int, y: int, z: int) -> dict:
    """Function with three parameters."""
    ctx.logger.info(f"Executing three_params with x={x}, y={y}, z={z}")
    return {"result": x * y + z, "params": 3}


__all__ = ["fn_01_no_params", "fn_02_one_param", "fn_03_two_params", "fn_04_three_params"]
