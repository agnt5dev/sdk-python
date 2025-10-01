"""
Example 3: Function Registry

This example demonstrates:
- How functions are automatically registered
- Inspecting registered functions
- Function discovery and introspection
- Custom function names
"""

import asyncio

from agnt5 import Context, FunctionRegistry, function


@function
async def process_order(ctx: Context, order_id: str) -> dict:
    """Process an order."""
    return {"order_id": order_id, "status": "processed"}


@function(name="analyze_sentiment")
async def sentiment_analysis(ctx: Context, text: str) -> dict:
    """Analyze sentiment with custom name."""
    # Simulate sentiment analysis
    score = 0.8 if "good" in text.lower() else 0.3
    return {"text": text, "sentiment": "positive" if score > 0.5 else "negative", "score": score}


@function
async def calculate_total(ctx: Context, items: list) -> float:
    """Calculate total price."""
    return sum(item.get("price", 0) for item in items)


def inspect_registry() -> None:
    """Inspect the function registry."""
    print("=== Registered Functions ===\n")

    all_functions = FunctionRegistry.all()

    for name, config in all_functions.items():
        print(f"Function: {name}")
        print(f"  Handler: {config.handler.__name__}")
        print(f"  Retries: {config.retries}")
        print(f"  Backoff: {config.backoff}")
        print()


async def call_by_name(function_name: str, ctx: Context, *args: object, **kwargs: object) -> object:
    """Call a function by name from registry."""
    config = FunctionRegistry.get(function_name)

    if not config:
        raise ValueError(f"Function '{function_name}' not found in registry")

    # Call the handler directly
    return await config.handler(ctx, *args, **kwargs)


async def main() -> None:
    """Run examples."""
    # Inspect registered functions
    inspect_registry()

    # Call function by original name
    print("=== Calling Functions ===\n")

    ctx1 = Context(run_id="registry-1")
    result1 = await call_by_name("process_order", ctx1, "order-123")
    print(f"process_order result: {result1}\n")

    # Call function by custom name
    ctx2 = Context(run_id="registry-2")
    result2 = await call_by_name("analyze_sentiment", ctx2, "This is a good product!")
    print(f"analyze_sentiment result: {result2}\n")

    # Direct function calls still work
    ctx3 = Context(run_id="registry-3")
    result3 = await calculate_total(
        ctx3, [{"name": "item1", "price": 10.0}, {"name": "item2", "price": 25.5}]
    )
    print(f"calculate_total result: {result3}\n")

    # Introspect function config
    print("=== Function Introspection ===\n")
    config = FunctionRegistry.get("analyze_sentiment")
    if config:
        print(f"Function name: {config.name}")
        print(f"Max retry attempts: {config.retries.max_attempts if config.retries else 'N/A'}")
        print(
            f"Initial retry interval: {config.retries.initial_interval_ms if config.retries else 'N/A'}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
