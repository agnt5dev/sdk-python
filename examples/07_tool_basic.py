"""
Example: Basic Tool Usage with Auto-Schema

This example demonstrates:
- Creating tools with @tool decorator
- Automatic schema extraction from type hints and docstrings
- Tool registration and discovery
- Invoking tools
"""

import asyncio
from typing import Dict, List

from agnt5 import Context, tool, ToolRegistry


# Define tools with auto-schema extraction
@tool(auto_schema=True)
async def search_web(ctx: Context, query: str, max_results: int = 10) -> List[Dict]:
    """Search the web for information.

    Args:
        query: The search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, url, and snippet
    """
    # Simulate web search
    ctx.logger.info(f"Searching web for: {query}")

    # Mock results
    results = [
        {
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a snippet for result {i+1}..."
        }
        for i in range(min(max_results, 3))
    ]

    return results


@tool(auto_schema=True)
async def calculate(ctx: Context, operation: str, a: float, b: float) -> float:
    """Perform mathematical calculations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Result of the calculation
    """
    ctx.logger.info(f"Calculating: {a} {operation} {b}")

    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    result = operations[operation](a, b)
    ctx.logger.info(f"Result: {result}")

    return result


@tool(auto_schema=True)
async def get_weather(ctx: Context, location: str, units: str = "celsius") -> Dict:
    """Get current weather for a location.

    Args:
        location: City name or location
        units: Temperature units (celsius or fahrenheit)

    Returns:
        Weather information including temperature, conditions, and humidity
    """
    ctx.logger.info(f"Getting weather for {location}")

    # Mock weather data
    weather = {
        "location": location,
        "temperature": 22 if units == "celsius" else 72,
        "units": units,
        "conditions": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 10
    }

    return weather


@tool(auto_schema=True)
def format_text(ctx: Context, text: str, style: str = "uppercase") -> str:
    """Format text in various styles.

    Args:
        text: The text to format
        style: Formatting style (uppercase, lowercase, title, reverse)

    Returns:
        Formatted text
    """
    # Note: This is a sync function - will be auto-converted to async
    styles = {
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "title": text.title(),
        "reverse": text[::-1]
    }

    return styles.get(style, text)


async def main():
    print("=== Basic Tool Example with Auto-Schema ===\n")

    # Create context
    ctx = Context(run_id="tool-demo")

    # 1. Discover registered tools
    print("1. Registered Tools:")
    for tool_name in ToolRegistry.list_names():
        tool_instance = ToolRegistry.get(tool_name)
        print(f"   - {tool_name}: {tool_instance.description}")

    print()

    # 2. Inspect tool schema
    print("2. Tool Schema Example (search_web):")
    search_tool = ToolRegistry.get("search_web")
    schema = search_tool.get_schema()

    print(f"   Name: {schema['name']}")
    print(f"   Description: {schema['description']}")
    print(f"   Required params: {schema['input_schema']['required']}")
    print(f"   Properties:")
    for prop_name, prop_schema in schema['input_schema']['properties'].items():
        print(f"     - {prop_name} ({prop_schema['type']}): {prop_schema['description']}")

    print()

    # 3. Invoke tools
    print("3. Invoking Tools:\n")

    print("   Searching web...")
    results = await search_tool.invoke(ctx, query="Python async programming", max_results=2)
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['title']}")
        print(f"      {result['url']}")

    print()

    print("   Performing calculations...")
    calc_tool = ToolRegistry.get("calculate")
    result = await calc_tool.invoke(ctx, operation="multiply", a=7, b=6)
    print(f"   7 × 6 = {result}")

    result = await calc_tool.invoke(ctx, operation="add", a=15.5, b=20.3)
    print(f"   15.5 + 20.3 = {result}")

    print()

    print("   Getting weather...")
    weather_tool = ToolRegistry.get("get_weather")
    weather = await weather_tool.invoke(ctx, location="San Francisco")
    print(f"   Location: {weather['location']}")
    print(f"   Temperature: {weather['temperature']}°{weather['units'][0].upper()}")
    print(f"   Conditions: {weather['conditions']}")

    print()

    print("   Formatting text...")
    format_tool = ToolRegistry.get("format_text")
    formatted = await format_tool.invoke(ctx, text="Hello World", style="reverse")
    print(f"   Original: 'Hello World'")
    print(f"   Reversed: '{formatted}'")

    print()

    # 4. Direct tool invocation (alternative syntax)
    print("4. Direct Tool Invocation:")
    result = await search_web(ctx, query="AGNT5 SDK")
    print(f"   Found {len(result)} results for 'AGNT5 SDK'")

    print()

    # 5. Tool with context state
    print("5. Tools with Context State:")

    @tool(auto_schema=True)
    async def counter_tool(ctx: Context, increment: int = 1) -> int:
        """Increment a counter stored in context.

        Args:
            increment: Amount to increment by
        """
        current = ctx.get("counter", 0)
        new_value = current + increment
        ctx.set("counter", new_value)
        return new_value

    # Use the tool multiple times
    for i in range(3):
        count = await counter_tool(ctx, increment=i+1)
        print(f"   Increment by {i+1}: counter = {count}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
