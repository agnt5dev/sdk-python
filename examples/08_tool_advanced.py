"""
Example: Advanced Tool Features

This example demonstrates:
- Tools with complex type annotations (List, Dict, Optional)
- Tools requiring confirmation
- Manual schema definition
- Tool composition and chaining
"""

import asyncio
from typing import Dict, List, Optional

from agnt5 import Context, Tool, tool, ToolRegistry


# Tools with complex types
@tool(auto_schema=True)
async def analyze_data(ctx: Context, dataset: List[float], operation: str = "average") -> Dict:
    """Analyze a dataset of numbers.

    Args:
        dataset: List of numbers to analyze
        operation: Type of analysis (average, sum, min, max, count)

    Returns:
        Analysis results including the requested operation and value
    """
    operations = {
        "average": lambda data: sum(data) / len(data) if data else 0,
        "sum": lambda data: sum(data),
        "min": lambda data: min(data) if data else None,
        "max": lambda data: max(data) if data else None,
        "count": lambda data: len(data)
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    value = operations[operation](dataset)

    return {
        "operation": operation,
        "value": value,
        "data_points": len(dataset)
    }


@tool(auto_schema=True)
async def filter_items(
    ctx: Context,
    items: List[Dict],
    filter_key: str,
    filter_value: Optional[str] = None
) -> List[Dict]:
    """Filter a list of dictionaries.

    Args:
        items: List of dictionaries to filter
        filter_key: Key to filter on
        filter_value: Value to match (if None, filters for key existence)

    Returns:
        Filtered list of items
    """
    if filter_value is None:
        # Filter for key existence
        return [item for item in items if filter_key in item]
    else:
        # Filter for key-value match
        return [item for item in items if item.get(filter_key) == filter_value]


@tool(auto_schema=True)
async def merge_dicts(ctx: Context, dict1: Dict, dict2: Dict, overwrite: bool = True) -> Dict:
    """Merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        overwrite: Whether dict2 values overwrite dict1 values on conflict

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key not in result or overwrite:
            result[key] = value

    return result


# Tool requiring confirmation
@tool(auto_schema=True, confirmation=True)
async def delete_records(ctx: Context, table_name: str, filter_condition: str) -> Dict:
    """Delete records from a table.

    Warning: This is a destructive operation that requires confirmation.

    Args:
        table_name: Name of the table
        filter_condition: SQL-like filter condition

    Returns:
        Deletion summary
    """
    # Note: Confirmation just logs a warning in this example
    ctx.logger.warning(f"DESTRUCTIVE: Would delete from {table_name} where {filter_condition}")

    return {
        "table": table_name,
        "condition": filter_condition,
        "status": "simulated",
        "records_deleted": 0  # Simulated
    }


# Manual schema tool
async def custom_handler(ctx: Context, **kwargs) -> Dict:
    """Custom handler with manual schema."""
    ctx.logger.info(f"Custom tool called with: {kwargs}")

    return {
        "processed": True,
        "input_keys": list(kwargs.keys())
    }


# Create tool with manual schema
custom_tool = Tool(
    name="custom_processor",
    description="Process data with custom schema",
    handler=custom_handler,
    input_schema={
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "Data to process",
                "items": {"type": "string"}
            },
            "options": {
                "type": "object",
                "description": "Processing options"
            }
        },
        "required": ["data"]
    },
    auto_schema=False
)


# Register manual tool
ToolRegistry.register(custom_tool)


# Tool composition example
@tool(auto_schema=True)
async def data_pipeline(
    ctx: Context,
    numbers: List[float],
    filter_threshold: Optional[float] = None
) -> Dict:
    """Execute a data processing pipeline.

    This tool composes multiple other tools.

    Args:
        numbers: Input numbers
        filter_threshold: Optional threshold to filter numbers

    Returns:
        Pipeline results
    """
    ctx.logger.info("Starting data pipeline...")

    # Step 1: Filter data if threshold provided
    if filter_threshold is not None:
        filtered = [n for n in numbers if n >= filter_threshold]
        ctx.logger.info(f"Filtered {len(numbers)} -> {len(filtered)} numbers")
    else:
        filtered = numbers

    # Step 2: Analyze the data using analyze_data tool
    analyze_tool = ToolRegistry.get("analyze_data")

    avg_result = await analyze_tool.invoke(ctx, dataset=filtered, operation="average")
    min_result = await analyze_tool.invoke(ctx, dataset=filtered, operation="min")
    max_result = await analyze_tool.invoke(ctx, dataset=filtered, operation="max")

    # Step 3: Combine results
    results = {
        "input_count": len(numbers),
        "processed_count": len(filtered),
        "statistics": {
            "average": avg_result["value"],
            "min": min_result["value"],
            "max": max_result["value"]
        }
    }

    ctx.logger.info("Pipeline completed")
    return results


async def main():
    print("=== Advanced Tool Features Example ===\n")

    ctx = Context(run_id="advanced-tools")

    # 1. Complex type annotations
    print("1. Tools with Complex Types:\n")

    print("   Analyzing dataset...")
    analysis = await analyze_data(
        ctx,
        dataset=[10.5, 20.3, 15.7, 30.2, 25.8],
        operation="average"
    )
    print(f"   Operation: {analysis['operation']}")
    print(f"   Result: {analysis['value']:.2f}")
    print(f"   Data points: {analysis['data_points']}")

    print()

    print("   Filtering items...")
    items = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "SF"},
        {"name": "Charlie", "age": 35, "city": "NYC"}
    ]

    filtered = await filter_items(ctx, items=items, filter_key="city", filter_value="NYC")
    print(f"   Filtered {len(items)} items -> {len(filtered)} matches")
    for item in filtered:
        print(f"     - {item['name']} from {item['city']}")

    print()

    print("   Merging dictionaries...")
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"b": 20, "d": 4}

    merged = await merge_dicts(ctx, dict1=dict1, dict2=dict2, overwrite=True)
    print(f"   dict1: {dict1}")
    print(f"   dict2: {dict2}")
    print(f"   merged: {merged}")

    print()

    # 2. Tool requiring confirmation
    print("2. Tool with Confirmation Requirement:\n")

    delete_tool = ToolRegistry.get("delete_records")
    schema = delete_tool.get_schema()

    print(f"   Tool: {schema['name']}")
    print(f"   Requires confirmation: {schema['requires_confirmation']}")

    print("   Invoking delete_records...")
    result = await delete_tool.invoke(
        ctx,
        table_name="users",
        filter_condition="created_at < '2020-01-01'"
    )
    print(f"   Status: {result['status']} (confirmation logged as warning)")

    print()

    # 3. Manual schema tool
    print("3. Tool with Manual Schema:\n")

    manual_tool = ToolRegistry.get("custom_processor")
    manual_schema = manual_tool.get_schema()

    print(f"   Tool: {manual_schema['name']}")
    print(f"   Description: {manual_schema['description']}")
    print(f"   Required fields: {manual_schema['input_schema']['required']}")

    result = await manual_tool.invoke(
        ctx,
        data=["item1", "item2", "item3"],
        options={"mode": "fast"}
    )
    print(f"   Result: {result}")

    print()

    # 4. Tool composition
    print("4. Tool Composition (Pipeline):\n")

    print("   Running data pipeline...")
    pipeline_result = await data_pipeline(
        ctx,
        numbers=[5.2, 15.7, 8.1, 20.3, 12.4, 30.6, 25.1],
        filter_threshold=10.0
    )

    print(f"   Input: {pipeline_result['input_count']} numbers")
    print(f"   After filtering (>= 10.0): {pipeline_result['processed_count']} numbers")
    print(f"   Statistics:")
    print(f"     Average: {pipeline_result['statistics']['average']:.2f}")
    print(f"     Min: {pipeline_result['statistics']['min']:.2f}")
    print(f"     Max: {pipeline_result['statistics']['max']:.2f}")

    print()

    # 5. Inspect all tools
    print("5. All Registered Tools:")
    for tool_name in sorted(ToolRegistry.list_names()):
        t = ToolRegistry.get(tool_name)
        required = t.input_schema.get("required", [])
        props = t.input_schema.get("properties", {})

        print(f"   {tool_name}")
        print(f"     Required: {required if required else 'none'}")
        print(f"     Parameters: {len(props)}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
