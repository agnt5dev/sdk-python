"""
Test Tools for Integration Testing

Provides reusable tools for testing tool and agent functionality.
"""

from typing import Dict, List

from agnt5 import Context, tool


@tool(auto_schema=True)
async def calculate_total(ctx: Context, numbers: List[float], operation: str = "sum") -> Dict:
    """Calculate statistical operations on a list of numbers.

    Args:
        numbers: List of numbers to process
        operation: Operation to perform (sum, average, min, max)

    Returns:
        Dictionary with operation result
    """
    ctx.logger.info(f"Calculating {operation} for {len(numbers)} numbers")

    if not numbers:
        return {"error": "No numbers provided", "result": None}

    operations = {
        "sum": lambda nums: sum(nums),
        "average": lambda nums: sum(nums) / len(nums),
        "min": lambda nums: min(nums),
        "max": lambda nums: max(nums),
    }

    if operation not in operations:
        return {"error": f"Unknown operation: {operation}", "result": None}

    result = operations[operation](numbers)

    return {
        "operation": operation,
        "result": result,
        "count": len(numbers),
    }


@tool(auto_schema=True)
async def search_database(ctx: Context, query: str, limit: int = 5) -> List[Dict]:
    """Search a mock database for records.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        List of matching records
    """
    ctx.logger.info(f"Searching database for: {query}")

    # Simulate database search with mock data
    all_records = [
        {"id": 1, "title": "Product Analytics Dashboard", "category": "analytics", "score": 0.95},
        {"id": 2, "title": "User Authentication System", "category": "security", "score": 0.88},
        {"id": 3, "title": "Analytics API Documentation", "category": "analytics", "score": 0.85},
        {"id": 4, "title": "Payment Processing Module", "category": "payment", "score": 0.82},
        {"id": 5, "title": "Security Best Practices", "category": "security", "score": 0.78},
    ]

    # Simple keyword matching
    query_lower = query.lower()
    filtered = [r for r in all_records if query_lower in r["title"].lower() or query_lower in r["category"]]

    # Sort by score and limit
    filtered.sort(key=lambda x: x["score"], reverse=True)

    return filtered[:limit]


@tool(auto_schema=True)
async def format_report(ctx: Context, data: Dict, format_style: str = "summary") -> str:
    """Format data into a structured report.

    Args:
        data: Data to format
        format_style: Report style (summary, detailed, compact)

    Returns:
        Formatted report as string
    """
    ctx.logger.info(f"Formatting report in {format_style} style")

    if format_style == "summary":
        lines = ["=== SUMMARY REPORT ==="]
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    elif format_style == "detailed":
        lines = ["=== DETAILED REPORT ==="]
        for key, value in data.items():
            lines.append(f"\n{key.upper()}:")
            if isinstance(value, (list, dict)):
                lines.append(f"  {value}")
            else:
                lines.append(f"  Value: {value}")
                lines.append(f"  Type: {type(value).__name__}")
        return "\n".join(lines)

    elif format_style == "compact":
        return " | ".join([f"{k}={v}" for k, v in data.items()])

    else:
        return f"Error: Unknown format style '{format_style}'"


@tool(auto_schema=True)
async def validate_data(ctx: Context, data: Dict, required_fields: List[str]) -> Dict:
    """Validate that data contains required fields.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names

    Returns:
        Validation result with status and missing fields
    """
    ctx.logger.info(f"Validating data with {len(required_fields)} required fields")

    missing_fields = [field for field in required_fields if field not in data]

    is_valid = len(missing_fields) == 0

    return {
        "is_valid": is_valid,
        "missing_fields": missing_fields,
        "provided_fields": list(data.keys()),
        "message": "Validation passed" if is_valid else f"Missing fields: {missing_fields}",
    }


__all__ = [
    "calculate_total",
    "search_database",
    "format_report",
    "validate_data",
]
