"""
Test Tools for Integration Testing

Provides reusable tools for testing tool and agent functionality.
"""

from typing import Dict, List, Union

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
async def validate_data(ctx: Context, data: Union[Dict, List[Dict]], required_fields: List[str]) -> Dict:
    """Validate that data contains required fields.

    Args:
        data: Data dictionary or list of dictionaries to validate
        required_fields: List of required field names

    Returns:
        Validation result with status and missing fields
    """
    ctx.logger.info(f"Validating data with {len(required_fields)} required fields")

    # Handle both single dict and list of dicts
    if isinstance(data, list):
        # Validate each item in the list
        results = []
        for idx, item in enumerate(data):
            missing_fields = [field for field in required_fields if field not in item]
            is_valid = len(missing_fields) == 0
            results.append({
                "index": idx,
                "is_valid": is_valid,
                "missing_fields": missing_fields,
                "provided_fields": list(item.keys()),
            })

        all_valid = all(r["is_valid"] for r in results)
        return {
            "is_valid": all_valid,
            "total_items": len(data),
            "valid_items": sum(1 for r in results if r["is_valid"]),
            "results": results,
            "message": f"Validated {len(data)} items. All valid: {all_valid}",
        }
    else:
        # Single dictionary validation
        missing_fields = [field for field in required_fields if field not in data]
        is_valid = len(missing_fields) == 0

        return {
            "is_valid": is_valid,
            "missing_fields": missing_fields,
            "provided_fields": list(data.keys()),
            "message": "Validation passed" if is_valid else f"Missing fields: {missing_fields}",
        }


@tool(auto_schema=True)
async def search_web(ctx: Context, query: str) -> List[Dict]:
    """
    Search the web for information (simulated).

    Args:
        query: Search query string

    Returns:
        List of search results with title and snippet
    """
    ctx.logger.info(f"[TOOL] search_web: searching for '{query}'")

    # Simulate search results
    results = [
        {
            "title": f"Article: {query}",
            "url": f"https://example.com/article-{query.replace(' ', '-')}",
            "snippet": f"Detailed information about {query}...",
        },
        {
            "title": f"Tutorial: {query}",
            "url": f"https://tutorial.com/{query.replace(' ', '-')}",
            "snippet": f"Learn all about {query} with examples...",
        },
    ]

    ctx.logger.info(f"[TOOL] search_web: found {len(results)} results")
    return results


@tool(auto_schema=True)
async def domain_specific_analysis(ctx: Context, data: str) -> Dict:
    """
    Perform domain-specific analysis on data (for specialist agent).

    Args:
        data: Data to analyze

    Returns:
        Analysis results with insights and recommendations
    """
    ctx.logger.info(f"[TOOL] domain_specific_analysis: analyzing '{data[:50]}...'")

    analysis = {
        "summary": f"Analysis of: {data[:50]}...",
        "key_insights": ["Insight 1: Important finding", "Insight 2: Notable pattern"],
        "recommendations": ["Recommendation 1: Action item", "Recommendation 2: Follow-up"],
        "confidence": 0.85,
    }

    ctx.logger.info(f"[TOOL] domain_specific_analysis: confidence = {analysis['confidence']}")
    return analysis


__all__ = [
    "calculate_total",
    "search_database",
    "format_report",
    "validate_data",
    "search_web",
    "domain_specific_analysis",
]
