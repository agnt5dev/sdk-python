"""
Example 20: Structured Output with LLMs

This example demonstrates:
- Using dataclasses for structured output
- Using Pydantic models for structured output
- Code analysis with structured responses
- Accessing structured output via .structured_output, .parsed, and .object properties
"""

import asyncio
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field

from agnt5 import lm


# ============================================================================
# Example 1: Dataclass-based Structured Output
# ============================================================================

@dataclass
class CodeReview:
    """Code review results using Python dataclass."""
    issues: List[str]
    suggestions: List[str]
    overall_quality: int  # 1-10 scale


async def analyze_code_with_dataclass():
    """Analyze code and get structured output using dataclass."""
    print("\n" + "=" * 80)
    print("Example 1: Code Analysis with Dataclass")
    print("=" * 80)

    code_to_review = """
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price'] * item['quantity']
    return total
"""

    response = await lm.generate(
        model="openai/gpt-4o",
        prompt=f"""Analyze this Python code and provide a structured review:

{code_to_review}

Provide:
1. issues: List of potential problems or bugs
2. suggestions: List of improvement suggestions
3. overall_quality: Quality rating from 1-10
""",
        response_format=CodeReview,
        temperature=0.3,
    )

    # Access via .structured_output (recommended)
    review = response.structured_output
    print("\n✅ Structured Output (dict):")
    print(f"   Issues: {review['issues']}")
    print(f"   Suggestions: {review['suggestions']}")
    print(f"   Quality Score: {review['overall_quality']}/10")

    # Access via .parsed (OpenAI SDK compatibility)
    print("\n✅ Parsed Output (same as structured_output):")
    print(f"   Type: {type(response.parsed)}")
    print(f"   Content: {response.parsed}")

    # Access via .object (generic alias)
    print("\n✅ Object Output (same as structured_output):")
    print(f"   Type: {type(response.object)}")


# ============================================================================
# Example 2: Pydantic-based Structured Output
# ============================================================================

class SecurityAnalysis(BaseModel):
    """Security analysis results using Pydantic model."""
    vulnerabilities: List[str] = Field(description="List of security vulnerabilities found")
    severity: str = Field(description="Overall severity: low, medium, high, critical")
    recommendations: List[str] = Field(description="Security recommendations")
    safe_to_deploy: bool = Field(description="Whether code is safe to deploy")


async def analyze_security_with_pydantic():
    """Analyze code security and get structured output using Pydantic."""
    print("\n" + "=" * 80)
    print("Example 2: Security Analysis with Pydantic")
    print("=" * 80)

    code_to_analyze = """
import pickle
import os

def load_user_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_command(user_input):
    os.system(user_input)
"""

    response = await lm.generate(
        model="openai/gpt-4o",
        prompt=f"""Perform a security analysis on this Python code:

{code_to_analyze}

Identify:
1. vulnerabilities: List of security issues
2. severity: Overall severity level (low/medium/high/critical)
3. recommendations: How to fix the issues
4. safe_to_deploy: Whether this code is safe for production
""",
        response_format=SecurityAnalysis,
        temperature=0.1,
    )

    # Access structured output
    analysis = response.structured_output
    print(f"\n🔒 Security Analysis:")
    print(f"   Severity: {analysis['severity']}")
    print(f"   Safe to Deploy: {analysis['safe_to_deploy']}")
    print(f"\n   Vulnerabilities:")
    for vuln in analysis['vulnerabilities']:
        print(f"   - {vuln}")
    print(f"\n   Recommendations:")
    for rec in analysis['recommendations']:
        print(f"   - {rec}")


# ============================================================================
# Example 3: Complex Nested Structures
# ============================================================================

class FunctionSignature(BaseModel):
    """Function signature details."""
    name: str
    parameters: List[str]
    return_type: str
    docstring: str


class CodebaseAnalysis(BaseModel):
    """Complete codebase analysis."""
    functions: List[FunctionSignature]
    complexity_score: int = Field(ge=1, le=10, description="Code complexity (1-10)")
    maintainability: str = Field(description="Maintainability assessment")
    test_coverage_estimate: int = Field(ge=0, le=100, description="Estimated test coverage %")


async def analyze_codebase_structure():
    """Analyze entire codebase structure with nested output."""
    print("\n" + "=" * 80)
    print("Example 3: Codebase Structure Analysis with Nested Pydantic")
    print("=" * 80)

    code = """
def fetch_user(user_id: int) -> dict:
    '''Fetch user data from database.'''
    # Implementation here
    pass

def update_profile(user_id: int, data: dict) -> bool:
    '''Update user profile information.'''
    # Implementation here
    pass

def send_notification(user_id: int, message: str) -> None:
    '''Send notification to user.'''
    # Implementation here
    pass
"""

    response = await lm.generate(
        model="openai/gpt-4o",
        prompt=f"""Analyze this Python codebase:

{code}

Extract:
1. functions: List of all functions with their signatures, parameters, return types, and docstrings
2. complexity_score: Overall complexity rating (1-10)
3. maintainability: Assessment of how maintainable the code is
4. test_coverage_estimate: Estimated percentage of test coverage needed
""",
        response_format=CodebaseAnalysis,
        temperature=0.2,
    )

    analysis = response.structured_output
    print(f"\n📊 Codebase Analysis:")
    print(f"   Complexity Score: {analysis['complexity_score']}/10")
    print(f"   Maintainability: {analysis['maintainability']}")
    print(f"   Estimated Test Coverage Needed: {analysis['test_coverage_estimate']}%")
    print(f"\n   Functions Detected:")
    for func in analysis['functions']:
        print(f"   - {func['name']}({', '.join(func['parameters'])}) -> {func['return_type']}")
        print(f"     {func['docstring']}")


# ============================================================================
# Example 4: Raw JSON Schema
# ============================================================================

async def analyze_with_raw_json_schema():
    """Use raw JSON schema for structured output."""
    print("\n" + "=" * 80)
    print("Example 4: Code Metrics with Raw JSON Schema")
    print("=" * 80)

    # Define schema directly as dict
    schema = {
        "type": "object",
        "properties": {
            "lines_of_code": {"type": "integer"},
            "cyclomatic_complexity": {"type": "integer"},
            "functions_count": {"type": "integer"},
            "classes_count": {"type": "integer"},
            "comments_ratio": {"type": "number"},
        },
        "required": ["lines_of_code", "cyclomatic_complexity", "functions_count", "classes_count"],
        "additionalProperties": False
    }

    code = """
class UserManager:
    def __init__(self):
        self.users = {}

    def add_user(self, user_id, name):
        self.users[user_id] = name

    def get_user(self, user_id):
        return self.users.get(user_id)
"""

    response = await lm.generate(
        model="openai/gpt-4o",
        prompt=f"""Calculate code metrics for this Python code:

{code}

Provide exact counts for:
- lines_of_code: Total lines
- cyclomatic_complexity: Complexity score
- functions_count: Number of functions
- classes_count: Number of classes
- comments_ratio: Ratio of comment lines to code lines (0-1)
""",
        response_format=schema,
        temperature=0.0,
    )

    metrics = response.structured_output
    print(f"\n📈 Code Metrics:")
    print(f"   Lines of Code: {metrics['lines_of_code']}")
    print(f"   Cyclomatic Complexity: {metrics['cyclomatic_complexity']}")
    print(f"   Functions: {metrics['functions_count']}")
    print(f"   Classes: {metrics['classes_count']}")
    print(f"   Comments Ratio: {metrics.get('comments_ratio', 0):.2%}")


async def main():
    """Run all structured output examples."""
    print("\n" + "=" * 80)
    print("🚀 Structured Output Examples with Code Analysis")
    print("=" * 80)

    # Run all examples
    await analyze_code_with_dataclass()
    await analyze_security_with_pydantic()
    await analyze_codebase_structure()
    await analyze_with_raw_json_schema()

    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Use dataclasses for simple structured output")
    print("  2. Use Pydantic for validation and complex schemas")
    print("  3. Use raw JSON schema for maximum control")
    print("  4. Access via .structured_output (recommended), .parsed, or .object")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
