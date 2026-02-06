# Batch Evaluation

## What is Batch Evaluation?

**Batch Evaluation** (`batch_eval`) enables you to evaluate a component against multiple inputs with scoring in a single operation. It combines the power of AGNT5's evaluation framework with parallel execution, making it ideal for:

- **Dataset evaluation**: Run your agent/function against a test dataset
- **Regression testing**: Verify outputs haven't changed across code updates
- **Quality benchmarking**: Measure performance with multiple scorers including LLM judges
- **A/B testing**: Compare outputs across different component versions

Batch evaluation runs `client.eval()` in parallel for each input item with controlled concurrency, collecting results and statistics.

## Why Use Batch Evaluation?

### 1. Efficient Parallel Execution

Instead of calling `eval()` sequentially for each test case, `batch_eval` runs evaluations in parallel with configurable concurrency. This significantly reduces the total time for large evaluation sets.

### 2. Rich Statistics

Get comprehensive statistics about your evaluation run:
- Total items processed
- Pass/fail rates
- Average duration per item
- Easy access to failing items for debugging

### 3. Multiple Scorers

Combine built-in scorers, custom scorers, and LLM judges in a single evaluation:
- `exact_match`, `contains`, `json_valid`, `regex_match`, `levenshtein`
- Custom Python scorers via `@scorer` decorator
- LLM-as-judge for semantic evaluation

### 4. Flexible Input Formats

Accept inputs in multiple formats to fit your workflow:
- Plain input dictionaries with separate expected values
- Structured `BatchEvalItem` objects for full control
- Mixed formats in the same batch

## How to Use Batch Evaluation

### Basic Usage

The simplest way to use batch evaluation with expected values:

```python
from agnt5 import Client

client = Client("http://localhost:34181")

result = client.batch_eval(
    component="greet",
    items=[
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Charlie"},
    ],
    expected=[
        "Hello, Alice!",
        "Hello, Bob!",
        "Hello, Charlie!",
    ],
    scorers=["exact_match"],
)

print(f"Pass rate: {result.pass_rate:.0%}")  # Pass rate: 100%
print(f"Total items: {result.stats.total_items}")
print(f"Passed: {result.stats.passed_items}")
```

### Using BatchEvalItem for More Control

For more control over individual items, use `BatchEvalItem`:

```python
from agnt5 import Client, BatchEvalItem

client = Client("http://localhost:34181")

result = client.batch_eval(
    component="add",
    items=[
        BatchEvalItem(
            input={"a": 1, "b": 2},
            expected=3,
            item_id="test-add-positive",
        ),
        BatchEvalItem(
            input={"a": -5, "b": 3},
            expected=-2,
            item_id="test-add-negative",
        ),
        BatchEvalItem(
            input={"a": 0, "b": 0},
            expected=0,
            item_id="test-add-zero",
        ),
    ],
    scorers=["exact_match"],
)

# Access results by item_id
for item in result.results:
    status = "PASS" if item.passed else "FAIL"
    print(f"{item.item_id}: {status}")
```

### Multiple Scorers

Combine multiple scorers to evaluate different aspects of the output:

```python
from agnt5 import Client

client = Client("http://localhost:34181")

result = client.batch_eval(
    component="generate_json_report",
    items=[
        {"data": [1, 2, 3]},
        {"data": [4, 5, 6]},
    ],
    scorers=[
        "json_valid",      # Check output is valid JSON
        "contains",        # Check for expected substrings
    ],
    expected=[
        '{"summary":',     # Expected substring for contains scorer
        '{"summary":',
    ],
)

# Check individual scorer results
for item in result.results:
    json_score = item.get_score("json_valid")
    contains_score = item.get_score("contains")
    print(f"Item {item.index}:")
    print(f"  JSON valid: {json_score.passed if json_score else 'N/A'}")
    print(f"  Contains: {contains_score.passed if contains_score else 'N/A'}")
```

### LLM-as-Judge Scoring

Use `LLMJudge` for semantic evaluation when exact matching isn't appropriate:

```python
from agnt5 import Client, BatchEvalItem
from agnt5.eval import LLMJudge

client = Client("http://localhost:34181")

result = client.batch_eval(
    component="summarize",
    component_type="agent",
    items=[
        BatchEvalItem(
            input={"text": "The quick brown fox jumps over the lazy dog. " * 10},
            item_id="doc-1",
        ),
        BatchEvalItem(
            input={"text": "Machine learning is a subset of artificial intelligence. " * 10},
            item_id="doc-2",
        ),
    ],
    scorers=[
        "json_valid",  # Ensure output is valid JSON
        LLMJudge(
            criteria="Is the summary concise (under 50 words) and accurate?",
            model="openai/gpt-4o-mini",
        ),
        LLMJudge(
            criteria="Does the summary capture the main topic without hallucination?",
            model="anthropic/claude-3-haiku-20240307",
        ),
    ],
)

print(f"Overall pass rate: {result.pass_rate:.0%}")

for item in result.results:
    print(f"\n{item.item_id}:")
    for score in item.scores:
        print(f"  {score.scorer}: {score.score:.2f} - {score.explanation or 'N/A'}")
```

### Controlling Concurrency

Limit parallel evaluations to avoid overwhelming external services:

```python
result = client.batch_eval(
    component="call_external_api",
    items=large_dataset,  # 1000 items
    scorers=["exact_match"],
    expected=expected_results,
    max_concurrency=5,  # Only 5 parallel requests
    timeout=30.0,       # 30 second timeout per item
)
```

### Async Client

For async applications, use `AsyncClient`:

```python
import asyncio
from agnt5 import AsyncClient, BatchEvalItem

async def run_evaluation():
    async with AsyncClient("http://localhost:34181") as client:
        result = await client.batch_eval(
            component="analyze",
            items=[
                BatchEvalItem(input={"text": "Hello"}, expected="greeting"),
                BatchEvalItem(input={"text": "Goodbye"}, expected="farewell"),
            ],
            scorers=["exact_match"],
            max_concurrency=10,
        )
        return result

result = asyncio.run(run_evaluation())
print(f"Pass rate: {result.pass_rate:.0%}")
```

## Input Formats

`batch_eval` accepts items in several formats:

### 1. Plain Input Dictionaries

Pass inputs directly with a separate `expected` list:

```python
result = client.batch_eval(
    component="add",
    items=[
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ],
    expected=[3, 7],
    scorers=["exact_match"],
)
```

### 2. Dictionaries with Input Key

Include `input` and optionally `expected` in each dict:

```python
result = client.batch_eval(
    component="add",
    items=[
        {"input": {"a": 1, "b": 2}, "expected": 3},
        {"input": {"a": 3, "b": 4}, "expected": 7, "item_id": "test-2"},
    ],
    scorers=["exact_match"],
)
```

### 3. BatchEvalItem Objects

Full control with typed objects:

```python
from agnt5 import BatchEvalItem

result = client.batch_eval(
    component="add",
    items=[
        BatchEvalItem(input={"a": 1, "b": 2}, expected=3, item_id="add-1"),
        BatchEvalItem(input={"a": 3, "b": 4}, expected=7, item_id="add-2"),
    ],
    scorers=["exact_match"],
)
```

### 4. Mixed Formats

You can mix formats in the same batch:

```python
result = client.batch_eval(
    component="add",
    items=[
        BatchEvalItem(input={"a": 1, "b": 2}, expected=3),
        {"input": {"a": 3, "b": 4}, "expected": 7},
        {"a": 5, "b": 6},  # Uses expected list
    ],
    expected=[None, None, 11],  # Only third item uses this
    scorers=["exact_match"],
)
```

## Available Scorers

### Built-in Scorers

| Scorer | Description | Requires Expected |
|--------|-------------|-------------------|
| `exact_match` | Exact equality check | Yes |
| `contains` | Substring check | Yes |
| `json_valid` | Valid JSON check | No |
| `regex_match` | Regex pattern match | Yes (pattern) |
| `levenshtein` | Edit distance similarity | Yes |

### LLMJudge

For semantic evaluation using language models:

```python
from agnt5.eval import LLMJudge

LLMJudge(
    criteria="Is the response helpful and accurate?",
    model="openai/gpt-4o-mini",      # Provider/model format
    include_input=True,               # Include input in judge prompt
    temperature=0.0,                  # Deterministic by default
)
```

**Supported Models:**
- OpenAI: `openai/gpt-4o-mini`, `openai/gpt-4o`, `openai/gpt-4-turbo`
- Anthropic: `anthropic/claude-3-haiku-20240307`, `anthropic/claude-3-5-sonnet-20241022`
- Any model supported by the platform's LM service

### Custom Scorers

Create custom scorers with the `@scorer` decorator:

```python
from agnt5 import scorer, ScorerContext, ScorerResult

@scorer(name="word_count")
def check_word_count(ctx: ScorerContext) -> ScorerResult:
    """Check if output has fewer than 100 words."""
    word_count = len(str(ctx.output).split())
    passed = word_count < 100
    return ScorerResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        explanation=f"Output has {word_count} words",
    )

# Use in batch_eval
result = client.batch_eval(
    component="summarize",
    items=test_items,
    scorers=["word_count", "json_valid"],
)
```

## Working with Results

### BatchEvalResult

The main result object provides:

```python
result = client.batch_eval(...)

# Overall status
result.batch_id      # Unique batch identifier
result.status        # "completed", "partial_failure", or "failed"
result.pass_rate     # Float 0.0-1.0 (passed items / total items)
result.is_success    # True if all items evaluated without errors
result.is_partial_failure  # True if some items had errors

# Access all results
result.results       # List[BatchEvalItemResult]
result.outputs       # List of outputs sorted by index

# Filter results
result.passing_items()  # Items where passed=True
result.failing_items()  # Items where passed=False
result.failed_items()   # Items with evaluation errors
```

### BatchEvalItemResult

Individual item results:

```python
for item in result.results:
    item.index       # Position in batch
    item.item_id     # Custom identifier (if provided)
    item.run_id      # Platform run ID
    item.output      # Component output
    item.passed      # True if all scorers passed
    item.scores      # List[ScorerResultSummary]
    item.duration_ms # Execution time
    item.trace_id    # OpenTelemetry trace ID
    item.error       # Error message (if failed)

    # Check specific scorer
    item.is_success  # True if no errors
    item.is_failed   # True if had errors
    item.get_score("exact_match")  # Get specific scorer result
```

### BatchEvalStats

Aggregate statistics:

```python
stats = result.stats

stats.total_items      # Total items in batch
stats.completed_items  # Items evaluated without errors
stats.failed_items     # Items with evaluation errors
stats.passed_items     # Items where passed=True
stats.avg_duration_ms  # Average duration per item
stats.duration_ms      # Total batch duration
```

## Best Practices

### 1. Use Item IDs for Debugging

Always assign meaningful `item_id` values for easier debugging:

```python
items = [
    BatchEvalItem(
        input=test_case["input"],
        expected=test_case["expected"],
        item_id=f"test-{test_case['name']}",
    )
    for test_case in test_suite
]
```

### 2. Start with Small Batches

When developing, start with small batches to iterate quickly:

```python
# Development: small batch, low concurrency
result = client.batch_eval(
    component="my_agent",
    items=test_items[:5],  # Just first 5
    scorers=["exact_match"],
    max_concurrency=2,
)

# Production: full dataset
result = client.batch_eval(
    component="my_agent",
    items=test_items,
    scorers=["exact_match"],
    max_concurrency=10,
)
```

### 3. Combine Fast and Slow Scorers

Put fast scorers first to fail fast, add LLM judges for semantic checks:

```python
scorers = [
    "json_valid",           # Fast: check structure first
    "contains",             # Fast: check for required content
    LLMJudge(criteria="..."),  # Slow: semantic evaluation last
]
```

### 4. Handle Partial Failures

Check for items that failed evaluation (not just failed scoring):

```python
result = client.batch_eval(...)

if result.status == "partial_failure":
    print("Some items failed to evaluate:")
    for item in result.failed_items():
        print(f"  {item.item_id}: {item.error}")

# Only consider successfully evaluated items
success_rate = result.stats.passed_items / result.stats.completed_items
```

### 5. Set Appropriate Timeouts

For LLM-based components, set realistic timeouts:

```python
result = client.batch_eval(
    component="slow_agent",
    component_type="agent",
    items=test_items,
    scorers=[LLMJudge(criteria="...")],
    timeout=60.0,  # 60 seconds per item
    max_concurrency=3,  # Limit parallel LLM calls
)
```

## API Reference

### Client.batch_eval()

```python
def batch_eval(
    self,
    component: str,
    items: List[Union[Dict[str, Any], BatchEvalItem]],
    scorers: Optional[List[Union[str, LLMJudge]]] = None,
    expected: Optional[List[Any]] = None,
    component_type: str = "function",
    max_concurrency: int = 10,
    timeout: Optional[float] = None,
) -> BatchEvalResult:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `component` | `str` | required | Name of the component to evaluate |
| `items` | `List[Dict \| BatchEvalItem]` | required | Input items to evaluate |
| `scorers` | `List[str \| LLMJudge]` | `None` | Scorers to run on each output |
| `expected` | `List[Any]` | `None` | Expected outputs (parallel to items) |
| `component_type` | `str` | `"function"` | `"function"`, `"workflow"`, or `"agent"` |
| `max_concurrency` | `int` | `10` | Maximum parallel evaluations |
| `timeout` | `float` | `None` | Per-item timeout in seconds |

### BatchEvalItem

```python
@dataclass
class BatchEvalItem:
    input: Dict[str, Any]          # Required: input data
    expected: Optional[Any] = None  # Expected output for comparison
    item_id: Optional[str] = None   # Custom identifier
    index: Optional[int] = None     # Position (auto-assigned)
```

### BatchEvalResult

```python
@dataclass
class BatchEvalResult:
    batch_id: str                           # Unique batch ID
    status: str                             # "completed", "partial_failure", "failed"
    results: List[BatchEvalItemResult]      # Individual results
    stats: Optional[BatchEvalStats]         # Aggregate statistics

    @property
    def pass_rate(self) -> float: ...       # Passed / total
    def passing_items(self) -> List[...]: ...
    def failing_items(self) -> List[...]: ...
    def failed_items(self) -> List[...]: ...
```

### BatchEvalItemResult

```python
@dataclass
class BatchEvalItemResult:
    index: int                              # Position in batch
    run_id: str                             # Platform run ID
    output: Any                             # Component output
    scores: List[ScorerResultSummary]       # Scorer results
    passed: bool                            # All scorers passed
    duration_ms: int                        # Execution time
    item_id: Optional[str] = None           # Custom identifier
    trace_id: Optional[str] = None          # OpenTelemetry trace
    error: Optional[str] = None             # Error message

    def get_score(self, name: str) -> Optional[ScorerResultSummary]: ...
```

### BatchEvalStats

```python
@dataclass
class BatchEvalStats:
    total_items: int = 0        # Total items
    completed_items: int = 0    # Evaluated without errors
    failed_items: int = 0       # Had evaluation errors
    passed_items: int = 0       # Passed all scorers
    avg_duration_ms: int = 0    # Average per item
    duration_ms: int = 0        # Total batch time
```

## See Also

- [Evaluation Framework](../eval/) - Scorers and LLMJudge details
- [Client API](../client.md) - Full client documentation
- [Agent Component](agent.md) - Evaluating agents
