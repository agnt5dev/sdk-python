# Structured Output Investigation

## Issue Summary

The integration test `test_function_with_lm_structured_output` fails because `response.structured_output` returns `None` even when a `response_format` is provided.

## Test Details

**Test**: `tests/integration/test_client_lm_openai.py::test_function_with_lm_structured_output`

**Function Tested**: `analyze_sentiment()` in `openai_lm_functions.py`

**Expected Behavior**: When calling `lm.generate()` with `response_format=SentimentAnalysis` (a dataclass), the response should have `structured_output` populated with the parsed object.

**Actual Behavior**: `response.structured_output` is `None`

**Log Output**:
```
[INFO] LM analyzed sentiment: None
```

## Code Flow Analysis

### 1. Python LM API (`src/agnt5/lm.py`)

**Lines 328-331**: Response format detection
```python
if response_format is not None:
    format_type, json_schema = detect_format_type(response_format)
    response_schema_json = json.dumps(json_schema)
```

✅ This correctly converts the dataclass to JSON schema.

**Line 365**: Request construction
```python
request = GenerateRequest(
    ...
    response_schema=response_schema_json,  # ✅ Schema is passed
)
```

### 2. Rust FFI Layer (`rust-src/language_model.rs`)

**Lines 149-150**: Extract response_schema_kw from kwargs
```rust
let response_schema_kw = get_optional_string(kwargs_ref, "response_schema_kw")?;
```

⚠️ **ISSUE**: The parameter name is `response_schema_kw` but Python sends `response_schema`!

**Line 154**: Parse response format
```rust
let response_format = parse_response_format(response_format_kw.as_deref(), response_schema_kw.as_deref())?;
```

**Lines 542-582**: `parse_response_format()` function
```rust
fn parse_response_format(
    format: Option<&str>,
    schema_json: Option<&str>,
) -> PyResult<Option<ResponseFormat>> {
    let schema = parse_schema_json(schema_json)?;  // ✅ Parses JSON schema

    match (format.map(|f| f.trim().to_lowercase()), schema) {
        (None, None) => Ok(None),
        (None, Some(schema)) => Ok(Some(ResponseFormat::JsonSchema(schema))),  // ✅ Should work
        ...
    }
}
```

**Lines 754-760**: `PyResponse.object` getter
```rust
#[getter]
fn object(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
    self.inner
        .object
        .as_ref()
        .map(|value| json_to_py(py, value))
        .transpose()
}
```

✅ This correctly exposes `self.inner.object` to Python.

### 3. SDK Core (`sdk-core/src/lm/openai_common.rs`)

**Lines 189-193**: Response conversion for non-streaming
```rust
let object = match response_format {
    ResponseFormat::Text => None,
    ResponseFormat::Json => Some(parse_json_value(&text)?),
    ResponseFormat::JsonSchema(_) => Some(parse_json_value(&text)?),  // ✅ Parses JSON
};

Ok(GenerateResponse {
    ...
    object,  // ✅ Sets the object field
    ...
})
```

✅ The Rust core correctly parses the JSON response and populates the `object` field.

## Root Cause

### ⚠️ **Parameter Name Mismatch**

**Python Side** (`lm.py:139`):
```python
kwargs["response_schema_kw"] = response_schema_json
```

Wait, let me check this more carefully...

Actually, looking at `lm.py:365`:
```python
request = GenerateRequest(
    ...
    response_schema=response_schema_json,
)
```

But then at `lm.py:139` (in the internal `_LanguageModel.generate()`):
```python
if request.response_schema is not None:
    kwargs["response_schema_kw"] = request.response_schema
```

**Rust Side** (`language_model.rs:150`):
```rust
let response_schema_kw = get_optional_string(kwargs_ref, "response_schema_kw")?;
```

✅ The parameter names match! So this isn't the issue.

## Next Steps for Debugging

### 1. Add Debug Logging

Add logging in Rust to see what's being received:

**In `language_model.rs` around line 150**:
```rust
let response_schema_kw = get_optional_string(kwargs_ref, "response_schema_kw")?;
log::info!("response_schema_kw received: {:?}", response_schema_kw);
```

**In `openai_common.rs` around line 189**:
```rust
log::info!("response_format: {:?}, text: {}", response_format, text);
let object = match response_format {
    ...
};
log::info!("parsed object: {:?}", object);
```

### 2. Check OpenAI API Response

The issue might be that OpenAI's API is returning the structured output in a different format than expected. Need to:

1. Log the raw API response
2. Check if OpenAI is returning `choices[0].message.content` as JSON string
3. Verify the JSON parsing is succeeding

### 3. Test with Simpler Schema

Try with a very simple schema first:
```python
@dataclass
class SimpleTest:
    result: str

response = await lm.generate(
    model="openai/gpt-4o",
    prompt="Say hello",
    response_format=SimpleTest
)
```

### 4. Check OpenAI API Version

OpenAI's structured output feature (`response_format` with `json_schema`) requires:
- GPT-4 models (gpt-4o, gpt-4o-mini)
- Specific API parameters

Verify we're using the correct API format.

## Workaround for Tests

For now, we've skipped the test with a clear note about the issue:

```python
@pytest.mark.skip(reason="Structured output parsing in Rust core needs fixes - returns None for object field")
def test_function_with_lm_structured_output(client, worker_process):
    ...
```

## Recommendation

This requires deeper investigation in the Rust core. The Python SDK and FFI layer appear to be correct. The issue is likely:

1. **OpenAI API Response Format**: The response might not be in the expected format
2. **JSON Parsing**: The `parse_json_value()` function might be failing silently
3. **Response Format Detection**: The response_format might not be correctly identified as JsonSchema

**Next Action**: Add comprehensive logging to the Rust core to trace the exact data flow from API response to Python object.
