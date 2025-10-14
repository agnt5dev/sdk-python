"""
Integration tests for distributed tracing and span collection.

Verifies that spans are properly created, exported via OTLP, and persisted
in the observability database for functions, workflows, tools, and agents.
"""

import time
import json
import requests
import pytest


def query_mcp_observability(mcp_endpoint: str, service: str, name: str = None):
    """
    Query spans from observability database via MCP HTTP endpoint.

    Args:
        mcp_endpoint: MCP HTTP endpoint URL
        service: Service name filter
        name: Optional span name filter

    Returns:
        List of spans matching the filter
    """
    # Call obs.list_traces via MCP HTTP endpoint
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "obs.list_traces",
            "arguments": {
                "service": service,
                "name": name,
                "limit": 10
            }
        }
    }

    response = requests.post(
        f"{mcp_endpoint}/mcp/rpc",
        json=payload,
        timeout=5
    )

    if response.status_code != 200:
        raise Exception(f"MCP request failed: {response.status_code} - {response.text}")

    result = response.json()

    if "error" in result:
        raise Exception(f"MCP error: {result['error']}")

    return result.get("result", {}).get("items", [])


def get_trace_spans(mcp_endpoint: str, trace_id: str):
    """
    Get all spans for a specific trace via MCP.

    Args:
        mcp_endpoint: MCP HTTP endpoint URL
        trace_id: Hex-encoded trace ID

    Returns:
        List of spans in the trace
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "obs.get_trace",
            "arguments": {
                "trace_id": trace_id
            }
        }
    }

    response = requests.post(
        f"{mcp_endpoint}/mcp/rpc",
        json=payload,
        timeout=5
    )

    if response.status_code != 200:
        raise Exception(f"MCP request failed: {response.status_code}")

    result = response.json()

    if "error" in result:
        raise Exception(f"MCP error: {result['error']}")

    return result.get("result", {}).get("spans", [])


def test_function_span_creation(client, worker_process, platform):
    """
    Test that function execution creates spans that are exported and stored.

    Verifies:
    - Function span is created in observability database
    - Span has correct attributes (service.name, function.name)
    - Span status is OK
    """
    # Execute a simple function
    result = client.run("greet", {"name": "TracingTest"})

    assert result is not None
    assert "message" in result

    # Wait for spans to be exported and stored
    time.sleep(3)

    # Query spans from observability database via MCP
    # Note: Span names are prefixed with component type (e.g., "function.greet")
    traces = query_mcp_observability(
        platform['mcp_endpoint'],
        service="test-service",
        name="function.greet"
    )

    # Verify function span was created
    assert len(traces) > 0, "No function traces found in observability database"

    trace = traces[0]
    assert trace['Service'] == 'test-service'
    assert trace['RootName'] == 'function.greet'
    assert trace['StatusCode'] == 1  # OK status


def test_workflow_with_tools_span_hierarchy(client, worker_process, platform):
    """
    Test that workflow execution with tools creates proper span hierarchy.

    This comprehensive test verifies:
    - Workflow span is created and exported
    - Tool/Agent spans are created as children
    - Parent-child relationships via trace_id
    - All spans have correct status codes
    """
    # Execute workflow that uses tools (from test-service blueprint)
    result = client.run("tool_orchestrated_workflow", {
        "numbers": [1.0, 2.0, 3.0, 4.0, 5.0],
        "query": "test data"
    })

    assert result is not None

    # Wait for spans to be exported and persisted
    time.sleep(3)

    # Query traces from observability database via MCP
    # Note: Span names are prefixed with component type (e.g., "workflow.tool_orchestrated_workflow")
    traces = query_mcp_observability(
        platform['mcp_endpoint'],
        service="test-service",
        name="workflow.tool_orchestrated_workflow"
    )

    # Verify workflow trace was created
    assert len(traces) > 0, "No workflow trace found in observability database"

    workflow_trace = traces[0]
    assert workflow_trace['Service'] == 'test-service'
    assert workflow_trace['RootName'] == 'workflow.tool_orchestrated_workflow'
    assert workflow_trace['StatusCode'] == 1  # OK status

    # Verify trace has multiple spans (workflow + children)
    assert workflow_trace['SpanCount'] >= 2, f"Expected at least 2 spans, found {workflow_trace['SpanCount']}"

    # Get full trace details to verify hierarchy
    trace_id = workflow_trace['TraceID']
    spans = get_trace_spans(platform['mcp_endpoint'], trace_id)

    # Verify we have parent-child relationships
    root_span = None
    child_spans = []

    for span in spans:
        if span['ParentSpanID'] == '' or span['ParentSpanID'] is None:
            root_span = span
        else:
            child_spans.append(span)

    assert root_span is not None, "No root span found"
    assert len(child_spans) > 0, "No child spans found"
    assert root_span['Name'] == 'workflow.tool_orchestrated_workflow', "Root span is not the workflow"

    # Verify all spans completed successfully
    for span in spans:
        assert span['StatusCode'] == 1, f"Span {span['Name']} failed with status {span['StatusCode']}"

    print(f"✅ Trace verification complete:")
    print(f"   Root span: {root_span['Name']}")
    print(f"   Child spans: {len(child_spans)}")
    for child in child_spans:
        print(f"     - {child['Name']}")
